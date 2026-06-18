# -*- coding: utf-8 -*-
"""D17-P3 mechanism-heavy inverse PINN smoke model.

P3 keeps the P2 no-state-label boundary but adds two generator-like mechanisms:
profile-wise latent offsets optimized only through observed V(t)/physics losses,
and a bounded low/transition voltage residual expert inspired by D12-S1K.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch import nn

from .electrochem_closure import ElectrochemicalClosure
from .latent_adapter import LatentBounds, ProfileLatentAdapter, observed_profile_features
from .p2dlite_prior import D17P2DlitePrior, FARADAY_C_PER_MOL
from .torch_ops import cumulative_trapezoid_torch, radial_volume_weights_torch, zero_volume_mean_project_torch
from .d12_transition_fade import D12TransitionFadeConfig, d12_transition_fade_basis, d12_transition_fade_gates


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 3, final_zero: bool = False) -> nn.Sequential:
    mods = []
    d = in_dim
    for _ in range(max(1, layers - 1)):
        mods.append(nn.Linear(d, hidden))
        mods.append(nn.Tanh())
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*mods)
    if final_zero and isinstance(net[-1], nn.Linear):
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class D17MechanisticPINN(nn.Module):
    def __init__(
        self,
        prior: D17P2DlitePrior,
        feature_dim: int,
        n_r: int = 17,
        hidden_dim: int = 64,
        latent_hidden_dim: int = 64,
        delta_layers: int = 3,
        delta_amp_fraction: float = 0.018,
        enable_low_transition_residual: bool = False,
        use_observed_voltage_for_gate: bool = True,
        enable_voltage_inverse_residual: bool = False,
        voltage_inverse_residual_amp_V: float = 0.12,
        voltage_inverse_residual_gate_mode: str = "low_transition",
        enable_voltage_basis_residual: bool = False,
        voltage_basis_residual_amp_V: float = 0.08,
        voltage_basis_count: int = 9,
        voltage_basis_formula_mode: str = "generic",
        d12_low_v: float = 2.75,
        d12_normal_v: float = 3.05,
        d12_low_width_v: float = 0.055,
        d12_transition_width_v: float = 0.080,
        d12_transition_gain: float = 0.70,
        d12_non_low_preservation_floor: float = 0.02,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.n_r = int(n_r)
        self.delta_amp_fraction = float(delta_amp_fraction)
        self.enable_low_transition_residual = bool(enable_low_transition_residual)
        self.use_observed_voltage_for_gate = bool(use_observed_voltage_for_gate)
        self.enable_voltage_inverse_residual = bool(enable_voltage_inverse_residual)
        self.voltage_inverse_residual_amp_V = float(voltage_inverse_residual_amp_V)
        self.voltage_inverse_residual_gate_mode = str(voltage_inverse_residual_gate_mode or "low_transition")
        self.enable_voltage_basis_residual = bool(enable_voltage_basis_residual)
        self.voltage_basis_residual_amp_V = float(voltage_basis_residual_amp_V)
        self.voltage_basis_count = int(voltage_basis_count)
        self.voltage_basis_formula_mode = str(voltage_basis_formula_mode or "generic")
        self.d12_fade_config = D12TransitionFadeConfig(
            low_v=float(d12_low_v),
            normal_v=float(d12_normal_v),
            low_width_v=float(d12_low_width_v),
            transition_width_v=float(d12_transition_width_v),
            transition_gain=float(d12_transition_gain),
            non_low_preservation_floor=float(d12_non_low_preservation_floor),
        )
        bounds = LatentBounds(
            theta_a0_min=prior.theta0_a_min,
            theta_a0_max=prior.theta0_a_max,
            theta_c0_min=prior.theta0_c_min,
            theta_c0_max=prior.theta0_c_max,
            qeff_min=prior.qeff_scale_min,
            qeff_max=prior.qeff_scale_max,
            theta_a0_init=prior.theta0_a_init,
            theta_c0_init=prior.theta0_c_init,
            qeff_init=prior.qeff_scale_init,
            Rohm_min=0.001,
            Rohm_max=max(0.002, min(0.20, prior.Rohm_Ohm * 4.0)),
            Rohm_init=prior.Rohm_Ohm,
            bV_abs_max=max(0.06, min(0.28, 0.08 + abs(float(prior.voltage_offset_V)))),
            bV_init_V=prior.voltage_offset_V,
            gauge_abs_max_V=prior.gauge_shift_max_V,
            residual_abs_max_V=max(float(prior.residual_coeff_max_V), float(voltage_inverse_residual_amp_V) * 0.5),
            ocp_phase_abs_max=prior.ocp_phase_shift_max,
        )
        self.latent_adapter = ProfileLatentAdapter(feature_dim, hidden_dim=latent_hidden_dim, bounds=bounds)
        # Inputs per grid point: t_norm, r_norm, q_norm, I_norm, V_obs_norm, latent summary 13.
        self.delta_net_a = _mlp(5 + 13, hidden_dim, 1, layers=delta_layers, final_zero=False)
        self.delta_net_c = _mlp(5 + 13, hidden_dim, 1, layers=delta_layers, final_zero=False)
        # Bounded observed-only inverse residual. It cannot freely copy V(t):
        # it is amplitude-limited and can be gated by low/transition, non-rest, or all-time smooth gates.
        # P3.2 also supports an optional low-dimensional smooth voltage-basis residual,
        # optimized only through observed voltage loss and physics audits.
        self.voltage_inverse_net = _mlp(5 + 13, hidden_dim, 1, layers=2, final_zero=True)
        self.closure = ElectrochemicalClosure(prior)

    @staticmethod
    def current_to_flux_torch(current_A: torch.Tensor, R_m: float, eps_s: float, V_m3: float, sign: float, qeff_scale: torch.Tensor) -> torch.Tensor:
        denom = 3.0 * float(eps_s) * FARADAY_C_PER_MOL * float(V_m3) * torch.clamp(qeff_scale, min=1.0e-6)
        return sign * current_A * float(R_m) / denom

    @staticmethod
    def integrate_cbar_torch(t_s: torch.Tensor, J: torch.Tensor, R_m: float, cbar0: torch.Tensor) -> torch.Tensor:
        integral = cumulative_trapezoid_torch(J, t_s)
        return cbar0 - (3.0 / float(R_m)) * integral

    @staticmethod
    def feasible_theta0_from_current(
        theta0: torch.Tensor,
        rel_theta: torch.Tensor,
        theta_min: float,
        theta_max: float,
        margin: float = 2.0e-3,
    ) -> torch.Tensor:
        """Project theta0 into the inventory-feasible interval.

        The generator-consistent cbar trajectory has the form
        theta_bar(t) = theta0 + rel_theta(t), where rel_theta(t) is fixed by
        the measured current integral and qeff.  A fixed theta0 can still make
        part of the trajectory leave the electrode stoichiometry window.  This
        helper performs a hard but differentiable-enough projection of theta0
        before cbar is constructed.  It does not use any state soft label.
        """
        lo = float(theta_min) + float(margin) - torch.min(rel_theta)
        hi = float(theta_max) - float(margin) - torch.max(rel_theta)
        # Normal case: enough room to place the current-integral trajectory.
        theta_clip = torch.clamp(theta0, min=lo, max=hi)
        # Degenerate case: requested current-integral swing is wider than the
        # prior window.  Choose the center; residual scaling below will still
        # keep cs in bounds.  This should be rare and is reported by physics
        # losses rather than hidden by state labels.
        mid = 0.5 * (lo + hi)
        return torch.where(hi >= lo, theta_clip, torch.clamp(mid, min=float(theta_min) + float(margin), max=float(theta_max) - float(margin)))

    @staticmethod
    def scale_zero_mean_delta_to_bounds(
        cbar: torch.Tensor,
        delta: torch.Tensor,
        csmax: float,
        theta_min: float,
        theta_max: float,
        margin: float = 2.0e-3,
    ) -> torch.Tensor:
        """Scale a zero-mean radial residual without changing its mean.

        Clamping cs after adding delta breaks mean(cs)=cbar.  The correct
        generator-consistent operation is to keep delta zero-mean and reduce
        its amplitude only when it would push any radial point outside the
        stoichiometry window.  Multiplication by a scalar per time row preserves
        the zero-volume-mean property exactly up to floating point error.
        """
        lo = (float(theta_min) + float(margin)) * float(csmax)
        hi = (float(theta_max) - float(margin)) * float(csmax)
        cb = cbar.reshape(-1, 1)
        d_min = torch.min(delta, dim=1, keepdim=True).values
        d_max = torch.max(delta, dim=1, keepdim=True).values
        eps = torch.tensor(1.0e-12, device=delta.device, dtype=delta.dtype)
        room_low = torch.clamp(cb - lo, min=0.0)
        room_high = torch.clamp(hi - cb, min=0.0)
        scale_low = torch.where(d_min < 0.0, room_low / torch.clamp(-d_min, min=eps), torch.ones_like(d_min))
        scale_high = torch.where(d_max > 0.0, room_high / torch.clamp(d_max, min=eps), torch.ones_like(d_max))
        scale = torch.minimum(torch.minimum(scale_low, scale_high), torch.ones_like(d_min))
        # small safety margin avoids single-precision audit overshoot
        return delta * torch.clamp(scale * 0.995, min=0.0, max=1.0)

    @staticmethod
    def low_transition_gate(voltage_exp: torch.Tensor, current_A: torch.Tensor) -> torch.Tensor:
        # D12-S1K-inspired smooth low/transition gate, bounded and observed-only.
        v_gate = torch.sigmoid((2.95 - voltage_exp) / 0.08)
        discharge_gate = torch.sigmoid((-current_A) / (0.20 * torch.clamp(torch.max(torch.abs(current_A)), min=1.0e-6)))
        return torch.clamp(v_gate * discharge_gate, 0.0, 1.0)

    def voltage_inverse_gate(self, low_gate: torch.Tensor, current_A: torch.Tensor) -> torch.Tensor:
        """Observed-only gate for bounded inverse-voltage residuals.

        P3.1 accidentally kept the default low-transition gate even when the
        config requested broader recovery.  P3.2 makes the gate explicit.  The
        gate uses only measured current and the observed voltage-derived low
        gate; it never uses cs/theta/phie/phis soft labels.
        """
        mode = self.voltage_inverse_residual_gate_mode.lower().strip()
        if mode in {"low", "low_transition", "d12", "d12_s1k", "d12_transition_fade"}:
            return torch.clamp(low_gate, 0.0, 1.0)
        max_i = torch.clamp(torch.max(torch.abs(current_A)), min=1.0e-6)
        non_rest = torch.sigmoid((torch.abs(current_A) / max_i - 0.025) / 0.020)
        if mode in {"non_rest", "loaded"}:
            return torch.clamp(non_rest, 0.0, 1.0)
        if mode in {"non_rest_plus_transition", "loaded_plus_transition", "p31", "p32"}:
            return torch.clamp(torch.maximum(low_gate, 0.55 * non_rest), 0.0, 1.0)
        if mode in {"all", "all_bounded", "global"}:
            return torch.ones_like(low_gate)
        return torch.clamp(low_gate, 0.0, 1.0)

    @staticmethod
    def voltage_basis_matrix(
        t_norm: torch.Tensor,
        q_norm: torch.Tensor,
        i_norm: torch.Tensor,
        voltage_exp: torch.Tensor,
        low_gate: torch.Tensor,
        inv_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Low-dimensional smooth residual basis for P3.2 voltage recovery.

        This is not a pointwise voltage copy.  Each profile gets a small number
        of coefficients multiplying smooth, observed-only basis functions.  It
        is closer to a D12-S1K-style voltage wrapper embedded in the inverse
        PINN, while hard inventory/zero-mean audits remain active.
        """
        t = torch.clamp(t_norm, 0.0, 1.0)
        q = torch.clamp(q_norm, -1.5, 1.5)
        i = torch.clamp(i_norm, -1.5, 1.5)
        max_i = torch.clamp(torch.max(torch.abs(i_norm)), min=1.0e-6)
        non_rest = torch.sigmoid((torch.abs(i_norm) / max_i - 0.025) / 0.020)
        rest = 1.0 - non_rest
        high_gate = torch.sigmoid((voltage_exp - 4.00) / 0.08)
        mid_gate = torch.clamp(inv_gate - low_gate, 0.0, 1.0)
        cols = [
            torch.ones_like(t),
            2.0 * t - 1.0,
            2.0 * q,
            2.0 * q * q - 1.0,
            i,
            low_gate,
            low_gate * (2.0 * t - 1.0),
            mid_gate,
            high_gate * non_rest,
            rest * (2.0 * t - 1.0),
        ]
        basis = torch.stack(cols, dim=-1)
        # Mild normalization keeps coefficient scale predictable.
        return basis / torch.sqrt(torch.tensor(float(len(cols)), device=basis.device, dtype=basis.dtype))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        t_s = batch["t_s"].float()
        I = batch["current_A"].float()
        V_obs = batch["voltage_exp"].float()
        T_C = batch["temperature_C"].float()
        r_norm = batch["r_norm"].float()
        features = batch["features"].float()
        raw_offset = batch.get("latent_raw_offset")
        device = t_s.device
        pa = self.prior.negative
        pc = self.prior.positive
        latent = self.latent_adapter(features, raw_offset=raw_offset)
        # Use first batch latent; P2 smoke is one profile per forward.
        qeff = latent["qeff_scale"].reshape(-1)[0]
        theta_a0 = latent["theta_a0"].reshape(-1)[0]
        theta_c0 = latent["theta_c0"].reshape(-1)[0]
        J_a = self.current_to_flux_torch(I, pa.particle_radius_m, pa.active_fraction, pa.electrode_volume_m3, pa.current_flux_sign, qeff)
        J_c = self.current_to_flux_torch(I, pc.particle_radius_m, pc.active_fraction, pc.electrode_volume_m3, pc.current_flux_sign, qeff)
        # Current integral gives relative inventory change only.  Before
        # constructing cbar, project theta0 into the feasible interval implied
        # by this observed I(t) segment and the electrode stoichiometry windows.
        rel_a = self.integrate_cbar_torch(t_s, J_a, pa.particle_radius_m, torch.zeros_like(theta_a0)) / float(pa.csmax_mol_m3)
        rel_c = self.integrate_cbar_torch(t_s, J_c, pc.particle_radius_m, torch.zeros_like(theta_c0)) / float(pc.csmax_mol_m3)
        theta_a0_safe = self.feasible_theta0_from_current(theta_a0, rel_a, pa.theta_min, pa.theta_max)
        theta_c0_safe = self.feasible_theta0_from_current(theta_c0, rel_c, pc.theta_min, pc.theta_max)
        cbar_a = (theta_a0_safe + rel_a) * float(pa.csmax_mol_m3)
        cbar_c = (theta_c0_safe + rel_c) * float(pc.csmax_mol_m3)
        # Numerical guard only on cbar, not on full cs.  It protects impossible
        # placeholder priors while keeping mean(cs)=cbar after residual scaling.
        cbar_a = torch.clamp(cbar_a, min=(float(pa.theta_min) + 2e-3) * float(pa.csmax_mol_m3), max=(float(pa.theta_max) - 2e-3) * float(pa.csmax_mol_m3))
        cbar_c = torch.clamp(cbar_c, min=(float(pc.theta_min) + 2e-3) * float(pc.csmax_mol_m3), max=(float(pc.theta_max) - 2e-3) * float(pc.csmax_mol_m3))
        # Build radial residual grid.
        N = t_s.numel()
        R = r_norm.numel()
        t_norm = (t_s - t_s[0]) / torch.clamp(t_s[-1] - t_s[0], min=1.0)
        q_Ah = cumulative_trapezoid_torch(I, t_s) / 3600.0
        q_norm = q_Ah / torch.clamp(torch.max(torch.abs(q_Ah)), min=1.0e-6)
        I_norm = I / torch.clamp(torch.max(torch.abs(I)), min=1.0e-6)
        V_norm = (V_obs - torch.mean(V_obs)) / torch.clamp(torch.std(V_obs), min=1.0e-6)
        z_vec = latent["raw_latent"].reshape(1, -1).expand(N * R, -1)
        grid = torch.stack([
            t_norm[:, None].expand(N, R),
            r_norm[None, :].expand(N, R),
            q_norm[:, None].expand(N, R),
            I_norm[:, None].expand(N, R),
            V_norm[:, None].expand(N, R),
        ], dim=-1).reshape(N * R, 5)
        x = torch.cat([grid, z_vec], dim=-1)
        raw_da = torch.tanh(self.delta_net_a(x)).reshape(N, R)
        raw_dc = torch.tanh(self.delta_net_c(x)).reshape(N, R)
        da = zero_volume_mean_project_torch(raw_da, r_norm)
        dc = zero_volume_mean_project_torch(raw_dc, r_norm)
        # Bound radial residual amplitude; current-dependent modulation encourages gradients during loaded segments.
        current_mod = (0.25 + 0.75 * torch.abs(I_norm)).reshape(N, 1)
        amp_a = self.delta_amp_fraction * float(pa.csmax_mol_m3)
        amp_c = self.delta_amp_fraction * float(pc.csmax_mol_m3)
        # Do NOT clamp the full cs field after adding the zero-mean residual.
        # A post-hoc clamp changes the spherical average and breaks the
        # generator-consistent identity mean(cs)=cbar.  Instead we scale the
        # zero-mean residual row-wise so every radial point remains inside the
        # physical theta window while mean(delta)=0 is preserved.
        delta_a = amp_a * current_mod * da
        delta_c = amp_c * current_mod * dc
        delta_a = self.scale_zero_mean_delta_to_bounds(cbar_a, delta_a, float(pa.csmax_mol_m3), pa.theta_min, pa.theta_max)
        delta_c = self.scale_zero_mean_delta_to_bounds(cbar_c, delta_c, float(pc.csmax_mol_m3), pc.theta_min, pc.theta_max)
        cs_a = cbar_a.reshape(N, 1) + delta_a
        cs_c = cbar_c.reshape(N, 1) + delta_c
        gate = self.low_transition_gate(V_obs, I) if self.use_observed_voltage_for_gate else torch.zeros_like(V_obs)
        close = self.closure(
            cs_a_surface=cs_a[:, -1],
            cs_c_surface=cs_c[:, -1],
            J_a=J_a,
            J_c=J_c,
            current_A=I,
            temperature_C=T_C,
            latent=latent,
            low_transition_gate=gate,
            enable_low_transition_residual=self.enable_low_transition_residual,
        )
        V_forward = close["V_pred"]
        inv_gate = self.voltage_inverse_gate(gate, I)
        if self.enable_voltage_inverse_residual:
            time_grid = torch.stack([t_norm, q_norm, I_norm, V_norm, gate], dim=-1)
            z_time = latent["raw_latent"].reshape(1, -1).expand(N, -1)
            v_res_inv = self.voltage_inverse_residual_amp_V * torch.tanh(
                self.voltage_inverse_net(torch.cat([time_grid, z_time], dim=-1))
            ).reshape(N)
            v_res_inv = torch.clamp(inv_gate, 0.0, 1.0) * v_res_inv
        else:
            v_res_inv = torch.zeros_like(V_forward)
        if self.enable_voltage_basis_residual:
            coeffs = batch.get("voltage_basis_raw_coeffs")
            if self.voltage_basis_formula_mode.lower().strip() in {"d12", "d12_s1k", "d12_transition_fade", "s1k"}:
                basis = d12_transition_fade_basis(t_norm, q_norm, I_norm, V_obs, I, self.d12_fade_config)
            else:
                basis = self.voltage_basis_matrix(t_norm, q_norm, I_norm, V_obs, gate, inv_gate)
            K = int(basis.shape[-1])
            if coeffs is None:
                coeffs_t = torch.zeros(K, device=device, dtype=V_forward.dtype)
            else:
                coeffs_t = coeffs.reshape(-1).to(device=device, dtype=V_forward.dtype)
                if coeffs_t.numel() < K:
                    coeffs_t = torch.cat([coeffs_t, torch.zeros(K - coeffs_t.numel(), device=device, dtype=V_forward.dtype)], dim=0)
                coeffs_t = coeffs_t[:K]
            v_res_basis = self.voltage_basis_residual_amp_V * torch.matmul(basis, torch.tanh(coeffs_t))
        else:
            v_res_basis = torch.zeros_like(V_forward)
        close["V_pred_forward"] = V_forward
        close["V_residual_inverse"] = v_res_inv
        close["V_residual_basis"] = v_res_basis
        close["V_residual_total"] = close["V_residual_local"] + v_res_inv + v_res_basis
        close["V_pred"] = V_forward + v_res_inv + v_res_basis
        if self.voltage_basis_formula_mode.lower().strip() in {"d12", "d12_s1k", "d12_transition_fade", "s1k"}:
            d12_gates = d12_transition_fade_gates(V_obs, I, self.d12_fade_config)
            close["d12_fade_gate"] = d12_gates["fade_gate"]
            close["d12_low_core_gate"] = d12_gates["low_core_gate"]
            close["d12_transition_gate"] = d12_gates["transition_gate"]
            close["d12_preserve_gate"] = d12_gates["preserve_gate"]
        close["phis_c"] = close["V_pred"]
        close["phie"] = close["phis_c"] - close["U_c"] - close["eta_c"] + latent["gauge_shift_V"].reshape(-1)[0]
        w = radial_volume_weights_torch(r_norm).to(device=device, dtype=cs_a.dtype)
        out: Dict[str, torch.Tensor] = {
            "cs_a": cs_a,
            "cs_c": cs_c,
            "theta_a": cs_a / float(pa.csmax_mol_m3),
            "theta_c": cs_c / float(pc.csmax_mol_m3),
            "cbar_a": cbar_a,
            "cbar_c": cbar_c,
            "delta_a": cs_a - torch.sum(cs_a * w.reshape(1, -1), dim=1, keepdim=True),
            "delta_c": cs_c - torch.sum(cs_c * w.reshape(1, -1), dim=1, keepdim=True),
            "J_a": J_a,
            "J_c": J_c,
            "r_m_a": r_norm * float(pa.particle_radius_m),
            "r_m_c": r_norm * float(pc.particle_radius_m),
            "latent_raw": latent["raw_latent"],
            "low_transition_gate": gate,
            "voltage_inverse_gate": inv_gate,
            "theta_a0_safe": theta_a0_safe.reshape(1),
            "theta_c0_safe": theta_c0_safe.reshape(1),
        }
        out.update(close)
        for k, v in latent.items():
            if k != "raw_latent":
                out[f"latent_{k}"] = v
        return out


def make_batch_from_profile(profile: Mapping[str, Any], n_r: int = 17, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    def arr(keys, default=0.0):
        for k in keys:
            if k in profile:
                return np.asarray(profile[k], dtype=np.float32).reshape(-1)
        return np.asarray([default], dtype=np.float32)
    t = arr(["t_global_s", "time_s"])
    I = arr(["I_profile", "current_A"])
    V = arr(["voltage_exp"], 3.6)
    T = arr(["temperature_C"], 25.0)
    n = min(len(t), len(I), len(V), len(T))
    if n < 8:
        raise ValueError("profile too short for D17-P2 smoke")
    t = t[:n]
    # Re-zero time for numerical stability.
    t = t - t[0]
    I, V, T = I[:n], V[:n], T[:n]
    r_norm = np.linspace(0.0, 1.0, int(n_r), dtype=np.float32)
    features = observed_profile_features(profile).reshape(1, -1)
    dev = torch.device(device)
    return {
        "t_s": torch.as_tensor(t, dtype=torch.float32, device=dev),
        "current_A": torch.as_tensor(I, dtype=torch.float32, device=dev),
        "voltage_exp": torch.as_tensor(V, dtype=torch.float32, device=dev),
        "temperature_C": torch.as_tensor(T, dtype=torch.float32, device=dev),
        "r_norm": torch.as_tensor(r_norm, dtype=torch.float32, device=dev),
        "features": torch.as_tensor(features, dtype=torch.float32, device=dev),
    }
