# -*- coding: utf-8 -*-
"""D17-P2 differentiable OCP/BV/Ohmic/gauge closure."""

from __future__ import annotations

from typing import Dict

import torch

from .p2dlite_prior import D17P2DlitePrior, FARADAY_C_PER_MOL, R_GAS_J_PER_MOL_K
from .torch_ops import interp1d_torch


def _tensor(x, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


class OCPTable:
    def __init__(self, theta, U, device=None, dtype=torch.float32) -> None:
        th = torch.as_tensor(theta, dtype=dtype, device=device).flatten()
        uu = torch.as_tensor(U, dtype=dtype, device=device).flatten()
        order = torch.argsort(th)
        self.theta = th[order]
        self.U = uu[order]

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return interp1d_torch(theta, self.theta, self.U)


class ElectrochemicalClosure(torch.nn.Module):
    """Convert predicted cs surface into theta/OCP/eta/V/phie/phis_c."""

    def __init__(self, prior: D17P2DlitePrior) -> None:
        super().__init__()
        self.prior = prior
        self.register_buffer("ocp_p_theta", torch.tensor(prior.ocp_positive_theta, dtype=torch.float32))
        self.register_buffer("ocp_p_U", torch.tensor(prior.ocp_positive_U, dtype=torch.float32))
        self.register_buffer("ocp_n_theta", torch.tensor(prior.ocp_negative_theta, dtype=torch.float32))
        self.register_buffer("ocp_n_U", torch.tensor(prior.ocp_negative_U, dtype=torch.float32))

    def _ocp_pos(self, theta: torch.Tensor) -> torch.Tensor:
        return interp1d_torch(theta, self.ocp_p_theta, self.ocp_p_U)

    def _ocp_neg(self, theta: torch.Tensor) -> torch.Tensor:
        return interp1d_torch(theta, self.ocp_n_theta, self.ocp_n_U)

    @staticmethod
    def _bv_inverse_eta(J_mol_m2_s: torch.Tensor, i0_A_m2: torch.Tensor, T_K: torch.Tensor) -> torch.Tensor:
        # Symmetric Butler-Volmer inverse: i = 2 i0 sinh(F eta / 2RT)
        # J is mol/m^2/s, i = F J.
        i_A_m2 = FARADAY_C_PER_MOL * J_mol_m2_s
        arg = i_A_m2 / torch.clamp(2.0 * i0_A_m2, min=1.0e-9)
        return (2.0 * R_GAS_J_PER_MOL_K * T_K / FARADAY_C_PER_MOL) * torch.asinh(arg)

    def forward(
        self,
        cs_a_surface: torch.Tensor,
        cs_c_surface: torch.Tensor,
        J_a: torch.Tensor,
        J_c: torch.Tensor,
        current_A: torch.Tensor,
        temperature_C: torch.Tensor,
        latent: Dict[str, torch.Tensor],
        low_transition_gate: torch.Tensor | None = None,
        enable_low_transition_residual: bool = False,
    ) -> Dict[str, torch.Tensor]:
        dev = cs_a_surface.device
        dtype = cs_a_surface.dtype
        pa = self.prior.negative
        pc = self.prior.positive
        # Map solid concentration to stoichiometry and apply bounded profile OCP phase shifts.
        theta_a = cs_a_surface / float(pa.csmax_mol_m3) + latent["ocp_phase_a"].reshape(-1)[0]
        theta_c = cs_c_surface / float(pc.csmax_mol_m3) + latent["ocp_phase_c"].reshape(-1)[0]
        theta_a = torch.clamp(theta_a, min=float(pa.theta_min), max=float(pa.theta_max))
        theta_c = torch.clamp(theta_c, min=float(pc.theta_min), max=float(pc.theta_max))
        U_a = self._ocp_neg(theta_a)
        U_c = self._ocp_pos(theta_c)
        T_K = temperature_C + 273.15
        i0_a = torch.clamp(latent["i0_scale_a"].reshape(-1)[0] * float(pa.i0_A_m2), min=1e-9)
        i0_c = torch.clamp(latent["i0_scale_c"].reshape(-1)[0] * float(pc.i0_A_m2), min=1e-9)
        eta_a = self._bv_inverse_eta(J_a, i0_a, T_K)
        eta_c = self._bv_inverse_eta(J_c, i0_c, T_K)
        Rohm = latent["Rohm_Ohm"].reshape(-1)[0]
        bV = latent["bV_V"].reshape(-1)[0]
        V_base = U_c - U_a + eta_c - eta_a + current_A * Rohm + bV
        residual = torch.zeros_like(V_base)
        if enable_low_transition_residual and low_transition_gate is not None:
            residual = low_transition_gate * latent["low_residual_coeff_V"].reshape(-1)[0]
        V_pred = V_base + residual
        # Gauge convention: phi_s,a = 0, terminal voltage ~= phi_s,c.
        phis_c = V_pred
        phie = phis_c - U_c - eta_c + latent["gauge_shift_V"].reshape(-1)[0]
        return {
            "theta_a_surface": theta_a,
            "theta_c_surface": theta_c,
            "U_a": U_a,
            "U_c": U_c,
            "eta_a": eta_a,
            "eta_c": eta_c,
            "V_base": V_base,
            "V_pred": V_pred,
            "V_residual_local": residual,
            "phie": phie,
            "phis_c": phis_c,
            "Rohm_Ohm": Rohm.expand_as(V_pred),
        }
