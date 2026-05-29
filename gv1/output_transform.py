"""D9.5.1 output transforms for GV1 conditioned effective-SPM PINN.

D9.5.1 keeps the D9.3 regime-aware voltage channels active and leaves branch
selection to auditable presets in ``gv1.profile_adaptive``.  Unlike D9.4/D9.4.1,
a low-rate 2C profile is no longer forced into a D9.2-like smooth branch.  The
main D9.5.1 improvement is in the loss: trend-preserving correlation plus rare-tail
coverage terms.

This is still a smoke/training-layer transform for one-profile diagnostics. It
is not yet a chemistry-specific NCM/graphite OCP implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass
class OutputTransformConfig:
    """Numerical constants for the GV1 effective-SPM output map."""

    nominal_capacity_Ah: float = 2.0
    voltage_min_V: float = 2.5
    voltage_max_V: float = 4.2
    current_scale_A: float = 1.0

    voltage_center_V: float = 3.6
    voltage_span_V: float = 1.7
    voltage_std_V: float = 0.35

    voltage_range_strategy: str = "profile_minmax"  # profile_minmax | percentile | fixed
    voltage_low_percentile: float = 0.1
    voltage_high_percentile: float = 99.9
    voltage_margin_V: float = 0.03
    voltage_floor_V: float = 2.35
    voltage_ceil_V: float = 4.35

    enable_voltage_hard_clamp: bool = False
    voltage_guard_low_V: float = 2.30
    voltage_guard_high_V: float = 4.40

    theta_a_init: float = 0.08
    theta_c_init: float = 0.92
    theta_a_min: float = 0.02
    theta_a_max: float = 0.98
    theta_c_min: float = 0.02
    theta_c_max: float = 0.98
    stoich_swing_a: float = 0.90
    stoich_swing_c: float = 0.90

    radial_scale_a: float = 0.035
    radial_scale_c: float = 0.050

    resistance_ohm: float = 0.035
    phie_current_scale_V: float = 0.020
    phie_correction_scale_V: float = 0.080

    phis_c_head_mode: str = "linear"  # linear | tanh | softsign
    phis_c_direct_scale: float = 0.52
    phis_c_correction_scale_V: float = 0.20
    ocv_baseline_mix: float = 0.18
    direct_voltage_mix: float = 0.82
    ohmic_mix: float = 1.0

    # D9.5.1 trend-first warmup gate. ``profile_event_gate`` is normally kept
    # near 1.0 so the D9.3 event/low-tail channels remain available; presets
    # may still lower it for ablations. ``profile_dynamic_event_gate`` lets
    # strong instantaneous current events add extra event-channel strength.
    profile_adaptive_mode: str = "off"
    profile_event_gate: float = 1.0
    profile_dynamic_event_gate: float = 0.0

    # D9.3/D9.5.1 regime-aware corrections. These are gates, not hard rules: the raw
    # correction sign is learned by the net, while the gate focuses capacity on
    # regimes that are hard to learn from uniform batches.
    low_voltage_gate_center_V: float = 3.08
    low_voltage_gate_width_V: float = 0.18
    phis_c_low_tail_scale_V: float = 0.85
    phis_c_event_scale_V: float = 0.24
    event_current_gain: float = 0.45
    temperature_polarization_scale_V: float = 0.035

    surface_flux_gain: float = 0.10
    eps: float = 1e-8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "OutputTransformConfig":
        if not data:
            return cls()
        valid = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def cumulative_trapezoid_numpy(t_s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with ``out[0] == 0``."""
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    v = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(t) != len(v):
        raise ValueError(f"t_s and y must have same length, got {len(t)} and {len(v)}")
    out = np.zeros_like(t, dtype=np.float64)
    if len(t) <= 1:
        return out
    dt = np.diff(t)
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    out[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)
    return out


def compute_cbar_baselines_numpy(
    t_s: np.ndarray,
    current_A: np.ndarray,
    cfg: OutputTransformConfig,
) -> dict[str, np.ndarray]:
    """Compute normalized anode/cathode cbar replay baselines from I(t).

    Current convention follows D8/XJTU: charge current is positive, discharge is
    negative.  Therefore anode stoichiometry increases during charge and cathode
    stoichiometry decreases during charge.
    """
    q_net_Ah = cumulative_trapezoid_numpy(t_s, current_A) / 3600.0
    cap = max(float(cfg.nominal_capacity_Ah), float(cfg.eps))
    dq_norm = q_net_Ah / cap
    theta_a = float(cfg.theta_a_init) + float(cfg.stoich_swing_a) * dq_norm
    theta_c = float(cfg.theta_c_init) - float(cfg.stoich_swing_c) * dq_norm
    theta_a = np.clip(theta_a, float(cfg.theta_a_min), float(cfg.theta_a_max))
    theta_c = np.clip(theta_c, float(cfg.theta_c_min), float(cfg.theta_c_max))
    return {
        "q_net_Ah_replay": q_net_Ah.astype(np.float32),
        "cbar_a_norm_replay": theta_a.astype(np.float32),
        "cbar_c_norm_replay": theta_c.astype(np.float32),
    }


def spherical_zero_mean_basis(rho: torch.Tensor) -> torch.Tensor:
    """Low-order basis with zero spherical average for constant coefficient."""
    return rho.pow(2) - 0.6


def smoothstep01(x: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(x, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def softsign(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.abs(x))


def _raw_or_zeros(raw: Mapping[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor:
    val = raw.get(key)
    if val is None:
        return torch.zeros_like(ref)
    return val


class GV1OutputTransform:
    """Apply effective-SPM structure and D9.3 regime-aware voltage map."""

    def __init__(self, config: OutputTransformConfig | Mapping[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, OutputTransformConfig) else OutputTransformConfig.from_mapping(config)

    def to_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    def ocv_baseline(self, cbar_a: torch.Tensor, cbar_c: torch.Tensor | None = None) -> torch.Tensor:
        """Smooth capacity-normalized OCV-like baseline.

        The D9.3 low-tail gate uses this weak baseline to identify deep-voltage
        regimes without using the target voltage during inference.
        """
        cfg = self.config
        soc = (cbar_a - cfg.theta_a_min) / max(cfg.theta_a_max - cfg.theta_a_min, cfg.eps)
        soc = smoothstep01(soc)
        return float(cfg.voltage_min_V) + (float(cfg.voltage_max_V) - float(cfg.voltage_min_V)) * soc

    def _voltage_head(self, raw_v: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        mode = str(cfg.phis_c_head_mode).lower().strip()
        if mode == "tanh":
            z = torch.tanh(raw_v)
        elif mode == "softsign":
            z = softsign(raw_v)
        elif mode == "linear":
            z = raw_v
        else:
            raise ValueError(f"Unsupported phis_c_head_mode: {cfg.phis_c_head_mode!r}")
        span = max(float(cfg.voltage_span_V), float(cfg.voltage_max_V) - float(cfg.voltage_min_V), 0.5)
        return float(cfg.voltage_center_V) + float(cfg.phis_c_direct_scale) * span * z

    def _low_voltage_gate(self, v_ocv: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        width = max(float(cfg.low_voltage_gate_width_V), 1e-3)
        return torch.sigmoid((float(cfg.low_voltage_gate_center_V) - v_ocv) / width)

    def __call__(
        self,
        raw: Mapping[str, torch.Tensor],
        *,
        r_norm: torch.Tensor,
        current_A: torch.Tensor,
        current_norm: torch.Tensor,
        cbar_a: torch.Tensor,
        cbar_c: torch.Tensor,
        temperature_norm: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rho = r_norm.reshape(-1, 1)
        cur_A = current_A.reshape(-1, 1)
        cur_norm = current_norm.reshape(-1, 1)
        temp_norm = torch.zeros_like(cur_norm) if temperature_norm is None else temperature_norm.reshape(-1, 1)
        cba = cbar_a.reshape(-1, 1)
        cbc = cbar_c.reshape(-1, 1)
        basis = spherical_zero_mean_basis(torch.clamp(rho, 0.0, 1.0))

        theta_a = cba + float(self.config.radial_scale_a) * torch.tanh(raw["raw_a"]) * basis
        theta_c = cbc + float(self.config.radial_scale_c) * torch.tanh(raw["raw_c"]) * basis
        theta_a = torch.clamp(theta_a, float(self.config.theta_a_min), float(self.config.theta_a_max))
        theta_c = torch.clamp(theta_c, float(self.config.theta_c_min), float(self.config.theta_c_max))

        phie = (
            float(self.config.phie_current_scale_V) * cur_norm
            + float(self.config.phie_correction_scale_V) * torch.tanh(raw["raw_phie"])
        )

        v_ocv = self.ocv_baseline(cba, cbc)
        v_direct = self._voltage_head(raw["raw_phis_c"])
        v_ohmic = float(self.config.ohmic_mix) * float(self.config.resistance_ohm) * cur_A
        v_correction = float(self.config.phis_c_correction_scale_V) * softsign(raw["raw_phis_c"])

        # D9.3 regime channels.
        raw_low = _raw_or_zeros(raw, "raw_voltage_low", raw["raw_phis_c"])
        raw_event = _raw_or_zeros(raw, "raw_voltage_event", raw["raw_phis_c"])
        low_gate = self._low_voltage_gate(v_ocv)
        current_event_gate = torch.clamp(torch.abs(cur_norm), 0.0, 1.5)
        v_low_tail = low_gate * float(self.config.phis_c_low_tail_scale_V) * softsign(raw_low)
        v_event = (
            float(self.config.phis_c_event_scale_V)
            * (0.25 + float(self.config.event_current_gain) * current_event_gate)
            * softsign(raw_event)
        )
        v_temperature = float(self.config.temperature_polarization_scale_V) * temp_norm * cur_norm

        mix_ocv = float(self.config.ocv_baseline_mix)
        mix_direct = float(self.config.direct_voltage_mix)
        norm = max(mix_ocv + mix_direct, float(self.config.eps))
        v_base = (
            (mix_ocv * v_ocv + mix_direct * v_direct) / norm
            + v_ohmic
            + v_correction
        )
        v_event_total = v_low_tail + v_event + v_temperature

        profile_gate = torch.clamp(
            torch.as_tensor(float(self.config.profile_event_gate), device=v_base.device, dtype=v_base.dtype),
            0.0,
            1.0,
        )
        dynamic_gate = float(self.config.profile_dynamic_event_gate) * torch.clamp(current_event_gate, 0.0, 1.0)
        hybrid_gate = torch.clamp(profile_gate + dynamic_gate, 0.0, 1.0)
        v_low_tail_scaled = hybrid_gate * v_low_tail
        v_event_scaled = hybrid_gate * v_event
        v_temperature_scaled = hybrid_gate * v_temperature
        phis_c = v_base + v_low_tail_scaled + v_event_scaled + v_temperature_scaled

        if bool(self.config.enable_voltage_hard_clamp):
            phis_c = torch.clamp(
                phis_c,
                float(self.config.voltage_guard_low_V),
                float(self.config.voltage_guard_high_V),
            )

        return {
            "theta_a": theta_a,
            "theta_c": theta_c,
            "cs_a": theta_a,
            "cs_c": theta_c,
            "phie": phie,
            "phis_c": phis_c,
            "cbar_a_norm_replay": cba,
            "cbar_c_norm_replay": cbc,
            "voltage_exp_pred": phis_c,
            "voltage_ocv_baseline": v_ocv,
            "voltage_direct_head": v_direct,
            "voltage_ohmic_baseline": v_ohmic,
            "voltage_softsign_correction": v_correction,
            "voltage_low_tail_correction": v_low_tail_scaled,
            "voltage_event_correction": v_event_scaled,
            "voltage_temperature_correction": v_temperature_scaled,
            "voltage_low_gate": low_gate,
            "voltage_current_event_gate": current_event_gate,
            "voltage_profile_event_gate": hybrid_gate,
            "voltage_base_branch": v_base,
            "voltage_event_branch_delta": v_event_total,
        }

    def surface_flux_targets(
        self,
        current_norm: torch.Tensor,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lightweight normalized surface-gradient targets."""
        cur = current_norm.reshape(-1, 1).to(device=ref.device, dtype=ref.dtype)
        gain = float(self.config.surface_flux_gain)
        return gain * cur, -gain * cur
