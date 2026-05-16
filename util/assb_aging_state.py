# -*- coding: utf-8 -*-
"""
Aging-state parameterization for ASSB ModelFin_109.

The purpose of this module is to introduce low-dimensional, monotone aging
states that can later be coupled to the effective SPM.  It does not solve any
SPM equation.  SOH is derived from the mechanism variables instead of being an
independent neural-network output.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple, Union

import math

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore


@dataclass
class AgingConfig:
    feature_dim: int = 7
    lam_max: float = 0.55
    r_ohm0: float = 105.0
    r_growth_max: float = 300.0
    theta_window0: float = 0.402
    theta_window_shrink_max: float = 0.45
    residual_gauge_max_v: float = 0.010
    dz_floor: float = 1.0e-7
    init_rate_logit: float = -3.0
    dtype: str = "float64"

    @classmethod
    def from_params(cls, params: Dict[str, object]) -> "AgingConfig":
        def get_float(*names: str, default: float) -> float:
            for name in names:
                if name in params and params[name] is not None:
                    try:
                        return float(params[name])
                    except Exception:
                        pass
            return float(default)

        def get_int(*names: str, default: int) -> int:
            for name in names:
                if name in params and params[name] is not None:
                    try:
                        return int(float(params[name]))
                    except Exception:
                        pass
            return int(default)

        theta_bottom = get_float("theta_c_bottom", "THETA_C_BOTTOM", default=0.834)
        theta_top = get_float("theta_c_top", "THETA_C_TOP", default=0.432)
        w0 = abs(theta_bottom - theta_top)
        if w0 <= 0:
            w0 = get_float("AGING_THETA_WINDOW0", default=0.402)
        return cls(
            feature_dim=get_int("AGING_FEATURE_DIM", default=7),
            lam_max=get_float("AGING_LAM_MAX", default=0.55),
            r_ohm0=get_float("AGING_R_OHM0", "R_ohm_eff", "R_OHM_EFF", default=105.0),
            r_growth_max=get_float("AGING_R_GROWTH_MAX", default=300.0),
            theta_window0=get_float("AGING_THETA_WINDOW0", default=w0),
            theta_window_shrink_max=get_float("AGING_THETA_WINDOW_SHRINK_MAX", default=0.45),
            residual_gauge_max_v=get_float("AGING_RESIDUAL_GAUGE_MAX_V", default=0.010),
            dz_floor=get_float("AGING_DZ_FLOOR", default=1.0e-7),
            init_rate_logit=get_float("AGING_INIT_RATE_LOGIT", default=-3.0),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class AgingProfiles:
    cycle_id: Tensor
    z: Tensor
    z_norm: Tensor
    f_lam_c: Tensor
    r_ohm: Tensor
    theta_window_c: Tensor
    soh_mech: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "cycle_id": self.cycle_id,
            "z": self.z,
            "z_norm": self.z_norm,
            "f_lam_c": self.f_lam_c,
            "R_ohm": self.r_ohm,
            "theta_window_c": self.theta_window_c,
            "SOH_mech": self.soh_mech,
        }


class AgingMechanismHead(torch.nn.Module if torch is not None else object):
    """Low-dimensional monotone aging mechanism head.

    The head maps cycle-level protocol/history features to a nonnegative aging
    increment dz(k).  Cumulative z(k) then drives:
    - f_LAM_c(k): effective positive-electrode active volume factor.
    - R_ohm(k): slowly increasing ASSB interfacial/solid-network resistance.
    - theta_window_c(k): slowly shrinking positive-electrode usable window.
    - SOH_mech(k): derived mechanism SOH, not an independent branch.
    """

    def __init__(self, cfg: Union[AgingConfig, Dict[str, object], None] = None):
        if torch is None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for AgingMechanismHead.")
        super().__init__()
        if cfg is None:
            cfg = AgingConfig()
        elif isinstance(cfg, dict):
            cfg = AgingConfig.from_params(cfg)
        self.cfg = cfg
        self.feature_dim = int(cfg.feature_dim)

        # The dz model is deliberately small.  softplus(linear features - bias)
        # provides a nonnegative increment without making the mapping a large
        # black-box curve fitter.
        self.raw_dz_bias = torch.nn.Parameter(torch.tensor(float(cfg.init_rate_logit), dtype=torch.float64))
        self.raw_dz_weights = torch.nn.Parameter(torch.zeros(self.feature_dim, dtype=torch.float64))

        # Positive rates for three mechanism mappings, initialized near slow aging.
        self.raw_a_lam = torch.nn.Parameter(torch.tensor(float(cfg.init_rate_logit), dtype=torch.float64))
        self.raw_a_r = torch.nn.Parameter(torch.tensor(float(cfg.init_rate_logit), dtype=torch.float64))
        self.raw_a_w = torch.nn.Parameter(torch.tensor(float(cfg.init_rate_logit), dtype=torch.float64))

    def extra_repr(self) -> str:
        return (
            f"feature_dim={self.feature_dim}, lam_max={self.cfg.lam_max}, "
            f"r0={self.cfg.r_ohm0}, r_growth_max={self.cfg.r_growth_max}, "
            f"theta_window0={self.cfg.theta_window0}"
        )

    def _match_feature_dim(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[-1] == self.feature_dim:
            return x
        if x.shape[-1] > self.feature_dim:
            return x[..., : self.feature_dim]
        pad = torch.zeros(*x.shape[:-1], self.feature_dim - x.shape[-1], dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)

    def forward(self, cycle_features: Tensor, cycle_id: Optional[Tensor] = None) -> AgingProfiles:
        if not torch.is_tensor(cycle_features):
            cycle_features = torch.as_tensor(cycle_features, dtype=torch.float64)
        dtype = cycle_features.dtype
        device = cycle_features.device
        x = self._match_feature_dim(cycle_features.to(dtype=torch.float64))
        weights = self.raw_dz_weights.to(device=device, dtype=torch.float64)
        bias = self.raw_dz_bias.to(device=device, dtype=torch.float64)

        logits = bias + torch.sum(x * weights, dim=-1)
        dz = torch.nn.functional.softplus(logits) + float(self.cfg.dz_floor)
        z = torch.cumsum(dz, dim=0)
        if z.numel() > 0:
            z = z - z[0]
        denom = torch.clamp(z[-1].detach() if z.numel() else torch.tensor(1.0, dtype=torch.float64, device=device), min=1.0e-12)
        z_norm = z / denom

        a_lam = torch.nn.functional.softplus(self.raw_a_lam.to(device=device, dtype=torch.float64))
        a_r = torch.nn.functional.softplus(self.raw_a_r.to(device=device, dtype=torch.float64))
        a_w = torch.nn.functional.softplus(self.raw_a_w.to(device=device, dtype=torch.float64))

        lam_max = torch.as_tensor(float(self.cfg.lam_max), dtype=torch.float64, device=device)
        r0 = torch.as_tensor(float(self.cfg.r_ohm0), dtype=torch.float64, device=device)
        r_growth = torch.as_tensor(float(self.cfg.r_growth_max), dtype=torch.float64, device=device)
        w0 = torch.as_tensor(float(self.cfg.theta_window0), dtype=torch.float64, device=device)
        w_shrink = torch.as_tensor(float(self.cfg.theta_window_shrink_max), dtype=torch.float64, device=device)

        f_lam = 1.0 - lam_max * (1.0 - torch.exp(-a_lam * z_norm))
        f_lam = torch.clamp(f_lam, min=1.0 - float(self.cfg.lam_max), max=1.0)
        r_ohm = r0 + r_growth * (1.0 - torch.exp(-a_r * z_norm))
        theta_window = w0 * (1.0 - w_shrink * (1.0 - torch.exp(-a_w * z_norm)))
        theta_window = torch.clamp(theta_window, min=w0 * (1.0 - float(self.cfg.theta_window_shrink_max)), max=w0)
        soh = f_lam * theta_window / torch.clamp(w0, min=1.0e-12)

        if cycle_id is None:
            cycle_id = torch.arange(cycle_features.shape[0], dtype=torch.long, device=device)
        else:
            cycle_id = cycle_id.to(device=device, dtype=torch.long)

        # Return tensors in the same dtype as the input unless the input was not floating.
        out_dtype = dtype if dtype.is_floating_point else torch.float64
        return AgingProfiles(
            cycle_id=cycle_id,
            z=z.to(dtype=out_dtype),
            z_norm=z_norm.to(dtype=out_dtype),
            f_lam_c=f_lam.to(dtype=out_dtype),
            r_ohm=r_ohm.to(dtype=out_dtype),
            theta_window_c=theta_window.to(dtype=out_dtype),
            soh_mech=soh.to(dtype=out_dtype),
        )


def build_aging_profiles(head_or_nn, params: Dict[str, object], cycle_features: Optional[Tensor] = None) -> AgingProfiles:
    """Build aging profiles from an AgingMechanismHead or a model containing one."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for build_aging_profiles().")
    from util.assb_cycle_table import cycle_features_from_params, _get_cycle_tensors

    head = head_or_nn
    if not isinstance(head, AgingMechanismHead):
        if hasattr(head_or_nn, "aging_head"):
            head = head_or_nn.aging_head
        else:
            raise AttributeError("Object does not contain an aging_head.")
    if cycle_features is None:
        device = next(head.parameters()).device
        dtype = next(head.parameters()).dtype
        cycle_features = cycle_features_from_params(params, device=device, dtype=dtype)
    table = _get_cycle_tensors(params, device=cycle_features.device, dtype=cycle_features.dtype)
    return head(cycle_features, cycle_id=table["cycle_id"])


def aging_profiles_to_numpy(profiles: AgingProfiles) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in profiles.as_dict().items():
        if hasattr(value, "detach"):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = value
    return out


def monotone_smoothness_losses(profiles: AgingProfiles) -> Dict[str, Tensor]:
    """Small mechanism priors used by the later _losses.py patch."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for monotone_smoothness_losses().")
    f = profiles.f_lam_c
    r = profiles.r_ohm
    w = profiles.theta_window_c
    losses: Dict[str, Tensor] = {}
    if f.numel() <= 1:
        zero = torch.zeros((), dtype=f.dtype, device=f.device)
        return {"mono": zero, "smooth": zero, "bounds": zero}
    df = f[1:] - f[:-1]
    dr = r[1:] - r[:-1]
    dw = w[1:] - w[:-1]
    # f_lam and window should not increase; R should not decrease.
    mono = torch.mean(torch.relu(df) ** 2) + torch.mean(torch.relu(-dr) ** 2) + torch.mean(torch.relu(dw) ** 2)
    if f.numel() > 2:
        smooth = (
            torch.mean((f[2:] - 2.0 * f[1:-1] + f[:-2]) ** 2)
            + torch.mean((r[2:] - 2.0 * r[1:-1] + r[:-2]) ** 2)
            + torch.mean((w[2:] - 2.0 * w[1:-1] + w[:-2]) ** 2)
        )
    else:
        smooth = torch.zeros((), dtype=f.dtype, device=f.device)
    bounds = torch.mean(torch.relu(0.0 - f) ** 2 + torch.relu(f - 1.0) ** 2)
    losses["mono"] = mono
    losses["smooth"] = smooth
    losses["bounds"] = bounds
    return losses
