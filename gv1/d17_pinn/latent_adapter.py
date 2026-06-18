# -*- coding: utf-8 -*-
"""Observed-only profile latent adapter for D17-P2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch import nn


@dataclass
class LatentBounds:
    theta_a0_min: float = 0.05
    theta_a0_max: float = 0.95
    theta_c0_min: float = 0.05
    theta_c0_max: float = 0.95
    qeff_min: float = 0.6
    qeff_max: float = 1.2
    # Center values make raw=0 decode to the generator/prior nominal choice.
    # This avoids starting every profile from the midpoint of a broad range.
    theta_a0_init: float = 0.72
    theta_c0_init: float = 0.48
    qeff_init: float = 1.0
    Rohm_min: float = 0.001
    Rohm_max: float = 0.120
    Rohm_init: float = 0.035
    bV_abs_max: float = 0.20
    bV_init_V: float = 0.0
    Ds_scale_min: float = 0.2
    Ds_scale_max: float = 5.0
    i0_scale_min: float = 0.2
    i0_scale_max: float = 5.0
    gauge_abs_max_V: float = 0.10
    residual_abs_max_V: float = 0.05
    ocp_phase_abs_max: float = 0.08


LATENT_NAMES = (
    "theta_a0",
    "theta_c0",
    "qeff_scale",
    "Rohm_Ohm",
    "bV_V",
    "Ds_scale_a",
    "Ds_scale_c",
    "i0_scale_a",
    "i0_scale_c",
    "gauge_shift_V",
    "low_residual_coeff_V",
    "ocp_phase_a",
    "ocp_phase_c",
)


def _safe_array(profile: Mapping[str, object], keys: Iterable[str], default: float = 0.0) -> np.ndarray:
    for k in keys:
        if k in profile:
            arr = np.asarray(profile[k], dtype=np.float64).reshape(-1)
            if arr.size:
                return arr
    return np.asarray([default], dtype=np.float64)


def observed_profile_features(profile: Mapping[str, object], max_initial_points: int = 256) -> np.ndarray:
    """Extract observed-only scalar features from I,V,T/time.

    No cs/theta/phie/phis keys are read here.  These features intentionally use
    only quantities that are measured in a real profile.
    """
    t = _safe_array(profile, ["t_global_s", "time_s"], 0.0)
    I = _safe_array(profile, ["I_profile", "current_A"], 0.0)
    V = _safe_array(profile, ["voltage_exp"], 3.6)
    T = _safe_array(profile, ["temperature_C"], 25.0)
    n = int(min(len(t), len(I), len(V), len(T)))
    t, I, V, T = t[:n], I[:n], V[:n], T[:n]
    if n == 0:
        raise ValueError("empty observed profile")
    dt = np.diff(t, prepend=t[0])
    dt = np.maximum(dt, 0.0)
    q_Ah = np.cumsum(I * dt) / 3600.0
    q_abs_Ah = np.cumsum(np.abs(I) * dt) / 3600.0
    k = max(2, min(max_initial_points, n))
    dV_early = (V[k - 1] - V[0]) / max(float(t[k - 1] - t[0]), 1.0)
    rest_fraction = float(np.mean(np.abs(I) < max(1.0e-6, 0.01 * np.nanmax(np.abs(I))))) if np.nanmax(np.abs(I)) > 0 else 1.0
    charge_fraction = float(np.mean(I > 0.0))
    discharge_fraction = float(np.mean(I < 0.0))
    features = np.asarray([
        float(V[0]), float(V[-1]), float(np.nanmin(V)), float(np.nanmax(V)), float(np.nanmean(V)), float(np.nanstd(V) + 1e-6),
        float(I[0]), float(I[-1]), float(np.nanmean(I)), float(np.nanstd(I) + 1e-9), float(np.nanmax(I)), float(np.nanmin(I)),
        float(q_Ah[-1]), float(q_abs_Ah[-1]), float(np.nanmax(q_Ah) - np.nanmin(q_Ah)),
        float(T[0]), float(np.nanmean(T)), float(np.nanstd(T) + 1e-6),
        float(t[-1] - t[0]), float(dV_early), rest_fraction, charge_fraction, discharge_fraction,
    ], dtype=np.float32)
    # Normalize roughly into numerically friendly ranges without using dataset-wide statistics.
    scale = np.asarray([
        4.2,4.2,4.2,4.2,4.2,0.5,
        5.0,5.0,5.0,5.0,5.0,5.0,
        2.5,5.0,2.5,
        40.0,40.0,10.0,
        2.0e5,1.0e-3,1.0,1.0,1.0,
    ], dtype=np.float32)
    return features / scale


class ProfileLatentAdapter(nn.Module):
    """Small observed-only MLP producing bounded profile latent variables."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, bounds: Optional[LatentBounds] = None) -> None:
        super().__init__()
        self.bounds = bounds or LatentBounds()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, len(LATENT_NAMES)),
        )
        # Start from conservative prior values rather than random extremes.
        nn.init.zeros_(self.net[-1].weight)  # type: ignore[index]
        nn.init.zeros_(self.net[-1].bias)    # type: ignore[index]

    @staticmethod
    def _range(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    @staticmethod
    def _range_centered(raw: torch.Tensor, lo: float, hi: float, center: float) -> torch.Tensor:
        """Map raw=0 to a prior center while preserving finite lower/upper bounds."""
        lo_f, hi_f = float(lo), float(hi)
        if not (hi_f > lo_f):
            return torch.full_like(raw, lo_f)
        c = min(max(float(center), lo_f + 1.0e-6 * (hi_f - lo_f)), hi_f - 1.0e-6 * (hi_f - lo_f))
        # logit of normalized center; raw offset then moves around the prior.
        p = (c - lo_f) / (hi_f - lo_f)
        center_logit = torch.logit(torch.as_tensor(p, device=raw.device, dtype=raw.dtype))
        return lo_f + (hi_f - lo_f) * torch.sigmoid(center_logit + raw)

    @staticmethod
    def _symmetric(raw: torch.Tensor, amp: float) -> torch.Tensor:
        return amp * torch.tanh(raw)

    @staticmethod
    def _symmetric_centered(raw: torch.Tensor, amp: float, center: float) -> torch.Tensor:
        # Centered bounded latent used by P3.4 resolved-spec alignment.  This
        # keeps raw=0 at the voltage-only fitted offset while retaining a finite
        # local adaptation band.  It uses observed V(t) only, not state labels.
        return float(center) + float(amp) * torch.tanh(raw)

    def decode_raw(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply D17 bounded transforms to raw profile latent values."""
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        b = self.bounds
        out: Dict[str, torch.Tensor] = {}
        out["theta_a0"] = self._range_centered(raw[:, 0], b.theta_a0_min, b.theta_a0_max, b.theta_a0_init)
        out["theta_c0"] = self._range_centered(raw[:, 1], b.theta_c0_min, b.theta_c0_max, b.theta_c0_init)
        out["qeff_scale"] = self._range_centered(raw[:, 2], b.qeff_min, b.qeff_max, b.qeff_init)
        out["Rohm_Ohm"] = self._range_centered(raw[:, 3], b.Rohm_min, b.Rohm_max, b.Rohm_init)
        out["bV_V"] = self._symmetric_centered(raw[:, 4], b.bV_abs_max, b.bV_init_V)
        out["Ds_scale_a"] = self._range(raw[:, 5], b.Ds_scale_min, b.Ds_scale_max)
        out["Ds_scale_c"] = self._range(raw[:, 6], b.Ds_scale_min, b.Ds_scale_max)
        out["i0_scale_a"] = self._range(raw[:, 7], b.i0_scale_min, b.i0_scale_max)
        out["i0_scale_c"] = self._range(raw[:, 8], b.i0_scale_min, b.i0_scale_max)
        out["gauge_shift_V"] = self._symmetric(raw[:, 9], b.gauge_abs_max_V)
        out["low_residual_coeff_V"] = self._symmetric(raw[:, 10], b.residual_abs_max_V)
        out["ocp_phase_a"] = self._symmetric(raw[:, 11], b.ocp_phase_abs_max)
        out["ocp_phase_c"] = self._symmetric(raw[:, 12], b.ocp_phase_abs_max)
        out["raw_latent"] = raw
        return out

    def forward(self, features: torch.Tensor, raw_offset: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Map observed-only features plus optional profile offset to bounded latents.

        `raw_offset` is used by D17-P3 for profile-wise inverse adaptation.  It is
        optimized only through observed voltage and physics losses, not through
        state soft labels.
        """
        raw = self.net(features)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw_offset is not None:
            if raw_offset.ndim == 1:
                raw_offset = raw_offset.unsqueeze(0)
            if raw_offset.shape[-1] != raw.shape[-1]:
                raise ValueError(f"latent raw_offset dim {raw_offset.shape[-1]} != {raw.shape[-1]}")
            raw = raw + raw_offset.to(device=raw.device, dtype=raw.dtype)
        return self.decode_raw(raw)
