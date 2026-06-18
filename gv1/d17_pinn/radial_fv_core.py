# -*- coding: utf-8 -*-
"""
D17-P1 radial finite-volume / audit helpers.

The main structure is:
    c_s(t, r) = cbar(t) + delta_c(t, r)
with delta_c projected to zero volume mean so it cannot change total inventory.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def radial_grid(n_r: int = 17, eps: float = 0.0) -> np.ndarray:
    if n_r < 3:
        raise ValueError("n_r must be >= 3")
    return np.linspace(eps, 1.0, n_r, dtype=float)


def radial_volume_weights(r_norm: np.ndarray) -> np.ndarray:
    """Return normalized spherical volume weights for a radial grid."""
    r = np.asarray(r_norm, dtype=float)
    if r.ndim != 1:
        raise ValueError("r_norm must be 1D")
    # Use shell-like weights proportional to r^2 dr.
    # For uniform grid, this is adequate for audit/smoke. FV training can refine later.
    w = np.maximum(r, 0.0) ** 2
    if float(np.sum(w)) <= 0:
        w = np.ones_like(r)
    return w / np.sum(w)


def zero_volume_mean_project(delta: np.ndarray, r_norm: np.ndarray) -> np.ndarray:
    """Project radial residual to zero spherical volume mean for each time row."""
    d = np.asarray(delta, dtype=float)
    r = np.asarray(r_norm, dtype=float)
    w = radial_volume_weights(r)
    if d.shape[-1] != len(r):
        raise ValueError(f"delta last dimension {d.shape[-1]} != len(r_norm) {len(r)}")
    mean = np.sum(d * w.reshape((1,) * (d.ndim - 1) + (-1,)), axis=-1, keepdims=True)
    return d - mean


def zero_mean_error(delta: np.ndarray, r_norm: np.ndarray) -> float:
    d = np.asarray(delta, dtype=float)
    w = radial_volume_weights(r_norm)
    mean = np.sum(d * w.reshape((1,) * (d.ndim - 1) + (-1,)), axis=-1)
    return float(np.max(np.abs(mean))) if mean.size else 0.0


def radial_gradient_audit(cs: np.ndarray, r_norm: np.ndarray, rest_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute simple radial gradient diagnostics."""
    arr = np.asarray(cs, dtype=float)
    if arr.ndim != 2:
        raise ValueError("cs must be shaped [time, r]")
    center = arr[:, 0]
    surface = arr[:, -1]
    diff = surface - center
    amp = np.abs(diff)
    out = {
        "surface_minus_center_mean": float(np.mean(diff)),
        "surface_minus_center_abs_mean": float(np.mean(amp)),
        "surface_minus_center_abs_max": float(np.max(amp)) if amp.size else 0.0,
        "surface_gt_center_fraction": float(np.mean(diff > 0)) if diff.size else 0.0,
        "surface_lt_center_fraction": float(np.mean(diff < 0)) if diff.size else 0.0,
    }
    if rest_mask is not None:
        m = np.asarray(rest_mask).astype(bool)
        if len(m) == len(diff) and np.any(m):
            out["rest_abs_gradient_mean"] = float(np.mean(amp[m]))
        else:
            out["rest_abs_gradient_mean"] = float("nan")
    return out


def diffusion_residual_spherical(
    cs: np.ndarray,
    t_s: np.ndarray,
    r_m: np.ndarray,
    D_s_m2_s: float,
) -> np.ndarray:
    """Approximate Fick spherical diffusion residual.

    residual = dc/dt - (1/r^2) d/dr(D r^2 dc/dr)

    This is for audit/smoke. D17-P2 can implement a more stable torch/FV residual.
    """
    cs = np.asarray(cs, dtype=float)
    t = np.asarray(t_s, dtype=float)
    r = np.asarray(r_m, dtype=float)
    if cs.ndim != 2 or cs.shape != (len(t), len(r)):
        raise ValueError("cs shape must be [len(t_s), len(r_m)]")
    dc_dt = np.gradient(cs, t, axis=0, edge_order=1)
    dc_dr = np.gradient(cs, r, axis=1, edge_order=1)
    r_safe = np.maximum(r, max(float(np.max(r)) * 1e-6, 1e-12))
    flux_like = D_s_m2_s * (r_safe ** 2)[None, :] * dc_dr
    dflux_dr = np.gradient(flux_like, r, axis=1, edge_order=1)
    rhs = dflux_dr / (r_safe ** 2)[None, :]
    # At r=0 use the first nonzero node approximation.
    rhs[:, 0] = rhs[:, 1]
    return dc_dt - rhs
