# -*- coding: utf-8 -*-
"""Torch numerical helpers for D17-P2."""

from __future__ import annotations

from typing import Tuple

import torch


def cumulative_trapezoid_torch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if y.ndim != 1 or x.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError("cumulative_trapezoid_torch expects 1D tensors with equal length")
    out = torch.zeros_like(y)
    if y.numel() > 1:
        dx = x[1:] - x[:-1]
        area = 0.5 * (y[:-1] + y[1:]) * dx
        out[1:] = torch.cumsum(area, dim=0)
    return out


def radial_volume_weights_torch(r_norm: torch.Tensor) -> torch.Tensor:
    if r_norm.ndim != 1:
        raise ValueError("r_norm must be 1D")
    w = torch.clamp(r_norm, min=0.0) ** 2
    if torch.sum(w) <= 0:
        w = torch.ones_like(w)
    return w / torch.sum(w)


def zero_volume_mean_project_torch(delta: torch.Tensor, r_norm: torch.Tensor) -> torch.Tensor:
    w = radial_volume_weights_torch(r_norm).to(delta.device, delta.dtype)
    shape = [1] * delta.ndim
    shape[-1] = -1
    mean = torch.sum(delta * w.reshape(shape), dim=-1, keepdim=True)
    return delta - mean


def interp1d_torch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation with boundary clamping."""
    xp = xp.to(device=x.device, dtype=x.dtype)
    fp = fp.to(device=x.device, dtype=x.dtype)
    x_clamped = torch.clamp(x, min=float(xp[0].detach().cpu()), max=float(xp[-1].detach().cpu()))
    idx = torch.searchsorted(xp, x_clamped, right=True) - 1
    idx = torch.clamp(idx, 0, xp.numel() - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    denom = torch.clamp(x1 - x0, min=torch.finfo(x.dtype).eps)
    frac = (x_clamped - x0) / denom
    return y0 + frac * (y1 - y0)


def safe_std(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return torch.clamp(torch.std(x), min=eps)


def finite_diff_time_centered(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Centered dy/dt for y [N, ...], returns [N-2, ...]."""
    if y.shape[0] < 3:
        return torch.zeros_like(y[:0])
    denom = (t[2:] - t[:-2]).reshape((-1,) + (1,) * (y.ndim - 1))
    return (y[2:] - y[:-2]) / torch.clamp(denom, min=1.0e-9)


def spherical_laplacian_uniform_r(cs: torch.Tensor, r_m: torch.Tensor) -> torch.Tensor:
    """Approximate spherical Laplacian on interior r nodes, returns [N, R-2]."""
    if cs.ndim != 2:
        raise ValueError("cs must be [time, r]")
    if cs.shape[1] < 3:
        return torch.zeros_like(cs[:, :0])
    dr = torch.mean(r_m[1:] - r_m[:-1])
    c_im1 = cs[:, :-2]
    c_i = cs[:, 1:-1]
    c_ip1 = cs[:, 2:]
    r_i = torch.clamp(r_m[1:-1], min=float(torch.max(r_m).detach().cpu()) * 1.0e-6).reshape(1, -1)
    d2 = (c_ip1 - 2.0 * c_i + c_im1) / torch.clamp(dr * dr, min=1.0e-30)
    d1 = (c_ip1 - c_im1) / torch.clamp(2.0 * dr, min=1.0e-30)
    return d2 + 2.0 * d1 / r_i
