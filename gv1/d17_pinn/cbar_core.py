# -*- coding: utf-8 -*-
"""
D17-P1 current-conserved cbar core.

This module implements the I(t)-cbar hard baseline idea in a simple NumPy form.
It is not yet a full torch/autograd model; D17-P2 can port the same equations
to torch after P0/P1 audits pass.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


FARADAY_C_PER_MOL = 96485.33212


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError("cumulative_trapezoid expects 1D arrays with equal length")
    out = np.zeros_like(y, dtype=float)
    if len(y) > 1:
        dx = np.diff(x)
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    return out


def current_to_surface_flux(
    current_A: np.ndarray,
    particle_radius_m: float,
    active_fraction: float,
    electrode_volume_m3: float,
    sign: float,
    faraday_C_per_mol: float = FARADAY_C_PER_MOL,
) -> np.ndarray:
    """Convert measured current to SPM-like molar surface flux.

    Formula follows the project convention:
        J_j = sign * I * R_j / (3 * eps_s_j * F * V_j)

    The sign should encode electrode convention:
        positive electrode often +1 under the adopted project convention,
        negative electrode often -1.
    """
    current_A = np.asarray(current_A, dtype=float)
    denom = 3.0 * active_fraction * faraday_C_per_mol * electrode_volume_m3
    if denom <= 0:
        raise ValueError("active_fraction, Faraday constant, and electrode_volume_m3 must be positive")
    return sign * current_A * particle_radius_m / denom


def integrate_cbar_from_flux(
    t_s: np.ndarray,
    J_mol_m2_s: np.ndarray,
    particle_radius_m: float,
    cbar0_mol_m3: float,
) -> np.ndarray:
    """Integrate sphere-average concentration from surface flux.

        d cbar / dt = -3 J / R

    Sign of J must already follow the electrode convention.
    """
    t_s = np.asarray(t_s, dtype=float)
    J_mol_m2_s = np.asarray(J_mol_m2_s, dtype=float)
    integral = cumulative_trapezoid(J_mol_m2_s, t_s)
    return cbar0_mol_m3 - (3.0 / particle_radius_m) * integral


def integrate_cbar_from_current(
    t_s: np.ndarray,
    current_A: np.ndarray,
    particle_radius_m: float,
    active_fraction: float,
    electrode_volume_m3: float,
    cbar0_mol_m3: float,
    sign: float,
    qeff_scale: float = 1.0,
) -> np.ndarray:
    """Convenience wrapper: I(t) -> J(t) -> cbar(t).

    qeff_scale scales effective electrode volume/capacity. Values should be
    bounded by a profile latent prior in later D17 stages.
    """
    effective_volume = electrode_volume_m3 * float(qeff_scale)
    J = current_to_surface_flux(
        current_A=current_A,
        particle_radius_m=particle_radius_m,
        active_fraction=active_fraction,
        electrode_volume_m3=effective_volume,
        sign=sign,
    )
    return integrate_cbar_from_flux(t_s, J, particle_radius_m, cbar0_mol_m3)
