from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


FARADAY_C_PER_MOL = 96485.33212


@dataclass
class ElectrodeRGParams:
    name: str
    radius_m: float
    diffusivity_m2_s: float
    csmax_mol_m3: float
    alpha_D: float = 1.0
    alpha_J: float = 1.0
    gradient_clip_normalized: float = 0.12
    theta_min_clip: float = 0.0
    theta_max_clip: float = 1.0

    @property
    def D_eff(self) -> float:
        return max(float(self.diffusivity_m2_s) * float(self.alpha_D), 1e-30)

    @property
    def R(self) -> float:
        return max(float(self.radius_m), 1e-12)

    @property
    def clip_abs(self) -> float:
        return abs(float(self.gradient_clip_normalized)) * float(self.csmax_mol_m3)


@dataclass
class RadialGrid:
    nr: int
    radius_m: float
    edges_m: np.ndarray
    centers_m: np.ndarray
    volumes_m3: np.ndarray
    areas_m2: np.ndarray
    weights: np.ndarray

    @classmethod
    def equal_interval(cls, nr: int, radius_m: float) -> 'RadialGrid':
        if nr < 3:
            raise ValueError('nr must be >= 3 for radial-gradient labels')
        R = float(radius_m)
        edges = np.linspace(0.0, R, nr + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
        areas = 4.0 * np.pi * edges ** 2
        weights = volumes / np.sum(volumes)
        return cls(nr=nr, radius_m=R, edges_m=edges, centers_m=centers, volumes_m3=volumes, areas_m2=areas, weights=weights)

    def normalized_centers(self) -> np.ndarray:
        return self.centers_m / self.radius_m


def _thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve tridiagonal Ax=b with lower/diag/upper vectors.

    lower[0] and upper[-1] are ignored.
    """
    n = diag.size
    c = upper.copy().astype(float)
    d = diag.copy().astype(float)
    b = rhs.copy().astype(float)
    a = lower.copy().astype(float)
    for i in range(1, n):
        m = a[i] / d[i - 1]
        d[i] -= m * c[i - 1]
        b[i] -= m * b[i - 1]
    x = np.empty(n, dtype=float)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    return x


def _implicit_diffusion_step(c_old: np.ndarray, dt_s: float, J_surface_mol_m2_s: float, grid: RadialGrid, D_m2_s: float) -> np.ndarray:
    """Backward-Euler finite-volume step for spherical solid diffusion.

    Sign convention: boundary condition is D * dc/dr|R = -J. Thus J > 0
    means lithium leaves the particle through the surface and the average
    concentration decreases.
    """
    nr = grid.nr
    dt = max(float(dt_s), 0.0)
    if dt <= 0.0:
        return c_old.copy()

    # Conductance at internal faces: D * A_face / distance_between_centers.
    g = np.zeros(nr + 1, dtype=float)
    for face in range(1, nr):
        dx = grid.centers_m[face] - grid.centers_m[face - 1]
        g[face] = D_m2_s * grid.areas_m2[face] / max(dx, 1e-30)

    lower = np.zeros(nr, dtype=float)
    diag = np.zeros(nr, dtype=float)
    upper = np.zeros(nr, dtype=float)
    rhs = grid.volumes_m3 * c_old

    # Boundary outflux through outer surface, treated explicitly in J.
    rhs[-1] -= dt * grid.areas_m2[-1] * float(J_surface_mol_m2_s)

    for i in range(nr):
        V_i = grid.volumes_m3[i]
        diag[i] = V_i
        if i > 0:
            # Influx from i-1: g_i * (c_{i-1} - c_i)
            diag[i] += dt * g[i]
            lower[i] = -dt * g[i]
        if i < nr - 1:
            # Outflux to i+1: g_{i+1} * (c_i - c_{i+1})
            diag[i] += dt * g[i + 1]
            upper[i] = -dt * g[i + 1]
    return _thomas_solve(lower, diag, upper, rhs)


def _project_to_cbar_and_clip(c: np.ndarray, target_cbar: float, weights: np.ndarray, params: ElectrodeRGParams) -> Tuple[np.ndarray, Dict[str, float]]:
    """Enforce zero-volume-mean deviation around target cbar and safe bounds."""
    info: Dict[str, float] = {}
    c = c.astype(float).copy()
    before_mean = float(np.sum(c * weights))
    c += float(target_cbar) - before_mean

    # Clip radial deviation first, then reproject to preserve mass.
    dev = c - float(target_cbar)
    clip_abs = params.clip_abs
    clipped_dev = np.clip(dev, -clip_abs, clip_abs)
    info['dev_clip_max_abs_before'] = float(np.max(np.abs(dev))) if dev.size else 0.0
    info['dev_clip_changed_fraction'] = float(np.mean(np.abs(clipped_dev - dev) > 1e-12)) if dev.size else 0.0
    c = float(target_cbar) + clipped_dev
    c += float(target_cbar) - float(np.sum(c * weights))

    lower = params.theta_min_clip * params.csmax_mol_m3
    upper = params.theta_max_clip * params.csmax_mol_m3
    c_phys = np.clip(c, lower, upper)
    info['physical_clip_changed_fraction'] = float(np.mean(np.abs(c_phys - c) > 1e-9)) if c.size else 0.0
    c = c_phys
    # Final cbar projection; if target itself is outside physical bounds, clipping wins.
    if lower <= target_cbar <= upper:
        c += float(target_cbar) - float(np.sum(c * weights))
        c = np.clip(c, lower, upper)
    info['cbar_after'] = float(np.sum(c * weights))
    info['surface_center'] = float(c[-1] - c[0])
    return c, info


def infer_surface_flux_from_cbar(t_s: np.ndarray, cbar: np.ndarray, radius_m: float) -> np.ndarray:
    """Infer J from d cbar / dt using d cbar / dt = -3J/R."""
    t = np.asarray(t_s, dtype=float)
    cb = np.asarray(cbar, dtype=float)
    if t.ndim != 1 or cb.ndim != 1 or t.size != cb.size:
        raise ValueError('t_s and cbar must be 1D arrays of equal length')
    dc_dt = np.zeros_like(cb)
    if cb.size > 1:
        dt = np.diff(t)
        dc = np.diff(cb)
        valid = dt > 0
        step = np.zeros_like(dc)
        step[valid] = dc[valid] / dt[valid]
        dc_dt[1:] = step
        dc_dt[0] = step[0] if step.size else 0.0
    J = -float(radius_m) * dc_dt / 3.0
    return J


def generate_rg_profile(
    t_s: np.ndarray,
    cbar_target: np.ndarray,
    J_profile: np.ndarray,
    initial_profile: Optional[np.ndarray],
    params: ElectrodeRGParams,
    nr: int,
    max_substep_s: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate radial-gradient-aware cs(t, r) with implicit FVM.

    The output preserves the source cbar(t) at every time point, while radial
    deviations are driven by surface flux and relaxed by solid diffusion.
    """
    t = np.asarray(t_s, dtype=float).reshape(-1)
    cbar = np.asarray(cbar_target, dtype=float).reshape(-1)
    J = np.asarray(J_profile, dtype=float).reshape(-1) * float(params.alpha_J)
    if t.size != cbar.size or t.size != J.size:
        raise ValueError('t_s, cbar_target, and J_profile must have the same length')
    if nr < 3:
        nr = 17
    grid = RadialGrid.equal_interval(nr=nr, radius_m=params.R)
    cs = np.empty((t.size, nr), dtype=np.float32)

    if initial_profile is not None:
        c0 = np.asarray(initial_profile, dtype=float).reshape(-1)
        if c0.size != nr:
            # Reinterpolate old initial profile onto the new equal-interval grid.
            x_old = np.linspace(0.0, 1.0, c0.size)
            x_new = grid.normalized_centers()
            c0 = np.interp(x_new, x_old, c0)
    else:
        c0 = np.full(nr, cbar[0], dtype=float)
    c0, _ = _project_to_cbar_and_clip(c0, cbar[0], grid.weights, params)
    cs[0, :] = c0.astype(np.float32)

    surface_center = np.zeros(t.size, dtype=np.float32)
    physical_clip_fraction = np.zeros(t.size, dtype=np.float32)
    dev_clip_fraction = np.zeros(t.size, dtype=np.float32)
    cbar_after = np.zeros(t.size, dtype=np.float32)
    surface_center[0] = c0[-1] - c0[0]
    cbar_after[0] = np.sum(c0 * grid.weights)
    c_prev = c0

    max_sub = max(float(max_substep_s), 0.0)
    for k in range(1, t.size):
        dt = float(t[k] - t[k - 1])
        if not np.isfinite(dt) or dt < 0.0:
            dt = 0.0
        n_sub = 1
        if max_sub > 0 and dt > max_sub:
            n_sub = int(np.ceil(dt / max_sub))
        sub_dt = dt / n_sub if n_sub > 0 else dt
        c_new = c_prev
        for _ in range(n_sub):
            c_new = _implicit_diffusion_step(c_new, sub_dt, J[k], grid, params.D_eff)
        c_new, info = _project_to_cbar_and_clip(c_new, cbar[k], grid.weights, params)
        cs[k, :] = c_new.astype(np.float32)
        surface_center[k] = info['surface_center']
        physical_clip_fraction[k] = info['physical_clip_changed_fraction']
        dev_clip_fraction[k] = info['dev_clip_changed_fraction']
        cbar_after[k] = info['cbar_after']
        c_prev = c_new

    diagnostics = {
        'surface_center': surface_center,
        'surface_mean': (cs[:, -1].astype(float) - np.sum(cs.astype(float) * grid.weights[None, :], axis=1)).astype(np.float32),
        'center_mean': (cs[:, 0].astype(float) - np.sum(cs.astype(float) * grid.weights[None, :], axis=1)).astype(np.float32),
        'cbar_after': cbar_after,
        'physical_clip_fraction': physical_clip_fraction,
        'dev_clip_fraction': dev_clip_fraction,
        'r_norm_centers': grid.normalized_centers().astype(np.float32),
        'volume_weights': grid.weights.astype(np.float32),
        'J_used': J.astype(np.float32),
        'D_eff': np.full(t.size, params.D_eff, dtype=np.float32),
    }
    return cs, diagnostics


def expected_surface_center_sign(current_A: np.ndarray, electrode: str) -> np.ndarray:
    """Expected sign of c_surface - c_center for I>0 charge convention."""
    s = np.sign(np.asarray(current_A, dtype=float))
    if electrode == 'c':
        return -s
    if electrode == 'a':
        return s
    raise ValueError(electrode)
