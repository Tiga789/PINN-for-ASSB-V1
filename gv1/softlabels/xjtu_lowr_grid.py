# -*- coding: utf-8 -*-
"""Low-resolution radial finite-volume grid utilities."""

from __future__ import annotations

import numpy as np


def make_spherical_fv_grid(radius_m: float, n_r: int) -> dict:
    faces = np.linspace(0.0, float(radius_m), int(n_r) + 1, dtype=float)
    centers = 0.5 * (faces[:-1] + faces[1:])
    volumes = (4.0 / 3.0) * np.pi * (faces[1:] ** 3 - faces[:-1] ** 3)
    areas = 4.0 * np.pi * faces ** 2
    return {"faces": faces, "centers": centers, "volumes": volumes, "areas": areas, "radius": float(radius_m), "n_r": int(n_r)}


def build_diffusion_matrix(radius_m: float, n_r: int, D_m2_s: float) -> tuple[np.ndarray, np.ndarray, dict]:
    grid = make_spherical_fv_grid(radius_m, n_r)
    centers = grid["centers"]
    faces = grid["faces"]
    volumes = grid["volumes"]
    areas = grid["areas"]
    n = int(n_r)

    A = np.zeros((n, n), dtype=float)
    for f in range(1, n):
        left = f - 1
        right = f
        dist = centers[right] - centers[left]
        conductance = float(D_m2_s) * areas[f] / max(dist, 1e-30)
        A[left, left] -= conductance / volumes[left]
        A[left, right] += conductance / volumes[left]
        A[right, right] -= conductance / volumes[right]
        A[right, left] += conductance / volumes[right]

    b_flux = np.zeros(n, dtype=float)
    # Boundary outward flux J at r=R removes Li from the outermost control volume.
    b_flux[-1] = -areas[-1] / volumes[-1]
    return A, b_flux, grid


def shell_average(c: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    return np.sum(c * volumes[None, :], axis=1) / np.sum(volumes)
