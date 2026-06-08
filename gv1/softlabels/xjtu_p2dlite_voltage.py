# -*- coding: utf-8 -*-
"""P2Dlite voltage closure, generic OCP functions, and bounded residual correction.

D14-P4A changes
---------------
1. `apply_s1k_style_residual` now supports a safe soft-label voltage bound
   read from the standalone prior through the resolved spec.
2. The function can optionally return the raw unbounded soft voltage so the
   audit can quantify how much correction was applied.
3. The bound is intentionally small around the nominal terminal voltage window;
   it prevents the generator from producing `phis_c_soft` labels that exceed
   the physically observed LR18650LA / XJTU voltage limits by a large amount.
"""

from __future__ import annotations

import numpy as np


def _clip_theta(theta):
    return np.clip(theta, 1e-5, 0.99999)


def ocp_graphite_generic(theta):
    th = _clip_theta(theta)
    return (
        0.14
        + 0.75 * np.exp(-35.61 * th)
        - 0.02 * np.tanh((th - 0.61) / 0.02)
        - 0.13 * np.tanh((th - 0.32) / 0.07)
        - 0.12 * np.tanh((th - 0.21) / 0.09)
        - 0.13 * np.tanh((th - 0.45) / 0.16)
        - 0.12 * np.tanh((th - 0.40) / 0.16)
        - 0.11 * np.tanh((th - 0.43) / 0.15)
        - 0.15 * np.tanh((th - 0.40) / 0.10)
        + 0.72 * np.tanh((th - 0.37) / 0.16)
    )


def ocp_nmc_generic(theta):
    th = _clip_theta(theta)
    x = 1.0 - th
    val = (
        2.11e-6
        + 110.52 * x
        - 1361.72 * x**2
        + 9188.4 * x**3
        - 37148.01 * x**4
        + 94012.19 * x**5
        - 150327.14 * x**6
        + 147704.4 * x**7
        - 81484.34 * x**8
        + 19336.88 * x**9
    )
    exp_arg = -57824.14 * np.power(th, 115)
    exp_arg = np.clip(exp_arg, -100.0, 20.0)
    val = val - 0.1 * np.exp(exp_arg)
    return np.clip(val, 2.5, 5.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def voltage_closure(theta_n_surf, theta_p_surf, j_n, j_p, I, temperature_K, resolved):
    R = float(resolved["constants"]["R_J_per_mol_K"])
    i0n = float(resolved["kinetics"]["i0_negative_A_m2"])
    i0p = float(resolved["kinetics"]["i0_positive_A_m2"])

    T = np.asarray(temperature_K, dtype=float)
    T = np.where(np.isfinite(T), T, float(resolved["cell"]["fallback_temperature_K"]))

    U_n = ocp_graphite_generic(theta_n_surf)
    U_p = ocp_nmc_generic(theta_p_surf)

    # Linearized Butler-Volmer v0: J ≈ i0 * eta / (R*T), hence eta ≈ J*R*T/i0.
    # This is a P2Dlite consistency term, not a full local P2D kinetic solve.
    eta_n = j_n * R * T / max(i0n, 1e-12)
    eta_p = j_p * R * T / max(i0p, 1e-12)

    R_ohm = float(resolved["voltage_closure"]["R_ohm_Ohm"])
    offset = float(resolved["voltage_closure"]["voltage_offset_V"])

    V_base = U_p - U_n + eta_p - eta_n + R_ohm * I + offset
    phie = -U_n - eta_n
    return V_base, phie, U_n, U_p, eta_n, eta_p


def _soft_voltage_bound_limits(resolved):
    cell = resolved.get("cell", {})
    vmin = float(cell.get("voltage_min_V", 2.5))
    vmax = float(cell.get("voltage_max_V", 4.2))
    bounds = resolved.get("voltage_closure", {}).get("soft_voltage_bounds", {}) or {}
    enabled = bool(bounds.get("enabled", True))
    lower_margin = float(bounds.get("lower_margin_V", 0.02))
    upper_margin = float(bounds.get("upper_margin_V", 0.02))
    lower = vmin - lower_margin
    upper = vmax + upper_margin
    return enabled, lower, upper


def apply_s1k_style_residual(V_base, V_exp, resolved, return_raw: bool = False):
    cfg = resolved.get("voltage_closure", {}).get("residual_correction", {})
    normal = float(cfg.get("normal_blend", 0.40))
    low = float(cfg.get("low_voltage_blend", 0.90))
    thr = float(cfg.get("low_voltage_threshold_V", 3.05))
    width = max(float(cfg.get("transition_width_V", 0.08)), 1e-6)
    max_abs = float(cfg.get("max_abs_residual_V", 1.0))

    residual = np.asarray(V_exp, dtype=float) - np.asarray(V_base, dtype=float)
    residual = np.clip(residual, -max_abs, max_abs)
    low_gate = sigmoid((thr - np.asarray(V_exp, dtype=float)) / width)
    weight = np.clip(normal + (low - normal) * low_gate, 0.0, 1.0)

    V_soft_raw = np.asarray(V_base, dtype=float) + weight * residual

    enabled, lower, upper = _soft_voltage_bound_limits(resolved)
    if enabled:
        V_soft = np.clip(V_soft_raw, lower, upper)
    else:
        V_soft = V_soft_raw.copy()

    if return_raw:
        return V_soft, residual, weight, V_soft_raw
    return V_soft, residual, weight
