# -*- coding: utf-8 -*-
"""
ASSB ModelFin_110 aging-fix1 thermo/geometry parameter utilities.

Complete replacement file. The positive/negative material identity
is fixed: c=NMC811 positive electrode, a=Li-In/In negative electrode.  Charging
and discharging change only the sign of I(t), hence flux and overpotential sign;
OCP functions and material parameters are not swapped.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

F_CONST = np.float64(96485.33212)
R_CONST = np.float64(8.31446261815324)
T_REF = np.float64(303.15)


def _as_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    s = str(path).strip().strip('"').strip("'")
    if not s or s.upper() in {"NONE", "NULL"}:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _find_ocp_dir() -> Optional[Path]:
    env = _as_path(os.environ.get("ASSB_OCP_DIR"))
    if env is not None and env.exists():
        return env
    candidates = [
        Path.cwd() / "ocp_estimation_outputs",
        Path.cwd().parent / "ocp_estimation_outputs",
        Path.cwd() / "Data" / "ocp_estimation_outputs",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_curve(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
        names = arr.dtype.names or ()
        if len(names) >= 2:
            x = np.asarray(arr[names[0]], dtype=float)
            y = np.asarray(arr[names[1]], dtype=float)
        else:
            raw = np.genfromtxt(path, delimiter=",", dtype=float)
            x, y = raw[:, 0], raw[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return None
        order = np.argsort(x[mask])
        return x[mask][order], y[mask][order]
    except Exception:
        return None


def _interp_curve(curve: Optional[Tuple[np.ndarray, np.ndarray]], x, fallback: Callable[[np.ndarray], np.ndarray]):
    if torch is not None and torch.is_tensor(x):
        # Torch fallback is smooth enough for diagnostics; true differentiable OCP
        # tables can be reintroduced later if OCP gradients are required.
        x_np = x.detach().cpu().numpy()
        y_np = _interp_curve(curve, x_np, fallback)
        return torch.as_tensor(y_np, dtype=x.dtype, device=x.device)
    x_np = np.asarray(x, dtype=float)
    if curve is None:
        return fallback(np.clip(x_np, 0.0, 1.0))
    xp, yp = curve
    return np.interp(np.clip(x_np, xp[0], xp[-1]), xp, yp)


_OCP_DIR = _find_ocp_dir()
_POS_CURVE = None
_NEG_CURVE = None
if _OCP_DIR is not None:
    for name in ("positive_ocp_curve.csv", "positive_ocp.csv", "pos_ocp.csv"):
        _POS_CURVE = _load_curve(_OCP_DIR / name)
        if _POS_CURVE is not None:
            break
    for name in ("negative_ocp_curve.csv", "negative_ocp.csv", "neg_ocp.csv"):
        _NEG_CURVE = _load_curve(_OCP_DIR / name)
        if _NEG_CURVE is not None:
            break


def Uocp_c_fun(theta):
    """NMC811 positive-electrode OCP prior, fixed material identity."""
    def fallback(x):
        # Smooth decreasing NMC-like shape over the usable window.
        return 4.35 - 0.85 * x + 0.05 * np.tanh((0.5 - x) * 8.0)
    return _interp_curve(_POS_CURVE, theta, fallback)


def Uocp_a_fun(theta):
    """Li-In/In negative-electrode OCP prior, fixed material identity."""
    def fallback(x):
        return 0.6246 + 0.015 * (np.clip(x, 0.0, 1.0) - 0.5)
    return _interp_curve(_NEG_CURVE, theta, fallback)


def i0_c_fun(theta, ce=1.2, T=T_REF):
    theta_arr = theta if torch is not None and torch.is_tensor(theta) else np.asarray(theta, dtype=float)
    prefac = np.float64(2.5)
    if torch is not None and torch.is_tensor(theta_arr):
        th = torch.clamp(theta_arr, 1.0e-6, 1.0 - 1.0e-6)
        return prefac * torch.sqrt(th * (1.0 - th))
    th = np.clip(theta_arr, 1.0e-6, 1.0 - 1.0e-6)
    return prefac * np.sqrt(th * (1.0 - th))


def i0_a_fun(theta, ce=1.2, T=T_REF, deg_i0_a=1.0):
    theta_arr = theta if torch is not None and torch.is_tensor(theta) else np.asarray(theta, dtype=float)
    ref = np.float64(1.0)
    if torch is not None and torch.is_tensor(theta_arr):
        th = torch.clamp(theta_arr, 1.0e-6, 1.0 - 1.0e-6)
        return ref * float(deg_i0_a) * torch.sqrt(th * (1.0 - th))
    th = np.clip(theta_arr, 1.0e-6, 1.0 - 1.0e-6)
    return ref * float(deg_i0_a) * np.sqrt(th * (1.0 - th))


def setParams(params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    p: Dict[str, object] = {} if params is None else dict(params)
    # Geometry and material constants from the ASSB effective-SPM prior.
    area = np.float64(np.pi * (5.0e-3) ** 2)  # 10 mm disk.
    p.setdefault("F", F_CONST)
    p.setdefault("R", R_CONST)
    p.setdefault("T", T_REF)
    p.setdefault("A_a", area)
    p.setdefault("A_c", area)
    p.setdefault("L_a", np.float64(100.0e-6))
    p.setdefault("L_c", np.float64(16.0e-6))
    p.setdefault("Rs_a", np.float64(50.0e-6))
    p.setdefault("Rs_c", np.float64(1.8e-6))
    p.setdefault("eps_s_a", np.float64(0.95))
    p.setdefault("eps_s_c", np.float64(0.55))
    p["V_a"] = np.float64(p.get("V_a", p["A_a"] * p["L_a"]))
    p["V_c"] = np.float64(p.get("V_c", p["A_c"] * p["L_c"]))

    # Concentration scales. c_s,max stays constant; aging uses active volume and
    # theta window, not c_s,max drift.
    p.setdefault("csanmax", np.float64(6.0))
    p.setdefault("cscamax", np.float64(51.8))
    p.setdefault("theta_c_bottom", np.float64(0.834))
    p.setdefault("theta_c_top", np.float64(0.432))
    p.setdefault("theta_c_window_mid0", np.float64(0.5 * (float(p["theta_c_bottom"]) + float(p["theta_c_top"]))))
    p.setdefault("cs_a0", np.float64(0.5 * float(p["csanmax"])))
    p.setdefault("cs_c0", np.float64(float(p["theta_c_bottom"]) * float(p["cscamax"])))
    p.setdefault("csa0", p["cs_a0"])
    p.setdefault("csc0", p["cs_c0"])
    p.setdefault("cs_a_min", np.float64(0.0))
    p.setdefault("cs_c_min", np.float64(0.0))
    p.setdefault("cs_a_upper", np.float64(p["csanmax"]))
    p.setdefault("cs_c_upper", np.float64(p["cscamax"]))

    # Transport/kinetics first-version priors.
    p.setdefault("Ds_a", np.float64(1.0e-13))
    p.setdefault("Ds_c", np.float64(1.0e-14))
    p.setdefault("R_ohm_eff", np.float64(105.0))
    p.setdefault("I_discharge", np.float64(3.3e-4))
    p.setdefault("ce", np.float64(1.2))
    p.setdefault("alpha_a", np.float64(0.5))
    p.setdefault("alpha_c", np.float64(0.5))
    p.setdefault("phie0", np.float64(0.0))
    p.setdefault("phis_c0", np.float64(3.2))
    p.setdefault("rescale_T", np.float64(p.get("tmax", 1.0)))
    p.setdefault("rescale_R", np.float64(max(float(p["Rs_a"]), float(p["Rs_c"]))))
    p.setdefault("rescale_R_a", p["Rs_a"])
    p.setdefault("rescale_R_c", p["Rs_c"])
    p.setdefault("use_per_electrode_rescale_R", True)
    p.setdefault("rescale_phie", np.float64(1.0))
    p.setdefault("rescale_phis_c", np.float64(1.0))
    p.setdefault("Uocp_a_fun", Uocp_a_fun)
    p.setdefault("Uocp_c_fun", Uocp_c_fun)
    p.setdefault("i0_a_fun", i0_a_fun)
    p.setdefault("i0_c_fun", i0_c_fun)

    # Baseline 107A-style transforms used by Stage C.
    p.setdefault("use_i_cbar_baseline_a", True)
    p.setdefault("use_i_cbar_baseline_c", True)
    p.setdefault("use_zero_mean_radial_deviation_a", True)
    p.setdefault("use_zero_mean_radial_deviation_c", True)
    p.setdefault("cbar_deviation_fraction_a", np.float64(0.15))
    p.setdefault("cbar_deviation_fraction_c", np.float64(0.10))
    p.setdefault("use_current_potential_baseline", True)
    p.setdefault("use_current_potential_baseline_phie", True)
    p.setdefault("use_current_potential_baseline_phis_c", True)
    p.setdefault("potential_baseline_correction_fraction_phie", np.float64(0.20))
    p.setdefault("potential_baseline_correction_fraction_phis_c", np.float64(0.20))
    p.setdefault("deg_i0_a_min_eff", np.float64(1.0))
    p.setdefault("deg_i0_a_max_eff", np.float64(1.0))
    p.setdefault("deg_ds_c_min_eff", np.float64(1.0))
    p.setdefault("deg_ds_c_max_eff", np.float64(1.0))
    return p


__all__ = ["setParams", "Uocp_a_fun", "Uocp_c_fun", "i0_a_fun", "i0_c_fun", "F_CONST", "R_CONST", "T_REF"]
