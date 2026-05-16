# -*- coding: utf-8 -*-
"""
ASSB ModelFin_110 aging-fix1 parameter entry point.

Complete replacement file. This file builds the effective-SPM
parameter dictionary for NMC811 || Li-In/In ASSB and optionally reads the
continuous solution.npz to populate t/I/cycle profiles.  It keeps positive and
negative material identity fixed; current sign only changes flux direction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (str(_ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .thermo_assb import setParams
except Exception:  # pragma: no cover
    from thermo_assb import setParams


def _normalize_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    s = str(path).strip().strip('"').strip("'")
    if not s or s.upper() in {"NONE", "NULL"}:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _maybe_soft_label_dir() -> Optional[Path]:
    for key in ("ASSB_SOFT_LABEL_DIR", "SOFT_LABEL_DIR"):
        p = _normalize_path(os.environ.get(key))
        if p is not None and (p / "solution.npz").exists():
            return p
    candidates = [
        Path.cwd() / "assb_soft_labels_cycle5_522_v2_massclosed_candidate",
        Path.cwd().parent / "assb_soft_labels_cycle5_522_v2_massclosed_candidate",
        Path.cwd() / "Data" / "assb_soft_labels_cycle5_522_v2_massclosed_candidate",
    ]
    for p in candidates:
        if (p / "solution.npz").exists():
            return p
    return None


def _load_solution_into_params(params: Dict[str, object], soft_dir: Path) -> Dict[str, object]:
    sol = soft_dir / "solution.npz"
    if not sol.exists():
        return params
    try:
        with np.load(sol, allow_pickle=True) as z:
            names = set(z.files)
            t_key = "t_global_s" if "t_global_s" in names else ("t" if "t" in names else "time_s")
            i_key = "I_profile" if "I_profile" in names else ("current_A" if "current_A" in names else "I")
            t = np.asarray(z[t_key], dtype=np.float64).reshape(-1)
            I = np.asarray(z[i_key], dtype=np.float64).reshape(-1)
            order = np.argsort(t)
            t = t[order]
            I = I[order]
            params["current_profile"] = (t, I)
            params["time_profile"] = t
            params["current_profile_A"] = I
            params["tmax"] = np.float64(float(np.nanmax(t)) if t.size else params.get("tmax", 1.0))
            params["rescale_T"] = np.float64(params["tmax"])
            params["soft_label_dir_runtime"] = str(soft_dir)
            params["soft_label_solution_runtime"] = str(sol)
            if "cycle_id" in names:
                params["cycle_id_profile"] = np.asarray(z["cycle_id"], dtype=np.int64).reshape(-1)[order]
            # Use first point as hard IC if available.  This preserves current
            # 107A/continuous soft-label state scale without treating labels as data loss.
            for key, pkey in (("cs_a", "cs_a0"), ("cs_c", "cs_c0")):
                if key in names:
                    arr = np.asarray(z[key], dtype=np.float64)
                    if arr.ndim >= 2:
                        params[pkey] = np.float64(np.nanmean(arr[0]))
                    elif arr.size:
                        params[pkey] = np.float64(arr[0])
            for key, pkey in (("phie", "phie0"), ("phis_c", "phis_c0")):
                if key in names:
                    arr = np.asarray(z[key], dtype=np.float64).reshape(-1)
                    if arr.size:
                        params[pkey] = np.float64(arr[0])
    except Exception as exc:
        params["soft_label_load_error"] = str(exc)
    return params


class Anode:
    def __init__(self):
        self.thickness = np.float64(100e-6)
        self.D50 = np.float64(100e-6)  # Rs_a = 50 µm effective Li-In/In diffusion length.
        self.eps = np.float64(0.95)


class Cathode:
    def __init__(self):
        self.thickness = np.float64(16e-6)
        self.D50 = np.float64(3.6e-6)  # Rs_c = 1.8 µm NMC811 representative particle.
        self.eps = np.float64(0.55)


def makeParams() -> Dict[str, object]:
    params = setParams({})
    # Explicit class-based values for compatibility with older project notes.
    an = Anode()
    ca = Cathode()
    params["L_a"] = an.thickness
    params["L_c"] = ca.thickness
    params["Rs_a"] = an.D50 / np.float64(2.0)
    params["Rs_c"] = ca.D50 / np.float64(2.0)
    params["eps_s_a"] = an.eps
    params["eps_s_c"] = ca.eps
    params["A_a"] = np.float64(np.pi * (5e-3) ** 2)
    params["A_c"] = np.float64(np.pi * (5e-3) ** 2)
    params["V_a"] = params["A_a"] * params["L_a"]
    params["V_c"] = params["A_c"] * params["L_c"]

    soft_dir = _maybe_soft_label_dir()
    if soft_dir is not None:
        _load_solution_into_params(params, soft_dir)
    else:
        # Minimal fallback for py_compile/import and smoke tests before data preparation.
        params.setdefault("tmax", np.float64(9232.0))
        params.setdefault("rescale_T", np.float64(params["tmax"]))
        t = np.linspace(0.0, float(params["tmax"]), 1024)
        I = np.full_like(t, 3.3e-4)
        I[t > t[-1] * 0.5] *= -1.0
        params["current_profile"] = (t, I)
        params["time_profile"] = t
        params["current_profile_A"] = I
    # Keep c_s,max constant. Aging variables modify V_c_eff, R_ohm and theta window.
    params.setdefault("csanmax", np.float64(6.0))
    params.setdefault("cscamax", np.float64(51.8))
    params.setdefault("cs_a0", np.float64(params.get("csa0", 3.0)))
    params.setdefault("cs_c0", np.float64(params.get("csc0", 0.834 * 51.8)))
    params.setdefault("csa0", params["cs_a0"])
    params.setdefault("csc0", params["cs_c0"])
    return params


__all__ = ["makeParams", "Anode", "Cathode"]
