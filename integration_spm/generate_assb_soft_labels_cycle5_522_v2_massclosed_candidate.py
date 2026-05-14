#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Generate ASSB cycle5--522 v2 mass-closed-candidate soft labels.

This script is designed for the QJW-2 / PINN-for-ASSB-V1 workflow.
It keeps the v1 continuous-soft-label philosophy:

  - one continuous trajectory, no per-cycle reset;
  - fixed electrode identities: a = Li-In/In negative, c = NMC811 positive;
  - +I charge, -I discharge, 0 rest;
  - fixed-B voltage alignment inherited from cycle5_v4 summary;
  - no explicit SOH/aging mechanism in this candidate.

New v2 change:

  - before saving, the positive-electrode cs_c trajectory is shifted at each
    time point so that its spherical volume average matches the I(t)-integrated
    positive-electrode cbar baseline:

        d<c_c>/dt = - I(t) / (eps_s_c * F * V_c)

    The radial shape is preserved by applying a uniform concentration shift to
    all positive-particle radial nodes at each time point. Then theta_c, Uocp_c,
    eta_c, phie and phis_c are recomputed using the existing project solver.

Default output directory required by the current D4 task:

  C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate

Run from repository root after placing this file at:

  integration_spm\generate_assb_soft_labels_cycle5_522_v2_massclosed_candidate.py

Typical command:

  D:\Anaconda\envs\torchgpu\python.exe .\integration_spm\generate_assb_soft_labels_cycle5_522_v2_massclosed_candidate.py ^
    --source_solution_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1" ^
    --record_csv "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv" ^
    --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" ^
    --output_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate" ^
    --cycle_from 5 --cycle_to 522 --overwrite
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires pandas.") from exc

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_DIR.parent if THIS_DIR.name == "integration_spm" else THIS_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

DEFAULT_RECORD_CSV = Path(r"C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv")
DEFAULT_OCP_DIR = Path(r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs")
DEFAULT_SOURCE_DIR = Path(r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate")
DEFAULT_FIXED_SUMMARY = REPO_ROOT / "Data" / "assb_soft_labels_cycle5_v4" / "soft_label_summary.json"

CYCLE_COLUMNS = ("循环号", "cycle", "Cycle", "cycle_index", "Cycle_Index", "循环", "循环序号")
TIME_COLUMNS = ("总时间", "总时间(s)", "time_s", "Time(s)", "time", "Time", "时间", "测试时间")
CURRENT_COLUMNS = ("电流(A)", "电流", "current_A", "Current(A)", "current", "Current", "I_A", "I(A)")
VOLTAGE_COLUMNS = ("电压(V)", "电压", "voltage_V", "Voltage(V)", "voltage", "Voltage", "V")
STEP_ID_COLUMNS = ("工步号", "step_id", "Step_ID", "step", "Step", "step_index", "Step_Index")
STEP_TYPE_COLUMNS = ("工步类型", "step_type", "Step_Type", "step_name", "Step_Name", "工步名称")
DATA_INDEX_COLUMNS = ("数据序号", "data_index", "Data_Index", "index", "Index")


# -----------------------------------------------------------------------------
# Numeric helpers
# -----------------------------------------------------------------------------
def _has_np_trapezoid() -> bool:
    return hasattr(np, "trapezoid")


def _trapz_1d(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if _has_np_trapezoid():
        return float(np.trapezoid(y, x))
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def _trapz_axis_last(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if _has_np_trapezoid():
        return np.trapezoid(y, x, axis=-1)
    if y.shape[-1] < 2:
        return np.zeros(y.shape[:-1], dtype=np.float64)
    dx = np.diff(x)
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)


def sphere_average(cs: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Spherical volume average using r^2 trapezoidal weights."""
    cs = np.asarray(cs, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if cs.shape[-1] != r.size:
        raise ValueError(f"cs last dimension {cs.shape[-1]} does not match r size {r.size}")
    w = r ** 2
    denom = _trapz_1d(w, r)
    if abs(denom) < 1e-300:
        raise ValueError("Invalid radial grid: integral of r^2 is zero.")
    return _trapz_axis_last(cs * w, r) / denom


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.nanstd(x)) <= 0.0 or float(np.nanstd(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def cast_array(arr: Any, dtype: str) -> np.ndarray:
    a = np.asarray(arr)
    if dtype == "float64":
        return a.astype(np.float64, copy=False) if np.issubdtype(a.dtype, np.floating) else a
    if dtype == "float32" and np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    return a


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return str(cols[name])
    norm = {str(c).strip().replace(" ", "").replace("\ufeff", ""): c for c in df.columns}
    for name in candidates:
        key = str(name).strip().replace(" ", "").replace("\ufeff", "")
        if key in norm:
            return str(norm[key])
    if required:
        raise KeyError(f"Could not find any of columns {tuple(candidates)}; available columns: {list(df.columns)}")
    return None


def parse_time_seconds(values: Any) -> np.ndarray:
    s = pd.Series(values)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() >= max(1, int(0.8 * len(s))):
        out = numeric.to_numpy(dtype=np.float64)
    else:
        td = pd.to_timedelta(s.astype(str), errors="coerce")
        if td.notna().sum() >= max(1, int(0.8 * len(s))):
            out = td.dt.total_seconds().to_numpy(dtype=np.float64)
        else:
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() == 0:
                raise ValueError("Could not parse time column as numeric, timedelta, or datetime.")
            t0 = dt.dropna().iloc[0]
            out = (dt - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(out)):
        mask = np.isfinite(out)
        if mask.sum() < 2:
            raise ValueError("Too few finite time values after parsing.")
        idx = np.arange(len(out), dtype=np.float64)
        out = np.interp(idx, idx[mask], out[mask])
    return out.astype(np.float64)


# -----------------------------------------------------------------------------
# Project imports and fixed-B loader
# -----------------------------------------------------------------------------
def load_project_solver() -> Any:
    errors: list[str] = []
    for module_name in ("integration_spm.spm_int_assb_cycle", "spm_int_assb_cycle"):
        try:
            mod = importlib.import_module(module_name)
            required = (
                "load_merged_cycle_params",
                "integrate_cycle",
                "surface_flux_from_current",
                "recompute_potentials_for_solution",
                "current_role_code_arrays",
                "charge_discharge_role_summary",
            )
            for name in required:
                if not hasattr(mod, name):
                    raise AttributeError(f"{module_name} does not expose {name}")
            return mod
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError(
        "Cannot import integration_spm/spm_int_assb_cycle.py.\n"
        "Run this script from the project root after copying it into integration_spm/.\n"
        + "\n".join(errors)
    )


def load_fixed_alignment(summary_path: Path, allow_fallback: bool = False) -> dict[str, float]:
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fit = meta.get("fit_report", {}) if isinstance(meta.get("fit_report", {}), dict) else {}
        values = {
            "theta_c_bottom": safe_float(meta.get("theta_c_bottom", fit.get("theta_c_bottom"))),
            "theta_c_top": safe_float(meta.get("theta_c_top", fit.get("theta_c_top"))),
            "U_p_offset_V": safe_float(meta.get("U_p_offset_V", fit.get("U_p_offset_V"))),
            "R_ohm_eff": safe_float(meta.get("R_ohm_eff", fit.get("R_ohm_eff"))),
            "csanmax": safe_float(meta.get("csanmax", 6.0), 6.0),
            "cscamax": safe_float(meta.get("cscamax", 51.8), 51.8),
        }
    elif allow_fallback:
        print(f"[WARN] fixed alignment summary not found: {summary_path}")
        print("[WARN] using embedded fallback values only because --allow_fallback_fixed_alignment was set.")
        values = {
            # These are v1-generator fallback values. Prefer the local summary JSON whenever possible.
            "theta_c_bottom": 0.9325,
            "theta_c_top": 0.4675,
            "U_p_offset_V": -0.2186690603079502,
            "R_ohm_eff": 143.6913493166367,
            "csanmax": 6.0,
            "cscamax": 51.8,
        }
    else:
        raise FileNotFoundError(
            f"Fixed alignment summary not found: {summary_path}\n"
            "Expected Data/assb_soft_labels_cycle5_v4/soft_label_summary.json. "
            "Use --allow_fallback_fixed_alignment only if intentional."
        )
    required = ("theta_c_bottom", "theta_c_top", "U_p_offset_V", "R_ohm_eff", "csanmax", "cscamax")
    missing = [k for k in required if not math.isfinite(values.get(k, float("nan")))]
    if missing:
        raise ValueError(f"Fixed alignment summary is missing required numeric fields: {missing}; file={summary_path}")
    return values


# -----------------------------------------------------------------------------
# Record metadata
# -----------------------------------------------------------------------------
def mode_string_from_current(I: float, raw_step: str | None = None) -> str:
    if raw_step:
        raw = str(raw_step)
        raw_low = raw.lower()
        if "搁置" in raw or "静置" in raw or raw_low in {"rest", "ocv"}:
            return "rest"
        if "充" in raw or "charge" in raw_low:
            return "constant_current_charge"
        if "放" in raw or "discharge" in raw_low:
            return "constant_current_discharge"
    if I > 1.0e-10:
        return "constant_current_charge"
    if I < -1.0e-10:
        return "constant_current_discharge"
    return "rest"


@dataclass
class RecordMetadata:
    t_global_s: np.ndarray
    cycle_id: np.ndarray
    step_id: np.ndarray
    step_type: np.ndarray
    data_index: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray | None


def extract_record_metadata(record_csv: str | Path, cycle_from: int, cycle_to: int | None) -> RecordMetadata:
    path = Path(record_csv)
    if not path.exists():
        raise FileNotFoundError(f"record_extracted.csv not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"record_extracted.csv is empty: {path}")

    cycle_col = find_column(df, CYCLE_COLUMNS, required=True)
    time_col = find_column(df, TIME_COLUMNS, required=True)
    current_col = find_column(df, CURRENT_COLUMNS, required=True)
    voltage_col = find_column(df, VOLTAGE_COLUMNS, required=False)
    step_id_col = find_column(df, STEP_ID_COLUMNS, required=False)
    step_type_col = find_column(df, STEP_TYPE_COLUMNS, required=False)
    data_index_col = find_column(df, DATA_INDEX_COLUMNS, required=False)

    cyc = pd.to_numeric(df[cycle_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(cyc) & (cyc >= int(cycle_from))
    if cycle_to is not None:
        mask = mask & (cyc <= int(cycle_to))
    selected = df.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No rows found for cycle_from={cycle_from}, cycle_to={cycle_to}")

    cyc_s = pd.to_numeric(selected[cycle_col], errors="coerce").to_numpy(dtype=np.float64)
    t_s = parse_time_seconds(selected[time_col])
    I = pd.to_numeric(selected[current_col], errors="coerce").to_numpy(dtype=np.float64)
    V = None if voltage_col is None else pd.to_numeric(selected[voltage_col], errors="coerce").to_numpy(dtype=np.float64)

    if step_id_col is None:
        step_id = np.arange(len(selected), dtype=np.float64)
    else:
        step_id = pd.to_numeric(selected[step_id_col], errors="coerce").to_numpy(dtype=np.float64)
    if data_index_col is None:
        data_idx = np.arange(len(selected), dtype=np.float64)
    else:
        data_idx = pd.to_numeric(selected[data_index_col], errors="coerce").to_numpy(dtype=np.float64)
    raw_step = np.array([None] * len(selected), dtype=object) if step_type_col is None else selected[step_type_col].astype(str).to_numpy(dtype=object)

    valid = np.isfinite(t_s) & np.isfinite(I) & np.isfinite(cyc_s)
    if V is not None:
        valid = valid & np.isfinite(V)
    t_s = t_s[valid]
    I = I[valid]
    cyc_s = cyc_s[valid]
    step_id = step_id[valid]
    data_idx = data_idx[valid]
    raw_step = raw_step[valid]
    if V is not None:
        V = V[valid]

    order = np.argsort(t_s, kind="mergesort")
    t_s = t_s[order]
    I = I[order]
    cyc_s = cyc_s[order]
    step_id = step_id[order]
    data_idx = data_idx[order]
    raw_step = raw_step[order]
    if V is not None:
        V = V[order]

    t_s = t_s - t_s[0]
    unique_t, inv = np.unique(t_s, return_inverse=True)
    n = int(unique_t.size)
    current_u = np.zeros(n, dtype=np.float64)
    cycle_u = np.zeros(n, dtype=np.float64)
    step_u = np.zeros(n, dtype=np.float64)
    data_u = np.zeros(n, dtype=np.float64)
    voltage_u = None if V is None else np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    step_type_first: list[str | None] = [None] * n
    for k, j in enumerate(inv):
        current_u[j] += I[k]
        cycle_u[j] += cyc_s[k]
        step_u[j] += step_id[k] if np.isfinite(step_id[k]) else 0.0
        data_u[j] += data_idx[k] if np.isfinite(data_idx[k]) else 0.0
        if voltage_u is not None and V is not None:
            voltage_u[j] += V[k]
        counts[j] += 1.0
        if step_type_first[j] is None:
            step_type_first[j] = str(raw_step[k]) if raw_step[k] is not None else None
    counts = np.maximum(counts, 1.0)
    current_u = current_u / counts
    cycle_id = np.rint(cycle_u / counts).astype(np.int32)
    step_id_i = np.rint(step_u / counts).astype(np.int32)
    data_i = np.rint(data_u / counts).astype(np.int64)
    voltage_out = None if voltage_u is None else voltage_u / counts
    step_type = np.array([mode_string_from_current(float(current_u[i]), step_type_first[i]) for i in range(n)], dtype="U32")

    return RecordMetadata(
        t_global_s=unique_t.astype(np.float64),
        cycle_id=cycle_id,
        step_id=step_id_i,
        step_type=step_type,
        data_index=data_i,
        current_A=current_u.astype(np.float64),
        voltage_V=None if voltage_out is None else voltage_out.astype(np.float64),
    )


# -----------------------------------------------------------------------------
# Load/integrate source solution
# -----------------------------------------------------------------------------
def load_solution_from_npz(source_dir_or_file: str | Path) -> tuple[dict, dict[str, Any]]:
    src = Path(source_dir_or_file)
    if src.is_dir():
        src = src / "solution.npz"
    if not src.exists():
        raise FileNotFoundError(f"source solution.npz not found: {src}")
    z = np.load(src, allow_pickle=True)

    def require(name: str) -> np.ndarray:
        if name not in z.files:
            raise KeyError(f"source solution.npz is missing required array: {name}")
        return np.asarray(z[name])

    t = require("t_global_s") if "t_global_s" in z.files else require("t")
    r_a = require("r_a")
    r_c = require("r_c")
    cs_a = require("cs_a").astype(np.float64)
    cs_c = require("cs_c").astype(np.float64)
    I_profile = require("I_profile").astype(np.float64)

    n_t = int(t.shape[0])
    n_r = int(r_c.shape[0])
    config = {
        "t": t.astype(np.float64),
        "r_a": r_a.astype(np.float64),
        "r_c": r_c.astype(np.float64),
        "n_t": n_t,
        "n_r": n_r,
        "dR_a": np.float64(r_a[1] - r_a[0]) if r_a.size > 1 else np.float64(0.0),
        "dR_c": np.float64(r_c[1] - r_c[0]) if r_c.size > 1 else np.float64(0.0),
    }
    sol = {
        "cs_a": cs_a,
        "cs_c": cs_c,
        "I_profile": I_profile,
        "phie": np.asarray(z["phie"], dtype=np.float64) if "phie" in z.files else np.zeros(n_t, dtype=np.float64),
        "phis_c": np.asarray(z["phis_c"], dtype=np.float64) if "phis_c" in z.files else np.zeros(n_t, dtype=np.float64),
        "j_a": np.asarray(z["j_a"], dtype=np.float64) if "j_a" in z.files else np.zeros(n_t, dtype=np.float64),
        "j_c": np.asarray(z["j_c"], dtype=np.float64) if "j_c" in z.files else np.zeros(n_t, dtype=np.float64),
        "eta_a": np.asarray(z["eta_a"], dtype=np.float64) if "eta_a" in z.files else np.zeros(n_t, dtype=np.float64),
        "eta_c": np.asarray(z["eta_c"], dtype=np.float64) if "eta_c" in z.files else np.zeros(n_t, dtype=np.float64),
        "Uocp_a": np.asarray(z["Uocp_a"], dtype=np.float64) if "Uocp_a" in z.files else np.zeros(n_t, dtype=np.float64),
        "Uocp_c": np.asarray(z["Uocp_c"], dtype=np.float64) if "Uocp_c" in z.files else np.zeros(n_t, dtype=np.float64),
    }
    aux: dict[str, Any] = {
        "source_solution": str(src),
        "cycle_id": np.asarray(z["cycle_id"], dtype=np.int32) if "cycle_id" in z.files else None,
        "cycle_profile": np.asarray(z["cycle_profile"], dtype=np.int32) if "cycle_profile" in z.files else None,
        "step_id": np.asarray(z["step_id"], dtype=np.int32) if "step_id" in z.files else None,
        "step_type": np.asarray(z["step_type"]) if "step_type" in z.files else None,
        "data_index": np.asarray(z["data_index"], dtype=np.int64) if "data_index" in z.files else None,
        "voltage_exp": np.asarray(z["voltage_exp"], dtype=np.float64) if "voltage_exp" in z.files else np.array([], dtype=np.float64),
        "all_source_keys": list(z.files),
    }
    return {"config": config, "sol": sol}, aux


def ensure_flux_arrays(solver: Any, sol: dict, params: dict) -> None:
    I = np.asarray(sol["I_profile"], dtype=np.float64)
    n_t = I.size
    if "j_a" not in sol or np.asarray(sol["j_a"]).shape[0] != n_t:
        sol["j_a"] = np.zeros(n_t, dtype=np.float64)
    if "j_c" not in sol or np.asarray(sol["j_c"]).shape[0] != n_t:
        sol["j_c"] = np.zeros(n_t, dtype=np.float64)
    if not np.all(np.isfinite(sol["j_a"])) or not np.all(np.isfinite(sol["j_c"])) or np.nanmax(np.abs(sol["j_c"])) == 0.0:
        for k, I_now in enumerate(I):
            sol["j_a"][k], sol["j_c"][k] = solver.surface_flux_from_current(float(I_now), params)


# -----------------------------------------------------------------------------
# Positive cbar mass closure repair
# -----------------------------------------------------------------------------
def integrated_positive_cbar_from_I(
    t: np.ndarray,
    I_profile: np.ndarray,
    params: dict,
    cbar0: float,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    I = np.asarray(I_profile, dtype=np.float64).reshape(-1)
    if t.size != I.size:
        raise ValueError(f"t and I length mismatch: {t.size} vs {I.size}")
    eps_c = float(params["eps_s_c"])
    F = float(params["F"])
    V_c = float(params.get("V_c", params["A_c"] * params["L_c"]))
    denom = eps_c * F * V_c
    if abs(denom) < 1e-300:
        raise ValueError("Invalid positive-electrode cbar denominator eps_s_c*F*V_c.")
    out = np.zeros_like(t, dtype=np.float64)
    out[0] = float(cbar0)
    for k in range(1, t.size):
        dt = float(t[k] - t[k - 1])
        if dt < 0:
            raise ValueError(f"Negative dt at k={k}: {dt}")
        # Match the project solver's step convention: I[k] drives the k-1 -> k update.
        out[k] = out[k - 1] - float(I[k]) * dt / denom
    return out


def apply_positive_mass_closure(
    config: dict,
    sol: dict,
    params: dict,
    clip: bool = False,
) -> dict[str, np.ndarray | float | int | bool]:
    t = np.asarray(config["t"], dtype=np.float64)
    r_c = np.asarray(config["r_c"], dtype=np.float64)
    cs_c_before = np.asarray(sol["cs_c"], dtype=np.float64)
    I = np.asarray(sol["I_profile"], dtype=np.float64)

    cbar_before = sphere_average(cs_c_before, r_c)
    cbar_from_I = integrated_positive_cbar_from_I(t, I, params, cbar0=float(cbar_before[0]))
    shift = cbar_from_I - cbar_before
    cs_c_after = cs_c_before + shift[:, None]

    cscamax = float(params["cscamax"])
    clipped_count = 0
    if clip:
        before_clip = cs_c_after.copy()
        cs_c_after = np.clip(cs_c_after, 0.0, cscamax)
        clipped_count = int(np.sum(np.abs(cs_c_after - before_clip) > 0.0))

    sol["cs_c"] = cs_c_after.astype(np.float64, copy=False)
    cbar_after = sphere_average(sol["cs_c"], r_c)

    return {
        "cbar_c_before_repair": cbar_before,
        "cbar_c_from_I": cbar_from_I,
        "cbar_c_after_repair": cbar_after,
        "cbar_c_shift_applied": shift,
        "max_abs_after_minus_I": float(np.nanmax(np.abs(cbar_after - cbar_from_I))),
        "mae_before_minus_I": float(np.nanmean(np.abs(cbar_before - cbar_from_I))),
        "bias_before_minus_I": float(np.nanmean(cbar_before - cbar_from_I)),
        "mae_after_minus_I": float(np.nanmean(np.abs(cbar_after - cbar_from_I))),
        "bias_after_minus_I": float(np.nanmean(cbar_after - cbar_from_I)),
        "max_abs_shift": float(np.nanmax(np.abs(shift))),
        "mean_shift": float(np.nanmean(shift)),
        "min_cs_c_after": float(np.nanmin(cs_c_after)),
        "max_cs_c_after": float(np.nanmax(cs_c_after)),
        "clip_enabled": bool(clip),
        "clipped_count": int(clipped_count),
    }


# -----------------------------------------------------------------------------
# Metrics and save helpers
# -----------------------------------------------------------------------------
def metrics_by_cycle_for_series(cycle_id: np.ndarray, y: np.ndarray, ref: np.ndarray, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cycle_id = np.asarray(cycle_id, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    for c in np.unique(cycle_id):
        idx = np.where(cycle_id == c)[0]
        if idx.size == 0:
            continue
        err = y[idx] - ref[idx]
        rows.append({
            "variable": name,
            "cycle_id": int(c),
            "n": int(idx.size),
            "mae": float(np.nanmean(np.abs(err))),
            "rmse": float(np.sqrt(np.nanmean(err ** 2))),
            "bias_mean": float(np.nanmean(err)),
            "max_abs": float(np.nanmax(np.abs(err))),
            "corr": corr_safe(y[idx], ref[idx]),
            "y_min": float(np.nanmin(y[idx])),
            "y_max": float(np.nanmax(y[idx])),
            "ref_min": float(np.nanmin(ref[idx])),
            "ref_max": float(np.nanmax(ref[idx])),
        })
    return rows


def voltage_metrics_by_cycle(cycle_id: np.ndarray, t: np.ndarray, I: np.ndarray, model_V: np.ndarray, exp_V: np.ndarray | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cycle_id = np.asarray(cycle_id, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    model_V = np.asarray(model_V, dtype=np.float64)
    exp = None if exp_V is None or len(exp_V) == 0 else np.asarray(exp_V, dtype=np.float64)
    for c in np.unique(cycle_id):
        idx = np.where(cycle_id == c)[0]
        if idx.size == 0:
            continue
        row: dict[str, Any] = {
            "cycle_id": int(c),
            "n_points": int(idx.size),
            "t_start_s": float(t[idx[0]]),
            "t_end_s": float(t[idx[-1]]),
            "duration_s": float(t[idx[-1]] - t[idx[0]]) if idx.size > 1 else 0.0,
            "I_min_A": float(np.nanmin(I[idx])),
            "I_max_A": float(np.nanmax(I[idx])),
            "V_model_min": float(np.nanmin(model_V[idx])),
            "V_model_max": float(np.nanmax(model_V[idx])),
        }
        if exp is not None and len(exp) == len(model_V):
            err = model_V[idx] - exp[idx]
            row.update({
                "V_exp_min": float(np.nanmin(exp[idx])),
                "V_exp_max": float(np.nanmax(exp[idx])),
                "V_mae_model_exp": float(np.nanmean(np.abs(err))),
                "V_rmse_model_exp": float(np.sqrt(np.nanmean(err ** 2))),
                "V_bias_model_exp": float(np.nanmean(err)),
                "V_max_abs_model_exp": float(np.nanmax(np.abs(err))),
                "V_corr_model_exp": corr_safe(model_V[idx], exp[idx]),
            })
        rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_step_type_from_I(I: np.ndarray) -> np.ndarray:
    return np.array([mode_string_from_current(float(x)) for x in np.asarray(I, dtype=np.float64)], dtype="U32")


def align_metadata(
    n_t: int,
    profile: Any,
    record_meta: RecordMetadata | None,
    aux: dict[str, Any] | None,
    I_profile: np.ndarray,
) -> dict[str, Any]:
    def good(arr: Any) -> bool:
        return arr is not None and len(arr) == n_t

    cycle_id = None
    step_id = None
    step_type = None
    data_index = None
    voltage_exp = None

    if aux:
        cycle_id = aux.get("cycle_id") if good(aux.get("cycle_id")) else aux.get("cycle_profile") if good(aux.get("cycle_profile")) else None
        step_id = aux.get("step_id") if good(aux.get("step_id")) else None
        step_type = aux.get("step_type") if good(aux.get("step_type")) else None
        data_index = aux.get("data_index") if good(aux.get("data_index")) else None
        voltage_exp = aux.get("voltage_exp") if good(aux.get("voltage_exp")) else None

    if cycle_id is None and record_meta is not None and good(record_meta.cycle_id):
        cycle_id = record_meta.cycle_id
    if step_id is None and record_meta is not None and good(record_meta.step_id):
        step_id = record_meta.step_id
    if step_type is None and record_meta is not None and good(record_meta.step_type):
        step_type = record_meta.step_type
    if data_index is None and record_meta is not None and good(record_meta.data_index):
        data_index = record_meta.data_index
    if voltage_exp is None and record_meta is not None and record_meta.voltage_V is not None and good(record_meta.voltage_V):
        voltage_exp = record_meta.voltage_V

    if cycle_id is None and hasattr(profile, "cycle_profile") and good(profile.cycle_profile):
        cycle_id = np.asarray(profile.cycle_profile, dtype=np.int32)
    if voltage_exp is None and hasattr(profile, "voltage_V") and profile.voltage_V is not None and good(profile.voltage_V):
        voltage_exp = np.asarray(profile.voltage_V, dtype=np.float64)

    if cycle_id is None:
        cycle_id = np.full(n_t, -1, dtype=np.int32)
    if step_id is None:
        step_id = np.arange(n_t, dtype=np.int32)
    if step_type is None:
        step_type = make_step_type_from_I(I_profile)
    if data_index is None:
        data_index = np.arange(n_t, dtype=np.int64)
    if voltage_exp is None:
        voltage_exp = np.array([], dtype=np.float64)

    return {
        "cycle_id": np.asarray(cycle_id, dtype=np.int32),
        "step_id": np.asarray(step_id, dtype=np.int32),
        "step_type": np.asarray(step_type),
        "data_index": np.asarray(data_index, dtype=np.int64),
        "voltage_exp": np.asarray(voltage_exp, dtype=np.float64),
    }


def save_solution_and_reports(
    output_dir: Path,
    config: dict,
    sol: dict,
    params: dict,
    profile: Any,
    fixed_alignment: dict[str, float],
    args: argparse.Namespace,
    audit: dict[str, Any],
    solver_checks: dict[str, Any],
    record_meta: RecordMetadata | None,
    aux: dict[str, Any] | None,
    source_mode: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(config["t"], dtype=np.float64)
    r_a = np.asarray(config["r_a"], dtype=np.float64)
    r_c = np.asarray(config["r_c"], dtype=np.float64)
    n_t = int(t.size)
    n_r = int(r_c.size)
    I_profile = np.asarray(sol["I_profile"], dtype=np.float64)
    meta = align_metadata(n_t, profile, record_meta, aux, I_profile)
    cycle_id = meta["cycle_id"]
    step_id = meta["step_id"]
    step_type = meta["step_type"]
    data_index = meta["data_index"]
    voltage_exp = meta["voltage_exp"]

    try:
        solver = load_project_solver()
        role_codes = solver.current_role_code_arrays(I_profile)
        role_summary = solver.charge_discharge_role_summary(I_profile)
    except Exception:
        role_codes = {
            "current_mode_code": np.where(I_profile > 1e-10, 1, np.where(I_profile < -1e-10, -1, 0)).astype(np.int8),
            "positive_electrode_role_code": np.zeros_like(I_profile, dtype=np.int8),
            "negative_electrode_role_code": np.zeros_like(I_profile, dtype=np.int8),
        }
        role_summary = {
            "current_sign_convention": "+I charge, -I discharge, 0 rest",
            "material_parameter_switching": "disabled",
        }

    theta_a = np.asarray(sol["cs_a"], dtype=np.float64) / float(params["csanmax"])
    theta_c = np.asarray(sol["cs_c"], dtype=np.float64) / float(params["cscamax"])
    phis_c_raw = np.asarray(sol.get("phis_c_raw", sol["phis_c"]), dtype=np.float64)
    voltage_alignment = np.asarray(sol.get("voltage_alignment_V", np.asarray(sol["phis_c"]) - phis_c_raw), dtype=np.float64)
    time_scale_s = float(t[-1] - t[0]) if n_t > 1 else 0.0
    dtype = str(args.dtype).lower()

    sol_path = output_dir / "solution.npz"
    np.savez_compressed(
        sol_path,
        t=cast_array(t, dtype),
        t_global_s=cast_array(t, dtype),
        time_scale_s=np.array([time_scale_s], dtype=np.float64),
        cycle_id=cycle_id.astype(np.int32, copy=False),
        cycle_profile=cycle_id.astype(np.int32, copy=False),
        step_id=step_id.astype(np.int32, copy=False),
        step_type=step_type,
        data_index=data_index.astype(np.int64, copy=False),
        r_a=cast_array(r_a, dtype),
        r_c=cast_array(r_c, dtype),
        cs_a=cast_array(sol["cs_a"], dtype),
        cs_c=cast_array(sol["cs_c"], dtype),
        theta_a=cast_array(theta_a, dtype),
        theta_c=cast_array(theta_c, dtype),
        phie=cast_array(sol["phie"], dtype),
        phis_c=cast_array(sol["phis_c"], dtype),
        phis_c_raw=cast_array(phis_c_raw, dtype),
        voltage_alignment_V=cast_array(voltage_alignment, dtype),
        phis_a=np.zeros(n_t, dtype=np.float32 if dtype == "float32" else np.float64),
        ce=np.full(n_t, float(params.get("ce0", 1.2)), dtype=np.float32 if dtype == "float32" else np.float64),
        I_profile=cast_array(I_profile, dtype),
        current_mode_code=role_codes["current_mode_code"],
        positive_electrode_role_code=role_codes["positive_electrode_role_code"],
        negative_electrode_role_code=role_codes["negative_electrode_role_code"],
        j_a=cast_array(sol["j_a"], dtype),
        j_c=cast_array(sol["j_c"], dtype),
        eta_a=cast_array(sol["eta_a"], dtype),
        eta_c=cast_array(sol["eta_c"], dtype),
        Uocp_a=cast_array(sol["Uocp_a"], dtype),
        Uocp_c=cast_array(sol["Uocp_c"], dtype),
        voltage_exp=cast_array(voltage_exp, dtype),
        # v2 audit arrays; these are small 1-D arrays and help downstream checks.
        cbar_c_before_repair=cast_array(audit["cbar_c_before_repair"], dtype),
        cbar_c_from_I=cast_array(audit["cbar_c_from_I"], dtype),
        cbar_c_after_repair=cast_array(audit["cbar_c_after_repair"], dtype),
        cbar_c_shift_applied=cast_array(audit["cbar_c_shift_applied"], dtype),
    )

    voltage_rows = voltage_metrics_by_cycle(cycle_id, t, I_profile, np.asarray(sol["phis_c"], dtype=np.float64), voltage_exp)
    voltage_metrics_path = output_dir / "metrics_voltage_fixedB_by_cycle.csv"
    save_csv(voltage_metrics_path, voltage_rows)

    by_cycle_rows: list[dict[str, Any]] = []
    by_cycle_rows += metrics_by_cycle_for_series(cycle_id, audit["cbar_c_before_repair"], audit["cbar_c_from_I"], "cbar_before_vs_I")
    by_cycle_rows += metrics_by_cycle_for_series(cycle_id, audit["cbar_c_after_repair"], audit["cbar_c_from_I"], "cbar_after_vs_I")
    by_cycle_rows += metrics_by_cycle_for_series(cycle_id, audit["cbar_c_shift_applied"], np.zeros_like(audit["cbar_c_shift_applied"]), "shift_applied")
    audit_by_cycle_path = output_dir / "mass_closure_audit_by_cycle.csv"
    save_csv(audit_by_cycle_path, by_cycle_rows)

    audit_global = {
        "version": "cycle5_522_v2_massclosed_candidate",
        "source_mode": source_mode,
        "source_solution": None if aux is None else aux.get("source_solution"),
        "repair_method": "uniform_shift_preserve_radial_shape",
        "positive_cbar_ode": "d<c_c>/dt = -I(t)/(eps_s_c*F*V_c)",
        "baseline_step_convention": "I[k] drives cbar[k-1] -> cbar[k], matching integration_spm.spm_int_assb_cycle.integrate_cycle",
        "clip_enabled": bool(audit["clip_enabled"]),
        "clipped_count": int(audit["clipped_count"]),
        "mae_before_minus_I": float(audit["mae_before_minus_I"]),
        "bias_before_minus_I": float(audit["bias_before_minus_I"]),
        "mae_after_minus_I": float(audit["mae_after_minus_I"]),
        "bias_after_minus_I": float(audit["bias_after_minus_I"]),
        "max_abs_after_minus_I": float(audit["max_abs_after_minus_I"]),
        "max_abs_shift": float(audit["max_abs_shift"]),
        "mean_shift": float(audit["mean_shift"]),
        "min_cs_c_after": float(audit["min_cs_c_after"]),
        "max_cs_c_after": float(audit["max_cs_c_after"]),
        "n_t": int(n_t),
        "n_r": int(n_r),
    }
    audit_global_path = output_dir / "mass_closure_audit_global.json"
    with open(audit_global_path, "w", encoding="utf-8") as f:
        json.dump(audit_global, f, ensure_ascii=False, indent=2)

    audit_timeseries_path = output_dir / "mass_closure_audit_timeseries.csv"
    if not args.no_audit_timeseries_csv:
        with open(audit_timeseries_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["t_global_s", "cycle_id", "I_profile_A", "cbar_before", "cbar_from_I", "cbar_after", "shift_applied", "after_minus_I"])
            cb = np.asarray(audit["cbar_c_before_repair"], dtype=np.float64)
            ci = np.asarray(audit["cbar_c_from_I"], dtype=np.float64)
            ca = np.asarray(audit["cbar_c_after_repair"], dtype=np.float64)
            sh = np.asarray(audit["cbar_c_shift_applied"], dtype=np.float64)
            for k in range(n_t):
                writer.writerow([float(t[k]), int(cycle_id[k]), float(I_profile[k]), float(cb[k]), float(ci[k]), float(ca[k]), float(sh[k]), float(ca[k] - ci[k])])

    unique_cycles = np.unique(cycle_id[cycle_id >= 0])
    exp_metrics: dict[str, Any] = {}
    if voltage_exp.size == n_t:
        err = np.asarray(sol["phis_c"], dtype=np.float64) - voltage_exp
        exp_metrics = {
            "V_exp_min": float(np.nanmin(voltage_exp)),
            "V_exp_max": float(np.nanmax(voltage_exp)),
            "V_mae_model_exp": float(np.nanmean(np.abs(err))),
            "V_rmse_model_exp": float(np.sqrt(np.nanmean(err ** 2))),
            "V_bias_model_exp": float(np.nanmean(err)),
            "V_max_abs_model_exp": float(np.nanmax(np.abs(err))),
            "V_corr_model_exp": corr_safe(np.asarray(sol["phis_c"], dtype=np.float64), voltage_exp),
        }

    summary = {
        "source": "generate_assb_soft_labels_cycle5_522_v2_massclosed_candidate.py",
        "based_on": "integration_spm/generate_assb_soft_labels_cycle5_522_v1.py + integration_spm/spm_int_assb_cycle.py",
        "version": "cycle5_to_522_continuous_fixedB_v2_massclosed_candidate",
        "record_csv": str(args.record_csv),
        "ocp_dir": str(args.ocp_dir),
        "output_dir": str(output_dir),
        "source_solution_dir": str(args.source_solution_dir),
        "source_mode": source_mode,
        "fixed_alignment_summary": str(args.fixed_alignment_summary),
        "continuous_state": True,
        "single_output": True,
        "cycle_from": int(args.cycle_from),
        "cycle_to": None if args.cycle_to is None else int(args.cycle_to),
        "selected_cycles_min": int(unique_cycles.min()) if unique_cycles.size else None,
        "selected_cycles_max": int(unique_cycles.max()) if unique_cycles.size else None,
        "n_selected_cycles": int(unique_cycles.size),
        "n_t": int(n_t),
        "n_r": int(n_r),
        "tmax_s": float(t[-1]) if n_t else 0.0,
        "time_scale_s": float(time_scale_s),
        "dt_median_s": float(np.median(np.diff(t))) if n_t > 1 else 0.0,
        "I_min_A": float(np.nanmin(I_profile)),
        "I_max_A": float(np.nanmax(I_profile)),
        "I_abs_max_A": float(np.nanmax(np.abs(I_profile))),
        "dtype_saved": dtype,
        "mass_closure_candidate": audit_global,
        "fixed_B_alignment": {
            "enabled": True,
            "theta_c_bottom": float(fixed_alignment["theta_c_bottom"]),
            "theta_c_top": float(fixed_alignment["theta_c_top"]),
            "U_p_offset_V": float(fixed_alignment["U_p_offset_V"]),
            "R_ohm_eff": float(fixed_alignment["R_ohm_eff"]),
            "per_cycle_voltage_refit": False,
        },
        "modeling_assumptions": {
            "positive_electrode": "NMC811 composite positive electrode; fixed material identity",
            "negative_electrode": "Li-In/In negative side represented as effective pseudo-particle; fixed material identity",
            "current_sign_convention": "+I charge, -I discharge, 0 rest",
            "surface_flux_closure": "J_a=-I*Rs_a/(3*eps_s_a*F*V_a), J_c=+I*Rs_c/(3*eps_s_c*F*V_c)",
            "aging_or_soh_mechanism": "not included in this v2 candidate dataset",
            "data_promotion_warning": "candidate only: audit OCP/voltage/theta consistency before using as final training target",
        },
        "parameters": {
            "theta_a0": float(params.get("theta_a0", np.nan)),
            "theta_c0": float(params.get("theta_c0", np.nan)),
            "csanmax": float(params["csanmax"]),
            "cscamax": float(params["cscamax"]),
            "Rs_a_m": float(params["Rs_a"]),
            "Rs_c_m": float(params["Rs_c"]),
            "eps_s_a": float(params["eps_s_a"]),
            "eps_s_c": float(params["eps_s_c"]),
            "V_a_m3": float(params.get("V_a", params["A_a"] * params["L_a"])),
            "V_c_m3": float(params.get("V_c", params["A_c"] * params["L_c"])),
            "T_K": float(params["T"]),
            "R_ohm_eff": float(params.get("R_ohm_eff", np.nan)),
            "U_p_offset_V": float(params.get("U_p_offset_V", np.nan)),
            "theta_c_bottom": float(params.get("theta_c_bottom", np.nan)),
            "theta_c_top": float(params.get("theta_c_top", np.nan)),
        },
        "charge_discharge_role_summary": role_summary,
        "global_voltage_metrics_fixedB": exp_metrics,
        "solver_sanity_checks": solver_checks,
        "files": {
            "solution": str(sol_path),
            "summary": str(output_dir / "soft_label_summary.json"),
            "record_profile_summary": str(output_dir / "record_profile_summary.json"),
            "metrics_voltage_fixedB_by_cycle": str(voltage_metrics_path),
            "mass_closure_audit_global": str(audit_global_path),
            "mass_closure_audit_by_cycle": str(audit_by_cycle_path),
            "mass_closure_audit_timeseries": None if args.no_audit_timeseries_csv else str(audit_timeseries_path),
        },
    }
    summary_path = output_dir / "soft_label_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    record_summary = {
        "record_csv": str(args.record_csv),
        "cycle_from": int(args.cycle_from),
        "cycle_to": None if args.cycle_to is None else int(args.cycle_to),
        "n_t": int(n_t),
        "n_selected_cycles": int(unique_cycles.size),
        "cycle_min": int(unique_cycles.min()) if unique_cycles.size else None,
        "cycle_max": int(unique_cycles.max()) if unique_cycles.size else None,
        "step_type_counts": {str(k): int(v) for k, v in zip(*np.unique(step_type, return_counts=True))},
        "current_unique_rounded_A_top20": sorted([float(x) for x in np.unique(np.round(I_profile, 12))])[:20],
        "note": "v2 candidate generated with positive cbar mass closure; candidate only until consistency audits pass.",
    }
    record_summary_path = output_dir / "record_profile_summary.json"
    with open(record_summary_path, "w", encoding="utf-8") as f:
        json.dump(record_summary, f, ensure_ascii=False, indent=2)

    return {
        "solution": sol_path,
        "summary": summary_path,
        "record_profile_summary": record_summary_path,
        "metrics_voltage_fixedB_by_cycle": voltage_metrics_path,
        "mass_closure_audit_global": audit_global_path,
        "mass_closure_audit_by_cycle": audit_by_cycle_path,
        "mass_closure_audit_timeseries": audit_timeseries_path,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate ASSB cycle5--522 v2 mass-closed-candidate soft labels."
    )
    p.add_argument("--record_csv", "--record-csv", dest="record_csv", type=str, default=str(DEFAULT_RECORD_CSV))
    p.add_argument("--ocp_dir", "--ocp-dir", dest="ocp_dir", type=str, default=str(DEFAULT_OCP_DIR))
    p.add_argument("--source_solution_dir", "--source-solution-dir", dest="source_solution_dir", type=str, default=str(DEFAULT_SOURCE_DIR), help="Existing v1 soft-label directory/file to repair. If missing or --force_regenerate_from_solver is set, v2 is regenerated from solver.")
    p.add_argument("--output_dir", "--out-dir", dest="output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--fixed_alignment_summary", "--fixed-alignment-summary", dest="fixed_alignment_summary", type=str, default=str(DEFAULT_FIXED_SUMMARY))
    p.add_argument("--allow_fallback_fixed_alignment", action="store_true", help="Use embedded v1 fallback values if the summary JSON is unavailable. Prefer the local summary.")
    p.add_argument("--cycle_from", "--cycle-from", dest="cycle_from", type=int, default=5)
    p.add_argument("--cycle_to", "--cycle-to", dest="cycle_to", type=int, default=522)
    p.add_argument("--n_r", "--n-r", dest="n_r", type=int, default=64)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--deg_i0_a", type=float, default=1.0)
    p.add_argument("--deg_ds_c", type=float, default=1.0)
    p.add_argument("--nonlinear_bv", action="store_true", help="Use nonlinear Butler-Volmer recomputation. Default keeps cycle5_v4 linearized BV.")
    p.add_argument("--no_ohmic", action="store_true", help="Disable I(t)*R_ohm_eff in recomputed terminal voltage; not recommended.")
    p.add_argument("--force_regenerate_from_solver", action="store_true", help="Ignore source_solution_dir and rerun the v1-style continuous integration before mass closure.")
    p.add_argument("--clip_cs_c", action="store_true", help="Clip repaired cs_c to [0, cscamax]. This can break exact mass closure; default is no clipping.")
    p.add_argument("--no_audit_timeseries_csv", action="store_true", help="Do not write the long mass_closure_audit_timeseries.csv file.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started = time.time()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}\n"
            "Use --overwrite only if you intentionally want to replace/update the candidate files."
        )

    print("[INFO] ASSB soft labels v2 mass-closed candidate generation")
    print(f"[INFO] repo_root           = {REPO_ROOT}")
    print(f"[INFO] record_csv          = {args.record_csv}")
    print(f"[INFO] ocp_dir             = {args.ocp_dir}")
    print(f"[INFO] source_solution_dir = {args.source_solution_dir}")
    print(f"[INFO] output_dir          = {output_dir}")
    print(f"[INFO] cycle range         = {args.cycle_from}--{args.cycle_to}")

    fixed = load_fixed_alignment(Path(args.fixed_alignment_summary), allow_fallback=bool(args.allow_fallback_fixed_alignment))
    print("[INFO] fixed-B alignment loaded:")
    for k in ("theta_c_bottom", "theta_c_top", "U_p_offset_V", "R_ohm_eff", "csanmax", "cscamax"):
        print(f"  {k:18s} = {fixed[k]}")

    solver = load_project_solver()
    params, profile = solver.load_merged_cycle_params(
        record_csv=args.record_csv,
        cycle_from=args.cycle_from,
        cycle_to=args.cycle_to,
        skip_activation_cycles=4,
        ocp_dir=args.ocp_dir,
        theta_c_bottom=fixed["theta_c_bottom"],
        theta_c_top=fixed["theta_c_top"],
        csanmax_eff=fixed["csanmax"],
        r_ohm_eff=fixed["R_ohm_eff"],
        up_offset_V=fixed["U_p_offset_V"],
    )
    params["deg_i0_a_current"] = np.float64(args.deg_i0_a)

    source_mode = "regenerated_from_solver"
    aux: dict[str, Any] | None = None
    src_path = Path(args.source_solution_dir) if str(args.source_solution_dir).strip() else Path("__missing__")
    use_source = (not args.force_regenerate_from_solver) and ((src_path / "solution.npz").exists() if src_path.is_dir() else src_path.exists())

    if use_source:
        print("[INFO] loading existing v1 solution and applying v2 mass closure")
        loaded, aux = load_solution_from_npz(src_path)
        config = loaded["config"]
        sol = loaded["sol"]
        source_mode = "repaired_existing_v1_solution"
        # Keep the solver profile aligned to the loaded solution arrays for potential recomputation.
        profile.time_s = np.asarray(config["t"], dtype=np.float64)
        profile.current_A = np.asarray(sol["I_profile"], dtype=np.float64)
        if aux.get("voltage_exp") is not None and len(aux.get("voltage_exp")) == len(profile.time_s):
            profile.voltage_V = np.asarray(aux.get("voltage_exp"), dtype=np.float64)
    else:
        print("[INFO] source solution unavailable or ignored; rerunning v1-style solver integration first")
        config, sol = solver.integrate_cycle(
            params=params,
            profile=profile,
            n_r=args.n_r,
            deg_i0_a=args.deg_i0_a,
            deg_ds_c=args.deg_ds_c,
            linearized_bv=not args.nonlinear_bv,
            include_ohmic=not args.no_ohmic,
            verbose=not args.quiet,
        )

    ensure_flux_arrays(solver, sol, params)

    # Record metadata is only needed if not already present in the source solution.
    record_meta: RecordMetadata | None = None
    try:
        record_meta = extract_record_metadata(args.record_csv, args.cycle_from, args.cycle_to)
    except Exception as exc:
        print(f"[WARN] record metadata extraction failed; using source/profile metadata only. Reason: {exc}")

    print("[INFO] applying positive-electrode mass closure")
    audit = apply_positive_mass_closure(config, sol, params, clip=bool(args.clip_cs_c))
    print(f"[INFO] positive cbar before-vs-I MAE = {audit['mae_before_minus_I']:.8g}")
    print(f"[INFO] positive cbar after -vs-I MAE = {audit['mae_after_minus_I']:.8g}")
    print(f"[INFO] max |after-I| = {audit['max_abs_after_minus_I']:.8g}")
    print(f"[INFO] max |shift|   = {audit['max_abs_shift']:.8g}")
    if float(audit["min_cs_c_after"]) < 0.0 or float(audit["max_cs_c_after"]) > float(params["cscamax"]):
        print("[WARN] repaired cs_c has values outside [0, cscamax]. This is not automatically clipped unless --clip_cs_c is used.")
        print(f"[WARN] min_cs_c_after={audit['min_cs_c_after']:.8g}, max_cs_c_after={audit['max_cs_c_after']:.8g}, cscamax={float(params['cscamax']):.8g}")
    if bool(audit["clip_enabled"]):
        print(f"[WARN] --clip_cs_c was used; clipped_count={audit['clipped_count']}. Exact mass closure may be weakened.")

    print("[INFO] recomputing phie/phis_c/Uocp/eta after cs_c repair")
    solver.recompute_potentials_for_solution(
        params=params,
        profile=profile,
        sol=sol,
        deg_i0_a=args.deg_i0_a,
        linearized_bv=not args.nonlinear_bv,
        include_ohmic=not args.no_ohmic,
    )

    checks = solver.sanity_check_solution(params, profile, sol) if hasattr(solver, "sanity_check_solution") else {}
    paths = save_solution_and_reports(
        output_dir=output_dir,
        config=config,
        sol=sol,
        params=params,
        profile=profile,
        fixed_alignment=fixed,
        args=args,
        audit=audit,
        solver_checks=checks,
        record_meta=record_meta,
        aux=aux,
        source_mode=source_mode,
    )

    print("[INFO] files written:")
    for k, p in paths.items():
        if k == "mass_closure_audit_timeseries" and args.no_audit_timeseries_csv:
            continue
        print(f"  {k:36s}: {p}")
    if checks:
        print("[INFO] solver sanity checks:")
        for k, v in checks.items():
            if isinstance(v, (float, int, np.floating, np.integer)):
                print(f"  {k:28s}: {float(v):.8g}")
            else:
                print(f"  {k:28s}: {v}")
    print(f"[INFO] completed in {time.time() - started:.2f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
