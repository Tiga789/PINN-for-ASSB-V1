#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Generate continuous ASSB soft labels for cycle 5--522 with fixed cycle5_v4
voltage-alignment parameters.

This script is intentionally a thin, conservative wrapper around the current
project solver:

    integration_spm/spm_int_assb_cycle.py

It does NOT change the electrode-fixed physics in that solver. It imports the
same parameter loader, OCP installer, current-flux closure, and implicit radial
integrator that produced the cycle5_v4 soft labels, then saves a compact single
continuous soft-label dataset with cycle_id retained.

Default output directory is fixed by the current experiment requirement:

    C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1

Main design choices
-------------------
1. Continuous-state solution: cycle k final concentration is cycle k+1 initial
   concentration; no per-cycle reset.
2. Fixed-B voltage closure: theta_c_bottom, theta_c_top, U_p_offset_V and
   R_ohm_eff are read from Data/assb_soft_labels_cycle5_v4/soft_label_summary.json.
   No per-cycle voltage fitting is performed.
3. Single output directory and single main solution.npz. cycle_id is stored in
   solution.npz for later per-cycle slicing.
4. Legacy data_*.npz files are NOT written by default because all-cycle
   concentration datasets would be very large. They can be requested explicitly
   with --write_legacy_data_files for short smoke ranges only.

Run from the repository root, for example:

D:\Anaconda\envs\torchgpu\python.exe .\integration_spm\generate_assb_soft_labels_cycle5_522_v1.py `
  --record_csv "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
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
    raise RuntimeError("This generator requires pandas to read record_extracted.csv.") from exc

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
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1")
DEFAULT_FIXED_SUMMARY = REPO_ROOT / "Data" / "assb_soft_labels_cycle5_v4" / "soft_label_summary.json"

# Column candidates aligned with the project solver, plus the extra record fields
# that are useful for all-cycle indexing.
CYCLE_COLUMNS = ("循环号", "cycle", "Cycle", "cycle_index", "Cycle_Index", "循环", "循环序号")
TIME_COLUMNS = ("总时间", "总时间(s)", "time_s", "Time(s)", "time", "Time", "时间", "测试时间")
CURRENT_COLUMNS = ("电流(A)", "电流", "current_A", "Current(A)", "current", "Current", "I_A", "I(A)")
VOLTAGE_COLUMNS = ("电压(V)", "电压", "voltage_V", "Voltage(V)", "voltage", "Voltage", "V")
STEP_ID_COLUMNS = ("工步号", "step_id", "Step_ID", "step", "Step", "step_index", "Step_Index")
STEP_TYPE_COLUMNS = ("工步类型", "step_type", "Step_Type", "step_name", "Step_Name", "工步名称")
DATA_INDEX_COLUMNS = ("数据序号", "data_index", "Data_Index", "index", "Index")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
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


def _parse_time_seconds(values: Any) -> np.ndarray:
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


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _load_project_solver() -> Any:
    """Import the existing project solver without modifying it."""
    errors: list[str] = []
    for module_name in ("integration_spm.spm_int_assb_cycle", "spm_int_assb_cycle"):
        try:
            mod = importlib.import_module(module_name)
            for required in ("load_merged_cycle_params", "integrate_cycle", "current_role_code_arrays", "charge_discharge_role_summary"):
                if not hasattr(mod, required):
                    raise AttributeError(f"{module_name} does not expose {required}")
            return mod
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    raise RuntimeError(
        "Cannot import the current solver integration_spm/spm_int_assb_cycle.py.\n"
        "Run this script from the project root after copying it into integration_spm/.\n"
        + "\n".join(errors)
    )


def load_fixed_alignment(summary_path: Path, allow_fallback: bool = False) -> dict[str, float]:
    """Load cycle5_v4 fixed-B alignment values."""
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fit = meta.get("fit_report", {}) if isinstance(meta.get("fit_report", {}), dict) else {}
        values = {
            "theta_c_bottom": _safe_float(meta.get("theta_c_bottom", fit.get("theta_c_bottom"))),
            "theta_c_top": _safe_float(meta.get("theta_c_top", fit.get("theta_c_top"))),
            "U_p_offset_V": _safe_float(meta.get("U_p_offset_V", fit.get("U_p_offset_V"))),
            "R_ohm_eff": _safe_float(meta.get("R_ohm_eff", fit.get("R_ohm_eff"))),
            "csanmax": _safe_float(meta.get("csanmax", 6.0), 6.0),
            "cscamax": _safe_float(meta.get("cscamax", 51.8), 51.8),
        }
    elif allow_fallback:
        print(f"[WARN] fixed alignment summary not found: {summary_path}")
        print("[WARN] using hard-coded cycle5_v4 values from the project record.")
        values = {
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
            "Use --allow_fallback_fixed_alignment only if you explicitly want the embedded cycle5_v4 values."
        )

    required = ("theta_c_bottom", "theta_c_top", "U_p_offset_V", "R_ohm_eff", "csanmax")
    missing = [k for k in required if not math.isfinite(values.get(k, float("nan")))]
    if missing:
        raise ValueError(f"Fixed alignment summary is missing required numeric fields: {missing}; file={summary_path}")
    return values


@dataclass
class RecordMetadata:
    t_global_s: np.ndarray
    cycle_id: np.ndarray
    step_id: np.ndarray
    step_type: np.ndarray
    data_index: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray | None


def _mode_string_from_current(I: float, raw_step: str | None = None) -> str:
    if raw_step:
        raw = str(raw_step)
        if "搁置" in raw or "静置" in raw or raw.lower() in {"rest", "ocv"}:
            return "rest"
        if "充" in raw or "charge" in raw.lower():
            return "constant_current_charge"
        if "放" in raw or "discharge" in raw.lower():
            return "constant_current_discharge"
    if I > 1.0e-10:
        return "constant_current_charge"
    if I < -1.0e-10:
        return "constant_current_discharge"
    return "rest"


def extract_record_metadata(
    record_csv: str | Path,
    cycle_from: int,
    cycle_to: int | None,
) -> RecordMetadata:
    """Read cycle_id / step_id / step_type aligned to the solver's merged profile."""
    path = Path(record_csv)
    if not path.exists():
        raise FileNotFoundError(f"record_extracted.csv not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"record_extracted.csv is empty: {path}")

    cycle_col = _find_column(df, CYCLE_COLUMNS, required=True)
    time_col = _find_column(df, TIME_COLUMNS, required=True)
    current_col = _find_column(df, CURRENT_COLUMNS, required=True)
    voltage_col = _find_column(df, VOLTAGE_COLUMNS, required=False)
    step_id_col = _find_column(df, STEP_ID_COLUMNS, required=False)
    step_type_col = _find_column(df, STEP_TYPE_COLUMNS, required=False)
    data_index_col = _find_column(df, DATA_INDEX_COLUMNS, required=False)

    cyc = pd.to_numeric(df[cycle_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(cyc) & (cyc >= int(cycle_from))
    if cycle_to is not None:
        mask = mask & (cyc <= int(cycle_to))
    selected = df.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No rows found for cycle_from={cycle_from}, cycle_to={cycle_to}")

    cyc_s = pd.to_numeric(selected[cycle_col], errors="coerce").to_numpy(dtype=np.float64)
    t_s = _parse_time_seconds(selected[time_col])
    I = pd.to_numeric(selected[current_col], errors="coerce").to_numpy(dtype=np.float64)
    V = None if voltage_col is None else pd.to_numeric(selected[voltage_col], errors="coerce").to_numpy(dtype=np.float64)
    step_id = (
        np.arange(len(selected), dtype=np.float64)
        if step_id_col is None
        else pd.to_numeric(selected[step_id_col], errors="coerce").to_numpy(dtype=np.float64)
    )
    if data_index_col is None:
        data_idx = np.arange(len(selected), dtype=np.float64)
    else:
        data_idx = pd.to_numeric(selected[data_index_col], errors="coerce").to_numpy(dtype=np.float64)

    if step_type_col is None:
        raw_step = np.array([None] * len(selected), dtype=object)
    else:
        raw_step = selected[step_type_col].astype(str).to_numpy(dtype=object)

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

    # Merge duplicate timestamps in the same way as the project solver: average
    # numeric quantities and round cycle/step identifiers.
    unique_t, inv = np.unique(t_s, return_inverse=True)
    n = unique_t.size
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
    step_type = np.array(
        [_mode_string_from_current(float(current_u[i]), step_type_first[i]) for i in range(n)],
        dtype="U32",
    )

    return RecordMetadata(
        t_global_s=unique_t.astype(np.float64),
        cycle_id=cycle_id,
        step_id=step_id_i,
        step_type=step_type,
        data_index=data_i,
        current_A=current_u.astype(np.float64),
        voltage_V=None if voltage_out is None else voltage_out.astype(np.float64),
    )


def _cast_array(arr: Any, dtype: str) -> np.ndarray:
    a = np.asarray(arr)
    if dtype == "float64":
        return a.astype(np.float64, copy=False) if np.issubdtype(a.dtype, np.floating) else a
    if dtype == "float32" and np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32, copy=False)
    return a


def _corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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
            row.update(
                {
                    "V_exp_min": float(np.nanmin(exp[idx])),
                    "V_exp_max": float(np.nanmax(exp[idx])),
                    "V_mae_model_exp": float(np.nanmean(np.abs(err))),
                    "V_rmse_model_exp": float(np.sqrt(np.nanmean(err**2))),
                    "V_bias_model_exp": float(np.nanmean(err)),
                    "V_max_abs_model_exp": float(np.nanmax(np.abs(err))),
                    "V_corr_model_exp": _corr_safe(model_V[idx], exp[idx]),
                }
            )
        rows.append(row)
    return rows


def save_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def save_continuous_solution(
    output_dir: Path,
    config: dict,
    sol: dict,
    params: dict,
    profile: Any,
    record_meta: RecordMetadata,
    fixed_alignment: dict[str, float],
    args: argparse.Namespace,
    solver_checks: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(config["t"], dtype=np.float64)
    n_t = int(config["n_t"])
    n_r = int(config["n_r"])

    cycle_id = record_meta.cycle_id
    if cycle_id.shape[0] != n_t:
        print(f"[WARN] metadata cycle_id length {cycle_id.shape[0]} != solver n_t {n_t}; falling back to profile.cycle_profile if available.")
        if hasattr(profile, "cycle_profile") and len(profile.cycle_profile) == n_t:
            cycle_id = np.asarray(profile.cycle_profile, dtype=np.int32)
        else:
            cycle_id = np.full(n_t, -1, dtype=np.int32)

    step_id = record_meta.step_id if record_meta.step_id.shape[0] == n_t else np.zeros(n_t, dtype=np.int32)
    step_type = record_meta.step_type if record_meta.step_type.shape[0] == n_t else np.array(["unknown"] * n_t, dtype="U32")
    data_index = record_meta.data_index if record_meta.data_index.shape[0] == n_t else np.arange(n_t, dtype=np.int64)

    # Prefer the solver profile voltage/current to guarantee alignment.
    voltage_exp = np.array([], dtype=np.float64) if profile.voltage_V is None else np.asarray(profile.voltage_V, dtype=np.float64)
    I_profile = np.asarray(sol["I_profile"], dtype=np.float64)

    role_codes = None
    try:
        solver_mod = _load_project_solver()
        role_codes = solver_mod.current_role_code_arrays(I_profile)
    except Exception:
        role_codes = {
            "current_mode_code": np.where(I_profile > 1e-10, 1, np.where(I_profile < -1e-10, -1, 0)).astype(np.int8),
            "positive_electrode_role_code": np.zeros_like(I_profile, dtype=np.int8),
            "negative_electrode_role_code": np.zeros_like(I_profile, dtype=np.int8),
        }

    theta_a = np.asarray(sol["cs_a"], dtype=np.float64) / float(params["csanmax"])
    theta_c = np.asarray(sol["cs_c"], dtype=np.float64) / float(params["cscamax"])
    phis_c_raw = np.asarray(sol.get("phis_c_raw", sol["phis_c"]), dtype=np.float64)
    voltage_alignment = np.asarray(sol.get("voltage_alignment_V", np.asarray(sol["phis_c"]) - phis_c_raw), dtype=np.float64)
    time_scale_s = float(t[-1] - t[0]) if n_t > 1 else 0.0

    sol_path = output_dir / "solution.npz"
    dtype = str(args.dtype).lower()
    np.savez_compressed(
        sol_path,
        # Time aliases: t keeps compatibility with older scripts; t_global_s is the new explicit long-sequence name.
        t=_cast_array(t, dtype),
        t_global_s=_cast_array(t, dtype),
        time_scale_s=np.array([time_scale_s], dtype=np.float64),
        cycle_id=cycle_id.astype(np.int32, copy=False),
        cycle_profile=cycle_id.astype(np.int32, copy=False),
        step_id=step_id.astype(np.int32, copy=False),
        step_type=step_type,
        data_index=data_index.astype(np.int64, copy=False),
        r_a=_cast_array(config["r_a"], dtype),
        r_c=_cast_array(config["r_c"], dtype),
        cs_a=_cast_array(sol["cs_a"], dtype),
        cs_c=_cast_array(sol["cs_c"], dtype),
        theta_a=_cast_array(theta_a, dtype),
        theta_c=_cast_array(theta_c, dtype),
        phie=_cast_array(sol["phie"], dtype),
        phis_c=_cast_array(sol["phis_c"], dtype),
        phis_c_raw=_cast_array(phis_c_raw, dtype),
        voltage_alignment_V=_cast_array(voltage_alignment, dtype),
        phis_a=np.zeros(n_t, dtype=np.float32 if dtype == "float32" else np.float64),
        ce=np.full(n_t, float(params.get("ce0", 1.2)), dtype=np.float32 if dtype == "float32" else np.float64),
        I_profile=_cast_array(I_profile, dtype),
        current_mode_code=role_codes["current_mode_code"],
        positive_electrode_role_code=role_codes["positive_electrode_role_code"],
        negative_electrode_role_code=role_codes["negative_electrode_role_code"],
        j_a=_cast_array(sol["j_a"], dtype),
        j_c=_cast_array(sol["j_c"], dtype),
        eta_a=_cast_array(sol["eta_a"], dtype),
        eta_c=_cast_array(sol["eta_c"], dtype),
        Uocp_a=_cast_array(sol["Uocp_a"], dtype),
        Uocp_c=_cast_array(sol["Uocp_c"], dtype),
        voltage_exp=_cast_array(voltage_exp, dtype),
    )

    per_cycle_rows = voltage_metrics_by_cycle(cycle_id, t, I_profile, np.asarray(sol["phis_c"], dtype=np.float64), voltage_exp)
    metrics_path = output_dir / "metrics_voltage_fixedB_by_cycle.csv"
    save_metrics_csv(metrics_path, per_cycle_rows)

    unique_cycles = np.unique(cycle_id[cycle_id >= 0])
    exp_metrics: dict[str, Any] = {}
    if voltage_exp.size == n_t:
        err = np.asarray(sol["phis_c"], dtype=np.float64) - voltage_exp
        exp_metrics = {
            "V_exp_min": float(np.nanmin(voltage_exp)),
            "V_exp_max": float(np.nanmax(voltage_exp)),
            "V_mae_model_exp": float(np.nanmean(np.abs(err))),
            "V_rmse_model_exp": float(np.sqrt(np.nanmean(err**2))),
            "V_bias_model_exp": float(np.nanmean(err)),
            "V_max_abs_model_exp": float(np.nanmax(np.abs(err))),
            "V_corr_model_exp": _corr_safe(np.asarray(sol["phis_c"], dtype=np.float64), voltage_exp),
        }

    role_summary: dict[str, Any]
    try:
        role_summary = _load_project_solver().charge_discharge_role_summary(I_profile)
    except Exception:
        role_summary = {
            "current_sign_convention": "+I charge, -I discharge, 0 rest",
            "material_parameter_switching": "disabled",
        }

    summary = {
        "source": "generate_assb_soft_labels_cycle5_522_v1.py",
        "uses_solver": "integration_spm/spm_int_assb_cycle.py",
        "version": "cycle5_to_522_continuous_fixedB_v1",
        "record_csv": str(args.record_csv),
        "ocp_dir": str(args.ocp_dir),
        "output_dir": str(output_dir),
        "fixed_output_path_required_by_experiment": str(DEFAULT_OUTPUT_DIR),
        "fixed_alignment_summary": str(args.fixed_alignment_summary),
        "continuous_state": True,
        "single_output": True,
        "legacy_data_files_written": bool(args.write_legacy_data_files),
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
            "aging_or_soh_mechanism": "not included in this v1 soft-label dataset",
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
            "V_a_m3": float(params["V_a"]),
            "V_c_m3": float(params["V_c"]),
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
            "metrics_voltage_fixedB_by_cycle": str(metrics_path),
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
        "note": "step_type is normalized to constant_current_charge / constant_current_discharge / rest for downstream scripts.",
    }
    with open(output_dir / "record_profile_summary.json", "w", encoding="utf-8") as f:
        json.dump(record_summary, f, ensure_ascii=False, indent=2)

    paths = {
        "solution": sol_path,
        "summary": summary_path,
        "record_profile_summary": output_dir / "record_profile_summary.json",
        "metrics_voltage_fixedB_by_cycle": metrics_path,
    }
    return paths


def maybe_write_legacy_data_files(
    output_dir: Path,
    config: dict,
    sol: dict,
    deg_i0_a: float,
    deg_ds_c: float,
    dtype: str,
) -> dict[str, Path]:
    """Optional old data_*.npz writer. Intended for short smoke ranges only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t = np.asarray(config["t"], dtype=np.float64)
    r_a = np.asarray(config["r_a"], dtype=np.float64)
    r_c = np.asarray(config["r_c"], dtype=np.float64)

    def params_matrix(n: int) -> np.ndarray:
        out = np.zeros((n, 2), dtype=np.float32 if dtype == "float32" else np.float64)
        out[:, 0] = deg_i0_a
        out[:, 1] = deg_ds_c
        return out

    def save_state(path: Path, x: np.ndarray, y: np.ndarray) -> None:
        y = y.reshape(-1, 1)
        if dtype == "float32":
            x = x.astype(np.float32, copy=False)
            y = y.astype(np.float32, copy=False)
        np.savez_compressed(path, x_train=x, y_train=y, x_params_train=params_matrix(y.shape[0]))

    x_t = t.reshape(-1, 1)
    save_state(output_dir / "data_phie.npz", x_t, np.asarray(sol["phie"]))
    save_state(output_dir / "data_phis_c.npz", x_t, np.asarray(sol["phis_c"]))

    tt_a, rr_a = np.meshgrid(t, r_a, indexing="ij")
    tt_c, rr_c = np.meshgrid(t, r_c, indexing="ij")
    save_state(output_dir / "data_cs_a.npz", np.column_stack([tt_a.reshape(-1), rr_a.reshape(-1)]), np.asarray(sol["cs_a"]).reshape(-1))
    save_state(output_dir / "data_cs_c.npz", np.column_stack([tt_c.reshape(-1), rr_c.reshape(-1)]), np.asarray(sol["cs_c"]).reshape(-1))
    return {
        "data_phie": output_dir / "data_phie.npz",
        "data_phis_c": output_dir / "data_phis_c.npz",
        "data_cs_a": output_dir / "data_cs_a.npz",
        "data_cs_c": output_dir / "data_cs_c.npz",
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate cycle5--522 continuous ASSB soft labels using cycle5_v4 fixed-B alignment."
    )
    p.add_argument("--record_csv", "--record-csv", dest="record_csv", type=str, default=str(DEFAULT_RECORD_CSV))
    p.add_argument("--ocp_dir", "--ocp-dir", dest="ocp_dir", type=str, default=str(DEFAULT_OCP_DIR))
    p.add_argument("--output_dir", "--out-dir", dest="output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--fixed_alignment_summary", "--fixed-alignment-summary", dest="fixed_alignment_summary", type=str, default=str(DEFAULT_FIXED_SUMMARY))
    p.add_argument("--allow_fallback_fixed_alignment", action="store_true", help="Use embedded cycle5_v4 fixed-B values if the summary JSON is unavailable.")
    p.add_argument("--cycle_from", "--cycle-from", dest="cycle_from", type=int, default=5)
    p.add_argument("--cycle_to", "--cycle-to", dest="cycle_to", type=int, default=522)
    p.add_argument("--n_r", "--n-r", dest="n_r", type=int, default=64)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Floating dtype for saved arrays; calculation remains float64.")
    p.add_argument("--deg_i0_a", type=float, default=1.0)
    p.add_argument("--deg_ds_c", type=float, default=1.0)
    p.add_argument("--nonlinear_bv", action="store_true", help="Use nonlinear Butler-Volmer inversion. Default keeps cycle5_v4 linearized BV.")
    p.add_argument("--no_ohmic", action="store_true", help="Disable I(t)*R_ohm_eff in terminal voltage closure; not recommended for fixed-B.")
    p.add_argument("--write_legacy_data_files", action="store_true", help="Also write data_phie/data_phis_c/data_cs_a/data_cs_c. Use only for short smoke ranges due to size.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t_start = time.time()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}\n"
            "Use --overwrite to replace/update files intentionally."
        )

    print("[INFO] Continuous ASSB soft-label generation: cycle5--522 fixed-B v1")
    print(f"[INFO] repo_root       = {REPO_ROOT}")
    print(f"[INFO] record_csv      = {args.record_csv}")
    print(f"[INFO] ocp_dir         = {args.ocp_dir}")
    print(f"[INFO] output_dir      = {output_dir}")
    print(f"[INFO] cycle range     = {args.cycle_from}--{args.cycle_to}")
    print(f"[INFO] n_r             = {args.n_r}")

    fixed = load_fixed_alignment(Path(args.fixed_alignment_summary), allow_fallback=bool(args.allow_fallback_fixed_alignment))
    print("[INFO] fixed-B cycle5_v4 alignment:")
    for k in ("theta_c_bottom", "theta_c_top", "U_p_offset_V", "R_ohm_eff", "csanmax"):
        print(f"       {k:18s} = {fixed[k]}")

    solver = _load_project_solver()

    # Load params using the project solver's merged profile path, but inject the
    # fixed cycle5_v4 voltage-closure values. This is the key 'B方案' behavior.
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

    print(f"[INFO] merged profile  = {getattr(profile, 'profile_label', 'merged')}")
    print(f"[INFO] n_t             = {profile.n_points}")
    print(f"[INFO] tmax            = {float(profile.tmax):.6g} s")
    print(f"[INFO] I range         = {float(np.min(profile.current_A)):.6g} A .. {float(np.max(profile.current_A)):.6g} A")

    meta = extract_record_metadata(args.record_csv, args.cycle_from, args.cycle_to)
    if meta.t_global_s.shape[0] != profile.n_points:
        print(f"[WARN] metadata length {meta.t_global_s.shape[0]} != solver profile length {profile.n_points}; cycle_id may be taken from solver profile.")

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

    checks = solver.sanity_check_solution(params, profile, sol) if hasattr(solver, "sanity_check_solution") else {}
    paths = save_continuous_solution(
        output_dir=output_dir,
        config=config,
        sol=sol,
        params=params,
        profile=profile,
        record_meta=meta,
        fixed_alignment=fixed,
        args=args,
        solver_checks=checks,
    )

    if args.write_legacy_data_files:
        print("[WARN] Writing legacy data_*.npz files. For full 5--522 this may consume large disk space.")
        paths.update(maybe_write_legacy_data_files(output_dir, config, sol, args.deg_i0_a, args.deg_ds_c, args.dtype))

    print("[INFO] files written:")
    for k, p in paths.items():
        print(f"       {k:34s}: {p}")
    print("[INFO] sanity checks:")
    for k, v in checks.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            print(f"       {k:24s}: {float(v):.8g}")
        else:
            print(f"       {k:24s}: {v}")
    print(f"[INFO] completed in {time.time() - t_start:.2f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
