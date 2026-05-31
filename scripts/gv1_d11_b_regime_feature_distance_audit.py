#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GV1 D11-B regime feature distance audit.

Report-only audit for B1_2C battery-8.  The script reads existing GV1/XJTU
profile/cache files and D10 policy artifacts, computes profile/window/segment
features, and measures battery-8 distance from B1_2C peers.

It intentionally does NOT modify gv1/model.py, gv1/output_transform.py,
gv1/losses.py, gv1/trainer.py, or scripts/gv1_train_conditioned_pinn.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    print(json.dumps({"ok": False, "error": "numpy_import_failed", "detail": str(exc)}, ensure_ascii=False))
    sys.exit(2)

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print(json.dumps({"ok": False, "error": "pandas_import_failed", "detail": str(exc)}, ensure_ascii=False))
    sys.exit(2)

EXPECTED_D10P5_VERDICT = "d10_p5_mainline_freeze_and_regime_policy_ready_for_d11"
EXPECTED_D10P0_VERDICT = "battery8_flagged_late_2C_discharge_regime_outlier_keep_D9_6_mainline"
EXPECTED_D10P3_VERDICT = "no_safe_lightweight_correction_keep_battery8_flagged"

ID_COLUMNS = [
    "profile_id", "dataset_id", "batch_id", "battery_id", "cell_uid", "protocol", "split",
    "profile_npz", "source_file", "source_path", "parquet_path", "solution_npz",
]

TIME_CANDIDATES = ["t_global_s", "time_s", "t_s", "test_time_s", "elapsed_time_s", "Time_s", "time"]
CURRENT_CANDIDATES = ["I_profile", "current_A", "I_A", "Current_A", "current", "Current"]
VOLTAGE_CANDIDATES = ["voltage_exp", "voltage_V", "Voltage_V", "voltage", "Voltage"]
TEMP_CANDIDATES = ["temperature_C", "T_C", "Temperature_C", "temperature", "Temperature"]
CYCLE_CANDIDATES = ["cycle_id", "cycle", "Cycle_Index", "cycle_index"]
STEP_CANDIDATES = ["step_id", "step", "Step_Index", "step_index"]
STEPTYPE_CANDIDATES = ["step_type", "Step_Type", "status", "mode"]

Q_DIS_CANDIDATES = ["q_discharge_Ah", "Q_dis_Ah", "discharge_capacity_Ah", "capacity_discharge_Ah", "Qd_Ah", "q_discharge"]
Q_CHG_CANDIDATES = ["q_charge_Ah", "Q_charge_Ah", "charge_capacity_Ah", "Qc_Ah", "q_charge"]
SOH_CANDIDATES = ["SOH", "soh", "SOH_label", "soh_label", "SOH_obs", "soh_obs"]
LABEL_CANDIDATES = ["use_for_soh", "label_eligible", "is_labeled", "complete_discharge", "is_full_discharge"]
PARTIAL_CANDIDATES = ["partial_discharge", "is_partial_discharge", "partial_or_unlabeled"]


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            return {"__read_error__": str(exc), "__path__": str(path)}
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception as exc:
        return {"__read_error__": str(exc), "__path__": str(path)}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def parse_markdown_verdict(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"verdict\s*=\s*([A-Za-z0-9_\-]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"Verdict:\s*\n\s*```text\s*\n\s*([^`\n]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"Verdict:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("` ")
    return ""


def first_existing_path(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def first_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    exact = {str(c): c for c in cols}
    lower = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in exact:
            return str(exact[cand])
        if cand.lower() in lower:
            return str(lower[cand.lower()])
    return None


def as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip().strip('"').strip("'")
    if not s:
        return None
    return Path(s)


def normalize_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def canonical_batch_from_any(*values: Any) -> str:
    text = " ".join(normalize_token(v) for v in values if normalize_token(v))
    m = re.search(r"Batch[-_\s]*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"Batch-{int(m.group(1))}"
    m = re.search(r"\bB(\d+)\b", text, flags=re.IGNORECASE)
    if m:
        return f"Batch-{int(m.group(1))}"
    return normalize_token(values[0]) if values else ""


def canonical_battery_from_any(*values: Any) -> str:
    text = " ".join(normalize_token(v) for v in values if normalize_token(v))
    m = re.search(r"battery[-_\s]*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"battery-{int(m.group(1))}"
    m = re.search(r"cell[-_\s]*(\d+)", text, flags=re.IGNORECASE)
    if m:
        return f"battery-{int(m.group(1))}"
    return normalize_token(values[0]) if values else ""


def canonical_protocol_from_any(*values: Any) -> str:
    text = " ".join(normalize_token(v) for v in values if normalize_token(v))
    # Order matters: R2.5 should not be parsed as 2C.
    m = re.search(r"\bR\s*([0-9]+(?:\.[0-9]+)?)\b", text, flags=re.IGNORECASE)
    if m:
        return f"R{m.group(1)}".replace(" ", "")
    m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*C\b", text, flags=re.IGNORECASE)
    if m:
        val = m.group(1)
        if val.endswith(".0"):
            val = val[:-2]
        return f"{val}C"
    return normalize_token(values[0]) if values else ""


def parse_profile_identifiers(row: Dict[str, Any]) -> Dict[str, str]:
    pathish = " ".join(str(row.get(c, "")) for c in ["profile_npz", "solution_npz", "source_file", "source_path", "parquet_path"])
    profile_id = normalize_token(row.get("profile_id"))
    dataset_id = normalize_token(row.get("dataset_id")) or "XJTU"
    batch_id = canonical_batch_from_any(row.get("batch_id"), row.get("batch"), row.get("cell_uid"), profile_id, pathish)
    battery_id = canonical_battery_from_any(row.get("battery_id"), row.get("cell_uid"), profile_id, pathish)
    protocol = canonical_protocol_from_any(row.get("protocol"), profile_id, pathish)
    cell_uid = normalize_token(row.get("cell_uid")) or f"{batch_id}_{battery_id}".strip("_")
    split = normalize_token(row.get("split"))
    if not profile_id:
        parts = [p for p in [batch_id, protocol, battery_id] if p]
        profile_id = "_".join(parts) if parts else cell_uid
    return {
        "profile_id": profile_id,
        "dataset_id": dataset_id,
        "batch_id": batch_id,
        "battery_id": battery_id,
        "cell_uid": cell_uid,
        "protocol": protocol,
        "split": split,
    }


def truthy_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype=bool)
    if s.dtype == bool:
        return s.fillna(False)
    text = s.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "eligible", "complete", "full"])


def safe_float_array(x: Any) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=np.float64)
    except Exception:
        arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def safe_str_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(str)
    except Exception:
        return None


def safe_percentile(x: np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, q))


def safe_mean(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def safe_min(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmin(arr))


def safe_max(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanmax(arr))


def add_basic_stats(out: Dict[str, Any], prefix: str, name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    out[f"{prefix}_{name}_n"] = int(a.size)
    if a.size == 0:
        for suffix in ["mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max"]:
            out[f"{prefix}_{name}_{suffix}"] = float("nan")
        return
    out[f"{prefix}_{name}_mean"] = float(np.nanmean(a))
    out[f"{prefix}_{name}_std"] = float(np.nanstd(a))
    out[f"{prefix}_{name}_min"] = float(np.nanmin(a))
    out[f"{prefix}_{name}_p05"] = float(np.nanpercentile(a, 5))
    out[f"{prefix}_{name}_p25"] = float(np.nanpercentile(a, 25))
    out[f"{prefix}_{name}_p50"] = float(np.nanpercentile(a, 50))
    out[f"{prefix}_{name}_p75"] = float(np.nanpercentile(a, 75))
    out[f"{prefix}_{name}_p95"] = float(np.nanpercentile(a, 95))
    out[f"{prefix}_{name}_max"] = float(np.nanmax(a))


def add_derivative_stats(out: Dict[str, Any], prefix: str, t: np.ndarray, v: np.ndarray, mask: np.ndarray) -> None:
    idx = np.where(mask & np.isfinite(t) & np.isfinite(v))[0]
    if idx.size < 3:
        out[f"{prefix}_dVdt_abs_p95"] = float("nan")
        out[f"{prefix}_dVdt_abs_max"] = float("nan")
        out[f"{prefix}_dVdt_abs_mean"] = float("nan")
        return
    # Keep derivative calculation bounded for multi-million-row profiles.
    stride = max(1, int(math.ceil(idx.size / 250000)))
    idx = idx[::stride]
    tt = t[idx]
    vv = v[idx]
    dt = np.diff(tt)
    dv = np.diff(vv)
    good = np.isfinite(dt) & np.isfinite(dv) & (dt > 0)
    if not np.any(good):
        out[f"{prefix}_dVdt_abs_p95"] = float("nan")
        out[f"{prefix}_dVdt_abs_max"] = float("nan")
        out[f"{prefix}_dVdt_abs_mean"] = float("nan")
        return
    d = np.abs(dv[good] / dt[good])
    out[f"{prefix}_dVdt_abs_p95"] = safe_percentile(d, 95)
    out[f"{prefix}_dVdt_abs_max"] = safe_max(d)
    out[f"{prefix}_dVdt_abs_mean"] = safe_mean(d)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        return np.array([], dtype=bool)
    n = min(len(a) for a in arrays if a is not None)
    if n <= 0:
        return np.array([], dtype=bool)
    m = np.ones(n, dtype=bool)
    for a in arrays:
        if a is None:
            continue
        aa = np.asarray(a[:n], dtype=np.float64)
        m &= np.isfinite(aa)
    return m


def load_npz_timeseries(path: Path) -> Dict[str, Any]:
    data = np.load(str(path), allow_pickle=True, mmap_mode="r")
    keys = set(data.files)

    def get_first(cands: Sequence[str]) -> Optional[Any]:
        for c in cands:
            if c in keys:
                return data[c]
        lower = {k.lower(): k for k in keys}
        for c in cands:
            if c.lower() in lower:
                return data[lower[c.lower()]]
        return None

    t = get_first(TIME_CANDIDATES)
    cur = get_first(CURRENT_CANDIDATES)
    vol = get_first(VOLTAGE_CANDIDATES)
    temp = get_first(TEMP_CANDIDATES)
    cyc = get_first(CYCLE_CANDIDATES)
    step = get_first(STEP_CANDIDATES)
    steptype = get_first(STEPTYPE_CANDIDATES)
    result = {
        "t": safe_float_array(t) if t is not None else np.array([], dtype=np.float64),
        "I": safe_float_array(cur) if cur is not None else np.array([], dtype=np.float64),
        "V": safe_float_array(vol) if vol is not None else np.array([], dtype=np.float64),
        "T": safe_float_array(temp) if temp is not None else np.array([], dtype=np.float64),
        "cycle": safe_float_array(cyc) if cyc is not None else np.array([], dtype=np.float64),
        "step": safe_float_array(step) if step is not None else np.array([], dtype=np.float64),
        "step_type": safe_str_array(steptype),
        "source_kind": "npz",
        "source_keys": sorted(keys),
    }
    return result


def parquet_columns(path: Path) -> List[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        return list(pq.ParquetFile(str(path)).schema_arrow.names)
    except Exception:
        try:
            return list(pd.read_parquet(path, engine="pyarrow").columns)
        except Exception:
            return []


def load_table_timeseries(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        cols = parquet_columns(path)
        use_cols: List[str] = []
        selected: Dict[str, Optional[str]] = {}
        for label, cands in [
            ("t", TIME_CANDIDATES), ("I", CURRENT_CANDIDATES), ("V", VOLTAGE_CANDIDATES),
            ("T", TEMP_CANDIDATES), ("cycle", CYCLE_CANDIDATES), ("step", STEP_CANDIDATES),
            ("step_type", STEPTYPE_CANDIDATES),
        ]:
            col = None
            lower = {c.lower(): c for c in cols}
            for cand in cands:
                if cand in cols:
                    col = cand
                    break
                if cand.lower() in lower:
                    col = lower[cand.lower()]
                    break
            selected[label] = col
            if col and col not in use_cols:
                use_cols.append(col)
        if not use_cols:
            raise ValueError(f"No usable columns found in {path}")
        df = pd.read_parquet(path, columns=use_cols)
    else:
        # CSV fallback.  This is not expected for the XJTU cache, but makes the
        # script usable for smaller debug profiles.
        header = pd.read_csv(path, nrows=0)
        cols = list(header.columns)
        selected = {}
        use_cols = []
        for label, cands in [
            ("t", TIME_CANDIDATES), ("I", CURRENT_CANDIDATES), ("V", VOLTAGE_CANDIDATES),
            ("T", TEMP_CANDIDATES), ("cycle", CYCLE_CANDIDATES), ("step", STEP_CANDIDATES),
            ("step_type", STEPTYPE_CANDIDATES),
        ]:
            col = None
            lower = {c.lower(): c for c in cols}
            for cand in cands:
                if cand in cols:
                    col = cand
                    break
                if cand.lower() in lower:
                    col = lower[cand.lower()]
                    break
            selected[label] = col
            if col and col not in use_cols:
                use_cols.append(col)
        df = pd.read_csv(path, usecols=use_cols)

    def arr(label: str) -> np.ndarray:
        col = selected.get(label)
        if col and col in df.columns:
            return safe_float_array(df[col].to_numpy())
        return np.array([], dtype=np.float64)

    st = None
    st_col = selected.get("step_type")
    if st_col and st_col in df.columns:
        st = safe_str_array(df[st_col].to_numpy())
    return {
        "t": arr("t"),
        "I": arr("I"),
        "V": arr("V"),
        "T": arr("T"),
        "cycle": arr("cycle"),
        "step": arr("step"),
        "step_type": st,
        "source_kind": suffix.lstrip(".") or "table",
        "source_keys": list(df.columns),
    }


def resolve_data_path(row: Dict[str, Any], cache_root: Path, project_root: Path) -> Optional[Path]:
    candidates = []
    for key in ["profile_npz", "solution_npz", "replay_profile_npz", "prediction_npz", "source_file", "source_path", "parquet_path"]:
        p = as_path(row.get(key))
        if p:
            candidates.append(p)
            if not p.is_absolute():
                candidates.append(project_root / p)
                candidates.append(cache_root / p)
    for p in candidates:
        try:
            if p.exists():
                return p
        except OSError:
            continue
    # Last resort: return the first path even if it does not exist so the error
    # report is informative.
    return candidates[0] if candidates else None


def load_timeseries(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz_timeseries(path)
    if suffix in [".parquet", ".pq", ".csv", ".txt"]:
        return load_table_timeseries(path)
    raise ValueError(f"Unsupported profile source extension: {path}")


def compute_window_features(
    out: Dict[str, Any],
    prefix: str,
    t: np.ndarray,
    I: np.ndarray,
    V: np.ndarray,
    T: np.ndarray,
    cycle: np.ndarray,
    step: np.ndarray,
    mask: np.ndarray,
    current_eps: float,
) -> None:
    n = int(np.sum(mask))
    out[f"{prefix}_n_points"] = n
    if n <= 0:
        return
    tt = t[mask]
    ii = I[mask]
    vv = V[mask] if V.size else np.array([], dtype=np.float64)
    temp = T[mask] if T.size else np.array([], dtype=np.float64)
    out[f"{prefix}_duration_s"] = float(np.nanmax(tt) - np.nanmin(tt)) if tt.size else float("nan")
    dt = np.diff(tt[np.isfinite(tt)]) if tt.size > 2 else np.array([], dtype=np.float64)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    out[f"{prefix}_dt_median_s"] = safe_percentile(dt, 50)
    out[f"{prefix}_dt_p95_s"] = safe_percentile(dt, 95)

    add_basic_stats(out, prefix, "I_A", ii)
    abs_i = np.abs(ii[np.isfinite(ii)])
    add_basic_stats(out, prefix, "I_abs_A", abs_i)
    if ii.size:
        out[f"{prefix}_I_pos_frac"] = float(np.nanmean(ii > current_eps))
        out[f"{prefix}_I_neg_frac"] = float(np.nanmean(ii < -current_eps))
        out[f"{prefix}_I_rest_frac"] = float(np.nanmean(np.abs(ii) <= current_eps))
    if vv.size:
        add_basic_stats(out, prefix, "V", vv)
        out[f"{prefix}_V_range"] = safe_max(vv) - safe_min(vv)
        out[f"{prefix}_V_upper_frac_ge_4p10"] = float(np.nanmean(vv >= 4.10))
        out[f"{prefix}_V_upper_frac_ge_4p20"] = float(np.nanmean(vv >= 4.20))
        out[f"{prefix}_V_upper_frac_ge_4p269"] = float(np.nanmean(vv >= 4.269))
        out[f"{prefix}_V_lower_frac_le_2p75"] = float(np.nanmean(vv <= 2.75))
        out[f"{prefix}_V_lower_frac_le_3p00"] = float(np.nanmean(vv <= 3.00))
        add_derivative_stats(out, prefix, t, V, mask)
    if temp.size:
        add_basic_stats(out, prefix, "T_C", temp)
        out[f"{prefix}_T_frac_ge_35C"] = float(np.nanmean(temp >= 35.0))
        out[f"{prefix}_T_frac_ge_40C"] = float(np.nanmean(temp >= 40.0))
        out[f"{prefix}_T_frac_ge_45C"] = float(np.nanmean(temp >= 45.0))
    if cycle.size:
        cc = cycle[mask]
        cc = cc[np.isfinite(cc)]
        out[f"{prefix}_cycle_count"] = int(np.unique(cc.astype(np.int64)).size) if cc.size else 0
    if step.size:
        ss = step[mask]
        ss = ss[np.isfinite(ss)]
        out[f"{prefix}_step_count"] = int(np.unique(ss.astype(np.int64)).size) if ss.size else 0

    for seg_name, seg_mask in [
        ("charge", mask & (I > current_eps)),
        ("discharge", mask & (I < -current_eps)),
        ("rest", mask & (np.abs(I) <= current_eps)),
    ]:
        sn = int(np.sum(seg_mask))
        sprefix = f"{prefix}_{seg_name}"
        out[f"{sprefix}_frac"] = float(sn / max(n, 1))
        out[f"{sprefix}_n_points"] = sn
        if sn <= 0:
            continue
        si = I[seg_mask]
        out[f"{sprefix}_I_abs_mean_A"] = safe_mean(np.abs(si))
        out[f"{sprefix}_I_abs_p95_A"] = safe_percentile(np.abs(si), 95)
        if V.size:
            sv = V[seg_mask]
            add_basic_stats(out, sprefix, "V", sv)
            out[f"{sprefix}_V_range"] = safe_max(sv) - safe_min(sv)
            out[f"{sprefix}_V_upper_frac_ge_4p10"] = float(np.nanmean(sv >= 4.10))
            out[f"{sprefix}_V_upper_frac_ge_4p20"] = float(np.nanmean(sv >= 4.20))
            out[f"{sprefix}_V_upper_frac_ge_4p269"] = float(np.nanmean(sv >= 4.269))
            out[f"{sprefix}_V_lower_frac_le_2p75"] = float(np.nanmean(sv <= 2.75))
            out[f"{sprefix}_V_lower_frac_le_3p00"] = float(np.nanmean(sv <= 3.00))
            add_derivative_stats(out, sprefix, t, V, seg_mask)
        if T.size:
            st = T[seg_mask]
            out[f"{sprefix}_T_mean_C"] = safe_mean(st)
            out[f"{sprefix}_T_p95_C"] = safe_percentile(st, 95)
            out[f"{sprefix}_T_max_C"] = safe_max(st)


def compute_profile_features(ts: Dict[str, Any]) -> Dict[str, Any]:
    t = ts.get("t", np.array([], dtype=np.float64))
    I = ts.get("I", np.array([], dtype=np.float64))
    V = ts.get("V", np.array([], dtype=np.float64))
    T = ts.get("T", np.array([], dtype=np.float64))
    cycle = ts.get("cycle", np.array([], dtype=np.float64))
    step = ts.get("step", np.array([], dtype=np.float64))
    n = min([len(a) for a in [t, I, V] if len(a) > 0] or [0])
    out: Dict[str, Any] = {
        "profile_read_ok": bool(n > 0),
        "source_kind": ts.get("source_kind", ""),
        "n_time_raw": int(n),
    }
    if n <= 0:
        return out
    t = t[:n]
    I = I[:n]
    V = V[:n]
    T = T[:n] if len(T) >= n else np.full(n, np.nan)
    cycle = cycle[:n] if len(cycle) >= n else np.full(n, np.nan)
    step = step[:n] if len(step) >= n else np.full(n, np.nan)
    base_mask = finite_mask(t, I, V)
    out["finite_core_frac"] = float(np.mean(base_mask)) if base_mask.size else 0.0
    if not np.any(base_mask):
        return out
    t0 = safe_min(t[base_mask])
    tmax = safe_max(t[base_mask])
    abs_i_p95 = safe_percentile(np.abs(I[base_mask]), 95)
    current_eps = max(1e-8, 0.005 * abs_i_p95 if np.isfinite(abs_i_p95) and abs_i_p95 > 0 else 1e-8)
    out["current_eps_A"] = current_eps
    windows = {
        "full": base_mask,
        "w40ks": base_mask & (t <= t0 + 40000.0),
        "w200ks": base_mask & (t <= t0 + 200000.0),
        "tail200ks": base_mask & (t >= max(t0, tmax - 200000.0)),
    }
    for prefix, mask in windows.items():
        compute_window_features(out, prefix, t, I, V, T, cycle, step, mask, current_eps)
    return out


def compute_cycle_manifest_features(cycle_manifest: Optional[pd.DataFrame]) -> pd.DataFrame:
    if cycle_manifest is None or cycle_manifest.empty:
        return pd.DataFrame()
    df = cycle_manifest.copy()
    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        ids = parse_profile_identifiers(row)
        new = dict(row)
        new.update({f"__{k}": v for k, v in ids.items()})
        rows.append(new)
    cdf = pd.DataFrame(rows)
    group_cols = ["__batch_id", "__battery_id", "__protocol"]
    q_dis_col = first_col(cdf, Q_DIS_CANDIDATES)
    q_chg_col = first_col(cdf, Q_CHG_CANDIDATES)
    soh_col = first_col(cdf, SOH_CANDIDATES)
    label_col = first_col(cdf, LABEL_CANDIDATES)
    partial_col = first_col(cdf, PARTIAL_CANDIDATES)
    cycle_col = first_col(cdf, CYCLE_CANDIDATES)
    feat_rows = []
    for key, g in cdf.groupby(group_cols, dropna=False):
        out: Dict[str, Any] = {"batch_id": key[0], "battery_id": key[1], "protocol": key[2]}
        out["cycle_manifest_rows"] = int(len(g))
        if cycle_col:
            cyc = pd.to_numeric(g[cycle_col], errors="coerce")
            out["cycle_manifest_cycle_count"] = int(cyc.dropna().nunique())
            out["cycle_manifest_cycle_min"] = float(cyc.min()) if cyc.notna().any() else float("nan")
            out["cycle_manifest_cycle_max"] = float(cyc.max()) if cyc.notna().any() else float("nan")
        if label_col:
            lab = truthy_series(g[label_col])
            out["cycle_manifest_label_eligible_count"] = int(lab.sum())
            out["cycle_manifest_label_eligible_frac"] = float(lab.mean()) if len(lab) else float("nan")
        if partial_col:
            part = truthy_series(g[partial_col])
            out["cycle_manifest_partial_count"] = int(part.sum())
            out["cycle_manifest_partial_frac"] = float(part.mean()) if len(part) else float("nan")
        for col, name in [(q_dis_col, "q_discharge_Ah"), (q_chg_col, "q_charge_Ah"), (soh_col, "SOH")]:
            if col and col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy(dtype=float)
                if vals.size:
                    out[f"cycle_manifest_{name}_mean"] = float(np.mean(vals))
                    out[f"cycle_manifest_{name}_std"] = float(np.std(vals))
                    out[f"cycle_manifest_{name}_min"] = float(np.min(vals))
                    out[f"cycle_manifest_{name}_p05"] = float(np.percentile(vals, 5))
                    out[f"cycle_manifest_{name}_p50"] = float(np.percentile(vals, 50))
                    out[f"cycle_manifest_{name}_p95"] = float(np.percentile(vals, 95))
                    out[f"cycle_manifest_{name}_max"] = float(np.max(vals))
                    out[f"cycle_manifest_{name}_range"] = float(np.max(vals) - np.min(vals))
        feat_rows.append(out)
    return pd.DataFrame(feat_rows)


def is_numeric_feature_col(col: str) -> bool:
    if col in ID_COLUMNS:
        return False
    if col.startswith("__"):
        return False
    lower = col.lower()
    if any(tok in lower for tok in ["path", "file", "source", "kind", "id", "uid", "protocol", "split", "status", "read_error"]):
        return False
    return True


def robust_scale(values: np.ndarray) -> Tuple[float, str]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan"), "insufficient"
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    scale = 1.4826 * mad
    if np.isfinite(scale) and scale > 1e-12:
        return float(scale), "mad"
    q75, q25 = np.nanpercentile(arr, [75, 25])
    iqr_scale = (q75 - q25) / 1.349 if np.isfinite(q75 - q25) else float("nan")
    if np.isfinite(iqr_scale) and iqr_scale > 1e-12:
        return float(iqr_scale), "iqr"
    std = np.nanstd(arr)
    if np.isfinite(std) and std > 1e-12:
        return float(std), "std"
    eps = max(1e-9, abs(float(med)) * 1e-6)
    return float(eps), "epsilon_zero_peer_variance"


def feature_group(feature: str) -> str:
    f = feature.lower()
    if "discharge" in f:
        return "discharge_segment"
    if "charge" in f:
        return "charge_segment"
    if "rest" in f:
        return "rest_segment"
    if "_t_c" in f or "temperature" in f or "t_frac" in f:
        return "temperature"
    if "_i_" in f or "current" in f:
        return "current"
    if "_v" in f or "voltage" in f or "dvdt" in f:
        return "voltage"
    if "cycle_manifest" in f or "soh" in f or "q_discharge" in f or "q_charge" in f:
        return "cycle_capacity_soh"
    if "cycle" in f or "step" in f:
        return "cycle_step"
    if "duration" in f or "dt_" in f or "n_points" in f:
        return "time_density"
    return "other"


def compute_battery8_distance(
    feature_df: pd.DataFrame,
    target_batch: str,
    target_battery: str,
    target_protocol: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if feature_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"ok": False, "reason": "empty_feature_table"}
    idf = feature_df.copy()
    for c in ["batch_id", "battery_id", "protocol"]:
        if c not in idf.columns:
            idf[c] = ""
    target_mask = (
        idf["batch_id"].astype(str).str.lower().eq(target_batch.lower())
        & idf["battery_id"].astype(str).str.lower().eq(target_battery.lower())
        & idf["protocol"].astype(str).str.lower().eq(target_protocol.lower())
    )
    peer_mask = (
        idf["batch_id"].astype(str).str.lower().eq(target_batch.lower())
        & idf["protocol"].astype(str).str.lower().eq(target_protocol.lower())
        & ~idf["battery_id"].astype(str).str.lower().eq(target_battery.lower())
    )
    if int(target_mask.sum()) < 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"ok": False, "reason": "target_not_found"}
    if int(peer_mask.sum()) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"ok": False, "reason": "insufficient_peers", "peer_count": int(peer_mask.sum())}
    target_row = idf[target_mask].iloc[0]
    peers = idf[peer_mask].copy()
    numeric_cols = []
    for c in idf.columns:
        if not is_numeric_feature_col(c):
            continue
        vals = pd.to_numeric(idf[c], errors="coerce")
        if vals.notna().sum() >= 3:
            numeric_cols.append(c)
    dist_rows: List[Dict[str, Any]] = []
    for c in numeric_cols:
        peer_vals = pd.to_numeric(peers[c], errors="coerce").to_numpy(dtype=float)
        target_val = pd.to_numeric(pd.Series([target_row.get(c)]), errors="coerce").iloc[0]
        peer_finite = peer_vals[np.isfinite(peer_vals)]
        if not np.isfinite(target_val) or peer_finite.size < 3:
            continue
        med = float(np.nanmedian(peer_finite))
        pmin = float(np.nanmin(peer_finite))
        pmax = float(np.nanmax(peer_finite))
        mad = float(np.nanmedian(np.abs(peer_finite - med)))
        scale, scale_source = robust_scale(peer_finite)
        if not np.isfinite(scale) or scale <= 0:
            continue
        z = float((float(target_val) - med) / scale)
        dist_rows.append({
            "feature": c,
            "feature_group": feature_group(c),
            "target_value": float(target_val),
            "peer_median": med,
            "peer_mad_raw": mad,
            "peer_scale": scale,
            "peer_scale_source": scale_source,
            "peer_min": pmin,
            "peer_max": pmax,
            "peer_n": int(peer_finite.size),
            "robust_z_vs_B1_2C_peers": z,
            "abs_robust_z": abs(z),
            "direction_vs_peer_median": "higher" if z > 0 else "lower" if z < 0 else "same",
        })
    distance_df = pd.DataFrame(dist_rows).sort_values("abs_robust_z", ascending=False) if dist_rows else pd.DataFrame()
    top_df = distance_df.head(50).copy() if not distance_df.empty else pd.DataFrame()

    # Pairwise robust distances within B1_2C group.
    group = idf[
        idf["batch_id"].astype(str).str.lower().eq(target_batch.lower())
        & idf["protocol"].astype(str).str.lower().eq(target_protocol.lower())
    ].copy()
    pair_rows: List[Dict[str, Any]] = []
    if len(group) >= 3 and numeric_cols:
        # Scale by non-target peers to keep target distance interpretable.
        scales: Dict[str, float] = {}
        centers: Dict[str, float] = {}
        for c in numeric_cols:
            vals = pd.to_numeric(peers[c], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 3:
                continue
            sc, _ = robust_scale(vals)
            if np.isfinite(sc) and sc > 0:
                scales[c] = sc
                centers[c] = float(np.nanmedian(vals))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ri = group.iloc[i]
                rj = group.iloc[j]
                diffs = []
                for c, sc in scales.items():
                    vi = pd.to_numeric(pd.Series([ri.get(c)]), errors="coerce").iloc[0]
                    vj = pd.to_numeric(pd.Series([rj.get(c)]), errors="coerce").iloc[0]
                    if np.isfinite(vi) and np.isfinite(vj):
                        diffs.append((float(vi) - float(vj)) / sc)
                if not diffs:
                    continue
                arr = np.asarray(diffs, dtype=float)
                bi = str(ri.get("battery_id", ""))
                bj = str(rj.get("battery_id", ""))
                pair_rows.append({
                    "profile_i": ri.get("profile_id", ""),
                    "battery_i": bi,
                    "profile_j": rj.get("profile_id", ""),
                    "battery_j": bj,
                    "includes_target": bool(bi.lower() == target_battery.lower() or bj.lower() == target_battery.lower()),
                    "feature_count": int(arr.size),
                    "distance_robust_euclidean_per_feature": float(np.sqrt(np.nanmean(arr ** 2))),
                    "distance_mean_abs_robust": float(np.nanmean(np.abs(arr))),
                    "distance_max_abs_robust": float(np.nanmax(np.abs(arr))),
                })
    pair_df = pd.DataFrame(pair_rows).sort_values("distance_robust_euclidean_per_feature", ascending=False) if pair_rows else pd.DataFrame()
    summary: Dict[str, Any] = {
        "ok": True,
        "target_profile_id": str(target_row.get("profile_id", "")),
        "target_batch_id": target_batch,
        "target_battery_id": target_battery,
        "target_protocol": target_protocol,
        "peer_count": int(peer_mask.sum()),
        "numeric_feature_count": int(len(numeric_cols)),
        "distance_feature_count": int(len(distance_df)) if not distance_df.empty else 0,
        "abs_z_ge_2_count": int((distance_df["abs_robust_z"] >= 2).sum()) if not distance_df.empty else 0,
        "abs_z_ge_3_count": int((distance_df["abs_robust_z"] >= 3).sum()) if not distance_df.empty else 0,
        "abs_z_ge_5_count": int((distance_df["abs_robust_z"] >= 5).sum()) if not distance_df.empty else 0,
        "max_abs_z": float(distance_df["abs_robust_z"].max()) if not distance_df.empty else float("nan"),
    }
    if not pair_df.empty:
        target_pairs = pair_df[pair_df["includes_target"] == True]
        non_target_pairs = pair_df[pair_df["includes_target"] == False]
        summary["target_pair_distance_median"] = float(target_pairs["distance_robust_euclidean_per_feature"].median()) if not target_pairs.empty else float("nan")
        summary["peer_pair_distance_median"] = float(non_target_pairs["distance_robust_euclidean_per_feature"].median()) if not non_target_pairs.empty else float("nan")
        if not target_pairs.empty and not non_target_pairs.empty:
            med = float(non_target_pairs["distance_robust_euclidean_per_feature"].median())
            vals = non_target_pairs["distance_robust_euclidean_per_feature"].to_numpy(dtype=float)
            sc, src = robust_scale(vals)
            summary["target_pair_distance_z_vs_peer_pairs"] = float((summary["target_pair_distance_median"] - med) / sc) if sc and np.isfinite(sc) else float("nan")
            summary["pair_distance_scale_source"] = src
    return distance_df, top_df, pair_df, summary


def choose_verdict(distance_summary: Dict[str, Any], top_df: pd.DataFrame, d10_context_ok: bool) -> Tuple[str, List[str], str]:
    if not distance_summary.get("ok"):
        reason = distance_summary.get("reason", "unknown")
        return (
            "d11_b_inconclusive_missing_or_insufficient_feature_evidence",
            [
                f"Feature audit could not complete: {reason}.",
                "Keep B1_2C battery-8 flagged/excluded under D10 policy.",
                "Do not proceed to D11-C model ablation until profile/peer feature evidence is available.",
            ],
            "fix_inputs_or_review_manifest",
        )
    max_abs_z = float(distance_summary.get("max_abs_z", float("nan")))
    ge3 = int(distance_summary.get("abs_z_ge_3_count", 0))
    ge5 = int(distance_summary.get("abs_z_ge_5_count", 0))
    pair_z = distance_summary.get("target_pair_distance_z_vs_peer_pairs", float("nan"))
    top_groups = []
    if top_df is not None and not top_df.empty:
        top_groups = list(top_df.head(20)["feature_group"].astype(str))
    has_discharge = "discharge_segment" in top_groups
    has_voltage = "voltage" in top_groups or any(g.endswith("segment") for g in top_groups)
    has_temperature = "temperature" in top_groups
    pair_strong = np.isfinite(pair_z) and float(pair_z) >= 3.0
    if ge5 >= 3 or max_abs_z >= 8.0 or pair_strong:
        notes = [
            "Battery-8 is separated from same Batch-1/2C peers by replay/profile features.",
            f"abs_z_ge_5_count={ge5}, abs_z_ge_3_count={ge3}, max_abs_z={max_abs_z:.3g}.",
            "Keep D9.6/D9.5.1 mainline frozen and keep battery-8 flagged/excluded.",
        ]
        if has_discharge:
            notes.append("Top-distance features include discharge-segment features, consistent with the D10/D9.7 discharge-regime diagnosis.")
        if has_voltage:
            notes.append("Top-distance features include voltage-shape or voltage-window features.")
        if has_temperature:
            notes.append("Top-distance features include temperature features; review measured T(t) as a possible regime marker.")
        if not d10_context_ok:
            notes.append("D10 policy context is incomplete; verify D10-P0/P1/P3/P5 artifacts before any D11-C model ablation.")
        return (
            "d11_b_battery8_feature_distance_boundary_supported_keep_flagged",
            notes,
            "manual_review_then_D11C_design_only_flag_aware_metadata_ablation",
        )
    if ge3 >= 3:
        return (
            "d11_b_battery8_feature_distance_weakly_supported_keep_flagged",
            [
                "Battery-8 shows moderate feature separation from B1_2C peers, but evidence is not strong enough for model changes.",
                f"abs_z_ge_3_count={ge3}, max_abs_z={max_abs_z:.3g}.",
                "Keep battery-8 flagged/excluded and manually review D11-B top features and D9.7 plots.",
            ],
            "manual_review_required_before_D11C",
        )
    return (
        "d11_b_battery8_not_isolated_by_available_replay_features_keep_flagged_no_model_change",
        [
            "The simple replay/profile feature audit does not isolate battery-8 strongly from B1_2C peers.",
            f"abs_z_ge_3_count={ge3}, max_abs_z={max_abs_z:.3g}.",
            "Keep battery-8 flagged/excluded because D10-P0/P3 evidence remains valid, but do not start D11-C based only on D11-B features.",
        ],
        "keep_flagged_no_D11C_yet",
    )


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v):
            return None
        if math.isinf(v):
            return str(v)
        return v
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return str(obj)
    return obj


def write_recommendation_md(
    path: Path,
    verdict: str,
    notes: List[str],
    next_action: str,
    summary: Dict[str, Any],
    top_df: pd.DataFrame,
    output_files: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# D11-B Regime Feature Distance Audit")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("```text")
    lines.append(verdict)
    lines.append("```")
    lines.append("")
    lines.append("## Recommended next action")
    lines.append("")
    lines.append("```text")
    lines.append(next_action)
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Distance summary")
    lines.append("")
    for key in [
        "target_profile_id", "target_batch_id", "target_battery_id", "target_protocol", "peer_count",
        "numeric_feature_count", "distance_feature_count", "abs_z_ge_2_count", "abs_z_ge_3_count",
        "abs_z_ge_5_count", "max_abs_z", "target_pair_distance_median", "peer_pair_distance_median",
        "target_pair_distance_z_vs_peer_pairs",
    ]:
        if key in summary:
            lines.append(f"- `{key}` = `{summary.get(key)}`")
    lines.append("")
    lines.append("## Top feature distances")
    lines.append("")
    if top_df is not None and not top_df.empty:
        cols = ["feature", "feature_group", "target_value", "peer_median", "robust_z_vs_B1_2C_peers", "abs_robust_z", "direction_vs_peer_median"]
        lines.append("| feature | group | target | peer median | robust z | abs z | direction |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for _, r in top_df.head(15).iterrows():
            lines.append(
                f"| `{r.get('feature','')}` | {r.get('feature_group','')} | {float(r.get('target_value', float('nan'))):.6g} | "
                f"{float(r.get('peer_median', float('nan'))):.6g} | {float(r.get('robust_z_vs_B1_2C_peers', float('nan'))):.4g} | "
                f"{float(r.get('abs_robust_z', float('nan'))):.4g} | {r.get('direction_vs_peer_median','')} |"
            )
    else:
        lines.append("No top feature table was generated.")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- Do not modify D9.6/D9.5.1 mainline from this audit.")
    lines.append("- Do not run direct 24-profile 200ks mainline claim while battery-8 remains unresolved.")
    lines.append("- D11-C is allowed only as design-only / flag-aware metadata ablation after manual review of this audit.")
    lines.append("- D11-D expert branch remains research-candidate only, not current mainline.")
    lines.append("")
    lines.append("## Generated files")
    lines.append("")
    lines.append("```text")
    for k, v in output_files.items():
        lines.append(f"{k}: {v}")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_make_plots(out_dir: Path, top_df: pd.DataFrame, pair_df: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return paths
    if top_df is not None and not top_df.empty:
        plot_df = top_df.head(20).copy().iloc[::-1]
        labels = [str(f)[:80] for f in plot_df["feature"].tolist()]
        values = plot_df["abs_robust_z"].astype(float).to_numpy()
        fig_h = max(4, 0.28 * len(labels) + 1.5)
        plt.figure(figsize=(10, fig_h))
        plt.barh(labels, values)
        plt.xlabel("abs robust z vs B1_2C peers")
        plt.title("D11-B battery-8 top feature distances")
        plt.tight_layout()
        p = out_dir / "d11_b_battery8_top_feature_distance_bar.png"
        plt.savefig(p, dpi=160)
        plt.close()
        paths.append(str(p))
    if pair_df is not None and not pair_df.empty:
        target_pairs = pair_df[pair_df["includes_target"] == True].copy()
        if not target_pairs.empty:
            target_pairs["pair_label"] = target_pairs["battery_i"].astype(str) + " vs " + target_pairs["battery_j"].astype(str)
            target_pairs = target_pairs.sort_values("distance_robust_euclidean_per_feature").tail(12)
            plt.figure(figsize=(9, max(3, 0.35 * len(target_pairs) + 1.5)))
            plt.barh(target_pairs["pair_label"], target_pairs["distance_robust_euclidean_per_feature"].astype(float))
            plt.xlabel("robust Euclidean distance per feature")
            plt.title("D11-B battery-8 pair distances to B1_2C peers")
            plt.tight_layout()
            p = out_dir / "d11_b_battery8_pair_distance_bar.png"
            plt.savefig(p, dpi=160)
            plt.close()
            paths.append(str(p))
    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GV1 D11-B battery-8 regime feature distance audit.")
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--profile_manifest", default="")
    ap.add_argument("--cycle_manifest", default="")
    ap.add_argument("--d10p5_dir", default="")
    ap.add_argument("--d10p0_dir", default="")
    ap.add_argument("--d10p1_dir", default="")
    ap.add_argument("--d10p3_dir", default="")
    ap.add_argument("--target_batch_id", default="Batch-1")
    ap.add_argument("--target_battery_id", default="battery-8")
    ap.add_argument("--target_protocol", default="2C")
    ap.add_argument("--make_plots", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Return non-zero when the audit is inconclusive or D10 context is missing.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = Path(args.cache_root)
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d11_b_regime_feature_distance_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile_manifest = Path(args.profile_manifest) if args.profile_manifest else first_existing_path([
        cache_root / "xjtu_batch134_training_ready" / "xjtu_batch134_profile_manifest.csv",
        cache_root / "xjtu_batch134_training_ready" / "profile_manifest.csv",
        cache_root / "xjtu_batch134_d10_p1_23profile_200ks_plan" / "d10_p1_24profile_manifest.csv",
    ])
    cycle_manifest = Path(args.cycle_manifest) if args.cycle_manifest else first_existing_path([
        cache_root / "xjtu_batch134_training_ready" / "xjtu_batch134_cycle_training_manifest.csv",
        cache_root / "xjtu_batch134_training_manifest" / "xjtu_batch134_cycle_manifest.csv",
    ])

    d10p5_dir = Path(args.d10p5_dir) if args.d10p5_dir else cache_root / "xjtu_batch134_d10_p5_regime_policy_d11_plan"
    d10p0_dir = Path(args.d10p0_dir) if args.d10p0_dir else cache_root / "xjtu_batch134_d10_p0_battery8_regime_judgement"
    d10p1_dir = Path(args.d10p1_dir) if args.d10p1_dir else cache_root / "xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8"
    d10p3_dir = Path(args.d10p3_dir) if args.d10p3_dir else cache_root / "xjtu_batch134_d10_p3_battery8_lightweight_correction"

    d10p5 = load_json(d10p5_dir / "d10_p5_regime_policy_summary.json") or {}
    d10p0 = load_json(d10p0_dir / "d10_p0_battery8_judgement_summary.json") or {}
    d10p1 = load_json(d10p1_dir / "scorecard_d10_p1_23profile_200ks.json") or {}
    d10p3_verdict = parse_markdown_verdict(read_text(d10p3_dir / "D10_P3_RECOMMENDATION.md"))
    d10_context = {
        "d10p5_verdict": d10p5.get("verdict", ""),
        "d10p5_ok": bool(d10p5.get("ok")) if isinstance(d10p5, dict) else False,
        "d10p0_verdict": d10p0.get("verdict", ""),
        "d10p1_status": d10p1.get("status", ""),
        "d10p1_profile_count": d10p1.get("profile_count", None),
        "d10p3_verdict": d10p3_verdict,
    }
    d10_context_ok = (
        d10_context["d10p5_verdict"] == EXPECTED_D10P5_VERDICT
        and d10_context["d10p0_verdict"] == EXPECTED_D10P0_VERDICT
        and d10_context["d10p1_status"] == "pass"
        and int(d10_context["d10p1_profile_count"] or -1) == 23
        and d10_context["d10p3_verdict"] == EXPECTED_D10P3_VERDICT
    )

    if not profile_manifest or not profile_manifest.exists():
        summary = {
            "ok": False,
            "stage": "D11-B regime feature distance audit",
            "verdict": "d11_b_inconclusive_missing_profile_manifest",
            "profile_manifest": str(profile_manifest) if profile_manifest else "",
            "out_dir": str(out_dir),
            "d10_context": d10_context,
        }
        summary_path = out_dir / "d11_b_regime_feature_distance_summary.json"
        summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
        return 2 if args.strict else 0

    manifest_df = pd.read_csv(profile_manifest)
    manifest_rows: List[Dict[str, Any]] = []
    for _, r in manifest_df.iterrows():
        raw = r.to_dict()
        ids = parse_profile_identifiers(raw)
        path = resolve_data_path(raw, cache_root, project_root)
        row = {**ids}
        row["profile_npz"] = normalize_token(raw.get("profile_npz")) or normalize_token(raw.get("solution_npz"))
        row["source_file"] = normalize_token(raw.get("source_file")) or normalize_token(raw.get("source_path")) or normalize_token(raw.get("parquet_path"))
        row["resolved_data_path"] = str(path) if path else ""
        row["resolved_data_path_exists"] = bool(path.exists()) if path else False
        row["raw_manifest_row_index"] = int(_)
        manifest_rows.append({**raw, **row})

    cycle_df = pd.read_csv(cycle_manifest) if cycle_manifest and cycle_manifest.exists() else None
    cycle_features = compute_cycle_manifest_features(cycle_df)

    feature_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for row in manifest_rows:
        base = {k: row.get(k, "") for k in ["profile_id", "dataset_id", "batch_id", "battery_id", "cell_uid", "protocol", "split", "resolved_data_path"]}
        p = Path(str(row.get("resolved_data_path", ""))) if row.get("resolved_data_path") else None
        if not p or not p.exists():
            err = {**base, "profile_read_ok": False, "profile_read_error": "resolved_data_path_missing"}
            feature_rows.append(err)
            errors.append(err)
            continue
        try:
            ts = load_timeseries(p)
            feats = compute_profile_features(ts)
            merged = {**base, **feats}
            feature_rows.append(merged)
        except Exception as exc:
            err = {**base, "profile_read_ok": False, "profile_read_error": repr(exc)}
            feature_rows.append(err)
            errors.append(err)

    feature_df = pd.DataFrame(feature_rows)
    if not cycle_features.empty and not feature_df.empty:
        feature_df = feature_df.merge(cycle_features, on=["batch_id", "battery_id", "protocol"], how="left")

    feature_table_path = out_dir / "d11_b_profile_feature_table.csv"
    feature_df.to_csv(feature_table_path, index=False, encoding="utf-8-sig")

    distance_df, top_df, pair_df, distance_summary = compute_battery8_distance(
        feature_df,
        target_batch=args.target_batch_id,
        target_battery=args.target_battery_id,
        target_protocol=args.target_protocol,
    )
    distance_csv = out_dir / "d11_b_battery8_vs_b1_2c_peer_distance.csv"
    top_csv = out_dir / "d11_b_battery8_top_distance_features.csv"
    pair_csv = out_dir / "d11_b_b1_2c_pairwise_distance_matrix.csv"
    distance_df.to_csv(distance_csv, index=False, encoding="utf-8-sig")
    top_df.to_csv(top_csv, index=False, encoding="utf-8-sig")
    pair_df.to_csv(pair_csv, index=False, encoding="utf-8-sig")

    verdict, notes, next_action = choose_verdict(distance_summary, top_df, d10_context_ok)
    plot_paths = maybe_make_plots(out_dir, top_df, pair_df) if args.make_plots else []

    output_files = {
        "recommendation_md": str(out_dir / "D11_B_RECOMMENDATION.md"),
        "summary_json": str(out_dir / "d11_b_regime_feature_distance_summary.json"),
        "profile_feature_table_csv": str(feature_table_path),
        "battery8_peer_distance_csv": str(distance_csv),
        "battery8_top_features_csv": str(top_csv),
        "b1_2c_pairwise_distance_csv": str(pair_csv),
    }
    if plot_paths:
        output_files["plots"] = "; ".join(plot_paths)

    summary: Dict[str, Any] = {
        "ok": True,
        "stage": "D11-B regime feature distance audit",
        "verdict": verdict,
        "next_action": next_action,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "out_dir": str(out_dir),
        "profile_manifest": str(profile_manifest),
        "cycle_manifest": str(cycle_manifest) if cycle_manifest else "",
        "target": {
            "batch_id": args.target_batch_id,
            "battery_id": args.target_battery_id,
            "protocol": args.target_protocol,
        },
        "d10_context": d10_context,
        "d10_context_ok": d10_context_ok,
        "profile_read": {
            "profile_manifest_rows": int(len(manifest_rows)),
            "feature_table_rows": int(len(feature_df)),
            "read_error_count": int(len(errors)),
        },
        "distance_summary": distance_summary,
        "top_feature_groups": top_df["feature_group"].value_counts().head(10).to_dict() if top_df is not None and not top_df.empty else {},
        "outputs": output_files,
        "notes": notes,
    }
    summary_path = out_dir / "d11_b_regime_feature_distance_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_recommendation_md(out_dir / "D11_B_RECOMMENDATION.md", verdict, notes, next_action, distance_summary, top_df, output_files)

    print(json.dumps(jsonable({
        "ok": True,
        "stage": summary["stage"],
        "verdict": verdict,
        "next_action": next_action,
        "out_dir": str(out_dir),
        "summary_json": str(summary_path),
        "recommendation_md": str(out_dir / "D11_B_RECOMMENDATION.md"),
        "read_error_count": len(errors),
        "distance_summary": distance_summary,
    }), ensure_ascii=False, indent=2))
    if args.strict and (not d10_context_ok or verdict.startswith("d11_b_inconclusive")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
