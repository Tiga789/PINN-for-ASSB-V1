#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D14-P3 XJTU Batch-5/6 feasibility audit.

This script is intentionally self-contained and conservative:
  - It does not train.
  - It does not patch GV1 mainline files.
  - It does not generate voltage soft labels.
  - It does not generate SOH labels in the voltage generator.
  - It only audits whether Batch-5/6 can be routed into the measured-current replay pipeline.

It can inspect .mat, .csv and .parquet files. For .mat files it uses scipy.io.loadmat
when available. If scipy cannot read a v7.3/HDF5 .mat file, it falls back to a shallow
h5py key listing and marks the file as requiring the existing GV1 reader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("D14-P3 requires numpy. Please install numpy before running.") from exc

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("D14-P3 requires pandas. Please install pandas before running.") from exc


ALIAS = {
    "time": [
        "time", "time_s", "test_time", "test_time_s", "relative_time", "relative_time_s",
        "elapsed_time", "elapsed_time_s", "data_time", "Time", "t"
    ],
    "system_time": [
        "system_time", "date_time", "datetime", "systemTime", "system time", "Date_Time"
    ],
    "voltage": [
        "voltage", "voltage_v", "Voltage", "voltage_V", "V", "voltage_measured",
        "Voltage_measured"
    ],
    "current": [
        "current", "current_a", "Current", "current_A", "I", "current_measured",
        "Current_measured"
    ],
    "temperature": [
        "temperature", "temperature_c", "temp", "temp_c", "Temperature",
        "temperature_C", "T", "Temperature_measured"
    ],
    "capacity": [
        "capacity", "Capacity", "capacity_ah", "Capacity_Ah", "capacity_Ah",
        "charge_capacity", "discharge_capacity", "Q", "q", "q_Ah"
    ],
    "cycle": ["cycle", "cycle_id", "cycle_index", "Cycle", "raw__mat_subrecord_index"],
    "step": ["step", "step_id", "step_type", "Step", "status", "mode"]
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def read_json(path: Path) -> Optional[dict]:
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(str(status).upper(), 1)


def combine_status(items: Sequence[dict]) -> str:
    worst = "PASS"
    for item in items:
        st = str(item.get("status", "WARN")).upper()
        if status_rank(st) > status_rank(worst):
            worst = st
    return worst


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_fingerprint(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    try:
        size = path.stat().st_size
        h.update(str(size).encode())
        h.update(str(int(path.stat().st_mtime)).encode())
        with path.open("rb") as f:
            h.update(f.read(block))
            if size > block:
                f.seek(max(0, size - block))
                h.update(f.read(block))
        return h.hexdigest()
    except Exception:
        return ""


def parse_batch(path: Path) -> str:
    s = str(path).replace("\\", "/")
    m = re.search(r"batch[\s_\-]*([56])", s, flags=re.I)
    if m:
        return f"Batch-{m.group(1)}"
    return "unknown"


def parse_protocol(batch: str) -> str:
    if batch == "Batch-5":
        return "random_walk"
    if batch == "Batch-6":
        return "GEO"
    return "unknown"


def parse_cell_id(path: Path, batch: str) -> str:
    stem = path.stem
    patterns = [
        r"battery[\s_\-]*([0-9]+)",
        r"cell[\s_\-]*([0-9]+)",
        r"bat[\s_\-]*([0-9]+)",
        r"([0-9]+)$",
    ]
    idx = None
    for pat in patterns:
        m = re.search(pat, stem, flags=re.I)
        if m:
            idx = int(m.group(1))
            break
    if idx is None:
        return f"{batch}_{stem}"
    return f"{batch}_battery-{idx}"


def should_skip(path: Path, skip_fragments: Sequence[str]) -> bool:
    s = str(path).replace("\\", "/").lower()
    for frag in skip_fragments:
        if frag.lower() in s:
            return True
    return False


def discover_files(data_root: Path, batches: List[str], exts: List[str], skip_fragments: List[str]) -> List[dict]:
    rows: List[dict] = []
    if not data_root.exists():
        return rows
    extset = {e.lower() for e in exts}
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        if should_skip(p, skip_fragments):
            continue
        if p.suffix.lower() not in extset:
            continue
        batch = parse_batch(p)
        if batch not in batches:
            continue
        cell_uid = parse_cell_id(p, batch)
        rows.append({
            "batch": batch,
            "protocol": parse_protocol(batch),
            "cell_uid": cell_uid,
            "file_name": p.name,
            "file_path": str(p),
            "extension": p.suffix.lower(),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
            "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            "fingerprint_sha256_partial": file_fingerprint(p),
        })
    rows.sort(key=lambda r: (r["batch"], r["cell_uid"], r["file_name"]))
    return rows


def as_numeric_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        arr = np.asarray(x)
        if arr.dtype.names:
            return None
        arr = np.ravel(arr)
        if arr.size == 0:
            return None
        if arr.dtype.kind in "biufc":
            out = arr.astype(float)
            if out.size == 0:
                return None
            return out
        # Try datetime parsing for object/string arrays.  Use a Series wrapper so
        # both pandas Series and DatetimeIndex cases expose a stable .dt accessor.
        # Warnings are suppressed locally because this is a schema audit, not a
        # strict datetime parser; uncertain time formats are reported downstream.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Could not infer format.*", category=UserWarning)
                dt = pd.to_datetime(pd.Series(arr), errors="coerce")
            valid = dt.notna()
            if int(valid.sum()) >= max(2, int(0.8 * len(dt))):
                first = dt[valid].iloc[0]
                secs = (dt - first).dt.total_seconds().to_numpy(dtype=float)
                return secs
        except Exception:
            pass
        # Try direct numeric conversion
        ser = pd.to_numeric(pd.Series(arr), errors="coerce")
        if ser.notna().sum() >= max(2, int(0.8 * len(ser))):
            return ser.to_numpy(dtype=float)
    except Exception:
        return None
    return None


def dict_get_alias(d: Dict[str, Any], alias_group: str) -> Tuple[Optional[str], Any]:
    if not isinstance(d, dict):
        return None, None
    norm_to_key = {norm_key(k): k for k in d.keys()}
    for alias in ALIAS[alias_group]:
        nk = norm_key(alias)
        if nk in norm_to_key:
            k = norm_to_key[nk]
            return k, d.get(k)
    # relaxed contains matching
    for alias in ALIAS[alias_group]:
        nk = norm_key(alias)
        for kk, orig in norm_to_key.items():
            if nk == kk or nk in kk:
                return orig, d.get(orig)
    return None, None


def record_has_core(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict):
        return False
    _, v = dict_get_alias(d, "voltage")
    _, i = dict_get_alias(d, "current")
    _, t = dict_get_alias(d, "time")
    _, st = dict_get_alias(d, "system_time")
    va = as_numeric_array(v)
    ia = as_numeric_array(i)
    ta = as_numeric_array(t)
    sta = as_numeric_array(st)
    return va is not None and ia is not None and (ta is not None or sta is not None or len(va) == len(ia))


def collect_records(obj: Any, out: List[Dict[str, Any]], max_records: int = 100000, depth: int = 0) -> None:
    if len(out) >= max_records or depth > 8:
        return
    if isinstance(obj, dict):
        if record_has_core(obj):
            out.append(obj)
            return
        for k, v in obj.items():
            if str(k).startswith("__"):
                continue
            collect_records(v, out, max_records=max_records, depth=depth + 1)
            if len(out) >= max_records:
                break
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            collect_records(item, out, max_records=max_records, depth=depth + 1)
            if len(out) >= max_records:
                break
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype == object or obj.size < 10000:
            for item in np.ravel(obj):
                collect_records(item, out, max_records=max_records, depth=depth + 1)
                if len(out) >= max_records:
                    break
        return


def load_mat_records(path: Path) -> Tuple[List[Dict[str, Any]], dict]:
    meta = {"loader": "scipy.io.loadmat", "loader_ok": False, "loader_error": "", "root_keys": []}
    try:
        import scipy.io  # type: ignore
        mat = scipy.io.loadmat(str(path), simplify_cells=True)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        meta["root_keys"] = keys
        records: List[Dict[str, Any]] = []
        if "data" in mat:
            collect_records(mat["data"], records)
        if not records:
            for k in keys:
                collect_records(mat[k], records)
        meta["loader_ok"] = True
        return records, meta
    except Exception as exc:
        meta["loader_error"] = f"{type(exc).__name__}: {exc}"
        # v7.3 fallback: shallow h5py inspection
        try:
            import h5py  # type: ignore
            with h5py.File(path, "r") as f:
                meta["loader"] = "h5py_shallow"
                meta["root_keys"] = list(f.keys())
                meta["loader_ok"] = True
                meta["loader_error"] = "scipy failed; h5py shallow key listing only"
        except Exception as exc2:
            meta["loader"] = "none"
            meta["loader_error"] += f" | h5py fallback failed: {type(exc2).__name__}: {exc2}"
        return [], meta


def load_table_records(path: Path, max_rows: int = 2_000_000) -> Tuple[List[Dict[str, Any]], dict]:
    meta = {"loader": "pandas", "loader_ok": False, "loader_error": "", "root_keys": []}
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, nrows=max_rows)
        meta["loader_ok"] = True
        meta["root_keys"] = list(df.columns)
        if len(df) == 0:
            return [], meta
        if "cycle_id" in df.columns:
            records = []
            for _, sub in df.groupby("cycle_id", sort=True):
                records.append({c: sub[c].to_numpy() for c in df.columns})
            return records, meta
        return [{c: df[c].to_numpy() for c in df.columns}], meta
    except Exception as exc:
        meta["loader_error"] = f"{type(exc).__name__}: {exc}"
        return [], meta


def derive_time_array(record: Dict[str, Any], n_ref: int) -> Tuple[np.ndarray, str, Optional[str]]:
    k, raw = dict_get_alias(record, "time")
    arr = as_numeric_array(raw)
    if arr is not None and len(arr) >= 2:
        return arr[:n_ref], "direct_time_field", k
    k2, raw2 = dict_get_alias(record, "system_time")
    arr2 = as_numeric_array(raw2)
    if arr2 is not None and len(arr2) >= 2:
        return arr2[:n_ref], "system_time_converted", k2
    return np.arange(n_ref, dtype=float), "implicit_index_time_assumption", None


def collapse_step_types(types: Sequence[str]) -> str:
    collapsed = []
    for t in types:
        if not collapsed or collapsed[-1] != t:
            collapsed.append(t)
    if len(collapsed) > 12:
        collapsed = collapsed[:12] + ["..."]
    return "/".join(collapsed)


def summarize_record(
    record: Dict[str, Any],
    record_index: int,
    file_row: dict,
    cfg: dict,
) -> dict:
    vk, vraw = dict_get_alias(record, "voltage")
    ik, iraw = dict_get_alias(record, "current")
    tk, traw = dict_get_alias(record, "temperature")
    ck, craw = dict_get_alias(record, "capacity")

    v = as_numeric_array(vraw)
    i = as_numeric_array(iraw)
    temp = as_numeric_array(traw)
    cap = as_numeric_array(craw)

    if v is None or i is None:
        return {
            **{k: file_row.get(k) for k in ["batch", "protocol", "cell_uid", "file_path", "file_name"]},
            "record_index": record_index,
            "record_ok": False,
            "record_error": "missing_voltage_or_current_array",
        }

    n = int(min(len(v), len(i)))
    if n < 2:
        return {
            **{k: file_row.get(k) for k in ["batch", "protocol", "cell_uid", "file_path", "file_name"]},
            "record_index": record_index,
            "record_ok": False,
            "record_error": "too_few_points",
        }

    v = v[:n].astype(float)
    i = i[:n].astype(float)
    t, time_quality, time_key = derive_time_array(record, n)
    t = t[:n].astype(float)

    finite = np.isfinite(v) & np.isfinite(i) & np.isfinite(t)
    if finite.sum() < 2:
        return {
            **{k: file_row.get(k) for k in ["batch", "protocol", "cell_uid", "file_path", "file_name"]},
            "record_index": record_index,
            "record_ok": False,
            "record_error": "too_few_finite_core_points",
        }

    v = v[finite]
    i = i[finite]
    t = t[finite]
    n = len(v)

    dt = np.diff(t)
    positive_dt = np.where(dt > 0, dt, np.nan)
    median_dt = float(np.nanmedian(positive_dt)) if np.isfinite(positive_dt).any() else 1.0
    dt_for_integral = np.where(dt > 0, dt, median_dt if median_dt > 0 else 1.0)

    max_abs_i = float(np.nanmax(np.abs(i))) if n else 0.0
    rest_abs = max(
        float(cfg["current_policy"].get("rest_current_abs_A", 0.02)),
        float(cfg["current_policy"].get("dynamic_rest_fraction_of_max_abs_current", 0.01)) * max_abs_i,
    )
    step = np.where(np.abs(i) <= rest_abs, "rest", np.where(i > 0, "charge_pos_current", "discharge_neg_current"))
    step_sequence = collapse_step_types(step.tolist())

    q_pos_Ah = float(np.nansum(np.maximum(i[:-1], 0.0) * dt_for_integral / 3600.0))
    q_neg_Ah = float(np.nansum(np.maximum(-i[:-1], 0.0) * dt_for_integral / 3600.0))
    q_abs_nonrest_Ah = max(q_pos_Ah, q_neg_Ah)

    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    cutoff = float(cfg["voltage_limits"].get("full_discharge_cutoff_V", 2.5))
    tol = float(cfg["voltage_limits"].get("full_discharge_tolerance_V", 0.05))
    partial_boundary = float(cfg["voltage_limits"].get("partial_discharge_boundary_V", 3.0))

    has_discharge_like = q_abs_nonrest_Ah > 0.01 and (np.nanmax(v) - np.nanmin(v)) > 0.05
    full_discharge_candidate = bool(has_discharge_like and vmin <= cutoff + tol)
    partial_discharge_candidate = bool(has_discharge_like and (cutoff + tol) < vmin <= partial_boundary + 0.25)

    if temp is not None and len(temp) >= 2:
        temp2 = temp[:min(len(temp), n)]
        temp_min = float(np.nanmin(temp2)) if np.isfinite(temp2).any() else math.nan
        temp_max = float(np.nanmax(temp2)) if np.isfinite(temp2).any() else math.nan
    else:
        temp_min = math.nan
        temp_max = math.nan

    if cap is not None and len(cap) >= 2:
        cap2 = cap[:min(len(cap), n)].astype(float)
        cap_range = float(np.nanmax(cap2) - np.nanmin(cap2)) if np.isfinite(cap2).any() else math.nan
    else:
        cap_range = math.nan

    nonmonotonic_count = int(np.sum(np.diff(t) < 0))
    repeated_count = int(np.sum(np.diff(t) == 0))
    monotonic_nondec = bool(nonmonotonic_count == 0)
    replay_ready_record = bool(n >= 10 and monotonic_nondec and np.isfinite(v).all() and np.isfinite(i).all())

    return {
        **{k: file_row.get(k) for k in ["batch", "protocol", "cell_uid", "file_path", "file_name"]},
        "record_index": record_index,
        "record_ok": True,
        "record_error": "",
        "n_points": n,
        "voltage_field": vk or "",
        "current_field": ik or "",
        "time_field": time_key or "",
        "temperature_field": tk or "",
        "capacity_field": ck or "",
        "time_quality": time_quality,
        "time_monotonic_nondec": monotonic_nondec,
        "time_nonmonotonic_count": nonmonotonic_count,
        "time_repeated_count": repeated_count,
        "dt_median_s": median_dt,
        "voltage_min_V": vmin,
        "voltage_max_V": vmax,
        "voltage_range_V": vmax - vmin,
        "current_min_A": float(np.nanmin(i)),
        "current_max_A": float(np.nanmax(i)),
        "current_max_abs_A": max_abs_i,
        "temperature_min_C": temp_min,
        "temperature_max_C": temp_max,
        "capacity_field_range_raw": cap_range,
        "q_pos_current_Ah": q_pos_Ah,
        "q_neg_current_Ah": q_neg_Ah,
        "q_abs_candidate_Ah": q_abs_nonrest_Ah,
        "step_type_sequence_from_current": step_sequence,
        "full_discharge_candidate": full_discharge_candidate,
        "partial_discharge_candidate": partial_discharge_candidate,
        "soh_label_eligible_candidate": full_discharge_candidate,
        "replay_ready_record": replay_ready_record,
    }


def inspect_file(file_row: dict, cfg: dict) -> Tuple[dict, List[dict]]:
    path = Path(file_row["file_path"])
    ext = path.suffix.lower()
    meta: dict
    records: List[Dict[str, Any]]
    if ext == ".mat":
        records, meta = load_mat_records(path)
    else:
        records, meta = load_table_records(path)

    cycle_rows: List[dict] = []
    for idx, record in enumerate(records):
        cycle_rows.append(summarize_record(record, idx, file_row, cfg))

    ok_records = [r for r in cycle_rows if r.get("record_ok") is True]
    n_total_points = int(sum(int(r.get("n_points", 0) or 0) for r in ok_records))
    n_full = int(sum(1 for r in ok_records if str(r.get("full_discharge_candidate")).lower() == "true" or r.get("full_discharge_candidate") is True))
    n_partial = int(sum(1 for r in ok_records if str(r.get("partial_discharge_candidate")).lower() == "true" or r.get("partial_discharge_candidate") is True))
    n_replay_ready = int(sum(1 for r in ok_records if r.get("replay_ready_record") is True))
    has_voltage = any(r.get("voltage_field") for r in ok_records)
    has_current = any(r.get("current_field") for r in ok_records)
    has_time = any(r.get("time_field") or r.get("time_quality") == "implicit_index_time_assumption" for r in ok_records)
    has_temperature = any(r.get("temperature_field") for r in ok_records)
    has_capacity = any(r.get("capacity_field") for r in ok_records)

    vmins = [float(r["voltage_min_V"]) for r in ok_records if r.get("voltage_min_V") not in ("", None) and math.isfinite(float(r["voltage_min_V"]))]
    vmaxs = [float(r["voltage_max_V"]) for r in ok_records if r.get("voltage_max_V") not in ("", None) and math.isfinite(float(r["voltage_max_V"]))]
    imins = [float(r["current_min_A"]) for r in ok_records if r.get("current_min_A") not in ("", None) and math.isfinite(float(r["current_min_A"]))]
    imaxs = [float(r["current_max_A"]) for r in ok_records if r.get("current_max_A") not in ("", None) and math.isfinite(float(r["current_max_A"]))]

    replay_ready_file = bool(has_voltage and has_current and has_time and n_replay_ready > 0)
    schema_status = "PASS" if replay_ready_file else ("WARN" if meta.get("loader_ok") else "FAIL")

    file_summary = {
        **file_row,
        "loader": meta.get("loader", ""),
        "loader_ok": bool(meta.get("loader_ok")),
        "loader_error": meta.get("loader_error", ""),
        "root_keys": "|".join(map(str, meta.get("root_keys", [])[:30])),
        "record_count_detected": len(records),
        "record_count_ok": len(ok_records),
        "n_points_total_estimate": n_total_points,
        "has_voltage": has_voltage,
        "has_current": has_current,
        "has_time_or_reconstructable_index": has_time,
        "has_temperature": has_temperature,
        "has_capacity_like_field": has_capacity,
        "voltage_min_V": min(vmins) if vmins else "",
        "voltage_max_V": max(vmaxs) if vmaxs else "",
        "current_min_A": min(imins) if imins else "",
        "current_max_A": max(imaxs) if imaxs else "",
        "full_discharge_candidate_records": n_full,
        "partial_discharge_candidate_records": n_partial,
        "soh_label_eligible_candidate_records": n_full,
        "replay_ready_records": n_replay_ready,
        "replay_ready_file": replay_ready_file,
        "schema_status": schema_status,
    }
    return file_summary, cycle_rows


def load_status_from_dir(path: Optional[Path], candidates: List[str]) -> Tuple[str, str]:
    if path is None:
        return "WARN", "directory not provided"
    if not path.exists():
        return "WARN", f"directory missing: {path}"
    for name in candidates:
        obj = read_json(path / name)
        if isinstance(obj, dict):
            status = obj.get("overall_status") or obj.get("status") or obj.get("summary", {}).get("overall_status")
            if status:
                return str(status).upper(), str(path / name)
    return "WARN", f"no known status json found in {path}"


def build_checks(
    cfg: dict,
    raw_rows: List[dict],
    file_summaries: List[dict],
    p0_status: Tuple[str, str],
    p1_status: Tuple[str, str],
    p2_status: Tuple[str, str],
) -> List[dict]:
    checks: List[dict] = []
    prereq_items = [
        {"name": "P0", "status": p0_status[0], "detail": p0_status[1]},
        {"name": "P1", "status": p1_status[0], "detail": p1_status[1]},
        {"name": "P2", "status": p2_status[0], "detail": p2_status[1]},
    ]
    prereq_worst = combine_status(prereq_items)
    checks.append({
        "check_id": "P3-C00",
        "name": "D14-P0/P1/P2 no-regression prerequisites",
        "status": "FAIL" if prereq_worst == "FAIL" else ("WARN" if prereq_worst == "WARN" else "PASS"),
        "detail": "; ".join([f"{x['name']}={x['status']} ({x['detail']})" for x in prereq_items]),
    })

    expected_batches = cfg.get("expected_batches", ["Batch-5", "Batch-6"])
    for batch in expected_batches:
        n = sum(1 for r in raw_rows if r.get("batch") == batch)
        expected_n = int(cfg.get("expected_files_per_batch", 8))
        if n == 0:
            st = "FAIL"
        elif n != expected_n:
            st = "WARN"
        else:
            st = "PASS"
        checks.append({
            "check_id": f"P3-C01-{batch}",
            "name": f"{batch} raw file discovery",
            "status": st,
            "detail": f"found {n} raw files; expected around {expected_n}",
        })

    inspected = [r for r in file_summaries if r.get("loader_ok")]
    checks.append({
        "check_id": "P3-C02",
        "name": "raw loader / schema inspection",
        "status": "PASS" if inspected else "FAIL",
        "detail": f"loader_ok files={len(inspected)} / discovered={len(raw_rows)}",
    })

    replay_ready_by_batch = {}
    full_by_batch = {}
    partial_by_batch = {}
    for batch in expected_batches:
        subset = [r for r in file_summaries if r.get("batch") == batch]
        replay_ready_by_batch[batch] = sum(1 for r in subset if r.get("replay_ready_file") is True)
        full_by_batch[batch] = sum(int(r.get("full_discharge_candidate_records", 0) or 0) for r in subset)
        partial_by_batch[batch] = sum(int(r.get("partial_discharge_candidate_records", 0) or 0) for r in subset)

    for batch in expected_batches:
        rr = replay_ready_by_batch.get(batch, 0)
        if rr <= 0:
            st = "FAIL"
        else:
            st = "PASS"
        checks.append({
            "check_id": f"P3-C03-{batch}",
            "name": f"{batch} replay-readiness",
            "status": st,
            "detail": f"replay_ready_files={rr}",
        })

    total_full = sum(full_by_batch.values())
    total_partial = sum(partial_by_batch.values())
    checks.append({
        "check_id": "P3-C04",
        "name": "SOH eligibility source policy",
        "status": "PASS",
        "detail": "SOH must come from original XJTU capacity/cycle data; voltage soft-label generator remains SOH-free.",
    })
    checks.append({
        "check_id": "P3-C05",
        "name": "complete-discharge / capacity-check candidate availability",
        "status": "PASS" if total_full > 0 else "WARN",
        "detail": f"full_discharge_candidate_records={total_full}; partial_discharge_candidate_records={total_partial}; partial cycles are replay-only.",
    })

    # Ensure this P3 package does not touch the known Batch-1 outlier policy.
    suspicious = []
    for r in raw_rows + file_summaries:
        text = " ".join(str(v) for v in r.values()).lower()
        if ("batch-1" in text or "batch_1" in text or "b1_2c" in text) and ("battery-8" in text or "battery_8" in text):
            suspicious.append(r.get("file_path") or r.get("file_name"))
    checks.append({
        "check_id": "P3-C06",
        "name": "Batch-1_2C_battery-8 outlier policy not modified by P3",
        "status": "FAIL" if suspicious else "PASS",
        "detail": "P3 scans Batch-5/6 only; no Batch-1_2C_battery-8 raw item should appear." if not suspicious else f"suspicious paths={suspicious[:5]}",
    })

    return checks


def md_table(rows: List[dict], cols: List[str]) -> str:
    if not rows:
        return ""
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D14-P3 Batch-5/6 feasibility audit")
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--p0_dir", default="")
    ap.add_argument("--p1_dir", default="")
    ap.add_argument("--p2_dir", default="")
    ap.add_argument("--batches", nargs="*", default=["Batch-5", "Batch-6"])
    ap.add_argument("--max_files_to_inspect", type=int, default=0, help="0 means inspect all discovered files")
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root)
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config) if args.config else project_root / "configs" / "d14_p3_batch56_feasibility_config.json"
    cfg = read_json(cfg_path) or {}
    if not cfg:
        cfg = {
            "expected_batches": args.batches,
            "expected_files_per_batch": 8,
            "raw_extensions": [".mat", ".csv", ".parquet"],
            "skip_path_fragments": ["_gv1_cache", "CacheGV1", "xjtu_d14_", ".git"],
            "voltage_limits": {"full_discharge_cutoff_V": 2.5, "full_discharge_tolerance_V": 0.05, "partial_discharge_boundary_V": 3.0},
            "current_policy": {"rest_current_abs_A": 0.02, "dynamic_rest_fraction_of_max_abs_current": 0.01},
        }
    cfg["expected_batches"] = args.batches or cfg.get("expected_batches", ["Batch-5", "Batch-6"])

    raw_rows = discover_files(
        data_root=data_root,
        batches=cfg["expected_batches"],
        exts=cfg.get("raw_extensions", [".mat", ".csv", ".parquet"]),
        skip_fragments=cfg.get("skip_path_fragments", ["_gv1_cache", "CacheGV1", "xjtu_d14_", ".git"]),
    )
    write_csv(output_dir / "D14_P3_RAW_FILE_INDEX.csv", raw_rows)

    inspect_rows = raw_rows
    if args.max_files_to_inspect and args.max_files_to_inspect > 0:
        inspect_rows = raw_rows[: args.max_files_to_inspect]

    file_summaries: List[dict] = []
    cycle_rows: List[dict] = []
    for fr in inspect_rows:
        try:
            fs, cr = inspect_file(fr, cfg)
            file_summaries.append(fs)
            cycle_rows.extend(cr)
        except Exception as exc:
            file_summaries.append({
                **fr,
                "loader": "exception",
                "loader_ok": False,
                "loader_error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=4),
                "record_count_detected": 0,
                "record_count_ok": 0,
                "schema_status": "FAIL",
                "replay_ready_file": False,
            })

    write_csv(output_dir / "D14_P3_FILE_SCHEMA_AUDIT.csv", file_summaries)
    write_csv(output_dir / "D14_P3_CYCLE_ELIGIBILITY_SUMMARY.csv", cycle_rows)

    replay_rows = []
    for r in file_summaries:
        replay_rows.append({
            "batch": r.get("batch"),
            "protocol": r.get("protocol"),
            "cell_uid": r.get("cell_uid"),
            "file_name": r.get("file_name"),
            "file_path": r.get("file_path"),
            "replay_ready_file": r.get("replay_ready_file"),
            "record_count_ok": r.get("record_count_ok"),
            "n_points_total_estimate": r.get("n_points_total_estimate"),
            "full_discharge_candidate_records": r.get("full_discharge_candidate_records"),
            "partial_discharge_candidate_records": r.get("partial_discharge_candidate_records"),
            "schema_status": r.get("schema_status"),
            "loader": r.get("loader"),
            "loader_error": r.get("loader_error"),
        })
    write_csv(output_dir / "D14_P3_REPLAY_READINESS.csv", replay_rows)

    batch_rows = []
    for batch in cfg["expected_batches"]:
        subset = [r for r in file_summaries if r.get("batch") == batch]
        discovered = [r for r in raw_rows if r.get("batch") == batch]
        batch_rows.append({
            "batch": batch,
            "protocol": parse_protocol(batch),
            "raw_file_count": len(discovered),
            "inspected_file_count": len(subset),
            "loader_ok_count": sum(1 for r in subset if r.get("loader_ok")),
            "replay_ready_file_count": sum(1 for r in subset if r.get("replay_ready_file")),
            "record_count_ok": sum(int(r.get("record_count_ok", 0) or 0) for r in subset),
            "full_discharge_candidate_records": sum(int(r.get("full_discharge_candidate_records", 0) or 0) for r in subset),
            "partial_discharge_candidate_records": sum(int(r.get("partial_discharge_candidate_records", 0) or 0) for r in subset),
            "n_points_total_estimate": sum(int(r.get("n_points_total_estimate", 0) or 0) for r in subset),
        })
    write_csv(output_dir / "D14_P3_BATCH_SUMMARY.csv", batch_rows)

    soh_policy_rows = [
        {
            "policy_item": "SOH source",
            "status": "PASS",
            "detail": "Use original XJTU cycle/capacity data or official capacity-like fields. Do not create SOH in voltage soft-label generator.",
        },
        {
            "policy_item": "Complete discharge eligibility",
            "status": "PASS",
            "detail": "Only complete discharge / capacity-check cycles are eligible SOH labels. Batch-5/6 partial cycles are replay-only unless a capacity-check full discharge is detected.",
        },
        {
            "policy_item": "Voltage replay separation",
            "status": "PASS",
            "detail": "All cycles with usable I(t), V(t), T(t) can be used for measured-current replay; SOH eligibility is a separate label policy.",
        },
    ]
    write_csv(output_dir / "D14_P3_SOH_POLICY.csv", soh_policy_rows)

    p0_status = load_status_from_dir(Path(args.p0_dir) if args.p0_dir else None, ["D14_P0_FREEZE_AUDIT.json"])
    p1_status = load_status_from_dir(Path(args.p1_dir) if args.p1_dir else None, ["D14_P1_EVIDENCE_BOUNDARY_REPORT.json"])
    p2_status = load_status_from_dir(Path(args.p2_dir) if args.p2_dir else None, ["D14_P2_GENERALIZATION_SCORECARD_REPORT.json"])

    checks = build_checks(cfg, raw_rows, file_summaries, p0_status, p1_status, p2_status)
    overall = combine_status(checks)

    if overall == "FAIL":
        recommendation = "Do not proceed to Batch-5/6 replay-profile generation until FAIL items are resolved."
    elif overall == "WARN":
        recommendation = "Proceed only with controlled smoke/profile build; document WARN items and do not train yet."
    else:
        recommendation = "Batch-5/6 appear feasible for the next controlled replay-profile generation step."

    report = {
        "package": "D14-P3 Batch-5/6 feasibility audit",
        "created_utc": utc_now(),
        "overall_status": overall,
        "recommendation": recommendation,
        "paths": {
            "project_root": str(project_root),
            "data_root": str(data_root),
            "cache_root": str(cache_root),
            "output_dir": str(output_dir),
            "config": str(cfg_path),
            "p0_dir": args.p0_dir,
            "p1_dir": args.p1_dir,
            "p2_dir": args.p2_dir,
        },
        "summary": {
            "raw_file_count": len(raw_rows),
            "inspected_file_count": len(file_summaries),
            "cycle_record_rows": len(cycle_rows),
            "batch_summary": batch_rows,
        },
        "checks": checks,
        "boundaries": {
            "does_train": False,
            "modifies_mainline": False,
            "generates_soh_in_voltage_soft_label_generator": False,
            "generates_p2d_internal_state_labels": False,
            "outlier_policy": "Batch-1_2C_battery-8 remains flagged/excluded; Batch-5/6 audit does not change this.",
        },
    }
    write_json(output_dir / "D14_P3_BATCH56_FEASIBILITY_REPORT.json", report)

    md = []
    md.append("# D14-P3 Batch-5/6 Feasibility Audit Report\n")
    md.append(f"Created UTC: `{report['created_utc']}`\n")
    md.append(f"Overall status: **{overall}**\n")
    md.append(f"Recommendation: {recommendation}\n")
    md.append("## Checks\n")
    md.append(md_table(checks, ["check_id", "name", "status", "detail"]))
    md.append("\n## Batch summary\n")
    md.append(md_table(batch_rows, [
        "batch", "protocol", "raw_file_count", "inspected_file_count",
        "loader_ok_count", "replay_ready_file_count", "full_discharge_candidate_records",
        "partial_discharge_candidate_records"
    ]))
    md.append("\n## Boundaries\n")
    md.append("- No training is performed.\n")
    md.append("- GV1 mainline code is not modified.\n")
    md.append("- XJTU voltage soft-label generator must remain SOH-free.\n")
    md.append("- SOH labels, if needed later, must come from original XJTU cycle/capacity data.\n")
    md.append("- Batch-1_2C_battery-8 remains flagged/excluded and is not affected by this audit.\n")
    (output_dir / "D14_P3_BATCH56_FEASIBILITY_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    readme_patch = f"""# README D14-P3 Patch

## D14-P3 Batch-5/6 feasibility audit

D14-P3 audits whether XJTU Batch-5 random-walk and Batch-6 GEO data can be
routed into the existing GV1 measured-current replay pipeline.

Status: **{overall}**

Recommendation: {recommendation}

Boundary:

- D14-P3 does not train any model.
- D14-P3 does not replace D9.6/D9.5.1 or D12-S1K.
- D14-P3 does not generate SOH inside the voltage soft-label generator.
- SOH labels must come from original XJTU cycle/capacity records.
- Partial-discharge cycles are replay-only unless they are capacity-check complete-discharge cycles.
- Batch-1_2C_battery-8 remains the only currently flagged battery-8 outlier.
"""
    (output_dir / "README_D14_P3_PATCH.md").write_text(readme_patch, encoding="utf-8")

    output_files = [
        "D14_P3_BATCH56_FEASIBILITY_REPORT.json",
        "D14_P3_BATCH56_FEASIBILITY_REPORT.md",
        "D14_P3_RAW_FILE_INDEX.csv",
        "D14_P3_FILE_SCHEMA_AUDIT.csv",
        "D14_P3_CYCLE_ELIGIBILITY_SUMMARY.csv",
        "D14_P3_REPLAY_READINESS.csv",
        "D14_P3_BATCH_SUMMARY.csv",
        "D14_P3_SOH_POLICY.csv",
        "README_D14_P3_PATCH.md",
    ]
    index = {
        "overall_status": overall,
        "output_dir": str(output_dir),
        "files": [{"name": name, "path": str(output_dir / name), "exists": (output_dir / name).exists()} for name in output_files],
    }
    write_json(output_dir / "D14_P3_OUTPUT_INDEX.json", index)

    run_summary = [
        f"D14-P3 Batch-5/6 feasibility audit",
        f"created_utc={report['created_utc']}",
        f"overall_status={overall}",
        f"raw_file_count={len(raw_rows)}",
        f"inspected_file_count={len(file_summaries)}",
        f"cycle_record_rows={len(cycle_rows)}",
        f"recommendation={recommendation}",
    ]
    (output_dir / "D14_P3_RUN_SUMMARY.txt").write_text("\n".join(run_summary) + "\n", encoding="utf-8")

    print("\n".join(run_summary))
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else (2 if overall == "WARN" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
