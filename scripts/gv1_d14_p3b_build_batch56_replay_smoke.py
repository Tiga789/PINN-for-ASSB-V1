#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D14-P3B Batch-5/6 controlled replay-profile build smoke.

This is a bounded smoke conversion:
  - selected files only;
  - limited subrecords and points;
  - no training;
  - no SOH generation in voltage soft-label generator;
  - no P2D internal-state labels.
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

warnings.filterwarnings("ignore", message="Could not infer format.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

import numpy as np
import pandas as pd


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
    "step": ["step", "step_id", "step_type", "Step", "status", "mode"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(str(status).upper(), 1)


def combine_status(items: Sequence[dict]) -> str:
    worst = "PASS"
    for item in items:
        st = str(item.get("status", "WARN")).upper()
        if status_rank(st) > status_rank(worst):
            worst = st
    return worst


def partial_sha256(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    try:
        st = path.stat()
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
        with path.open("rb") as f:
            h.update(f.read(block))
            if st.st_size > block:
                f.seek(max(0, st.st_size - block))
                h.update(f.read(block))
        return h.hexdigest()
    except Exception:
        return ""


def should_skip(path: Path, skip_fragments: Sequence[str]) -> bool:
    s = str(path).replace("\\", "/").lower()
    return any(frag.lower() in s for frag in skip_fragments)


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


def parse_cell_uid(path: Path, batch: str) -> str:
    s = path.stem
    for pat in [r"battery[\s_\-]*([0-9]+)", r"cell[\s_\-]*([0-9]+)", r"bat[\s_\-]*([0-9]+)", r"([0-9]+)$"]:
        m = re.search(pat, s, flags=re.I)
        if m:
            return f"{batch}_battery-{int(m.group(1))}"
    return f"{batch}_{path.stem}"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def discover_files(data_root: Path, batches: Sequence[str], exts: Sequence[str], skip: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    extset = {e.lower() for e in exts}
    if not data_root.exists():
        return rows
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        if should_skip(p, skip):
            continue
        if p.suffix.lower() not in extset:
            continue
        batch = parse_batch(p)
        if batch not in batches:
            continue
        st = p.stat()
        rows.append({
            "batch": batch,
            "protocol": parse_protocol(batch),
            "cell_uid": parse_cell_uid(p, batch),
            "file_name": p.name,
            "file_path": str(p),
            "extension": p.suffix.lower(),
            "size_mb": round(st.st_size / 1024 / 1024, 3),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "partial_sha256": partial_sha256(p),
        })
    rows.sort(key=lambda r: (r["batch"], float(r.get("size_mb") or 0), r["cell_uid"], r["file_name"]))
    return rows


def select_files(rows: List[dict], batches: Sequence[str], files_per_batch: int) -> List[dict]:
    selected: List[dict] = []
    for batch in batches:
        subset = [r for r in rows if r.get("batch") == batch]
        subset.sort(key=lambda r: (float(r.get("size_mb") or 0), r["file_name"]))
        selected.extend(subset[:files_per_batch])
    return selected


def dict_get_alias(d: Dict[str, Any], group: str) -> Tuple[Optional[str], Any]:
    if not isinstance(d, dict):
        return None, None
    norm_to_key = {norm_key(k): k for k in d.keys()}
    for alias in ALIAS[group]:
        nk = norm_key(alias)
        if nk in norm_to_key:
            k = norm_to_key[nk]
            return k, d.get(k)
    for alias in ALIAS[group]:
        nk = norm_key(alias)
        for kk, orig in norm_to_key.items():
            if nk in kk or kk in nk:
                return orig, d.get(orig)
    return None, None


def to_1d_float(x: Any) -> Optional[np.ndarray]:
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
            return out if out.size else None
        ser = pd.to_numeric(pd.Series(arr), errors="coerce")
        if ser.notna().sum() >= max(2, int(0.6 * len(ser))):
            return ser.to_numpy(dtype=float)
    except Exception:
        return None
    return None


def to_time_seconds(x: Any, n_ref: int) -> Optional[np.ndarray]:
    arr = to_1d_float(x)
    if arr is not None and arr.size >= 2:
        return arr[:n_ref].astype(float)
    try:
        raw = np.ravel(np.asarray(x))
        if raw.size >= 2:
            dt = pd.to_datetime(pd.Series(raw), errors="coerce")
            if dt.notna().sum() >= max(2, int(0.6 * len(dt))):
                secs = (dt - dt.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
                return secs[:n_ref]
    except Exception:
        return None
    return None


def is_record_dict(d: Dict[str, Any]) -> bool:
    _, v = dict_get_alias(d, "voltage")
    _, i = dict_get_alias(d, "current")
    va = to_1d_float(v)
    ia = to_1d_float(i)
    if va is None or ia is None:
        return False
    return min(len(va), len(ia)) >= 2


def collect_records(obj: Any, out: List[Dict[str, Any]], max_records: int, depth: int = 0) -> None:
    if len(out) >= max_records or depth > 8:
        return
    if isinstance(obj, dict):
        if is_record_dict(obj):
            out.append(obj)
            return
        # prioritize data-like keys
        items = list(obj.items())
        items.sort(key=lambda kv: 0 if str(kv[0]).lower() in ("data", "summary") else 1)
        for k, v in items:
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


def load_records_from_mat(path: Path, max_records: int) -> Tuple[List[Dict[str, Any]], dict]:
    meta = {"loader": "scipy.io.loadmat", "loader_ok": False, "loader_error": "", "root_keys": []}
    try:
        import scipy.io  # type: ignore
        mat = scipy.io.loadmat(str(path), simplify_cells=True)
        keys = [k for k in mat.keys() if not str(k).startswith("__")]
        meta["root_keys"] = keys
        records: List[Dict[str, Any]] = []
        if "data" in mat:
            collect_records(mat["data"], records, max_records=max_records)
        if not records:
            for k in keys:
                collect_records(mat[k], records, max_records=max_records)
                if len(records) >= max_records:
                    break
        meta["loader_ok"] = True
        return records[:max_records], meta
    except Exception as exc:
        meta["loader_error"] = f"{type(exc).__name__}: {exc}"
        return [], meta


def load_records_from_table(path: Path, max_records: int) -> Tuple[List[Dict[str, Any]], dict]:
    meta = {"loader": "pandas", "loader_ok": False, "loader_error": "", "root_keys": []}
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        meta["root_keys"] = list(df.columns)
        meta["loader_ok"] = True
        if "cycle_id" in df.columns:
            records: List[Dict[str, Any]] = []
            for _, sub in df.groupby("cycle_id", sort=True):
                records.append({c: sub[c].to_numpy() for c in df.columns})
                if len(records) >= max_records:
                    break
            return records, meta
        return [{c: df[c].to_numpy() for c in df.columns}], meta
    except Exception as exc:
        meta["loader_error"] = f"{type(exc).__name__}: {exc}"
        return [], meta


def record_to_arrays(record: Dict[str, Any], cycle_idx: int, cfg: dict) -> Optional[dict]:
    vk, vraw = dict_get_alias(record, "voltage")
    ik, iraw = dict_get_alias(record, "current")
    tk, traw = dict_get_alias(record, "time")
    sk, sraw = dict_get_alias(record, "system_time")
    tempk, tempraw = dict_get_alias(record, "temperature")

    v = to_1d_float(vraw)
    i = to_1d_float(iraw)
    if v is None or i is None:
        return None
    n = int(min(len(v), len(i)))
    if n < 2:
        return None
    v = v[:n].astype(float)
    i = i[:n].astype(float)

    t = to_time_seconds(traw, n) if traw is not None else None
    if t is None and sraw is not None:
        t = to_time_seconds(sraw, n)
    if t is None or len(t) < n:
        t = np.arange(n, dtype=float)
    else:
        t = t[:n].astype(float)

    temp = to_1d_float(tempraw)
    if temp is None or len(temp) < n:
        temp_arr = np.full(n, np.nan, dtype=float)
    else:
        temp_arr = temp[:n].astype(float)

    finite = np.isfinite(v) & np.isfinite(i) & np.isfinite(t)
    if finite.sum() < 2:
        return None
    v = v[finite]
    i = i[finite]
    t = t[finite]
    temp_arr = temp_arr[finite] if temp_arr.shape[0] == finite.shape[0] else np.full(len(v), np.nan)

    # local time repair
    t = t - t[0]
    dt = np.diff(t)
    if np.any(dt < 0) or np.nanmax(t) <= 0:
        t = np.arange(len(v), dtype=float)

    max_abs_i = float(np.nanmax(np.abs(i))) if len(i) else 0.0
    rest_abs = max(
        float(cfg.get("current_policy", {}).get("rest_current_abs_A", 0.02)),
        float(cfg.get("current_policy", {}).get("dynamic_rest_fraction_of_max_abs_current", 0.01)) * max_abs_i,
    )
    step_type = np.where(np.abs(i) <= rest_abs, "rest", np.where(i > 0, "charge", "discharge")).astype("<U16")

    step_id = np.zeros(len(i), dtype=np.int32)
    sid = 1
    step_id[0] = sid
    for k in range(1, len(i)):
        if step_type[k] != step_type[k-1]:
            sid += 1
        step_id[k] = sid

    return {
        "t_local_s": t.astype(float),
        "voltage_exp": v.astype(float),
        "I_profile": i.astype(float),
        "temperature_C": temp_arr.astype(float),
        "cycle_id": np.full(len(i), cycle_idx, dtype=np.int32),
        "step_id": step_id,
        "step_type": step_type,
        "fields": {"voltage": vk or "", "current": ik or "", "time": tk or sk or "", "temperature": tempk or ""},
    }


def build_profile_from_file(row: dict, output_root: Path, cfg: dict, max_records: int, max_points: int) -> Tuple[dict, Optional[Path]]:
    path = Path(row["file_path"])
    if path.suffix.lower() == ".mat":
        records, meta = load_records_from_mat(path, max_records=max_records)
    else:
        records, meta = load_records_from_table(path, max_records=max_records)

    pieces: List[dict] = []
    fields_seen: List[dict] = []
    for idx, rec in enumerate(records, 1):
        arr = record_to_arrays(rec, idx, cfg)
        if arr is None:
            continue
        pieces.append(arr)
        fields_seen.append(arr["fields"])
        if sum(len(p["I_profile"]) for p in pieces) >= max_points:
            break

    summary = {
        **row,
        "loader": meta.get("loader", ""),
        "loader_ok": meta.get("loader_ok", False),
        "loader_error": meta.get("loader_error", ""),
        "root_keys": "|".join(map(str, meta.get("root_keys", [])[:30])),
        "records_detected": len(records),
        "records_used": len(pieces),
        "profile_ok": False,
        "profile_error": "",
        "n_points": 0,
        "cycle_count": 0,
        "step_count": 0,
        "time_monotonic_nondec": False,
        "voltage_min_V": "",
        "voltage_max_V": "",
        "current_min_A": "",
        "current_max_A": "",
        "temperature_available": False,
        "npz_path": "",
    }

    if not pieces:
        summary["profile_error"] = "no usable records with voltage/current"
        return summary, None

    t_all = []
    v_all = []
    i_all = []
    temp_all = []
    cycle_all = []
    step_all = []
    stype_all = []
    offset = 0.0
    global_step_offset = 0
    for p in pieces:
        local_t = p["t_local_s"].astype(float)
        if len(local_t) >= 2:
            pos_dt = np.diff(local_t)
            pos_dt = pos_dt[pos_dt > 0]
            gap = float(np.nanmedian(pos_dt)) if len(pos_dt) else 1.0
        else:
            gap = 1.0
        t_piece = local_t + offset
        offset = float(t_piece[-1] + max(gap, 1.0))

        step_piece = p["step_id"].astype(np.int32) + global_step_offset
        global_step_offset = int(step_piece.max())

        t_all.append(t_piece)
        v_all.append(p["voltage_exp"])
        i_all.append(p["I_profile"])
        temp_all.append(p["temperature_C"])
        cycle_all.append(p["cycle_id"])
        step_all.append(step_piece)
        stype_all.append(p["step_type"])

    t = np.concatenate(t_all)
    v = np.concatenate(v_all)
    i = np.concatenate(i_all)
    temp = np.concatenate(temp_all)
    cycle = np.concatenate(cycle_all)
    step = np.concatenate(step_all)
    stype = np.concatenate(stype_all).astype("<U16")

    if len(t) > max_points:
        t = t[:max_points]
        v = v[:max_points]
        i = i[:max_points]
        temp = temp[:max_points]
        cycle = cycle[:max_points]
        step = step[:max_points]
        stype = stype[:max_points]

    finite = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    if finite.sum() < 10:
        summary["profile_error"] = "too few finite points"
        return summary, None

    out_dir = output_root / "profiles" / safe_name(f"{row['batch']}_{row['cell_uid']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "solution_replay_profile.npz"

    np.savez_compressed(
        npz_path,
        t_global_s=t.astype(float),
        I_profile=i.astype(float),
        voltage_exp=v.astype(float),
        temperature_C=temp.astype(float),
        cycle_id=cycle.astype(np.int32),
        step_id=step.astype(np.int32),
        step_type=stype,
        source_file=str(path),
        batch=str(row.get("batch", "")),
        protocol=str(row.get("protocol", "")),
        cell_uid=str(row.get("cell_uid", "")),
    )

    summary.update({
        "profile_ok": True,
        "profile_error": "",
        "n_points": int(len(t)),
        "cycle_count": int(len(np.unique(cycle))),
        "step_count": int(len(np.unique(step))),
        "time_monotonic_nondec": bool(np.all(np.diff(t) >= 0)),
        "voltage_min_V": float(np.nanmin(v)),
        "voltage_max_V": float(np.nanmax(v)),
        "current_min_A": float(np.nanmin(i)),
        "current_max_A": float(np.nanmax(i)),
        "temperature_available": bool(np.isfinite(temp).any()),
        "npz_path": str(npz_path),
    })
    write_json(out_dir / "profile_summary.json", summary)
    return summary, npz_path


def validate_npz(path: Path, required_keys: Sequence[str]) -> dict:
    row = {"npz_path": str(path), "exists": path.exists(), "status": "FAIL", "detail": ""}
    if not path.exists():
        row["detail"] = "missing npz"
        return row
    try:
        data = np.load(path, allow_pickle=True)
        keys = set(data.files)
        missing = [k for k in required_keys if k not in keys]
        if missing:
            row["detail"] = "missing keys: " + ",".join(missing)
            return row
        t = data["t_global_s"]
        i = data["I_profile"]
        v = data["voltage_exp"]
        cycle = data["cycle_id"]
        step = data["step_id"]
        ok = (
            len(t) == len(i) == len(v) == len(cycle) == len(step)
            and len(t) >= 10
            and np.all(np.diff(t) >= 0)
            and np.isfinite(t).all()
            and np.isfinite(i).all()
            and np.isfinite(v).all()
        )
        row.update({
            "status": "PASS" if ok else "FAIL",
            "detail": "ok" if ok else "length/time/finite check failed",
            "n_points": int(len(t)),
            "cycle_count": int(len(np.unique(cycle))),
            "step_count": int(len(np.unique(step))),
            "voltage_min_V": float(np.nanmin(v)),
            "voltage_max_V": float(np.nanmax(v)),
            "current_min_A": float(np.nanmin(i)),
            "current_max_A": float(np.nanmax(i)),
        })
    except Exception as exc:
        row["detail"] = f"{type(exc).__name__}: {exc}"
    return row


def load_status_from_dir(path_s: str, filenames: Sequence[str]) -> Tuple[str, str]:
    if not path_s:
        return "WARN", "directory not provided"
    d = Path(path_s)
    if not d.exists():
        return "WARN", f"directory missing: {d}"
    for fn in filenames:
        obj = read_json(d / fn)
        if isinstance(obj, dict):
            st = obj.get("overall_status") or obj.get("status") or obj.get("summary", {}).get("overall_status")
            if st:
                return str(st).upper(), str(d / fn)
    return "WARN", f"no known status json found in {d}"


def md_table(rows: List[dict], cols: List[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--p3_fast_dir", default="")
    ap.add_argument("--files_per_batch", type=int, default=1)
    ap.add_argument("--max_subrecords_per_file", type=int, default=30)
    ap.add_argument("--max_total_points_per_profile", type=int, default=120000)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config) if args.config else project_root / "configs" / "d14_p3b_batch56_replay_smoke_config.json"
    cfg = read_json(cfg_path) or {}
    batches = cfg.get("batches", ["Batch-5", "Batch-6"])
    exts = cfg.get("raw_extensions", [".mat", ".csv", ".parquet"])
    skip = cfg.get("skip_path_fragments", ["_gv1_cache", "CacheGV1", "xjtu_d14_", ".git", "__pycache__"])
    required_keys = cfg.get("required_npz_keys", ["t_global_s", "I_profile", "voltage_exp", "temperature_C", "cycle_id", "step_id", "step_type"])

    print(f"[D14-P3B] start {utc_now()}", flush=True)
    raw_rows = discover_files(Path(args.data_root), batches, exts, skip)
    selected = select_files(raw_rows, batches, args.files_per_batch)
    write_csv(output_dir / "D14_P3B_SELECTED_FILES.csv", selected)
    print(f"[D14-P3B] discovered={len(raw_rows)} selected={len(selected)}", flush=True)

    profile_rows: List[dict] = []
    npz_paths: List[Path] = []
    for idx, row in enumerate(selected, 1):
        print(f"[D14-P3B] build profile {idx}/{len(selected)}: {row['batch']} {row['file_name']}", flush=True)
        try:
            summary, npz_path = build_profile_from_file(
                row=row,
                output_root=output_dir,
                cfg=cfg,
                max_records=args.max_subrecords_per_file,
                max_points=args.max_total_points_per_profile,
            )
        except Exception as exc:
            summary = {
                **row,
                "profile_ok": False,
                "profile_error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc(limit=6),
                "npz_path": "",
            }
            npz_path = None
        profile_rows.append(summary)
        if npz_path is not None:
            npz_paths.append(npz_path)

    write_csv(output_dir / "D14_P3B_PROFILE_SMOKE_SUMMARY.csv", profile_rows)
    manifest_rows = []
    for row in profile_rows:
        manifest_rows.append({
            "batch": row.get("batch"),
            "protocol": row.get("protocol"),
            "cell_uid": row.get("cell_uid"),
            "source_file": row.get("file_path"),
            "profile_npz": row.get("npz_path"),
            "profile_ok": row.get("profile_ok"),
            "n_points": row.get("n_points"),
            "cycle_count": row.get("cycle_count"),
            "step_count": row.get("step_count"),
        })
    write_csv(output_dir / "D14_P3B_PROFILE_MANIFEST.csv", manifest_rows)

    validation_rows = [validate_npz(p, required_keys) for p in npz_paths]
    write_csv(output_dir / "D14_P3B_REPLAY_VALIDATION.csv", validation_rows)

    soh_rows = [
        {"policy_item": "SOH source", "status": "PASS", "detail": "Use original XJTU capacity/cycle data; P3B replay smoke does not generate SOH labels."},
        {"policy_item": "Voltage replay profile", "status": "PASS", "detail": "Replay NPZ contains I(t), V(t), optional T(t), cycle_id, step_id, step_type."},
        {"policy_item": "Partial discharge", "status": "PASS", "detail": "Batch-5/6 partial cycles are replay-only unless later identified as capacity-check full discharge cycles."},
    ]
    write_csv(output_dir / "D14_P3B_SOH_POLICY.csv", soh_rows)

    p3_status = load_status_from_dir(args.p3_fast_dir, ["D14_P3_BATCH56_FEASIBILITY_REPORT.json"])
    checks: List[dict] = []
    checks.append({
        "check_id": "P3B-C00",
        "name": "P3 FAST prerequisite",
        "status": "FAIL" if p3_status[0] == "FAIL" else ("WARN" if p3_status[0] == "WARN" else "PASS"),
        "detail": f"P3_FAST={p3_status[0]} ({p3_status[1]})",
    })
    for batch in batches:
        n_sel = sum(1 for r in selected if r.get("batch") == batch)
        n_ok = sum(1 for r in profile_rows if r.get("batch") == batch and r.get("profile_ok") is True)
        checks.append({
            "check_id": f"P3B-C01-{batch}",
            "name": f"{batch} replay profile smoke build",
            "status": "PASS" if n_ok >= min(args.files_per_batch, max(1, n_sel)) else "FAIL",
            "detail": f"selected={n_sel}; profile_ok={n_ok}",
        })
    val_fail = [r for r in validation_rows if r.get("status") != "PASS"]
    checks.append({
        "check_id": "P3B-C02",
        "name": "replay NPZ validation",
        "status": "PASS" if validation_rows and not val_fail else "FAIL",
        "detail": f"validated={len(validation_rows)}; failed={len(val_fail)}",
    })
    checks.append({
        "check_id": "P3B-C03",
        "name": "SOH-free voltage generator boundary",
        "status": "PASS",
        "detail": "P3B did not generate SOH labels; SOH remains external capacity/cycle data.",
    })
    checks.append({
        "check_id": "P3B-C04",
        "name": "GV1 mainline not modified",
        "status": "PASS",
        "detail": "P3B only writes output profiles and audit CSV/JSON under OutputDir.",
    })

    overall = combine_status(checks)
    if overall == "FAIL":
        recommendation = "Do not expand Batch-5/6 profile generation until FAIL checks are fixed."
    elif overall == "WARN":
        recommendation = "Controlled replay-profile smoke succeeded with warnings; review inherited P3 FAST WARN before full Batch-5/6 build."
    else:
        recommendation = "Controlled Batch-5/6 replay-profile smoke succeeded. Next step can build a small full-profile manifest."

    report = {
        "package": "D14-P3B Batch-5/6 controlled replay-profile build smoke",
        "created_utc": utc_now(),
        "overall_status": overall,
        "recommendation": recommendation,
        "paths": {
            "project_root": args.project_root,
            "data_root": args.data_root,
            "cache_root": args.cache_root,
            "output_dir": args.output_dir,
            "config": str(cfg_path),
            "p3_fast_dir": args.p3_fast_dir,
        },
        "summary": {
            "raw_discovered": len(raw_rows),
            "selected_files": len(selected),
            "profile_ok_count": sum(1 for r in profile_rows if r.get("profile_ok") is True),
            "validation_pass_count": sum(1 for r in validation_rows if r.get("status") == "PASS"),
            "profile_manifest_rows": len(manifest_rows),
        },
        "checks": checks,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(output_dir / "D14_P3B_REPLAY_SMOKE_REPORT.json", report)

    md = []
    md.append("# D14-P3B Batch-5/6 Replay-Profile Smoke Report\n")
    md.append(f"Created UTC: `{report['created_utc']}`\n")
    md.append(f"Overall status: **{overall}**\n")
    md.append(f"Recommendation: {recommendation}\n")
    md.append("## Checks\n")
    md.append(md_table(checks, ["check_id", "name", "status", "detail"]))
    md.append("\n## Profile summary\n")
    md.append(md_table(profile_rows, ["batch", "cell_uid", "profile_ok", "n_points", "cycle_count", "step_count", "voltage_min_V", "voltage_max_V", "current_min_A", "current_max_A", "npz_path", "profile_error"]))
    md.append("\n## Boundaries\n")
    md.append("- No training.\n")
    md.append("- No GV1 mainline modification.\n")
    md.append("- No SOH generation in the XJTU voltage soft-label generator.\n")
    md.append("- No P2D internal-state labels.\n")
    (output_dir / "D14_P3B_REPLAY_SMOKE_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    readme_patch = f"""# README D14-P3B Patch

D14-P3B performs a controlled Batch-5/6 replay-profile smoke build.

Status: **{overall}**

Recommendation: {recommendation}

Boundary:

- No training.
- No GV1 mainline changes.
- No SOH generation in XJTU voltage soft-label generator.
- Replay profile smoke only verifies `time/I/V/T/cycle/step` expansion for selected Batch-5/6 files.
"""
    (output_dir / "README_D14_P3B_PATCH.md").write_text(readme_patch, encoding="utf-8")

    outputs = [
        "D14_P3B_REPLAY_SMOKE_REPORT.json",
        "D14_P3B_REPLAY_SMOKE_REPORT.md",
        "D14_P3B_SELECTED_FILES.csv",
        "D14_P3B_PROFILE_MANIFEST.csv",
        "D14_P3B_PROFILE_SMOKE_SUMMARY.csv",
        "D14_P3B_REPLAY_VALIDATION.csv",
        "D14_P3B_SOH_POLICY.csv",
        "D14_P3B_OUTPUT_INDEX.json",
        "D14_P3B_RUN_SUMMARY.txt",
        "README_D14_P3B_PATCH.md",
    ]
    index = {
        "overall_status": overall,
        "output_dir": args.output_dir,
        "files": [{"name": f, "exists": (output_dir / f).exists()} for f in outputs],
    }
    write_json(output_dir / "D14_P3B_OUTPUT_INDEX.json", index)
    (output_dir / "D14_P3B_RUN_SUMMARY.txt").write_text(
        "\n".join([
            "D14-P3B Batch-5/6 replay-profile smoke",
            f"created_utc={report['created_utc']}",
            f"overall_status={overall}",
            f"raw_discovered={len(raw_rows)}",
            f"selected_files={len(selected)}",
            f"profile_ok_count={report['summary']['profile_ok_count']}",
            f"validation_pass_count={report['summary']['validation_pass_count']}",
            f"recommendation={recommendation}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[D14-P3B] overall_status={overall}", flush=True)
    print(f"[D14-P3B] recommendation={recommendation}", flush=True)
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else (2 if overall == "WARN" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
