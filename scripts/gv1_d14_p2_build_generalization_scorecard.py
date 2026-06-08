
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D14-P2 XJTU real-data generalization scorecard builder.

Purpose
-------
Build a unified, auditable scorecard for the already-completed XJTU voltage
replay evidence. This script does NOT train a model and does NOT modify GV1
mainline files. It consolidates D10-P1 / D12-S1K scorecards plus D14-P0/P1
audits into global, protocol, cell, segment, candidate-comparison and outlier
policy tables.

Expected use
------------
python scripts/gv1_d14_p2_build_generalization_scorecard.py ^
  --project-root "C:/.../PINN-for-ASSB-V1" ^
  --cache-root "E:/XJTU battery dataset/_gv1_cache" ^
  --p0-dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d14_p0_freeze_audit_v2" ^
  --p1-dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d14_p1_evidence_boundary_v2" ^
  --output-dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d14_p2_generalization_scorecard"
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    print("[D14-P2][ERROR] pandas is required. Install with: pip install pandas", file=sys.stderr)
    raise

SCHEMA_VERSION = "D14-P2-generalization-scorecard-v1"

DEFAULT_D10P1_DIRNAME = "xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8"
DEFAULT_D12S1K_200KS_DIRNAME = "xjtu_batch134_d12_s1k_two_candidate_23x200ks_scorecard"
DEFAULT_D12S1K_40KS_DIRNAME = "xjtu_batch134_d12_s1k_two_candidate_23x40ks_scorecard"
DEFAULT_D13_DIRNAME = "xjtu_batch134_d13_segment_protocol_diagnosis"

RUN_METRIC_CANDIDATES = {
    "mae": ["mae_v", "mae", "MAE_V", "MAE", "mean_mae", "mean_MAE", "global_mae", "voltage_mae"],
    "rmse": ["rmse_v", "rmse", "RMSE_V", "RMSE", "mean_rmse", "mean_RMSE"],
    "corr": ["corr", "correlation", "mean_corr", "pearson", "r"],
    "bias": ["bias_v", "bias", "mean_bias", "BIAS_V", "BIAS"],
    "status": ["status", "run_status", "metrics_status", "verdict"],
    "candidate": ["mode", "candidate", "variant", "method", "wrapper", "case"],
    "profile": ["profile", "profile_id", "cell", "cell_id", "cell_uid", "run", "run_id"],
    "segment": ["segment", "segment_name", "region", "voltage_segment", "step_segment"],
    "protocol": ["protocol", "protocol_id", "batch_protocol"],
    "time_window": ["time_window_s", "window_s", "max_time_s", "duration_s"],
}

SEGMENT_HINTS = ("segment", "low_target", "low_le_2p75", "high_target", "rest", "charge", "discharge")

# ---------------- utilities ----------------

def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_col(columns: Sequence[str], kind: str) -> Optional[str]:
    exact_norm = {norm_col(c): c for c in columns}
    for cand in RUN_METRIC_CANDIDATES.get(kind, []):
        if norm_col(cand) in exact_norm:
            return exact_norm[norm_col(cand)]
    # fuzzy fallbacks
    ncols = [(norm_col(c), c) for c in columns]
    if kind == "mae":
        for nc, c in ncols:
            if "mae" in nc and not any(x in nc for x in ["delta", "improve", "change"]):
                return c
    if kind == "rmse":
        for nc, c in ncols:
            if "rmse" in nc:
                return c
    if kind == "corr":
        for nc, c in ncols:
            if nc in ("r", "corr") or "corr" in nc or "pearson" in nc:
                return c
    if kind == "bias":
        for nc, c in ncols:
            if "bias" in nc:
                return c
    if kind == "candidate":
        for nc, c in ncols:
            if nc in ("mode", "variant", "candidate", "method", "case"):
                return c
    if kind == "profile":
        for nc, c in ncols:
            if "profile" in nc or nc in ("cell", "cell_id", "cell_uid"):
                return c
    if kind == "segment":
        for nc, c in ncols:
            if "segment" in nc or "region" in nc:
                return c
    if kind == "protocol":
        for nc, c in ncols:
            if "protocol" in nc:
                return c
    return None


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip().replace("%", "")
            if s == "" or s.lower() in {"nan", "none", "null", "inf", "-inf"}:
                return None
            return float(s)
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def status_ok(value: Any) -> Optional[bool]:
    s = safe_str(value).lower()
    if not s:
        return None
    if any(k in s for k in ["fail", "error", "read_error", "not_ok", "bad"]):
        return False
    if any(k in s for k in ["pass", "ok", "metrics_ok", "completed", "success"]):
        return True
    return None


def infer_stage_from_path(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    if "d12_s1k" in text or "s1k" in text:
        return "D12-S1K"
    if "d10_p1" in text or "d10p1" in text or "23profile_200ks" in text:
        return "D10-P1"
    if "d13" in text:
        return "D13-segment-diagnosis"
    if "d14_p0" in text:
        return "D14-P0"
    return "unknown"


def infer_time_window_from_path(path: Path, row: Optional[dict] = None) -> Optional[int]:
    if row:
        for key in ["time_window_s", "window_s", "max_time_s", "duration_s"]:
            if key in row:
                val = safe_float(row.get(key))
                if val is not None:
                    return int(round(val))
    text = str(path).lower()
    m = re.search(r"(40|200|500)ks", text)
    if m:
        return int(m.group(1)) * 1000
    m = re.search(r"(\d+)x(40|200|500)ks", text)
    if m:
        return int(m.group(2)) * 1000
    return None


def infer_profile_parts(profile: str, path: Optional[Path] = None) -> Dict[str, Any]:
    text = " ".join([profile or "", str(path or "")])
    norm = text.replace("\\", "/")
    batch = None
    battery = None
    protocol = None

    # Batch forms: Batch-1_battery-8, Batch-1/battery-8, B1_2C_battery-8
    m = re.search(r"Batch[-_ ]?(\d+)", norm, flags=re.I)
    if m:
        batch = f"Batch-{int(m.group(1))}"
    m = re.search(r"\bB(\d+)\b", norm, flags=re.I)
    if batch is None and m:
        batch = f"Batch-{int(m.group(1))}"

    m = re.search(r"battery[-_ ]?(\d+)", norm, flags=re.I)
    if m:
        battery = int(m.group(1))
    else:
        # Sometimes profile is like Batch-1_battery-8; handled above.
        pass

    if re.search(r"\b2C\b", norm, flags=re.I):
        protocol = "2C"
    if re.search(r"R2\.5|R2p5|R25", norm, flags=re.I):
        protocol = "R2.5"
    if re.search(r"\bR3\b", norm, flags=re.I):
        protocol = "R3"
    if protocol is None and batch:
        protocol = {"Batch-1": "2C", "Batch-3": "R2.5", "Batch-4": "R3"}.get(batch)

    profile_id = profile
    if not profile_id and batch and battery:
        profile_id = f"{batch}_battery-{battery}"

    is_b1_b8 = (batch == "Batch-1" and protocol == "2C" and battery == 8)
    return {
        "profile_id_norm": profile_id,
        "batch": batch or "unknown",
        "protocol": protocol or "unknown",
        "battery_index": battery,
        "is_batch1_2c_battery8": bool(is_b1_b8),
        "is_any_battery8": bool(battery == 8),
    }


def classify_csv(path: Path, df: pd.DataFrame) -> str:
    cols = list(df.columns)
    segment_col = find_col(cols, "segment")
    mae_col = find_col(cols, "mae")
    corr_col = find_col(cols, "corr")
    profile_col = find_col(cols, "profile")
    cand_col = find_col(cols, "candidate")
    text = path.name.lower()
    if segment_col or "segment" in text:
        return "segment_metrics"
    if mae_col and (profile_col or cand_col or corr_col):
        if "training_history" in text or "history" in text:
            return "training_history"
        return "run_metrics"
    if cand_col and ("decision" in text or "summary" in text):
        return "candidate_summary"
    return "other"


def discover_csv_files(dirs: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root in dirs:
        if not root or not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".csv":
            key = str(root.resolve()).lower()
            if key not in seen:
                out.append(root); seen.add(key)
            continue
        for p in root.rglob("*.csv"):
            # Skip giant or irrelevant files by name where possible.
            name = p.name.lower()
            if any(skip in name for skip in ["training_history", "history"]):
                # inventory but not scoring; keep optional? Skip to avoid noisy scorecards.
                continue
            key = str(p.resolve()).lower()
            if key not in seen:
                out.append(p); seen.add(key)
    return out


def read_csv_limited(path: Path, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    try:
        if max_rows:
            return pd.read_csv(path, nrows=max_rows)
        return pd.read_csv(path)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8-sig", nrows=max_rows)
        except Exception:
            return None
    except Exception:
        return None

# ---------------- P0/P1 fallback ----------------

def load_p0_p1(p0_dir: Path, p1_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p0 = read_json(p0_dir / "D14_P0_FREEZE_AUDIT.json", {}) or {}
    p1 = read_json(p1_dir / "D14_P1_EVIDENCE_BOUNDARY_REPORT.json", {}) or {}
    return p0, p1


def default_dirs(cache_root: Path, args: argparse.Namespace, p0_dir: Path) -> Dict[str, Path]:
    dirs = {
        "d10_p1_dir": Path(args.d10_p1_dir) if args.d10_p1_dir else cache_root / DEFAULT_D10P1_DIRNAME,
        "d12_s1k_200ks_dir": Path(args.d12_s1k_200ks_dir) if args.d12_s1k_200ks_dir else cache_root / DEFAULT_D12S1K_200KS_DIRNAME,
        "d12_s1k_40ks_dir": Path(args.d12_s1k_40ks_dir) if args.d12_s1k_40ks_dir else cache_root / DEFAULT_D12S1K_40KS_DIRNAME,
        "d13_segment_dir": Path(args.d13_segment_dir) if args.d13_segment_dir else cache_root / DEFAULT_D13_DIRNAME,
    }
    # P0 contains authoritative dirs from last audit; use them as hints if default not found.
    p0_json = read_json(p0_dir / "D14_P0_FREEZE_AUDIT.json", {}) or {}
    for key, val in [("d10_p1_dir", p0_json.get("d10_p1_dir")), ("d12_s1k_200ks_dir", p0_json.get("d12_s1k_dir"))]:
        if val and not dirs[key].exists():
            dirs[key] = Path(val)
    return dirs


def fallback_rows_from_p0_index(p0_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """Fallback aggregate records from D14_P0_SCORECARD_INDEX.json when raw CSV dirs are unavailable.

    Returns (run_rows, segment_rows, inventory_rows). These rows have aggregate granularity.
    """
    idx_path = p0_dir / "D14_P0_SCORECARD_INDEX.json"
    data = read_json(idx_path, []) or []
    run_rows: List[dict] = []
    seg_rows: List[dict] = []
    inventory: List[dict] = []
    if not isinstance(data, list):
        return run_rows, seg_rows, inventory
    for item in data:
        if not isinstance(item, dict):
            continue
        csv_path = Path(str(item.get("csv_path", "")))
        stage = infer_stage_from_path(csv_path)
        inv = {
            "source_stage": stage,
            "csv_path": str(csv_path),
            "root": str(item.get("root", "")),
            "row_count_scanned": item.get("row_count_scanned"),
            "field_count": item.get("field_count"),
            "candidate_field": item.get("candidate_field"),
            "profile_field": item.get("profile_field"),
            "mae_field": item.get("mae_field"),
            "corr_field": item.get("corr_field"),
            "battery8_rows": item.get("battery8_rows"),
            "unflagged_battery8_rows": item.get("unflagged_battery8_rows"),
        }
        inventory.append(inv)
        summaries = item.get("candidate_summary") or []
        is_segment = "segment" in csv_path.name.lower()
        if isinstance(summaries, list):
            for s in summaries:
                if not isinstance(s, dict):
                    continue
                cand = s.get("candidate") or "baseline_or_unknown"
                row = {
                    "source_stage": stage,
                    "source_file": str(csv_path),
                    "record_granularity": "aggregate_from_p0_index",
                    "candidate": cand,
                    "profile_id": "<aggregate>",
                    "batch": "aggregate",
                    "protocol": "aggregate",
                    "battery_index": None,
                    "time_window_s": infer_time_window_from_path(csv_path),
                    "MAE_V": s.get("mean_mae_from_csv"),
                    "RMSE_V": None,
                    "corr": s.get("mean_corr_from_csv"),
                    "bias_V": None,
                    "status": "aggregate_from_p0_index",
                    "metrics_ok": True,
                    "is_batch1_2c_battery8": False,
                    "is_any_battery8": False,
                }
                if is_segment:
                    seg = dict(row)
                    seg["segment"] = "<aggregate_all_segments>"
                    seg_rows.append(seg)
                else:
                    run_rows.append(row)
    return run_rows, seg_rows, inventory

# ---------------- parsers ----------------

def parse_run_metrics_csv(path: Path, df: pd.DataFrame) -> List[dict]:
    cols = list(df.columns)
    mae_col = find_col(cols, "mae")
    rmse_col = find_col(cols, "rmse")
    corr_col = find_col(cols, "corr")
    bias_col = find_col(cols, "bias")
    status_col = find_col(cols, "status")
    cand_col = find_col(cols, "candidate")
    profile_col = find_col(cols, "profile")
    protocol_col = find_col(cols, "protocol")
    tw_col = find_col(cols, "time_window")

    if not mae_col and not corr_col:
        return []

    rows: List[dict] = []
    for _, r in df.iterrows():
        profile = safe_str(r.get(profile_col)) if profile_col else ""
        parts = infer_profile_parts(profile, path)
        cand = safe_str(r.get(cand_col)) if cand_col else ""
        if not cand:
            # Provide a stable default for D10-P1 baseline.
            cand = "d10p1_d96_baseline" if infer_stage_from_path(path) == "D10-P1" else "baseline_or_unknown"
        protocol = safe_str(r.get(protocol_col)) if protocol_col else ""
        if protocol:
            parts["protocol"] = protocol
        status = safe_str(r.get(status_col)) if status_col else ""
        ok = status_ok(status)
        rows.append({
            "source_stage": infer_stage_from_path(path),
            "source_file": str(path),
            "record_granularity": "per_run_or_profile",
            "candidate": cand,
            "profile_id": profile or parts["profile_id_norm"] or "unknown_profile",
            "batch": parts["batch"],
            "protocol": parts["protocol"],
            "battery_index": parts["battery_index"],
            "time_window_s": safe_float(r.get(tw_col)) if tw_col else infer_time_window_from_path(path),
            "MAE_V": safe_float(r.get(mae_col)) if mae_col else None,
            "RMSE_V": safe_float(r.get(rmse_col)) if rmse_col else None,
            "corr": safe_float(r.get(corr_col)) if corr_col else None,
            "bias_V": safe_float(r.get(bias_col)) if bias_col else None,
            "status": status,
            "metrics_ok": True if ok is None else ok,
            "is_batch1_2c_battery8": parts["is_batch1_2c_battery8"],
            "is_any_battery8": parts["is_any_battery8"],
        })
    return rows


def parse_segment_metrics_csv(path: Path, df: pd.DataFrame) -> List[dict]:
    cols = list(df.columns)
    seg_col = find_col(cols, "segment")
    mae_col = find_col(cols, "mae")
    rmse_col = find_col(cols, "rmse")
    corr_col = find_col(cols, "corr")
    bias_col = find_col(cols, "bias")
    cand_col = find_col(cols, "candidate")
    profile_col = find_col(cols, "profile")
    status_col = find_col(cols, "status")
    protocol_col = find_col(cols, "protocol")
    tw_col = find_col(cols, "time_window")
    if not mae_col and not corr_col:
        return []
    rows: List[dict] = []
    for _, r in df.iterrows():
        profile = safe_str(r.get(profile_col)) if profile_col else ""
        parts = infer_profile_parts(profile, path)
        cand = safe_str(r.get(cand_col)) if cand_col else ""
        if not cand:
            cand = "d10p1_d96_baseline" if infer_stage_from_path(path) == "D10-P1" else "baseline_or_unknown"
        protocol = safe_str(r.get(protocol_col)) if protocol_col else ""
        if protocol:
            parts["protocol"] = protocol
        status = safe_str(r.get(status_col)) if status_col else ""
        ok = status_ok(status)
        rows.append({
            "source_stage": infer_stage_from_path(path),
            "source_file": str(path),
            "record_granularity": "per_segment",
            "candidate": cand,
            "profile_id": profile or parts["profile_id_norm"] or "unknown_profile",
            "batch": parts["batch"],
            "protocol": parts["protocol"],
            "battery_index": parts["battery_index"],
            "segment": safe_str(r.get(seg_col)) if seg_col else "<unknown_segment>",
            "time_window_s": safe_float(r.get(tw_col)) if tw_col else infer_time_window_from_path(path),
            "MAE_V": safe_float(r.get(mae_col)) if mae_col else None,
            "RMSE_V": safe_float(r.get(rmse_col)) if rmse_col else None,
            "corr": safe_float(r.get(corr_col)) if corr_col else None,
            "bias_V": safe_float(r.get(bias_col)) if bias_col else None,
            "status": status,
            "metrics_ok": True if ok is None else ok,
            "is_batch1_2c_battery8": parts["is_batch1_2c_battery8"],
            "is_any_battery8": parts["is_any_battery8"],
        })
    return rows


def collect_from_csvs(csv_paths: List[Path]) -> Tuple[List[dict], List[dict], List[dict]]:
    run_rows: List[dict] = []
    segment_rows: List[dict] = []
    inventory_rows: List[dict] = []
    for path in csv_paths:
        df = read_csv_limited(path)
        if df is None or df.empty:
            inventory_rows.append({"csv_path": str(path), "status": "read_error_or_empty"})
            continue
        kind = classify_csv(path, df)
        inv = {
            "csv_path": str(path),
            "source_stage": infer_stage_from_path(path),
            "kind": kind,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": ";".join(map(str, df.columns[:30])),
        }
        inventory_rows.append(inv)
        try:
            if kind == "segment_metrics":
                segment_rows.extend(parse_segment_metrics_csv(path, df))
            elif kind == "run_metrics":
                run_rows.extend(parse_run_metrics_csv(path, df))
            elif kind == "candidate_summary":
                # Treat mode summaries as aggregate run-level rows if they include mean columns.
                rows = parse_run_metrics_csv(path, df)
                if rows:
                    for r in rows:
                        r["record_granularity"] = "candidate_summary"
                    run_rows.extend(rows)
        except Exception as exc:
            inv["parse_error"] = repr(exc)
    return run_rows, segment_rows, inventory_rows

# ---------------- aggregation ----------------

def to_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate_metrics(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    base_cols = ["MAE_V", "RMSE_V", "corr", "bias_V"]
    expected_cols = list(group_cols) + [
        "n_rows", "n_profiles", "n_metrics_ok", "n_metrics_fail", "mean_MAE_V", "median_MAE_V", "max_MAE_V",
        "mean_RMSE_V", "mean_corr", "min_corr", "mean_bias_V", "max_abs_bias_V"
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=expected_cols)
    d = df.copy()
    d = to_numeric(d, base_cols)
    if "metrics_ok" not in d.columns:
        d["metrics_ok"] = True
    for g in group_cols:
        if g not in d.columns:
            d[g] = "unknown"
    grouped = []
    for keys, sub in d.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {g: k for g, k in zip(group_cols, keys)}
        rec["n_rows"] = int(len(sub))
        if "profile_id" in sub.columns:
            rec["n_profiles"] = int(sub["profile_id"].nunique(dropna=True))
        else:
            rec["n_profiles"] = None
        rec["n_metrics_ok"] = int((sub["metrics_ok"] == True).sum())
        rec["n_metrics_fail"] = int((sub["metrics_ok"] == False).sum())
        mae = sub["MAE_V"] if "MAE_V" in sub else pd.Series(dtype=float)
        rmse = sub["RMSE_V"] if "RMSE_V" in sub else pd.Series(dtype=float)
        corr = sub["corr"] if "corr" in sub else pd.Series(dtype=float)
        bias = sub["bias_V"] if "bias_V" in sub else pd.Series(dtype=float)
        rec["mean_MAE_V"] = float(mae.mean()) if mae.notna().any() else None
        rec["median_MAE_V"] = float(mae.median()) if mae.notna().any() else None
        rec["max_MAE_V"] = float(mae.max()) if mae.notna().any() else None
        rec["mean_RMSE_V"] = float(rmse.mean()) if rmse.notna().any() else None
        rec["mean_corr"] = float(corr.mean()) if corr.notna().any() else None
        rec["min_corr"] = float(corr.min()) if corr.notna().any() else None
        rec["mean_bias_V"] = float(bias.mean()) if bias.notna().any() else None
        rec["max_abs_bias_V"] = float(bias.abs().max()) if bias.notna().any() else None
        grouped.append(rec)
    return pd.DataFrame(grouped, columns=expected_cols)


def build_candidate_comparison(run_df: pd.DataFrame, segment_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if run_df.empty:
        return pd.DataFrame(columns=["candidate", "n_profiles", "mean_MAE_V", "mean_corr", "delta_MAE_vs_baseline_V", "delta_corr_vs_baseline", "recommendation"]), {"recommended_candidate": None, "reason": "no run metrics"}
    global_by_candidate = aggregate_metrics(run_df, ["candidate"])
    # Identify baseline and transition-fade if present.
    candidates = list(global_by_candidate["candidate"].astype(str)) if not global_by_candidate.empty else []
    baseline_name = None
    for c in candidates:
        cl = c.lower()
        if "baseline" in cl or "d951" in cl or "d96" in cl:
            baseline_name = c
            break
    transition_name = None
    for c in candidates:
        if "transition" in c.lower() or "fade" in c.lower():
            transition_name = c
            break
    low_only_name = None
    for c in candidates:
        if "low_only" in c.lower():
            low_only_name = c
            break

    base_mae = None; base_corr = None
    if baseline_name:
        b = global_by_candidate[global_by_candidate["candidate"].astype(str) == baseline_name]
        if not b.empty:
            base_mae = safe_float(b.iloc[0].get("mean_MAE_V"))
            base_corr = safe_float(b.iloc[0].get("mean_corr"))
    rows = []
    for _, r in global_by_candidate.iterrows():
        cand = str(r.get("candidate"))
        mae = safe_float(r.get("mean_MAE_V"))
        corr = safe_float(r.get("mean_corr"))
        delta_mae = (mae - base_mae) if mae is not None and base_mae is not None else None
        delta_corr = (corr - base_corr) if corr is not None and base_corr is not None else None
        recommendation = "candidate"
        if baseline_name and cand == baseline_name:
            recommendation = "baseline_reference"
        if transition_name and cand == transition_name:
            recommendation = "recommended_voltage_wrapper_if_no_segment_regression"
        if low_only_name and cand == low_only_name:
            recommendation = "conservative_low_only_reference"
        rows.append({
            "candidate": cand,
            "n_profiles": r.get("n_profiles"),
            "n_rows": r.get("n_rows"),
            "mean_MAE_V": mae,
            "mean_corr": corr,
            "delta_MAE_vs_baseline_V": delta_mae,
            "delta_corr_vs_baseline": delta_corr,
            "recommendation": recommendation,
        })
    cmp_df = pd.DataFrame(rows)

    # Add segment low-target summary if available.
    if not segment_df.empty:
        sd = segment_df.copy()
        sd["segment_l"] = sd.get("segment", "").astype(str).str.lower()
        low = sd[sd["segment_l"].str.contains("low", na=False)]
        if not low.empty:
            low_by_cand = aggregate_metrics(low, ["candidate"])[["candidate", "mean_MAE_V", "mean_corr", "n_rows"]]
            low_by_cand = low_by_cand.rename(columns={"mean_MAE_V": "low_segment_mean_MAE_V", "mean_corr": "low_segment_mean_corr", "n_rows": "low_segment_n_rows"})
            cmp_df = cmp_df.merge(low_by_cand, on="candidate", how="left")

    rec = {
        "recommended_candidate": transition_name or low_only_name or baseline_name,
        "baseline_candidate": baseline_name,
        "transition_fade_candidate": transition_name,
        "low_only_candidate": low_only_name,
        "reason": "transition-fade is preferred when present because D12-S1K established the best voltage-wrapper candidate; low-only remains conservative reference.",
    }
    return cmp_df, rec


def build_outlier_policy(run_df: pd.DataFrame, segment_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "profile_pattern": "Batch-1_2C_battery-8 / Batch-1_battery-8 under 2C / B1_2C_battery-8",
            "policy": "flagged_excluded_from_mainline",
            "scope": "D14-P2 voltage generalization scorecard",
            "rationale": "D9.7/D10/D12 diagnose it as a late-2C discharge boundary/outlier; do not count it in 23-profile mainline success.",
            "mainline_inclusion": "no",
            "stress_test_allowed": "yes",
        },
        {
            "profile_pattern": "Batch-3_battery-8",
            "policy": "not_the_current_battery8_outlier_by_name_alone",
            "scope": "D14-P2 voltage generalization scorecard",
            "rationale": "Only Batch-1 / 2C / battery-8 is flagged. Do not match all battery-8 strings.",
            "mainline_inclusion": "yes_if_non_outlier_metrics_present",
            "stress_test_allowed": "not_required",
        },
        {
            "profile_pattern": "Batch-4_battery-8",
            "policy": "not_the_current_battery8_outlier_by_name_alone",
            "scope": "D14-P2 voltage generalization scorecard",
            "rationale": "Only Batch-1 / 2C / battery-8 is flagged. Do not match all battery-8 strings.",
            "mainline_inclusion": "yes_if_non_outlier_metrics_present",
            "stress_test_allowed": "not_required",
        },
    ]
    if run_df is not None and not run_df.empty:
        b1 = run_df[run_df.get("is_batch1_2c_battery8", False) == True]
        rows.append({
            "profile_pattern": "observed Batch-1_2C_battery-8 rows in parsed mainline files",
            "policy": "should_be_zero_in_mainline",
            "scope": "parsed scorecard rows",
            "rationale": f"Parsed {len(b1)} rows matching Batch-1/2C/battery-8.",
            "mainline_inclusion": "fail_if_nonzero",
            "stress_test_allowed": "yes",
        })
    return pd.DataFrame(rows)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def file_index(paths: List[Path]) -> List[dict]:
    rows = []
    for p in paths:
        rows.append({
            "name": p.name,
            "path": str(p),
            "exists": p.exists(),
            "size_bytes": p.stat().st_size if p.exists() else None,
            "sha256": sha256_file(p),
        })
    return rows


def make_markdown(report: Dict[str, Any], output_files: List[Path]) -> str:
    lines = []
    lines.append("# D14-P2 XJTU Generalization Scorecard Report")
    lines.append("")
    lines.append(f"Generated: `{report.get('generated_at')}`")
    lines.append(f"Overall status: **{report.get('overall_status')}**")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("D14-P2 consolidates existing XJTU voltage-replay evidence into auditable scorecards. It does not train a model, does not generate SOH labels, and does not turn XJTU voltage fitting into internal-state ground truth.")
    lines.append("")
    lines.append("## Prerequisites")
    lines.append("")
    for k in ["p0_status", "p1_status"]:
        lines.append(f"- {k}: `{report.get(k)}`")
    lines.append("")
    lines.append("## Parsed evidence")
    lines.append("")
    lines.append(f"- CSV files scanned: `{report.get('csv_files_scanned')}`")
    lines.append(f"- Run rows parsed: `{report.get('run_rows')}`")
    lines.append(f"- Segment rows parsed: `{report.get('segment_rows')}`")
    lines.append(f"- Batch-1 / 2C / battery-8 rows in mainline parse: `{report.get('batch1_2c_battery8_mainline_rows')}`")
    lines.append("")
    rec = report.get("candidate_recommendation") or {}
    lines.append("## Candidate recommendation")
    lines.append("")
    lines.append(f"- Recommended voltage-wrapper candidate: `{rec.get('recommended_candidate')}`")
    lines.append(f"- Baseline candidate: `{rec.get('baseline_candidate')}`")
    lines.append(f"- Reason: {rec.get('reason')}")
    lines.append("")
    lines.append("## Warnings / failures")
    lines.append("")
    fails = report.get("failures") or []
    warns = report.get("warnings") or []
    if not fails and not warns:
        lines.append("No warnings or failures.")
    else:
        for f in fails:
            lines.append(f"- **FAIL**: {f}")
        for w in warns:
            lines.append(f"- **WARN**: {w}")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for p in output_files:
        lines.append(f"- `{p.name}`")
    lines.append("")
    lines.append("## Interpretation boundary")
    lines.append("")
    lines.append("This scorecard supports non-outlier XJTU measured-current voltage replay / voltage surrogate generalization. It does not establish XJTU `cs_a/cs_c/phie/phis_c` as experimental internal-state truth. SOH must be read or computed from XJTU cycle/capacity data separately, not generated by the voltage soft-label generator.")
    lines.append("")
    return "\n".join(lines)

# ---------------- main ----------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build D14-P2 XJTU generalization scorecards from existing D10/D12/D13/P0/P1 evidence.")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--p0-dir", required=True)
    ap.add_argument("--p1-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--d10-p1-dir", default=None)
    ap.add_argument("--d12-s1k-200ks-dir", default=None)
    ap.add_argument("--d12-s1k-40ks-dir", default=None)
    ap.add_argument("--d13-segment-dir", default=None)
    ap.add_argument("--strict-evidence", action="store_true", help="Fail if required D10/D12 detailed scorecards cannot be parsed.")
    ap.add_argument("--allow-p0-p1-warn", action="store_true", help="Treat P0/P1 WARN as acceptable.")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root)
    cache_root = Path(args.cache_root)
    p0_dir = Path(args.p0_dir)
    p1_dir = Path(args.p1_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    p0, p1 = load_p0_p1(p0_dir, p1_dir)
    p0_status = str(p0.get("overall_status", "MISSING"))
    p1_status = str(p1.get("overall_status", "MISSING"))

    warnings: List[str] = []
    failures: List[str] = []
    if p0_status == "FAIL":
        failures.append("D14-P0 status is FAIL; freeze/no-regression must be fixed before P2.")
    elif p0_status not in {"PASS", "WARN"}:
        failures.append(f"D14-P0 status is not PASS/WARN: {p0_status}")
    elif p0_status == "WARN" and not args.allow_p0_p1_warn:
        warnings.append("D14-P0 is WARN. Pass --allow-p0-p1-warn if this is the accepted hard-clamp-default-off warning.")

    if p1_status == "FAIL":
        failures.append("D14-P1 status is FAIL; evidence-boundary audit must be fixed before P2.")
    elif p1_status not in {"PASS", "WARN"}:
        failures.append(f"D14-P1 status is not PASS/WARN: {p1_status}")
    elif p1_status == "WARN" and not args.allow_p0_p1_warn:
        warnings.append("D14-P1 is WARN. Pass --allow-p0-p1-warn if inherited from accepted P0 warning.")

    dirs = default_dirs(cache_root, args, p0_dir)
    evidence_dirs = [v for v in dirs.values() if v and v.exists()]
    for key, path in dirs.items():
        if not path.exists():
            msg = f"Evidence directory not found: {key}={path}"
            if args.strict_evidence and key in {"d10_p1_dir", "d12_s1k_200ks_dir"}:
                failures.append(msg)
            else:
                warnings.append(msg)

    csv_paths = discover_csv_files(evidence_dirs)
    run_rows, segment_rows, inventory_rows = collect_from_csvs(csv_paths)

    # Fallback from P0 scorecard index when detailed raw dirs are absent or not readable.
    fb_run, fb_seg, fb_inv = fallback_rows_from_p0_index(p0_dir)
    if not run_rows and fb_run:
        warnings.append("No detailed run metric CSV rows parsed; using aggregate fallback from D14_P0_SCORECARD_INDEX.json.")
        run_rows.extend(fb_run)
    if not segment_rows and fb_seg:
        warnings.append("No detailed segment metric CSV rows parsed; using aggregate fallback from D14_P0_SCORECARD_INDEX.json.")
        segment_rows.extend(fb_seg)
    inventory_rows.extend(fb_inv)

    run_df = pd.DataFrame(run_rows)
    segment_df = pd.DataFrame(segment_rows)
    inventory_df = pd.DataFrame(inventory_rows)

    if run_df.empty:
        if args.strict_evidence:
            failures.append("No run metrics parsed from detailed scorecards or P0 index fallback.")
        else:
            warnings.append("No run metrics parsed from detailed scorecards or P0 index fallback.")
    if segment_df.empty:
        warnings.append("No segment metrics parsed. D14-P2 can proceed, but segment-aware claims should remain limited.")

    # Mainline outlier check: only Batch-1 / 2C / battery-8 is excluded; all other battery-8 names are allowed.
    b1_rows = 0
    if not run_df.empty and "is_batch1_2c_battery8" in run_df.columns:
        b1_rows = int((run_df["is_batch1_2c_battery8"] == True).sum())
        if b1_rows > 0:
            failures.append(f"Parsed {b1_rows} Batch-1/2C/battery-8 rows in mainline scorecard rows; this outlier should be excluded.")

    # Build scorecards.
    global_df = aggregate_metrics(run_df, ["source_stage", "candidate", "time_window_s"])
    by_protocol_df = aggregate_metrics(run_df, ["candidate", "protocol"])
    by_cell_df = aggregate_metrics(run_df, ["candidate", "profile_id", "batch", "protocol", "battery_index"])
    by_segment_df = aggregate_metrics(segment_df, ["candidate", "segment"]) if not segment_df.empty else pd.DataFrame()
    by_protocol_segment_df = aggregate_metrics(segment_df, ["candidate", "protocol", "segment"]) if not segment_df.empty else pd.DataFrame()
    candidate_cmp_df, recommendation = build_candidate_comparison(run_df, segment_df)
    outlier_policy_df = build_outlier_policy(run_df, segment_df)

    # Write outputs.
    outputs: Dict[str, Path] = {
        "report_json": output_dir / "D14_P2_GENERALIZATION_SCORECARD_REPORT.json",
        "report_md": output_dir / "D14_P2_GENERALIZATION_SCORECARD_REPORT.md",
        "source_inventory": output_dir / "D14_P2_SOURCE_INVENTORY.csv",
        "run_metrics_normalized": output_dir / "D14_P2_RUN_METRICS_NORMALIZED.csv",
        "segment_metrics_normalized": output_dir / "D14_P2_SEGMENT_METRICS_NORMALIZED.csv",
        "global_scorecard": output_dir / "D14_P2_GLOBAL_SCORECARD.csv",
        "by_protocol": output_dir / "D14_P2_BY_PROTOCOL.csv",
        "by_cell": output_dir / "D14_P2_BY_CELL.csv",
        "by_segment": output_dir / "D14_P2_BY_SEGMENT.csv",
        "by_protocol_segment": output_dir / "D14_P2_BY_PROTOCOL_SEGMENT.csv",
        "candidate_comparison": output_dir / "D14_P2_CANDIDATE_COMPARISON.csv",
        "outlier_policy": output_dir / "D14_P2_OUTLIER_POLICY.csv",
        "readme_patch": output_dir / "README_D14_P2_PATCH.md",
        "output_index": output_dir / "D14_P2_OUTPUT_INDEX.json",
        "run_summary": output_dir / "D14_P2_RUN_SUMMARY.txt",
    }

    write_csv(outputs["source_inventory"], inventory_df)
    write_csv(outputs["run_metrics_normalized"], run_df)
    write_csv(outputs["segment_metrics_normalized"], segment_df)
    write_csv(outputs["global_scorecard"], global_df)
    write_csv(outputs["by_protocol"], by_protocol_df)
    write_csv(outputs["by_cell"], by_cell_df)
    write_csv(outputs["by_segment"], by_segment_df)
    write_csv(outputs["by_protocol_segment"], by_protocol_segment_df)
    write_csv(outputs["candidate_comparison"], candidate_cmp_df)
    write_csv(outputs["outlier_policy"], outlier_policy_df)

    status = "PASS"
    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "overall_status": status,
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "p0_dir": str(p0_dir),
        "p1_dir": str(p1_dir),
        "p0_status": p0_status,
        "p1_status": p1_status,
        "evidence_dirs": {k: str(v) for k, v in dirs.items()},
        "csv_files_scanned": len(csv_paths),
        "run_rows": int(len(run_df)),
        "segment_rows": int(len(segment_df)),
        "batch1_2c_battery8_mainline_rows": b1_rows,
        "warnings": warnings,
        "failures": failures,
        "candidate_recommendation": recommendation,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "row_counts": {
            "source_inventory": int(len(inventory_df)),
            "global_scorecard": int(len(global_df)),
            "by_protocol": int(len(by_protocol_df)),
            "by_cell": int(len(by_cell_df)),
            "by_segment": int(len(by_segment_df)),
            "candidate_comparison": int(len(candidate_cmp_df)),
        },
    }

    write_json(outputs["report_json"], report)
    md = make_markdown(report, list(outputs.values()))
    outputs["report_md"].write_text(md, encoding="utf-8")

    readme_patch = f"""# README D14-P2 Patch: XJTU Generalization Scorecard

D14-P2 consolidates D10-P1 and D12-S1K XJTU voltage replay evidence into an auditable multi-cell / multi-protocol / multi-segment scorecard.

Recommended wording:

- XJTU evidence supports non-outlier real public liquid-cell measured-current voltage replay / voltage-surrogate generalization.
- D12-S1K `low_plus_transition_fade_to_baseline` remains the recommended voltage-wrapper candidate when available.
- Batch-1 / 2C / battery-8 remains flagged/excluded from mainline and should only be used as a stress-test.
- Batch-3/Batch-4 battery-8 strings are not the flagged outlier by name alone.
- XJTU voltage soft-label generator does not generate SOH labels. XJTU SOH must be read or computed from official cycle/capacity data.
- XJTU voltage success does not by itself prove `cs_a/cs_c/phie/phis_c` as experimental internal-state truth.

D14-P2 status from this run: `{status}`.
"""
    outputs["readme_patch"].write_text(readme_patch, encoding="utf-8")

    idx = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "overall_status": status,
        "files": file_index(list(outputs.values())),
    }
    write_json(outputs["output_index"], idx)

    summary_lines = [
        f"schema_version={SCHEMA_VERSION}",
        f"generated_at={report['generated_at']}",
        f"overall_status={status}",
        f"p0_status={p0_status}",
        f"p1_status={p1_status}",
        f"csv_files_scanned={len(csv_paths)}",
        f"run_rows={len(run_df)}",
        f"segment_rows={len(segment_df)}",
        f"batch1_2c_battery8_mainline_rows={b1_rows}",
        f"recommended_candidate={recommendation.get('recommended_candidate')}",
        f"warnings={len(warnings)}",
        f"failures={len(failures)}",
    ]
    outputs["run_summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Recompute index after run_summary exists.
    idx["files"] = file_index(list(outputs.values()))
    write_json(outputs["output_index"], idx)

    print(f"[D14-P2] overall_status={status}")
    print(f"[D14-P2] output_dir={output_dir}")
    if warnings:
        print("[D14-P2] warnings:")
        for w in warnings:
            print(f"  - {w}")
    if failures:
        print("[D14-P2] failures:")
        for f in failures:
            print(f"  - {f}")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
