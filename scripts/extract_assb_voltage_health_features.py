# -*- coding: utf-8 -*-
"""Extract cycle-level voltage health features for ASSB strict30 SOH work.

The default output contains only online current-cycle voltage/current features.
It does not use observed SOH or discharge-capacity labels.  When merged with an
existing ASSB111 dataset, the script keeps existing labels/splits intact and
adds G2/G3 feature columns for D7 feature audit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

try:
    from util.assb_soh_feature_schema import (
        G2_VOLTAGE,
        G3_SWITCH_POLARIZATION,
        audit_feature_frame,
        write_schema_json,
        _json_clean,  # type: ignore
    )
except Exception:  # pragma: no cover
    from assb_soh_feature_schema import G2_VOLTAGE, G3_SWITCH_POLARIZATION, audit_feature_frame, write_schema_json, _json_clean  # type: ignore


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract ASSB voltage health features by cycle")
    p.add_argument("--record_csv", required=True, help="Path to record_extracted.csv")
    p.add_argument("--base_dataset_csv", default="", help="Optional ASSB111 dataset.csv to merge on cycle_id")
    p.add_argument("--output_csv", default=r"Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv")
    p.add_argument("--output_dir", default=r"Data\assb112_feature_audit_v1")
    p.add_argument("--cycle_from", type=int, default=5)
    p.add_argument("--cycle_to", type=int, default=522)
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--allow_capacity_columns_in_output", action="store_true", help="Diagnostic only; default drops raw capacity columns")
    return p.parse_args(argv)


def _save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


def _find_col(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols:
            return str(cols[key])
    # fallback fuzzy contains
    for cand in candidates:
        low = str(cand).strip().lower()
        for c in df.columns:
            if low in str(c).strip().lower():
                return str(c)
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}; available={list(df.columns)}")
    return None


def _time_to_seconds_one(x: Any) -> float:
    if pd.isna(x):
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if not s:
        return float("nan")
    # HH:MM:SS or DD HH:MM:SS-like.  Pandas timedelta handles most cases.
    try:
        td = pd.to_timedelta(s)
        return float(td.total_seconds())
    except Exception:
        pass
    parts = s.split(":")
    try:
        nums = [float(p) for p in parts]
        if len(nums) == 3:
            return nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
        if len(nums) == 2:
            return nums[0] * 60.0 + nums[1]
        if len(nums) == 1:
            return nums[0]
    except Exception:
        return float("nan")
    return float("nan")


def _time_to_seconds(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray([_time_to_seconds_one(v) for v in values], dtype=float)
    # If a cycle crosses 24h and instrument resets string, enforce monotonic by
    # carrying forward increments where possible.
    if arr.size > 1 and np.any(np.diff(arr[np.isfinite(arr)]) < 0):
        fixed = arr.copy()
        offset = 0.0
        prev = fixed[0]
        for i in range(1, len(fixed)):
            if not np.isfinite(fixed[i]) or not np.isfinite(prev):
                prev = fixed[i]
                continue
            val = fixed[i] + offset
            if val < prev:
                offset += 24.0 * 3600.0
                val = fixed[i] + offset
            fixed[i] = val
            prev = val
        arr = fixed
    return arr


def _safe_num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else float("nan")


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    if int(np.sum(m)) < 3:
        return float("nan")
    tt = t[m] - np.nanmean(t[m])
    yy = y[m] - np.nanmean(y[m])
    denom = float(np.sum(tt * tt))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(tt * yy) / denom)


def _interp_by_tfrac(t: np.ndarray, v: np.ndarray, frac: float) -> float:
    m = np.isfinite(t) & np.isfinite(v)
    if int(np.sum(m)) == 0:
        return float("nan")
    tt = t[m]
    vv = v[m]
    if tt.size == 1 or float(np.nanmax(tt) - np.nanmin(tt)) <= 1e-12:
        return float(vv[0])
    order = np.argsort(tt)
    tt = tt[order]
    vv = vv[order]
    target = float(np.nanmin(tt) + frac * (np.nanmax(tt) - np.nanmin(tt)))
    return float(np.interp(target, tt, vv))


def _transition_features(i: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    i = np.asarray(i, dtype=float)
    v = np.asarray(v, dtype=float)
    if len(i) < 3:
        return {
            "step_voltage_jump_abs_mean": float("nan"),
            "step_voltage_jump_signed_mean": float("nan"),
            "r_step_proxy_abs_mean": float("nan"),
        }
    di = np.diff(i)
    dv = np.diff(v)
    m = np.isfinite(di) & np.isfinite(dv) & (np.abs(di) > 1e-8)
    if not np.any(m):
        return {
            "step_voltage_jump_abs_mean": 0.0,
            "step_voltage_jump_signed_mean": 0.0,
            "r_step_proxy_abs_mean": 0.0,
        }
    jumps = dv[m]
    return {
        "step_voltage_jump_abs_mean": float(np.mean(np.abs(jumps))),
        "step_voltage_jump_signed_mean": float(np.mean(jumps)),
        "r_step_proxy_abs_mean": float(np.mean(np.abs(jumps / di[m]))),
    }


def _rest_recovery_features(i: np.ndarray, v: np.ndarray, step_type: Sequence[Any]) -> Dict[str, float]:
    i = np.asarray(i, dtype=float)
    v = np.asarray(v, dtype=float)
    step = np.asarray([str(x) for x in step_type])
    rest = (np.abs(i) < 1e-10) | np.char.find(step.astype(str), "搁置") >= 0
    vals: List[float] = []
    # contiguous rest runs
    idx = np.where(rest)[0]
    if idx.size == 0:
        return {"rest_voltage_recovery_mean": 0.0, "rest_voltage_recovery_abs_mean": 0.0}
    starts = [idx[0]]
    ends: List[int] = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            ends.append(a)
            starts.append(b)
    ends.append(idx[-1])
    for s, e in zip(starts, ends):
        if e > s and np.isfinite(v[s]) and np.isfinite(v[e]):
            vals.append(float(v[e] - v[s]))
    if not vals:
        return {"rest_voltage_recovery_mean": 0.0, "rest_voltage_recovery_abs_mean": 0.0}
    arr = np.asarray(vals, dtype=float)
    return {"rest_voltage_recovery_mean": float(np.mean(arr)), "rest_voltage_recovery_abs_mean": float(np.mean(np.abs(arr)))}


def extract_features(record: pd.DataFrame, cycle_from: int, cycle_to: int) -> pd.DataFrame:
    cycle_col = _find_col(record, ["cycle_id", "循环号"])
    step_col = _find_col(record, ["step_id", "工步号"], required=False)
    type_col = _find_col(record, ["step_type", "工步类型"], required=False)
    time_col = _find_col(record, ["t_global_s", "总时间", "时间"])
    current_col = _find_col(record, ["I_profile", "电流(A)", "current"])
    voltage_col = _find_col(record, ["voltage_exp", "电压(V)", "voltage"])

    df = record.copy()
    df["cycle_id"] = pd.to_numeric(df[cycle_col], errors="coerce").astype("Int64")
    df = df[(df["cycle_id"] >= int(cycle_from)) & (df["cycle_id"] <= int(cycle_to))].copy()
    if df.empty:
        raise RuntimeError("No rows selected by cycle_from/cycle_to")

    rows: List[Dict[str, Any]] = []
    for cid, g in df.groupby("cycle_id", sort=True):
        cycle = int(cid)
        t = _time_to_seconds(g[time_col].tolist())
        if np.isfinite(t).any():
            t = t - float(np.nanmin(t))
        i = _safe_num(g[current_col])
        v = _safe_num(g[voltage_col])
        stype = g[type_col].tolist() if type_col and type_col in g.columns else [""] * len(g)
        charge = i > 1e-10
        discharge = i < -1e-10
        row: Dict[str, Any] = {
            "cycle_id": cycle,
            "voltage_start": float(v[np.where(np.isfinite(v))[0][0]]) if np.isfinite(v).any() else float("nan"),
            "voltage_end": float(v[np.where(np.isfinite(v))[0][-1]]) if np.isfinite(v).any() else float("nan"),
            "voltage_mean": _nanmean(v),
            "voltage_std": _nanstd(v),
            "voltage_min": float(np.nanmin(v)) if np.isfinite(v).any() else float("nan"),
            "voltage_max": float(np.nanmax(v)) if np.isfinite(v).any() else float("nan"),
            "charge_voltage_mean": _nanmean(v[charge]),
            "discharge_voltage_mean": _nanmean(v[discharge]),
            "charge_voltage_slope": _slope(t[charge], v[charge]),
            "discharge_voltage_slope": _slope(t[discharge], v[discharge]),
            "v_at_tfrac_010": _interp_by_tfrac(t, v, 0.10),
            "v_at_tfrac_025": _interp_by_tfrac(t, v, 0.25),
            "v_at_tfrac_050": _interp_by_tfrac(t, v, 0.50),
            "v_at_tfrac_075": _interp_by_tfrac(t, v, 0.75),
            "v_at_tfrac_090": _interp_by_tfrac(t, v, 0.90),
        }
        row.update(_transition_features(i, v))
        row.update(_rest_recovery_features(i, v, stype))
        cv = row.get("charge_voltage_mean", float("nan"))
        dv = row.get("discharge_voltage_mean", float("nan"))
        row["charge_discharge_voltage_gap"] = float(cv - dv) if np.isfinite(cv) and np.isfinite(dv) else float("nan")
        row["voltage_efficiency_proxy"] = float(dv / cv) if np.isfinite(cv) and abs(cv) > 1e-12 and np.isfinite(dv) else float("nan")
        # Extra provenance columns for audit only, not selected by strict schema.
        row["n_points_record"] = int(len(g))
        row["n_charge_points"] = int(np.sum(charge))
        row["n_discharge_points"] = int(np.sum(discharge))
        row["n_rest_points"] = int(np.sum(np.abs(i) < 1e-10))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cycle_id").reset_index(drop=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = pd.read_csv(args.record_csv)
    features = extract_features(record, args.cycle_from, args.cycle_to)
    feature_cols = [s.name for s in (G2_VOLTAGE + G3_SWITCH_POLARIZATION)]
    audit = audit_feature_frame(features, feature_cols, allow_upper_bound=False)

    if args.base_dataset_csv:
        base = pd.read_csv(args.base_dataset_csv)
        if "cycle_id" not in base.columns:
            raise KeyError("base_dataset_csv must contain cycle_id")
        # Drop any previous G2/G3 columns before merge to keep reruns deterministic.
        drop_cols = [c for c in feature_cols if c in base.columns]
        if drop_cols:
            base = base.drop(columns=drop_cols)
        merged = base.merge(features[["cycle_id"] + feature_cols], on="cycle_id", how="left", validate="one_to_one")
    else:
        merged = features

    if not args.allow_capacity_columns_in_output:
        # The record has raw capacity columns; make sure we did not accidentally
        # preserve them when no base dataset is used.
        bad_tokens = ["容量", "capacity", "q_discharge", "q_charge"]
        keep = [c for c in merged.columns if not any(tok.lower() in str(c).lower() for tok in bad_tokens)]
        if "cycle_id" not in keep:
            keep.insert(0, "cycle_id")
        merged = merged.loc[:, sorted(set(keep), key=keep.index)]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    features.to_csv(out_dir / "voltage_health_features_by_cycle.csv", index=False, encoding="utf-8-sig")
    write_schema_json(out_dir / "voltage_health_feature_schema.json", "g4_all_strict")
    _save_json(
        {
            "ok": bool(audit["ok"]),
            "record_csv": str(args.record_csv),
            "base_dataset_csv": str(args.base_dataset_csv),
            "output_csv": str(output_csv),
            "cycle_from": int(args.cycle_from),
            "cycle_to": int(args.cycle_to),
            "n_cycles": int(len(features)),
            "feature_columns_added": feature_cols,
            "feature_audit": audit,
            "strict_note": "No SOH/capacity labels are used in extracted G2/G3 features.",
        },
        out_dir / "voltage_feature_extraction_audit.json",
    )
    print(f"[OK] wrote {output_csv}")
    print(f"[OK] added {len(feature_cols)} voltage-health features for {len(features)} cycles")
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
