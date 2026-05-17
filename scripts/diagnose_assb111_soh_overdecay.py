# -*- coding: utf-8 -*-
"""Diagnose ASSB-111 SOH over-decay and numeric clamp behavior.

This script is deliberately evaluation-only. It reads ``soh_pred_by_cycle.csv``
written by ``evaluate_assb111_five_state.py`` and summarizes whether the SOH
head is still over-decaying in the held-out cycles.

Typical use
-----------
python scripts/diagnose_assb111_soh_overdecay.py \
  --pred_csv EvalFin_111_smoke/soh_pred_by_cycle.csv \
  --output_dir EvalFin_111_smoke
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


def _as_float_array(values: Sequence[Any]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)


def _metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(obs) & np.isfinite(pred)
    out: Dict[str, Any] = {
        "n": int(mask.sum()),
        "MAE": float("nan"),
        "RMSE": float("nan"),
        "BIAS": float("nan"),
        "MAX_ABS": float("nan"),
        "R2": float("nan"),
        "corr": float("nan"),
        "NMAE": float("nan"),
        "NRMSE": float("nan"),
    }
    if not mask.any():
        return out
    y = obs[mask]
    p = pred[mask]
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    out.update(MAE=mae, RMSE=rmse, BIAS=float(np.mean(e)), MAX_ABS=float(np.max(np.abs(e))))
    denom = float(np.nanmax(y) - np.nanmin(y))
    if denom > 1e-30:
        out["NMAE"] = float(mae / denom)
        out["NRMSE"] = float(rmse / denom)
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot > 1e-30:
        out["R2"] = float(1.0 - np.sum(e * e) / ss_tot)
    if y.size >= 2 and float(np.std(y)) > 1e-30 and float(np.std(p)) > 1e-30:
        out["corr"] = float(np.corrcoef(y, p)[0, 1])
    return out


def _linear_slope(cycle: np.ndarray, value: np.ndarray) -> float:
    cycle = np.asarray(cycle, dtype=np.float64).reshape(-1)
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    mask = np.isfinite(cycle) & np.isfinite(value)
    if mask.sum() < 2:
        return float("nan")
    x = cycle[mask]
    y = value[mask]
    x = x - float(np.mean(x))
    den = float(np.sum(x * x))
    if den <= 1e-30:
        return float("nan")
    return float(np.sum(x * (y - float(np.mean(y)))) / den)


def _parse_segments(text: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            a = b = int(part)
        else:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
        if b < a:
            a, b = b, a
        out.append((a, b))
    return out


def _bool_series(values: Sequence[Any]) -> np.ndarray:
    s = pd.Series(values)
    if s.dtype == bool:
        return s.to_numpy(dtype=bool)
    low = s.astype(str).str.strip().str.lower()
    return low.isin({"1", "true", "yes", "y", "t"}).to_numpy(dtype=bool)


def _infer_clamp_mask(df: pd.DataFrame, pred_col: str, *, active_col: str, clamp_value: Optional[float], clamp_tol: float, numeric_min: Optional[float]) -> Tuple[np.ndarray, Dict[str, Any]]:
    pred = _as_float_array(df[pred_col])
    info: Dict[str, Any] = {"method": ""}
    if active_col in df.columns:
        mask = _bool_series(df[active_col])
        info.update(method=f"explicit column {active_col}", explicit_col=active_col)
        return mask, info

    candidates: List[Tuple[str, float]] = []
    if clamp_value is not None:
        candidates.append(("user_clamp_value", float(clamp_value)))
    # Old failed ASSB-111 smoke commonly hit 0.4. New saturating_v2 should not.
    candidates.append(("legacy_0p4", 0.4))
    if numeric_min is not None:
        candidates.append(("numeric_min", float(numeric_min)))

    best_name = "none"
    best_value = float("nan")
    best_count = 0
    best_mask = np.zeros_like(pred, dtype=bool)
    for name, value in candidates:
        mask = np.isfinite(pred) & (pred <= value + float(clamp_tol))
        count = int(mask.sum())
        if count > best_count:
            best_name, best_value, best_count, best_mask = name, value, count, mask
    info.update(method=best_name, clamp_value=best_value if math.isfinite(best_value) else None, count=best_count)
    return best_mask, info


def _split_summary(df: pd.DataFrame, *, cycle_col: str, split_col: str, pred_col: str, obs_col: str, clamp_mask: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in sorted(str(x) for x in df[split_col].dropna().unique()):
        m = df[split_col].astype(str).to_numpy() == split
        cyc = _as_float_array(df.loc[m, cycle_col])
        pred = _as_float_array(df.loc[m, pred_col])
        obs = _as_float_array(df.loc[m, obs_col]) if obs_col in df.columns else np.full_like(pred, np.nan)
        cm = clamp_mask[m]
        diffs = np.diff(pred[np.isfinite(pred)])
        out[split] = {
            "n": int(m.sum()),
            "cycle_min": int(np.nanmin(cyc)) if cyc.size and np.isfinite(cyc).any() else None,
            "cycle_max": int(np.nanmax(cyc)) if cyc.size and np.isfinite(cyc).any() else None,
            "SOH_obs_min": float(np.nanmin(obs)) if np.isfinite(obs).any() else None,
            "SOH_obs_max": float(np.nanmax(obs)) if np.isfinite(obs).any() else None,
            "SOH_pred_min": float(np.nanmin(pred)) if np.isfinite(pred).any() else None,
            "SOH_pred_max": float(np.nanmax(pred)) if np.isfinite(pred).any() else None,
            "SOH_pred_linear_slope_per_cycle": _linear_slope(cyc, pred),
            "SOH_pred_step_min": float(np.nanmin(diffs)) if diffs.size else None,
            "SOH_pred_step_median": float(np.nanmedian(diffs)) if diffs.size else None,
            "SOH_pred_step_max": float(np.nanmax(diffs)) if diffs.size else None,
            "active_clamp_count": int(cm.sum()),
            "active_clamp_fraction": float(cm.mean()) if cm.size else 0.0,
            "metrics": _metrics(obs, pred) if np.isfinite(obs).any() else {},
        }
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose ASSB111 SOH over-decay in soh_pred_by_cycle.csv")
    p.add_argument("--pred_csv", default=r"EvalFin_111_saturating_v2_strict30_test70\soh_pred_by_cycle.csv")
    p.add_argument("--output_dir", default="", help="Defaults to parent directory of --pred_csv")
    p.add_argument("--output_json", default="")
    p.add_argument("--output_csv", default="")
    p.add_argument("--cycle_col", default="cycle_id")
    p.add_argument("--split_col", default="split")
    p.add_argument("--obs_col", default="SOH_obs")
    p.add_argument("--pred_col", default="SOH_pred")
    p.add_argument("--active_clamp_col", default="active_clamp_mask")
    p.add_argument("--clamp_value", type=float, default=None, help="Optional explicit clamp value to count, e.g. 0.4 or 0.6")
    p.add_argument("--numeric_min", type=float, default=0.60, help="Expected saturating_v2 numeric lower bound; diagnostic only")
    p.add_argument("--clamp_tol", type=float, default=1e-9)
    p.add_argument("--tail_n", type=int, default=20)
    p.add_argument("--segments", default="160-250,251-400,401-521")
    p.add_argument("--fail_on_active_clamp", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pred_csv = Path(args.pred_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)
    out_dir = Path(args.output_dir) if args.output_dir else pred_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else out_dir / "soh_overdecay_diagnostic.json"
    output_csv = Path(args.output_csv) if args.output_csv else out_dir / "soh_overdecay_diagnostic_segments.csv"

    df = pd.read_csv(pred_csv)
    required = [args.cycle_col, args.split_col, args.pred_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {pred_csv}: {missing}")
    df = df.copy()
    df[args.cycle_col] = pd.to_numeric(df[args.cycle_col], errors="coerce").astype("Int64")
    df = df.sort_values([args.cycle_col]).reset_index(drop=True)

    clamp_mask, clamp_info = _infer_clamp_mask(
        df,
        args.pred_col,
        active_col=args.active_clamp_col,
        clamp_value=args.clamp_value,
        clamp_tol=float(args.clamp_tol),
        numeric_min=args.numeric_min,
    )
    split_summary = _split_summary(df, cycle_col=args.cycle_col, split_col=args.split_col, pred_col=args.pred_col, obs_col=args.obs_col, clamp_mask=clamp_mask)

    segment_rows: List[Dict[str, Any]] = []
    for a, b in _parse_segments(args.segments):
        m = (pd.to_numeric(df[args.cycle_col], errors="coerce") >= a) & (pd.to_numeric(df[args.cycle_col], errors="coerce") <= b)
        sub = df.loc[m].copy()
        pred = _as_float_array(sub[args.pred_col]) if len(sub) else np.asarray([], dtype=float)
        obs = _as_float_array(sub[args.obs_col]) if args.obs_col in sub.columns and len(sub) else np.asarray([], dtype=float)
        row: Dict[str, Any] = {
            "segment": f"{a}-{b}",
            "cycle_min": a,
            "cycle_max": b,
            "n": int(len(sub)),
            "SOH_pred_min": float(np.nanmin(pred)) if pred.size and np.isfinite(pred).any() else None,
            "SOH_pred_max": float(np.nanmax(pred)) if pred.size and np.isfinite(pred).any() else None,
            "SOH_pred_slope_per_cycle": _linear_slope(_as_float_array(sub[args.cycle_col]) if len(sub) else [], pred),
            "active_clamp_count": int(clamp_mask[m.to_numpy()].sum()) if len(df) else 0,
        }
        if obs.size and np.isfinite(obs).any():
            row.update({f"metric_{k}": v for k, v in _metrics(obs, pred).items()})
        segment_rows.append(row)
    pd.DataFrame(segment_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    test_df = df[df[args.split_col].astype(str) == "test"].copy()
    tail = test_df.tail(max(0, int(args.tail_n)))
    tail_cols = [c for c in [args.cycle_col, args.split_col, args.obs_col, args.pred_col, "SOH_struct", "SOH_base", "SOH_pred_unclipped", "remaining_degradable", "damage_rate_gated"] if c in tail.columns]

    diagnostic: Dict[str, Any] = {
        "pred_csv": str(pred_csv),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "clamp_detection": clamp_info,
        "active_clamp_count_all": int(clamp_mask.sum()),
        "active_clamp_fraction_all": float(clamp_mask.mean()) if clamp_mask.size else 0.0,
        "split_summary": split_summary,
        "segments_csv": str(output_csv),
        "segments": segment_rows,
        "test_tail_n": int(len(tail)),
        "test_tail_rows": tail[tail_cols].to_dict(orient="records") if tail_cols else [],
        "pass_flags": {
            "no_active_clamp_all": int(clamp_mask.sum()) == 0,
            "no_active_clamp_test": int(clamp_mask[df[args.split_col].astype(str).to_numpy() == "test"].sum()) == 0 if args.split_col in df.columns else None,
        },
    }
    _write_json(output_json, diagnostic)
    print(json.dumps(_json_clean({
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "active_clamp_count_all": diagnostic["active_clamp_count_all"],
        "test_summary": split_summary.get("test", {}),
        "pass_flags": diagnostic["pass_flags"],
    }), ensure_ascii=False, indent=2))
    if args.fail_on_active_clamp and int(clamp_mask.sum()) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
