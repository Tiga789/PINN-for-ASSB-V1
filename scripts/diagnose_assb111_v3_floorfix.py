# -*- coding: utf-8 -*-
"""Diagnose ASSB-111 saturating_v3_floorfix initializer and SOH outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd


def _load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        y = float(x)
        return None if not np.isfinite(y) else y
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    return x


def _metrics(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(obs) & np.isfinite(pred)
    out = {"n": int(m.sum())}
    if m.sum() == 0:
        return out
    o = obs[m]
    p = pred[m]
    err = p - o
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    out.update({
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "BIAS": float(np.mean(err)),
        "R2": float("nan") if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot,
        "corr": float("nan") if o.size < 2 or np.std(o) <= 1e-15 or np.std(p) <= 1e-15 else float(np.corrcoef(o, p)[0, 1]),
        "SOH_obs_min": float(np.min(o)),
        "SOH_obs_max": float(np.max(o)),
        "SOH_pred_min": float(np.min(p)),
        "SOH_pred_max": float(np.max(p)),
    })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initializer_json", required=True)
    ap.add_argument("--pred_csv", default="")
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--expected_floor", type=float, default=0.72)
    ap.add_argument("--floor_tol", type=float, default=0.005)
    ap.add_argument("--numeric_min", type=float, default=0.60)
    args = ap.parse_args(argv)

    init_payload = _load_json(args.initializer_json)
    initializer = init_payload.get("initializer", {}) if isinstance(init_payload, dict) else {}
    best = init_payload.get("best", {}) if isinstance(init_payload, dict) else {}
    floor = float(initializer.get("soh_floor", best.get("floor", np.nan)))
    mode = str(initializer.get("floor_selection_mode", init_payload.get("floor_selection_mode", "")))
    report: Dict[str, Any] = {
        "initializer_json": str(args.initializer_json),
        "floor_selection_mode": mode,
        "floor": floor,
        "expected_floor": float(args.expected_floor),
        "floor_abs_error": None if not np.isfinite(floor) else abs(floor - float(args.expected_floor)),
        "floor_ok": bool(np.isfinite(floor) and abs(floor - float(args.expected_floor)) <= float(args.floor_tol)),
        "k_per_cycle": initializer.get("k_per_cycle", best.get("k_per_cycle")),
        "soh0": initializer.get("soh0", best.get("soh0")),
        "selected_by": init_payload.get("selected_by"),
        "leakage_statement": init_payload.get("leakage_statement"),
    }

    if args.pred_csv:
        df = pd.read_csv(args.pred_csv)
        report["pred_csv"] = str(args.pred_csv)
        if "active_clamp_mask" in df.columns:
            active = pd.to_numeric(df["active_clamp_mask"], errors="coerce").fillna(0).to_numpy(dtype=float) != 0
            report["active_clamp_count_all"] = int(active.sum())
        if "SOH_pred" in df.columns:
            pred = pd.to_numeric(df["SOH_pred"], errors="coerce").to_numpy(dtype=float)
            report["SOH_pred_min_all"] = float(np.nanmin(pred))
            report["SOH_pred_max_all"] = float(np.nanmax(pred))
            report["near_numeric_min_count_all"] = int(np.sum(np.isfinite(pred) & (pred <= float(args.numeric_min) + 1e-8)))
        if {"split", "SOH_obs", "SOH_pred"}.issubset(df.columns):
            split_metrics = {}
            for split, g in df.groupby(df["split"].astype(str).str.lower()):
                split_metrics[str(split)] = _metrics(g["SOH_obs"].to_numpy(dtype=float), g["SOH_pred"].to_numpy(dtype=float))
            report["metrics_by_split"] = split_metrics

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(_clean(report), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(_clean(report), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
