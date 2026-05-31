#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute GV1 voltage metrics from a D9/D10 prediction.npz.

The D9.5.1 trainer writes prediction.npz with `voltage_exp` and
`voltage_exp_pred`. This script scores global and regime-specific errors in a
format that can be aggregated by D10-P1 scorecards.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    a = x[mask].astype(float)
    b = y[mask].astype(float)
    if float(np.nanstd(a)) <= 1e-12 or float(np.nanstd(b)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(name: str, y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = mask & np.isfinite(y) & np.isfinite(p)
    n = int(mask.sum())
    if n == 0:
        return {"label": name, "n": 0}
    err = p[mask] - y[mask]
    return {
        "label": name,
        "n": n,
        "mae_V": float(np.nanmean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.nanmean(err ** 2))),
        "bias_V": float(np.nanmean(err)),
        "corr": _corr(y[mask], p[mask]),
        "target_min_V": float(np.nanmin(y[mask])),
        "target_max_V": float(np.nanmax(y[mask])),
        "pred_min_V": float(np.nanmin(p[mask])),
        "pred_max_V": float(np.nanmax(p[mask])),
        "pred_upper_frac_ge_4p269": float(np.nanmean(p[mask] >= 4.269)),
        "pred_overshoot_frac_gt_4p35": float(np.nanmean(p[mask] > 4.35)),
        "pred_low_frac_le_2p75": float(np.nanmean(p[mask] <= 2.75)),
        "target_low_frac_le_2p75": float(np.nanmean(y[mask] <= 2.75)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_npz", required=True)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--output_csv", default=None)
    ap.add_argument("--current_eps_A", type=float, default=1e-8)
    ap.add_argument("--low_voltage_threshold_V", type=float, default=2.75)
    ap.add_argument("--high_voltage_threshold_V", type=float, default=4.10)
    ap.add_argument("--high_current_quantile", type=float, default=0.90)
    args = ap.parse_args()

    pred_path = Path(args.prediction_npz)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    with np.load(pred_path, allow_pickle=True) as z:
        keys = set(z.files)
        y = np.asarray(z["voltage_exp"], dtype=float).reshape(-1)
        if "voltage_exp_pred" in keys:
            p = np.asarray(z["voltage_exp_pred"], dtype=float).reshape(-1)
        elif "phis_c_pred" in keys:
            p = np.asarray(z["phis_c_pred"], dtype=float).reshape(-1)
        else:
            raise KeyError("prediction.npz must contain voltage_exp_pred or phis_c_pred")
        I = np.asarray(z["I_profile"], dtype=float).reshape(-1) if "I_profile" in keys else np.zeros_like(y)
        t = np.asarray(z["t_global_s"], dtype=float).reshape(-1) if "t_global_s" in keys else np.arange(len(y), dtype=float)

    n = min(len(y), len(p), len(I), len(t))
    y, p, I, t = y[:n], p[:n], I[:n], t[:n]
    finite = np.isfinite(y) & np.isfinite(p)
    absI = np.abs(I[np.isfinite(I)])
    i_thr = float(np.nanquantile(absI, args.high_current_quantile)) if len(absI) else 0.0
    masks = {
        "all": np.ones(n, dtype=bool),
        "charge_I_pos": I > float(args.current_eps_A),
        "discharge_I_neg": I < -float(args.current_eps_A),
        "rest_I_zero": np.abs(I) <= float(args.current_eps_A),
        "low_target": y <= float(args.low_voltage_threshold_V),
        "high_target": y >= float(args.high_voltage_threshold_V),
        "mid_target": (y > float(args.low_voltage_threshold_V)) & (y < float(args.high_voltage_threshold_V)),
        "high_current_abs": np.abs(I) >= max(i_thr, float(args.current_eps_A)),
    }
    rows = [_metrics(name, y, p, mask & finite) for name, mask in masks.items()]
    summary = {
        "ok": True,
        "prediction_npz": str(pred_path),
        "n": int(n),
        "current_high_quantile_threshold_A": i_thr,
        "metrics": {row["label"]: row for row in rows},
    }
    out_json = Path(args.output_json) if args.output_json else pred_path.with_name("d10_voltage_metrics.json")
    out_csv = Path(args.output_csv) if args.output_csv else pred_path.with_name("d10_voltage_metrics_by_segment.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, rows)
    print(json.dumps({"ok": True, "output_json": str(out_json), "output_csv": str(out_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
