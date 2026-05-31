#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect D10-P1 profile metrics into a scorecard."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _to_float(x: Any, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _classify(row: dict[str, Any]) -> str:
    mae = _to_float(row.get("mae_V"))
    corr = _to_float(row.get("corr"))
    upper = _to_float(row.get("pred_upper_frac_ge_4p269"), 0.0)
    overshoot = _to_float(row.get("pred_overshoot_frac_gt_4p35"), 0.0)
    # Conservative D9-style thresholds for medium-window voltage replay.
    if np.isfinite(mae) and np.isfinite(corr) and mae <= 0.10 and corr >= 0.90 and upper <= 0.08 and overshoot <= 0.02:
        return "pass"
    if np.isfinite(mae) and np.isfinite(corr) and mae <= 0.14 and corr >= 0.85 and overshoot <= 0.05:
        return "borderline"
    return "fail"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir) if args.out_dir else runs_root
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.rglob("d10_voltage_metrics.json")):
        try:
            data = _read_json(metrics_path)
            all_metrics = data.get("metrics", {}).get("all", {})
            row = dict(all_metrics)
            row["run_dir"] = str(metrics_path.parent)
            row["profile_id"] = metrics_path.parent.name
            row["status"] = _classify(row)
            rows.append(row)
        except Exception as exc:
            rows.append({"profile_id": metrics_path.parent.name, "run_dir": str(metrics_path.parent), "status": "read_error", "error": str(exc)})
    counts = {"pass": 0, "borderline": 0, "fail": 0, "read_error": 0}
    for row in rows:
        counts[str(row.get("status", "fail"))] = counts.get(str(row.get("status", "fail")), 0) + 1
    mae_vals = [_to_float(r.get("mae_V")) for r in rows if np.isfinite(_to_float(r.get("mae_V")))]
    rmse_vals = [_to_float(r.get("rmse_V")) for r in rows if np.isfinite(_to_float(r.get("rmse_V")))]
    corr_vals = [_to_float(r.get("corr")) for r in rows if np.isfinite(_to_float(r.get("corr")))]
    summary = {
        "ok": True,
        "stage": "D10-P1 23-profile 200ks scorecard",
        "runs_root": str(runs_root),
        "profile_count": len(rows),
        "counts": counts,
        "mean_mae_V": float(np.nanmean(mae_vals)) if mae_vals else math.nan,
        "mean_rmse_V": float(np.nanmean(rmse_vals)) if rmse_vals else math.nan,
        "mean_corr": float(np.nanmean(corr_vals)) if corr_vals else math.nan,
        "status": "pass" if counts.get("fail", 0) == 0 and counts.get("read_error", 0) == 0 else "review_needed",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "scorecard_d10_p1_23profile_200ks.csv", rows)
    (out_dir / "scorecard_d10_p1_23profile_200ks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
