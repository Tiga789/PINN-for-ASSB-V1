# -*- coding: utf-8 -*-
"""Aggregate ASSB-111 saturating_v3 multi-seed strict30 evaluation results.

This script is post-evaluation only. It is allowed to read test metrics because
it is not used inside the training loss or checkpoint selection. Its purpose is
to prevent single-seed cherry-picking by reporting stability across all supplied
seeds/evaluation directories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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
        val = float(x)
        return None if not np.isfinite(val) else val
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    return x


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_soh_row(eval_dir: Path) -> Dict[str, Any]:
    score = eval_dir / "five_state_scorecard.csv"
    row: Dict[str, Any] = {"eval_dir": str(eval_dir), "available": False}
    if not score.exists():
        row["failure"] = f"missing {score}"
        return row
    df = pd.read_csv(score)
    if "variable" not in df.columns:
        row["failure"] = "five_state_scorecard.csv has no variable column"
        return row
    soh = df[df["variable"].astype(str).str.upper() == "SOH"]
    if soh.empty:
        row["failure"] = "SOH row not found"
        return row
    src = soh.iloc[0].to_dict()
    row.update({k: src.get(k) for k in src.keys()})
    for key in ["n", "MAE", "RMSE", "NMAE", "NRMSE", "R2", "corr"]:
        if key in row:
            try:
                row[key] = float(row[key])
            except Exception:
                pass
    row["available"] = True
    diag = _load_json(eval_dir / "soh_overdecay_diagnostic.json")
    if diag:
        row["active_clamp_count_all"] = diag.get("active_clamp_count_all")
        row["active_clamp_fraction_all"] = diag.get("active_clamp_fraction_all")
        split = diag.get("split_summary", {}).get("test", {}) if isinstance(diag.get("split_summary"), dict) else {}
        row["test_pred_min_diag"] = split.get("SOH_pred_min")
        row["test_pred_max_diag"] = split.get("SOH_pred_max")
        segments = diag.get("segments", [])
        if isinstance(segments, list):
            for seg in segments:
                name = str(seg.get("segment", ""))
                if name == "401-521":
                    row["tail_401_521_R2"] = seg.get("metric_R2")
                    row["tail_401_521_MAE"] = seg.get("metric_MAE")
                    row["tail_401_521_BIAS"] = seg.get("metric_BIAS")
    return row


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dirs", nargs="+", required=True, help="Evaluation directories to aggregate.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_r2", type=float, default=0.98)
    p.add_argument("--max_r2_std", type=float, default=0.01)
    p.add_argument("--max_mae_mean", type=float, default=0.006)
    p.add_argument("--require_no_clamp", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = [_read_soh_row(Path(p)) for p in args.eval_dirs]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "seed_stability_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    valid = df[df.get("available", False) == True].copy()  # noqa: E712
    summary: Dict[str, Any] = {
        "eval_dirs": list(args.eval_dirs),
        "n_eval_dirs": len(args.eval_dirs),
        "n_available": int(len(valid)),
        "target_r2": float(args.target_r2),
        "max_r2_std": float(args.max_r2_std),
        "max_mae_mean": float(args.max_mae_mean),
        "csv": str(csv_path),
    }
    failures: List[str] = []
    if len(valid) == 0:
        failures.append("No available SOH rows found.")
    else:
        r2 = pd.to_numeric(valid["R2"], errors="coerce").to_numpy(dtype=float)
        mae = pd.to_numeric(valid["MAE"], errors="coerce").to_numpy(dtype=float)
        summary.update(
            R2_min=float(np.nanmin(r2)),
            R2_mean=float(np.nanmean(r2)),
            R2_std=float(np.nanstd(r2, ddof=0)),
            MAE_mean=float(np.nanmean(mae)),
            MAE_max=float(np.nanmax(mae)),
        )
        bad_r2 = valid[pd.to_numeric(valid["R2"], errors="coerce") < float(args.target_r2)]
        if not bad_r2.empty:
            failures.append("Some seeds are below target_r2: " + ", ".join(str(x) for x in bad_r2["eval_dir"].tolist()))
        if float(summary["R2_std"]) > float(args.max_r2_std):
            failures.append(f"R2_std={summary['R2_std']:.6g} exceeds max_r2_std={args.max_r2_std}")
        if float(summary["MAE_mean"]) > float(args.max_mae_mean):
            failures.append(f"MAE_mean={summary['MAE_mean']:.6g} exceeds max_mae_mean={args.max_mae_mean}")
        if args.require_no_clamp and "active_clamp_count_all" in valid.columns:
            clamp = pd.to_numeric(valid["active_clamp_count_all"], errors="coerce").fillna(0).to_numpy(dtype=float)
            if np.any(clamp > 0):
                failures.append("At least one seed has active clamp hits.")
    summary["ok"] = len(failures) == 0
    summary["failures"] = failures
    json_path = out_dir / "seed_stability_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(summary), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
