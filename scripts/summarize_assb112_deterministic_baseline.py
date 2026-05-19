# -*- coding: utf-8 -*-
"""Summarize ASSB-112 deterministic SOH baseline outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _get(d: Mapping[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="ModelFin_112_deterministicSOH_ridge_g4")
    p.add_argument("--output_dir", default="EvalFin_112_deterministicSOH_ridge_g4")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_path = model_dir / "metrics_soh_by_split_final_report.json"
    summary_path = model_dir / "train_summary.json"
    pred_path = model_dir / "soh_pred_by_cycle.csv"
    if not final_path.exists():
        raise FileNotFoundError(final_path)
    with final_path.open("r", encoding="utf-8") as f:
        final = json.load(f)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f) if summary_path.exists() else {}

    rows = []
    metrics = final.get("metrics_by_split_after_selection", {})
    for split, m in metrics.items():
        row = {"split": split}
        row.update(m)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "deterministic_soh_scorecard.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("ASSB-112 deterministic SOH baseline summary")
    lines.append(f"model_dir = {model_dir}")
    lines.append(f"model_variant = {summary.get('model_variant')}")
    lines.append(f"feature_mode = {summary.get('feature_mode')}")
    lines.append(f"selected_alpha = {summary.get('selected_alpha')}")
    lines.append(f"device_used = {summary.get('device_used')}; gpu_reserved_actual_gb = {summary.get('gpu_reserved_actual_gb')}")
    lines.append(f"no_test_metrics_in_training_history = {summary.get('no_test_metrics_in_training_history')}")
    lines.append(f"test_metrics_used_for_selection = {summary.get('test_metrics_used_for_selection')}")
    test = metrics.get("test", {})
    lines.append(
        "TEST: "
        f"R2={test.get('SOH_R2')} MAE={test.get('SOH_MAE')} "
        f"RMSE={test.get('SOH_RMSE')} BIAS={test.get('SOH_BIAS')} corr={test.get('SOH_corr')}"
    )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)

    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        # compact cycle-level error file for quick plotting/checking
        pred[["cycle_id", "split", "SOH_obs", "SOH_pred", "SOH_err"]].to_csv(out_dir / "soh_pred_by_cycle.csv", index=False, encoding="utf-8-sig")
    print(f"[OK] summary written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
