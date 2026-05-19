# -*- coding: utf-8 -*-
"""Summarize ASSB-112 guarded SOH seed sweep outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_args():
    p = argparse.ArgumentParser(description="Summarize guarded ASSB-112 SOH seed sweep")
    p.add_argument("--model_prefix", default=r".\ModelFin_112_guardedSOH_seed")
    p.add_argument("--seeds", default="7,42,2026,3407,7890")
    p.add_argument("--output_dir", default=r".\EvalFin_112_guarded_soh_sweep_v2")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        model_dir = Path(f"{args.model_prefix}{seed}")
        summary_path = model_dir / "train_summary.json"
        audit_path = model_dir / "selected_checkpoint_audit.json"
        final_path = model_dir / "metrics_soh_by_split_final_report.json"
        row: Dict[str, Any] = {"seed": seed, "model_dir": str(model_dir)}
        if not summary_path.exists():
            row.update({"status": "MISSING_TRAIN_SUMMARY"})
            rows.append(row)
            continue
        summary = _read_json(summary_path)
        audit = _read_json(audit_path) if audit_path.exists() else {}
        final = _read_json(final_path) if final_path.exists() else {}
        row.update({
            "status": "OK" if audit.get("ok", False) else "AUDIT_NOT_OK",
            "best_epoch": summary.get("best_epoch"),
            "best_status": summary.get("best_selection_status"),
            "audit_ok": audit.get("ok"),
            "hard_visible_guard_ok": audit.get("hard_visible_guard_ok", summary.get("hard_visible_guard_ok")),
            "selection_strategy": audit.get("selection_strategy", summary.get("selection_strategy")),
            "no_test_history": summary.get("no_test_metrics_in_training_history"),
            "test_used_for_selection": summary.get("test_metrics_used_for_selection"),
            "train_R2": _get(summary, "final_visible_metrics.train_r2"),
            "train_MAE": _get(summary, "final_visible_metrics.train_mae"),
            "val_R2": _get(summary, "final_visible_metrics.val_r2"),
            "val_corr": _get(summary, "final_visible_metrics.val_corr"),
            "val_MAE": _get(summary, "final_visible_metrics.val_mae"),
            "val_bias_abs": _get(summary, "final_visible_metrics.val_bias_abs"),
            "val_slope_mae": _get(summary, "final_visible_metrics.val_slope_mae"),
            "val_range_ratio": _get(summary, "final_visible_metrics.val_range_ratio"),
            "test_R2": _get(final, "metrics_by_split_after_selection.test.SOH_R2"),
            "test_MAE": _get(final, "metrics_by_split_after_selection.test.SOH_MAE"),
            "test_RMSE": _get(final, "metrics_by_split_after_selection.test.SOH_RMSE"),
            "test_BIAS": _get(final, "metrics_by_split_after_selection.test.SOH_BIAS"),
            "test_corr": _get(final, "metrics_by_split_after_selection.test.SOH_corr"),
            "guard_reasons": ";".join(audit.get("visible_guard", {}).get("guard_reasons", [])) if isinstance(audit.get("visible_guard"), dict) else "",
        })
        rows.append(row)

    csv_path = out_dir / "guarded_soh_seed_summary.csv"
    keys = sorted({k for r in rows for k in r.keys()})
    preferred = [
        "seed", "status", "best_epoch", "best_status", "audit_ok", "hard_visible_guard_ok", "selection_strategy", "no_test_history", "test_used_for_selection",
        "train_R2", "train_MAE", "val_R2", "val_corr", "val_MAE", "val_bias_abs", "val_slope_mae", "val_range_ratio",
        "test_R2", "test_MAE", "test_RMSE", "test_BIAS", "test_corr", "guard_reasons", "model_dir",
    ]
    fieldnames = preferred + [k for k in keys if k not in preferred]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "OK" and isinstance(r.get("test_R2"), (int, float))]
    aggregate: Dict[str, Any] = {"n_seeds": len(rows), "n_ok": len(ok_rows), "csv": str(csv_path)}
    if ok_rows:
        vals = [float(r["test_R2"]) for r in ok_rows]
        maes = [float(r["test_MAE"]) for r in ok_rows]
        aggregate.update({
            "test_R2_mean": sum(vals) / len(vals),
            "test_R2_min": min(vals),
            "test_MAE_mean": sum(maes) / len(maes),
            "test_MAE_max": max(maes),
            "pass_mean_R2_ge_0p98": (sum(vals) / len(vals)) >= 0.98,
            "pass_worst_R2_ge_0p96": min(vals) >= 0.96,
        })
    with (out_dir / "guarded_soh_seed_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    print("\nRows:")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
