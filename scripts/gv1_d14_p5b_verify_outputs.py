#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify D14-P5B closed-set precision benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = [
    "D14_P5B_MANIFEST_REPORT.json",
    "D14_P5B_CLOSEDSET_MANIFEST.csv",
    "D14_P5B_CLOSEDSET_MANIFEST.json",
    "D14_P5B_MANIFEST_CHECKS.csv",
    "ModelFin_D14_P5B_8cell_closedset_precision/best.pt",
    "ModelFin_D14_P5B_8cell_closedset_precision/training_summary.json",
    "ModelFin_D14_P5B_8cell_closedset_precision/feature_stats.json",
    "ModelFin_D14_P5B_8cell_closedset_precision/tensor_memory_summary.json",
    "ModelFin_D14_P5B_8cell_closedset_precision/loss_history.csv",
    "EvalFin_D14_P5B_8cell_closedset_precision/D14_P5B_EVAL_REPORT.json",
    "EvalFin_D14_P5B_8cell_closedset_precision/metrics_by_profile.csv",
    "EvalFin_D14_P5B_8cell_closedset_precision/metrics_by_batch.csv",
    "EvalFin_D14_P5B_8cell_closedset_precision/metrics_by_protocol.csv",
    "EvalFin_D14_P5B_8cell_closedset_precision/metrics_global.json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    missing = [p for p in REQUIRED if not (out / p).exists()]
    manifest = json.loads((out / "D14_P5B_MANIFEST_REPORT.json").read_text(encoding="utf-8")) if (out / "D14_P5B_MANIFEST_REPORT.json").exists() else {}
    train = json.loads((out / "ModelFin_D14_P5B_8cell_closedset_precision/training_summary.json").read_text(encoding="utf-8")) if (out / "ModelFin_D14_P5B_8cell_closedset_precision/training_summary.json").exists() else {}
    eval_report = json.loads((out / "EvalFin_D14_P5B_8cell_closedset_precision/D14_P5B_EVAL_REPORT.json").read_text(encoding="utf-8")) if (out / "EvalFin_D14_P5B_8cell_closedset_precision/D14_P5B_EVAL_REPORT.json").exists() else {}

    status = "PASS"
    reasons = []
    if missing:
        status = "FAIL"
        reasons.append("missing_required_outputs")
    if manifest.get("overall_status") == "FAIL":
        status = "FAIL"
        reasons.append("manifest_failed")
    if eval_report.get("overall_status") == "FAIL":
        status = "FAIL"
        reasons.append("eval_failed")
    elif eval_report.get("overall_status") == "WARN" and status != "FAIL":
        status = "WARN"
        reasons.append("eval_warn")

    verify = {
        "overall_status": status,
        "reasons": reasons,
        "missing": missing,
        "manifest_status": manifest.get("overall_status"),
        "eval_status": eval_report.get("overall_status"),
        "eval_warn_reasons": eval_report.get("warn_reasons", []),
        "eval_fail_reasons": eval_report.get("fail_reasons", []),
        "best_epoch": train.get("best_epoch"),
        "best_loss": train.get("best_loss"),
        "point_count": train.get("point_count"),
        "profile_count": train.get("profile_count"),
        "global_metrics": eval_report.get("global_metrics", {}),
        "boundaries": eval_report.get("boundaries", {}),
    }
    (out / "D14_P5B_VERIFY_REPORT.json").write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D14-P5B VERIFY status={status} reasons={reasons}")
    if missing:
        print("Missing:")
        for m in missing:
            print("  -", m)
    if status == "FAIL":
        return 1
    if status == "WARN" and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
