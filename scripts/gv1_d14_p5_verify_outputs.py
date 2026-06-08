#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D14-P5A verifier for eval/verify closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    required = [
        "D14_P5_MANIFEST_REPORT.json",
        "D14_P5_SOFTLABEL_NN_MANIFEST.csv",
        "D14_P5_SOFTLABEL_NN_MANIFEST.json",
        "ModelFin_D14_P5_p2dlite_nn_smoke/best.pt",
        "ModelFin_D14_P5_p2dlite_nn_smoke/training_summary.json",
        "ModelFin_D14_P5_p2dlite_nn_smoke/feature_stats.json",
        "ModelFin_D14_P5_p2dlite_nn_smoke/loss_history.csv",
        "EvalFin_D14_P5_p2dlite_nn_smoke/D14_P5_EVAL_REPORT.json",
        "EvalFin_D14_P5_p2dlite_nn_smoke/metrics_by_profile.csv",
        "EvalFin_D14_P5_p2dlite_nn_smoke/metrics_by_split.csv",
    ]
    missing = [p for p in required if not (out / p).exists()]
    manifest = json.loads((out / "D14_P5_MANIFEST_REPORT.json").read_text(encoding="utf-8")) if (out / "D14_P5_MANIFEST_REPORT.json").exists() else {}
    eval_report = json.loads((out / "EvalFin_D14_P5_p2dlite_nn_smoke/D14_P5_EVAL_REPORT.json").read_text(encoding="utf-8")) if (out / "EvalFin_D14_P5_p2dlite_nn_smoke/D14_P5_EVAL_REPORT.json").exists() else {}
    train_summary = json.loads((out / "ModelFin_D14_P5_p2dlite_nn_smoke/training_summary.json").read_text(encoding="utf-8")) if (out / "ModelFin_D14_P5_p2dlite_nn_smoke/training_summary.json").exists() else {}

    status = "PASS"
    reasons = []
    if missing:
        status = "FAIL"
        reasons.append("missing_required_outputs")
    if manifest.get("overall_status") == "FAIL":
        status = "FAIL"
        reasons.append("manifest_failed")
    eval_status = eval_report.get("overall_status")
    if eval_status == "FAIL":
        status = "FAIL"
        reasons.append("eval_failed")
    elif eval_status == "WARN" and status != "FAIL":
        status = "WARN"
        reasons.append("eval_warn")

    verify = {
        "overall_status": status,
        "reasons": reasons,
        "missing": missing,
        "manifest_status": manifest.get("overall_status"),
        "eval_status": eval_status,
        "eval_warn_reasons": eval_report.get("warn_reasons", []),
        "eval_fail_reasons": eval_report.get("fail_reasons", []),
        "best_epoch": train_summary.get("best_epoch"),
        "best_val_loss": train_summary.get("best_val_loss"),
        "eval_report": str(out / "EvalFin_D14_P5_p2dlite_nn_smoke/D14_P5_EVAL_REPORT.json"),
    }
    (out / "D14_P5_VERIFY_REPORT.json").write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D14-P5A VERIFY status={status} reasons={reasons}")
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
