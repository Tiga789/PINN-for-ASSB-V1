#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify D14-P5C closed-set precision audit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = [
    "D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.json",
    "D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.md",
    "D14_P5C_CHECKS.csv",
    "D14_P5C_PROFILE_METRICS_COMPACT.csv",
    "D14_P5C_LOSS_SUMMARY.json",
    "D14_P5C_BATCH_METRICS.json",
    "D14_P5C_PROTOCOL_METRICS.json",
    "D14_P5C_OUTPUT_INDEX.json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    missing = [name for name in REQUIRED if not (out / name).exists()]
    report_path = out / "D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    status = str(report.get("overall_status", "FAIL")).upper()
    checks = report.get("checks", [])
    fail = [c for c in checks if str(c.get("status", "")).upper() == "FAIL"]
    warn = [c for c in checks if str(c.get("status", "")).upper() == "WARN"]

    verify = {
        "overall_status": "FAIL" if missing or status == "FAIL" or fail else ("WARN" if status == "WARN" or warn else "PASS"),
        "missing": missing,
        "audit_status": status,
        "warn_count": len(warn),
        "fail_count": len(fail),
    }
    (out / "D14_P5C_VERIFY_REPORT.json").write_text(json.dumps(verify, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"D14-P5C VERIFY status={verify['overall_status']} audit_status={status} warn={len(warn)} fail={len(fail)} missing={len(missing)}")
    if verify["overall_status"] == "FAIL":
        return 1
    if verify["overall_status"] == "WARN" and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
