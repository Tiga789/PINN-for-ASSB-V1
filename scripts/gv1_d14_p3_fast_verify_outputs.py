#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED = [
    "D14_P3_BATCH56_FEASIBILITY_REPORT.json",
    "D14_P3_BATCH56_FEASIBILITY_REPORT.md",
    "D14_P3_RAW_FILE_INDEX.csv",
    "D14_P3_FILE_SCHEMA_AUDIT.csv",
    "D14_P3_CYCLE_ELIGIBILITY_SUMMARY.csv",
    "D14_P3_REPLAY_READINESS.csv",
    "D14_P3_BATCH_SUMMARY.csv",
    "D14_P3_SOH_POLICY.csv",
    "D14_P3_OUTPUT_INDEX.json",
    "D14_P3_RUN_SUMMARY.txt",
    "README_D14_P3_PATCH.md",
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    missing = [f for f in EXPECTED if not (out / f).exists()]
    if missing:
        print("D14-P3 FAST VERIFY FAIL: missing files:")
        for f in missing:
            print("  -", f)
        return 1
    report = json.loads((out / "D14_P3_BATCH56_FEASIBILITY_REPORT.json").read_text(encoding="utf-8"))
    status = str(report.get("overall_status", "WARN")).upper()
    checks = report.get("checks", [])
    fail = [c for c in checks if str(c.get("status", "")).upper() == "FAIL"]
    warn = [c for c in checks if str(c.get("status", "")).upper() == "WARN"]
    print(f"D14-P3 FAST VERIFY status={status}")
    print(f"PASS={sum(1 for c in checks if str(c.get('status','')).upper()=='PASS')} WARN={len(warn)} FAIL={len(fail)}")
    for c in warn:
        print(f"WARN {c.get('check_id')}: {c.get('detail')}")
    for c in fail:
        print(f"FAIL {c.get('check_id')}: {c.get('detail')}")
    if status == "FAIL":
        return 1
    if status == "WARN" and not args.allow_warn:
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
