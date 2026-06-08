#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify D14-P1 evidence-boundary output directory.
This script does not modify project files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED = [
    "D14_P1_EVIDENCE_BOUNDARY_REPORT.json",
    "D14_P1_EVIDENCE_BOUNDARY_REPORT.md",
    "D14_P1_CLAIMS_MATRIX.csv",
    "D14_P1_TERMINOLOGY_GUARDRAILS.csv",
    "README_D14_P1_PATCH.md",
    "D14_P1_RUN_SUMMARY.txt",
    "D14_P1_OUTPUT_INDEX.json",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--allow-warn", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    missing = [x for x in REQUIRED if not (out / x).exists()]
    if missing:
        print("D14-P1 VERIFY: FAIL")
        print("Missing files:")
        for x in missing:
            print("  -", x)
        return 1

    report = read_json(out / "D14_P1_EVIDENCE_BOUNDARY_REPORT.json")
    status = str(report.get("overall_status", "UNKNOWN")).upper()
    print(f"D14-P1 VERIFY: status={status}")
    print(f"status_reasons={report.get('status_reasons')}")
    print(f"risky_wording_counts={report.get('risky_wording_counts')}")
    print(f"p0_overall_status={report.get('p0', {}).get('p0_overall_status')}")

    if status == "PASS":
        return 0
    if status == "WARN" and args.allow_warn:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
