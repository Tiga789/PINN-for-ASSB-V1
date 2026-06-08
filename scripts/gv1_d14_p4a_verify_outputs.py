#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify D14-P4A output files and hard guardrails."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED = [
    "D14_P4A_SOFTLABEL_SMOKE_REPORT.json",
    "D14_P4A_SOFTLABEL_SMOKE_REPORT.md",
    "D14_P4A_SELECTED_PROFILES.csv",
    "D14_P4A_SOFTLABEL_MANIFEST.csv",
    "D14_P4A_SOFTLABEL_AUDIT.csv",
    "D14_P4A_PRIOR_RESOLVED.json",
    "D14_P4A_PRIOR_HASH.txt",
    "D14_P4A_OUTPUT_INDEX.json",
    "D14_P4A_RUN_SUMMARY.txt",
    "README_D14_P4A_PATCH.md",
]


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(x, default=0.0):
    try:
        if x in ("", None):
            return default
        return float(x)
    except Exception:
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    missing = [f for f in EXPECTED if not (out / f).exists()]
    if missing:
        print("D14-P4A VERIFY FAIL: missing files:")
        for f in missing:
            print("  -", f)
        return 1

    report = json.loads((out / "D14_P4A_SOFTLABEL_SMOKE_REPORT.json").read_text(encoding="utf-8"))
    status = str(report.get("overall_status", "WARN")).upper()
    checks = report.get("checks", [])
    fail_checks = [c for c in checks if str(c.get("status", "")).upper() == "FAIL"]
    warn_checks = [c for c in checks if str(c.get("status", "")).upper() == "WARN"]

    audit_rows = read_csv_rows(out / "D14_P4A_SOFTLABEL_AUDIT.csv")
    audit_rerun_rows = read_csv_rows(out / "D14_P4A_SOFTLABEL_AUDIT_RERUN.csv")

    rows_to_check = audit_rerun_rows or audit_rows
    hard_fail = []
    warn_rows = []
    metadata_fail = []
    voltage_fail = []
    for r in rows_to_check:
        st = str(r.get("status", "")).upper()
        if st == "FAIL":
            hard_fail.append(r)
        elif st == "WARN":
            warn_rows.append(r)
        if str(r.get("metadata_ok", "")).lower() not in ("true", "1", "yes"):
            metadata_fail.append(r)
        if as_float(r.get("voltage_upper_fail_count")) > 0 or as_float(r.get("voltage_lower_fail_count")) > 0:
            voltage_fail.append(r)

    print(f"D14-P4A VERIFY report_status={status}")
    print(f"checks: PASS={sum(1 for c in checks if str(c.get('status','')).upper()=='PASS')} WARN={len(warn_checks)} FAIL={len(fail_checks)}")
    print(f"audit_rows={len(rows_to_check)} audit_WARN={len(warn_rows)} audit_FAIL={len(hard_fail)} metadata_fail={len(metadata_fail)} voltage_fail={len(voltage_fail)}")

    for c in warn_checks:
        print(f"WARN check {c.get('check_id')}: {c.get('detail')}")
    for c in fail_checks:
        print(f"FAIL check {c.get('check_id')}: {c.get('detail')}")
    for r in warn_rows[:10]:
        print(f"WARN audit {r.get('npz_path')}: {r.get('detail')}")
    for r in hard_fail[:10]:
        print(f"FAIL audit {r.get('npz_path')}: {r.get('detail')}")

    if status == "FAIL" or fail_checks or hard_fail or metadata_fail or voltage_fail:
        return 1
    if (status == "WARN" or warn_checks or warn_rows) and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
