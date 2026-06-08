#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify D14-P4B-v3 output files and audit status."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED = [
    "D14_P4B_SOFTLABEL_MULTIPROFILE_REPORT.json",
    "D14_P4B_SOFTLABEL_MULTIPROFILE_REPORT.md",
    "D14_P4B_DISCOVERED_PROFILES.csv",
    "D14_P4B_SELECTED_PROFILES.csv",
    "D14_P4B_SOFTLABEL_MANIFEST.csv",
    "D14_P4B_SOFTLABEL_AUDIT.csv",
    "D14_P4B_BY_BATCH_PROTOCOL.csv",
    "D14_P4B_PRIOR_RESOLVED.json",
    "D14_P4B_PRIOR_HASH.txt",
    "D14_P4B_OUTPUT_INDEX.json",
    "D14_P4B_RUN_SUMMARY.txt",
    "README_D14_P4B_PATCH.md",
]


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_warn", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    missing = [name for name in EXPECTED if not (out / name).exists()]
    if missing:
        print("D14-P4B-v3 VERIFY FAIL: missing files:")
        for name in missing:
            print("  -", name)
        return 1

    report = json.loads((out / "D14_P4B_SOFTLABEL_MULTIPROFILE_REPORT.json").read_text(encoding="utf-8"))
    status = str(report.get("overall_status", "WARN")).upper()
    checks = report.get("checks", [])
    fail_checks = [c for c in checks if str(c.get("status", "")).upper() == "FAIL"]
    warn_checks = [c for c in checks if str(c.get("status", "")).upper() == "WARN"]

    audit_rows = read_rows(out / "D14_P4B_SOFTLABEL_AUDIT.csv")
    rerun_rows = read_rows(out / "D14_P4B_SOFTLABEL_AUDIT_RERUN.csv")
    rows = rerun_rows or audit_rows
    audit_fail = [r for r in rows if str(r.get("status", "")).upper() == "FAIL"]
    audit_warn = [r for r in rows if str(r.get("status", "")).upper() == "WARN"]

    index = json.loads((out / "D14_P4B_OUTPUT_INDEX.json").read_text(encoding="utf-8"))
    index_missing = [f for f in index.get("files", []) if not f.get("exists")]

    print(f"D14-P4B-v3 VERIFY status={status}")
    print(f"checks PASS={sum(1 for c in checks if str(c.get('status','')).upper()=='PASS')} WARN={len(warn_checks)} FAIL={len(fail_checks)}")
    print(f"audit rows={len(rows)} WARN={len(audit_warn)} FAIL={len(audit_fail)}")
    print(f"output_index_missing={len(index_missing)}")

    for c in warn_checks:
        print(f"WARN check {c.get('check_id')}: {c.get('detail')}")
    for c in fail_checks:
        print(f"FAIL check {c.get('check_id')}: {c.get('detail')}")
    for r in audit_warn[:10]:
        print(f"WARN audit {r.get('npz_path')}: {r.get('detail')}")
    for r in audit_fail[:10]:
        print(f"FAIL audit {r.get('npz_path')}: {r.get('detail')}")
    if index_missing:
        for f in index_missing[:10]:
            print(f"INDEX MISSING {f.get('name')}")

    if status == "FAIL" or fail_checks or audit_fail or index_missing:
        return 1
    if (status == "WARN" or warn_checks or audit_warn) and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
