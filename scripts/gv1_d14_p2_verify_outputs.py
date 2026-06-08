
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify D14-P2 scorecard outputs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REQUIRED = [
    "D14_P2_GENERALIZATION_SCORECARD_REPORT.json",
    "D14_P2_GENERALIZATION_SCORECARD_REPORT.md",
    "D14_P2_SOURCE_INVENTORY.csv",
    "D14_P2_RUN_METRICS_NORMALIZED.csv",
    "D14_P2_GLOBAL_SCORECARD.csv",
    "D14_P2_BY_PROTOCOL.csv",
    "D14_P2_BY_CELL.csv",
    "D14_P2_CANDIDATE_COMPARISON.csv",
    "D14_P2_OUTLIER_POLICY.csv",
    "D14_P2_OUTPUT_INDEX.json",
    "D14_P2_RUN_SUMMARY.txt",
]
OPTIONAL = [
    "D14_P2_SEGMENT_METRICS_NORMALIZED.csv",
    "D14_P2_BY_SEGMENT.csv",
    "D14_P2_BY_PROTOCOL_SEGMENT.csv",
    "README_D14_P2_PATCH.md",
]

def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--allow-warn", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.output_dir)
    failures=[]; warnings=[]; rows=[]
    for name in REQUIRED:
        p=out/name
        if not p.exists():
            failures.append(f"missing required file: {name}")
        elif p.stat().st_size <= 0:
            failures.append(f"empty required file: {name}")
        else:
            rows.append((name,p.stat().st_size))
    for name in OPTIONAL:
        p=out/name
        if not p.exists():
            warnings.append(f"missing optional file: {name}")
    report=read_json(out/"D14_P2_GENERALIZATION_SCORECARD_REPORT.json")
    status="MISSING"
    if isinstance(report,dict):
        status=str(report.get("overall_status","MISSING"))
        if status == "FAIL": failures.append("report overall_status=FAIL")
        if status == "WARN" and not args.allow_warn: warnings.append("report overall_status=WARN")
        if int(report.get("run_rows",0) or 0) <= 0:
            failures.append("report run_rows <= 0")
        if int(report.get("batch1_2c_battery8_mainline_rows",0) or 0) > 0:
            failures.append("Batch-1/2C/battery-8 rows found in mainline scorecards")
    else:
        failures.append("cannot parse D14_P2_GENERALIZATION_SCORECARD_REPORT.json")
    # Basic CSV sanity.
    for name in ["D14_P2_GLOBAL_SCORECARD.csv","D14_P2_CANDIDATE_COMPARISON.csv","D14_P2_OUTLIER_POLICY.csv"]:
        p=out/name
        if p.exists():
            try:
                df=pd.read_csv(p)
                if df.empty and args.strict:
                    failures.append(f"{name} is empty")
            except Exception as exc:
                failures.append(f"cannot read {name}: {exc!r}")
    final="PASS"
    if failures: final="FAIL"
    elif warnings: final="WARN"
    print(f"[D14-P2-VERIFY] status={final}, report_status={status}")
    for f in failures: print(f"[FAIL] {f}")
    for w in warnings: print(f"[WARN] {w}")
    return 2 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
