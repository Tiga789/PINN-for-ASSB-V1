#!/usr/bin/env python3
"""Audit ASSB-111 seed42-locked candidate selection for test-set usage."""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

FORBIDDEN_PATTERNS = [
    re.compile(r"five_state_scorecard\.csv", re.I),
    re.compile(r"soh_pred_by_cycle\.csv", re.I),
    re.compile(r"soh_overdecay", re.I),
    re.compile(r"EvalFin_", re.I),
    re.compile(r"\btest_R2\b|\bSOH_test_R2\b|\btest_mae\b", re.I),
]
ALLOWED_REPORT_ONLY = ["show_ModelFin111_seed42locked_metrics.ps1"]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scan_file(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    hits: List[str] = []
    for pat in FORBIDDEN_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    return hits


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection_dir", required=True)
    p.add_argument("--candidate_root", default="ASSB111_seed42_locked_candidates")
    p.add_argument("--output_json", required=True)
    p.add_argument("--strict", action="store_true", help="Exit nonzero if audit fails.")
    args = p.parse_args(argv)

    selection_dir = Path(args.selection_dir)
    selected_path = selection_dir / "selected_candidate.json"
    score_path = selection_dir / "candidate_visible_score.csv"
    selected = load_json(selected_path)

    failures: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {
        "selection_dir": str(selection_dir),
        "candidate_root": args.candidate_root,
        "selected_candidate_json_exists": selected_path.exists(),
        "candidate_visible_score_csv_exists": score_path.exists(),
    }

    if not selected_path.exists():
        failures.append("missing selected_candidate.json")
    if selected.get("test_metrics_used_for_selection") is True or selected.get("selection_used_test_metrics") is True:
        failures.append("selected_candidate.json reports test metrics were used")

    files_read = selected.get("files_read", [])
    forbidden_files_read = [p for p in files_read if any(pat.search(str(p)) for pat in FORBIDDEN_PATTERNS)]
    if forbidden_files_read:
        failures.append("selection files_read contains forbidden test/eval files")
        details["forbidden_files_read"] = forbidden_files_read

    if score_path.exists():
        with score_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        details["candidate_visible_score_rows"] = len(rows)
        for i, row in enumerate(rows):
            if str(row.get("selection_used_test_metrics", "")).lower() in {"true", "1", "yes"}:
                failures.append(f"candidate_visible_score row {i} reports test metric usage")

    # Scan only selection scripts/configs, not final reporting scripts. Reporting is allowed to display test metrics after selection.
    scripts_to_scan = [
        Path("scripts/optimize_assb111_seed42_locked_trainval.py"),
        Path("scripts/compare_assb111_seed42locked_candidates.py"),
        Path("scripts/run_ModelFin111_saturating_v2_seed42locked.ps1"),
    ]
    script_hits: Dict[str, List[str]] = {}
    for path in scripts_to_scan:
        if path.exists():
            hits = scan_file(path)
            # The optimizer contains forbidden file names as blocklist constants. That is allowed if the file also declares them forbidden.
            if hits:
                txt = path.read_text(encoding="utf-8", errors="ignore")
                if "FORBIDDEN" in txt or "forbidden" in txt:
                    warnings.append(f"{path} mentions forbidden patterns as guard/blocklist")
                else:
                    script_hits[str(path)] = hits
    if script_hits:
        failures.append("selection scripts contain forbidden test/eval metric references outside guard context")
        details["script_forbidden_hits"] = script_hits

    ok = len(failures) == 0
    report = {
        "protocol": "ASSB111_seed42_locked_trainval_only_selection_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ok": ok,
        "failures": failures,
        "warnings": warnings,
        "details": details,
        "statement": "Candidate/checkpoint selection must be based on train/val visible metrics only. Final test metrics may be reported only after selection is frozen.",
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
