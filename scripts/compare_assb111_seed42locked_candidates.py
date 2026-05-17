#!/usr/bin/env python3
"""Compare ASSB-111 seed42-locked candidates using train/val visible metrics only."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate_model_dirs", nargs="+", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--input_file", default="input_assb111_strict30_saturating_v2_seed42locked")
    p.add_argument("--selection_mode", default="visible_train_val_only")
    p.add_argument("--max_val_mae", type=float, default=0.00150)
    p.add_argument("--min_train_r2", type=float, default=0.990)
    p.add_argument("--require_no_test_history", action="store_true")
    args = p.parse_args(argv)

    if args.selection_mode != "visible_train_val_only":
        raise SystemExit("Only visible_train_val_only selection is allowed for seed42-locked candidates.")

    this_dir = Path(__file__).resolve().parent
    helper = this_dir / "optimize_assb111_seed42_locked_trainval.py"
    if not helper.exists():
        raise SystemExit(f"Missing helper script: {helper}")

    cmd = [sys.executable, str(helper), "--output_dir", args.output_dir, "--max_val_mae", str(args.max_val_mae), "--min_train_r2", str(args.min_train_r2), "--candidate_model_dirs", *args.candidate_model_dirs]
    if args.require_no_test_history:
        cmd.append("--require_no_test_history")
    subprocess.check_call(cmd)

    out = Path(args.output_dir)
    selected_path = out / "selected_candidate.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
    selected.update({
        "input_file": args.input_file,
        "selection_driver": "compare_assb111_seed42locked_candidates.py",
        "selection_mode": "visible_train_val_only",
        "selection_boundary": "No test metrics were requested or read by this comparator.",
    })
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps({"selected_candidate_json": str(selected_path), "selected": selected.get("candidate_tag")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
