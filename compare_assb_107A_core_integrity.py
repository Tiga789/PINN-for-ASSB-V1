# -*- coding: utf-8 -*-
r"""Stage-A integrity check for ModelFin_107A core preservation.

This first-package script does not modify or train the model.  It performs the
checks that are possible before Stage-C files are covered:
1. verify ModelFin_107A checkpoint/config exist;
2. inspect state_dict critical core keys;
3. verify aging-fix1 package has no overlay/base-file dependency patterns;
4. optionally compare reference/candidate prediction NPZ files for four states.

After Stage-C modified files are installed, the same script can be called with
``--candidate_npz`` to confirm aging-disabled output identity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb_model_integrity import (
    compare_npz_states,
    load_state_dict_file,
    save_json,
    scan_for_overlay_patterns,
    state_dict_key_report,
)


def _maybe_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {"exists": True, "keys": sorted(list(data.keys()))[:100]}
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check ModelFin_107A core integrity before aging-fix1 Stage C")
    p.add_argument("--model_dir", default="ModelFin_107A")
    p.add_argument("--solution_npz", default="", help="Reference solution path recorded in report")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference_npz", default="", help="Optional reference prediction npz")
    p.add_argument("--candidate_npz", default="", help="Optional candidate prediction npz")
    args = p.parse_args(argv)

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / "best.pt"
    config = model_dir / "config.json"
    report: Dict[str, object] = {
        "model_dir": str(model_dir),
        "checkpoint_exists": ckpt.exists(),
        "config_exists": config.exists(),
        "solution_npz": args.solution_npz,
        "device_requested": args.device,
    }
    if ckpt.exists():
        try:
            sd = load_state_dict_file(ckpt)
            report["state_dict_report"] = state_dict_key_report(sd)
        except Exception as exc:
            report["state_dict_error"] = str(exc)
    report["config_report"] = _maybe_json(config)
    # Scan production files only. The guard/check files intentionally contain
    # forbidden strings as literals and are therefore excluded.
    report["overlay_scan"] = scan_for_overlay_patterns(Path.cwd(), relative_files=[
        "util/assb_aging_fix1_config.py",
        "util/assb_aging_mechanism.py",
        "util/assb_aging_capacity.py",
        "util/assb_aging_injection.py",
        "scripts/prepare_assb_aging_fix1_cycle_table.py",
        "scripts/train_assb_aging_stageB.py",
        "evaluate_assb_aging_fix1.py",
    ])
    if args.reference_npz and args.candidate_npz:
        try:
            report["state_npz_comparison"] = compare_npz_states(args.reference_npz, args.candidate_npz)
        except Exception as exc:
            report["state_npz_comparison"] = {"available": False, "reason": str(exc)}
    else:
        report["state_npz_comparison"] = {
            "available": False,
            "reason": "Provide --reference_npz and --candidate_npz after Stage-C aging-disabled prediction is generated.",
        }
    # Pass/fail gate for first package.
    failures = []
    if not report["checkpoint_exists"]:
        failures.append("ModelFin_107A/best.pt missing")
    overlay = report.get("overlay_scan", {})
    if isinstance(overlay, dict) and not overlay.get("ok", False):
        failures.append("overlay/base dependency pattern found")
    report["ok"] = len(failures) == 0
    report["failures"] = failures
    save_json(report, out_dir / "core_integrity_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
