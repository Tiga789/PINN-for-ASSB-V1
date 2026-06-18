from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.generator_equivalence import json_load, run_g0_audit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D17-G0 generator equivalence audit. No training is performed.")
    p.add_argument("--project_root", default=".", help="Local PINN-for-ASSB-V1 project root.")
    p.add_argument("--config", default="configs/d17_g0_generator_equivalence_audit.json")
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--softlabel_root", default="", help="D15 ALL55 FINAL soft-label root. Optional if manifest paths are valid.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--profile_limit", type=int, default=0, help="0 means all records. Use small number for quick smoke.")
    p.add_argument("--exclude_flagged_probe", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config: Dict[str, Any] = json_load(config_path, default={}) or {}
    report = run_g0_audit(
        project_root=project_root,
        split_manifest=Path(args.split_manifest),
        softlabel_root=Path(args.softlabel_root) if args.softlabel_root else None,
        out_dir=Path(args.out_dir),
        config=config,
        profile_limit=int(args.profile_limit),
        include_flagged_probe=not bool(args.exclude_flagged_probe),
    )
    profile = report.get("profile_audit", {})
    summary = {
        "status": report.get("status"),
        "g1_ready": report.get("g1_ready"),
        "reasons": report.get("reasons", []),
        "out_dir": str(args.out_dir),
        "code_scan_status": report.get("code_scan", {}).get("status"),
        "profile_audit_status": profile.get("status"),
        "profile_count_audited": profile.get("profile_count_audited"),
        "semantic_branch_counts": profile.get("semantic_branch_counts"),
        "semantics_known_fraction": profile.get("semantics_known_fraction"),
        "semantic_known_profile_count": profile.get("semantic_known_profile_count"),
        "audit_json": str(Path(args.out_dir) / "D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json"),
        "profile_semantics_csv": profile.get("profile_semantics_csv"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
