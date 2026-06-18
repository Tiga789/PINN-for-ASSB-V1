
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g5_release import run_g5


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G5 final release / reproducibility freeze")
    ap.add_argument("--config", required=True)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_audit", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--g21_summary", required=True)
    ap.add_argument("--g21_dir", required=True)
    ap.add_argument("--g3_summary", required=True)
    ap.add_argument("--g3_scorecard", required=True)
    ap.add_argument("--g3_dir", required=True)
    ap.add_argument("--g4_scorecard", required=True)
    ap.add_argument("--g4_dir", required=True)
    ap.add_argument("--no_state_label_audit", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_copy_small_artifacts", action="store_true")
    ap.add_argument("--max_hash_mb", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    manifest = run_g5(
        project_root=args.project_root,
        out_dir=args.out_dir,
        split_manifest=args.split_manifest,
        g0_audit=args.g0_audit,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        g21_summary=args.g21_summary,
        g21_dir=args.g21_dir,
        g3_summary=args.g3_summary,
        g3_scorecard=args.g3_scorecard,
        g3_dir=args.g3_dir,
        g4_scorecard=args.g4_scorecard,
        g4_dir=args.g4_dir,
        no_state_label_audit=args.no_state_label_audit or None,
        copy_small_artifacts=not args.no_copy_small_artifacts and bool(cfg.get("copy_small_artifacts", True)),
        max_hash_mb=args.max_hash_mb if args.max_hash_mb is not None else float(cfg.get("max_hash_mb", 512.0)),
    )
    print(json.dumps({
        "status": manifest.get("status"),
        "final_release_ready": manifest.get("final_release_ready"),
        "candidate_id": manifest.get("candidate_id"),
        "recommendation": manifest.get("recommendation"),
        "reasons": manifest.get("reasons"),
        "frozen_test_mean_r2": manifest.get("key_metrics", {}).get("frozen_test_mean_r2"),
        "frozen_test_min_r2": manifest.get("key_metrics", {}).get("frozen_test_min_r2"),
        "samples_per_second": manifest.get("key_metrics", {}).get("samples_per_second"),
        "manifest_json": manifest.get("output_files", {}).get("manifest_json"),
        "final_report_md": manifest.get("output_files", {}).get("final_report_md"),
    }, ensure_ascii=False, indent=2))
    return 0 if manifest.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
