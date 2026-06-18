from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g62_p4d_inventory_patch import run_p4d_equivalence_smoke


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G6.2 P4D deterministic inventory equivalence smoke; no training.")
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--p4d_config", default="")
    ap.add_argument("--prior_json", default="")
    ap.add_argument("--profile_contains", action="append", default=["Batch-6_GEO_battery-2", "Batch-6_GEO_battery-5"], help="Substring of canonical/cell UID to test. Can be repeated.")
    ap.add_argument("--max_time_points", type=int, default=0)
    ap.add_argument("--time_window_s", type=float, default=0.0)
    ap.add_argument("--r2_mean_threshold", type=float, default=0.98)
    ap.add_argument("--r2_min_threshold", type=float, default=0.95)
    args = ap.parse_args()
    summary = run_p4d_equivalence_smoke(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        out_dir=args.out_dir,
        project_root=args.project_root,
        p4d_config=args.p4d_config,
        prior_json=args.prior_json,
        profile_contains=args.profile_contains,
        max_time_points=args.max_time_points,
        time_window_s=args.time_window_s,
        r2_mean_threshold=args.r2_mean_threshold,
        r2_min_threshold=args.r2_min_threshold,
    )
    print(json.dumps({
        "status": summary.get("status"),
        "promotion_status": summary.get("promotion_status"),
        "g62_patch_ready": summary.get("g62_patch_ready"),
        "recommendation": summary.get("recommendation"),
        "blockers": summary.get("blockers"),
        "aggregate": summary.get("aggregate"),
        "worst_profile_target": summary.get("worst_profile_target"),
        "summary_json": summary.get("files", {}).get("summary_json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
