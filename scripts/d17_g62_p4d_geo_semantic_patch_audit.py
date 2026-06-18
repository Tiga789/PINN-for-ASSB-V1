from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g62_p4d_inventory_patch import run_p4d_patched_audit


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G6.2 P4D/GEO semantic inventory patched full-cycle audit; no training.")
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--candidate_dir", required=True)
    ap.add_argument("--candidate_summary", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--p4d_config", default="")
    ap.add_argument("--prior_json", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--splits", default="all", help="Comma-separated splits, e.g. train,validation or all")
    ap.add_argument("--include_flagged_probe", action="store_true")
    ap.add_argument("--profile_limit", type=int, default=0)
    ap.add_argument("--profile_contains", action="append", default=[], help="Optional substring filter. Repeatable.")
    ap.add_argument("--max_time_points", type=int, default=0)
    ap.add_argument("--time_window_s", type=float, default=0.0)
    ap.add_argument("--predict_batch_size", type=int, default=8192)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--apply_protocol", action="append", default=[], help="Optional protocol filter for P4D override, e.g. GEO or random_walk. Empty means all P4D.")
    ap.add_argument("--r2_mean_threshold", type=float, default=0.95)
    ap.add_argument("--r2_min_threshold", type=float, default=0.90)
    args = ap.parse_args()
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    summary = run_p4d_patched_audit(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        candidate_dir=args.candidate_dir,
        candidate_summary=args.candidate_summary,
        out_dir=args.out_dir,
        project_root=args.project_root,
        p4d_config=args.p4d_config,
        prior_json=args.prior_json,
        checkpoint_path=args.checkpoint,
        splits=splits,
        include_flagged_probe=bool(args.include_flagged_probe),
        profile_limit=int(args.profile_limit),
        profile_contains=args.profile_contains,
        max_time_points=int(args.max_time_points),
        time_window_s=float(args.time_window_s),
        predict_batch_size=int(args.predict_batch_size),
        device_arg=args.device,
        apply_protocols=args.apply_protocol,
        r2_mean_threshold=float(args.r2_mean_threshold),
        r2_min_threshold=float(args.r2_min_threshold),
    )
    print(json.dumps({
        "status": summary.get("status"),
        "promotion_status": summary.get("promotion_status"),
        "g6_streaming_ready": summary.get("g6_streaming_ready"),
        "recommendation": summary.get("recommendation"),
        "blockers": summary.get("blockers"),
        "aggregate": summary.get("aggregate"),
        "worst_profile_target": summary.get("worst_profile_target"),
        "patch_application_counts": summary.get("patch_application_counts"),
        "summary_json": summary.get("files", {}).get("summary_json"),
        "scorecard_json": summary.get("files", {}).get("scorecard_json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
