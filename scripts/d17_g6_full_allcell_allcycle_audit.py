from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g3_frozen_audit import read_json
from gv1.d17_g.g6_full_cycle_audit import run_g6_full_cycle_audit


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G6 full all-cell all-cycle report-only audit")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--candidate_g21_dir", required=True)
    ap.add_argument("--candidate_g21_summary", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--splits", default="all", help="Comma-separated: all,train,validation,frozen_test,flagged_probe")
    ap.add_argument("--include_flagged_probe", action="store_true", default=True)
    ap.add_argument("--exclude_flagged_probe", action="store_false", dest="include_flagged_probe")
    ap.add_argument("--profile_limit", type=int, default=0)
    ap.add_argument("--max_time_points", type=int, default=0, help="0 means full soft-label time grid")
    ap.add_argument("--time_window_s", type=float, default=0.0, help="0 means no time-window truncation")
    ap.add_argument("--predict_batch_size", type=int, default=8192)
    ap.add_argument("--save_predictions", choices=["none", "compressed_npz"], default="none")
    ap.add_argument("--no_cycle_metrics", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg: Dict[str, Any] = read_json(args.config, default={}) or {}
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    summary = run_g6_full_cycle_audit(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        candidate_g21_dir=args.candidate_g21_dir,
        candidate_g21_summary=args.candidate_g21_summary,
        out_dir=args.out_dir,
        config=cfg,
        checkpoint_path=args.checkpoint,
        splits=splits,
        include_flagged_probe=bool(args.include_flagged_probe),
        profile_limit=int(args.profile_limit),
        max_time_points=int(args.max_time_points),
        time_window_s=float(args.time_window_s),
        predict_batch_size=int(args.predict_batch_size),
        save_predictions=args.save_predictions,
        cycle_metrics=not bool(args.no_cycle_metrics),
        device_arg=args.device,
    )
    print(json.dumps({
        "status": summary.get("status"),
        "promotion_status": summary.get("promotion_status"),
        "full_cycle_all55_ready": summary.get("full_cycle_all55_ready"),
        "recommendation": summary.get("recommendation"),
        "blockers": summary.get("blockers"),
        "selected_record_count": (summary.get("dataset") or {}).get("selected_record_count"),
        "evaluated_profile_count": (summary.get("dataset") or {}).get("evaluated_profile_count"),
        "total_time_points_evaluated": (summary.get("dataset") or {}).get("total_time_points_evaluated"),
        "all_profile_target_r2_mean": summary.get("all_profile_target_r2_mean"),
        "all_profile_target_r2_min": summary.get("all_profile_target_r2_min"),
        "per_target_profile_r2_summary": summary.get("per_target_profile_r2_summary"),
        "worst_profile_target": summary.get("worst_profile_target"),
        "worst_cycle_target": summary.get("worst_cycle_target"),
        "points_per_second": (summary.get("runtime") or {}).get("points_per_second"),
        "summary_json": (summary.get("files") or {}).get("summary_json"),
        "scorecard_json": (summary.get("files") or {}).get("scorecard_json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
