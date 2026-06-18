from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g3_frozen_audit import read_json, run_g3_frozen_test_audit


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G3 frozen-test report-only state audit")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--candidate_g21_dir", required=True)
    ap.add_argument("--candidate_g21_summary", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoint", default="", help="Optional override. Defaults to best_model_pt in G2.1 summary.")
    ap.add_argument("--max_time_points", type=int, default=512)
    ap.add_argument("--time_window_s", type=float, default=40000.0)
    ap.add_argument("--frozen_test_profile_limit", type=int, default=0, help="0 means all frozen-test profiles.")
    ap.add_argument("--flagged_probe_profile_limit", type=int, default=1)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg: Dict[str, Any] = read_json(args.config, default={}) or {}
    summary = run_g3_frozen_test_audit(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        candidate_g21_dir=args.candidate_g21_dir,
        candidate_g21_summary=args.candidate_g21_summary,
        out_dir=args.out_dir,
        config=cfg,
        checkpoint_path=args.checkpoint,
        max_time_points=args.max_time_points,
        time_window_s=args.time_window_s,
        frozen_test_profile_limit=args.frozen_test_profile_limit,
        flagged_probe_profile_limit=args.flagged_probe_profile_limit,
        device_arg=args.device,
    )
    print(json.dumps({
        "status": summary.get("status"),
        "promotion_status": summary.get("promotion_status"),
        "g4_ready": summary.get("g4_ready"),
        "recommendation": summary.get("recommendation"),
        "g4_blockers": summary.get("g4_blockers"),
        "frozen_test_profile_count": (summary.get("dataset") or {}).get("frozen_test_profile_count"),
        "frozen_test_mean_r2": (summary.get("frozen_test_per_target_aggregate") or {}).get("all_target_profile_r2_mean"),
        "frozen_test_min_r2": (summary.get("frozen_test_per_target_aggregate") or {}).get("all_target_profile_r2_min"),
        "frozen_test_phie_min_r2": (summary.get("frozen_test_per_target_aggregate") or {}).get("phie_r2_min"),
        "worst_frozen_test_target_profile": summary.get("worst_frozen_test_target_profile"),
        "summary_json": (summary.get("files") or {}).get("summary_json"),
        "scorecard_json": str(Path(args.out_dir) / "D17_G3_SCORECARD.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
