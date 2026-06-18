from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g4_finalization import read_json, run_g4_finalization


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G4 final scorecard, freeze manifest, and speed audit")
    ap.add_argument("--config", required=True)
    ap.add_argument("--g0_audit", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--g21_summary", required=True)
    ap.add_argument("--g21_dir", required=True)
    ap.add_argument("--g3_summary", required=True)
    ap.add_argument("--g3_scorecard", required=True)
    ap.add_argument("--g3_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--checkpoint", default="", help="Optional override. Defaults to best_model_pt in G2.1 summary.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--speed_trials", type=int, default=200)
    ap.add_argument("--speed_batch_size", type=int, default=8192)
    ap.add_argument("--hash_large_artifacts", action="store_true")
    args = ap.parse_args()
    cfg: Dict[str, Any] = read_json(args.config, default={}) or {}
    scorecard = run_g4_finalization(
        config=cfg,
        g0_audit=args.g0_audit,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        g21_summary=args.g21_summary,
        g21_dir=args.g21_dir,
        g3_summary=args.g3_summary,
        g3_scorecard=args.g3_scorecard,
        g3_dir=args.g3_dir,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        speed_trials=args.speed_trials,
        speed_batch_size=args.speed_batch_size,
        hash_large_artifacts=bool(args.hash_large_artifacts),
    )
    g3 = scorecard.get("g3") or {}
    speed = scorecard.get("speed_audit") or {}
    print(json.dumps({
        "status": scorecard.get("status"),
        "final_candidate_ready": scorecard.get("final_candidate_ready"),
        "recommendation": scorecard.get("recommendation"),
        "blockers": scorecard.get("blockers"),
        "frozen_test_mean_r2": g3.get("frozen_test_mean_r2"),
        "frozen_test_min_r2": g3.get("frozen_test_min_r2"),
        "frozen_test_phie_min_r2": g3.get("frozen_test_phie_min_r2"),
        "speed_status": speed.get("status"),
        "samples_per_second": speed.get("samples_per_second"),
        "final_scorecard_json": (scorecard.get("files") or {}).get("final_scorecard_json"),
        "final_report_md": (scorecard.get("files") or {}).get("final_report_md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
