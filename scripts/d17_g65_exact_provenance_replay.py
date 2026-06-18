from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.d17_g.g65_exact_replay import run_exact_replay


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="D17-G6.5 exact provenance replay test for P4D/GEO theta inventory; no training, no radial solver.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--config", default="configs/d17_g65_exact_provenance_replay.json")
    ap.add_argument("--g64_dir", required=True, help="Directory containing D17_G64_PROFILE_PROVENANCE_DETAILS.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--profile_contains", action="append", default=[], help="Optional profile substring; repeatable. Defaults to profiles from G64 details.")
    ap.add_argument("--d15_p4d_config", default=None, help="Optional override for exact D15-P4D config path")
    ap.add_argument("--prior_json", default=None, help="Optional override for prior JSON")
    ap.add_argument("--max_time_points", type=int, default=4096)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_exact_replay(args)
    print(json.dumps({
        "status": summary.get("status"),
        "exact_replay_ready": summary.get("exact_replay_ready"),
        "patch_ready": summary.get("patch_ready"),
        "recommendation": summary.get("recommendation"),
        "blockers": summary.get("blockers"),
        "selected_profile_count": summary.get("selected_profile_count"),
        "evaluated_profile_count": summary.get("evaluated_profile_count"),
        "min_deployable_formula_r2": summary.get("min_deployable_formula_r2"),
        "min_any_formula_r2": summary.get("min_any_formula_r2"),
        "elapsed_s": summary.get("elapsed_s"),
        "summary_json": summary.get("outputs", {}).get("summary_json"),
        "candidate_metrics_csv": summary.get("outputs", {}).get("candidate_metrics_csv"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
