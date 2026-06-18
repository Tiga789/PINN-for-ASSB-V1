from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.d17_g.g63_forensics import run_forensics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D17-G6.3 P4D/GEO generator formula forensics. No training, no radial solver, no full 55-cell run.")
    p.add_argument("--config", default="configs/d17_g63_p4d_generator_formula_forensics.json")
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--g0_profile_semantics_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--profile_contains", action="append", default=[], help="Substring such as Batch-6_GEO_battery-2. Can be repeated.")
    p.add_argument("--max_time_points", type=int, default=4096, help="Uniformly sample this many points from full profile. 0 means all points, not recommended for smoke.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_forensics(args)
    print(json.dumps({
        "status": summary.get("status"),
        "patch_ready": summary.get("patch_ready"),
        "recommendation": summary.get("recommendation"),
        "blockers": summary.get("blockers"),
        "selected_profile_count": summary.get("selected_profile_count"),
        "evaluated_profile_count": summary.get("evaluated_profile_count"),
        "min_best_formula_r2": summary.get("min_best_formula_r2"),
        "min_config_formula_r2": summary.get("min_config_formula_r2"),
        "elapsed_s": summary.get("elapsed_s"),
        "summary_json": str(Path(args.out_dir) / "D17_G63_P4D_FORMULA_FORENSICS_SUMMARY.json"),
        "candidate_metrics_csv": str(Path(args.out_dir) / "D17_G63_FORMULA_CANDIDATE_METRICS.csv"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
