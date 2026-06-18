from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g62_lite_patch import read_json, run_formula_only


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G6.2L fast P4D/GEO formula-only inventory check. No training, no model, no radial solver.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--profile_contains", action="append", default=[])
    ap.add_argument("--splits", default="train,validation")
    ap.add_argument("--include_flagged_probe", action="store_true")
    ap.add_argument("--profile_limit", type=int, default=0)
    ap.add_argument("--max_time_points", type=int, default=4096)
    ap.add_argument("--time_window_s", type=float, default=0.0)
    args = ap.parse_args()
    cfg = read_json(args.config, default={}) or {}
    terms = args.profile_contains or cfg.get("default_profile_contains", ["Batch-6_GEO_battery-2", "Batch-6_GEO_battery-5"])
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    summary = run_formula_only(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        out_dir=args.out_dir,
        config=cfg,
        profile_contains=terms,
        splits=splits,
        include_flagged_probe=bool(args.include_flagged_probe),
        profile_limit=int(args.profile_limit),
        max_time_points=int(args.max_time_points),
        time_window_s=float(args.time_window_s),
    )
    print(json.dumps({
        "status": summary.get("status"),
        "promotion_status": summary.get("promotion_status"),
        "g62_patch_formula_ready": summary.get("g62_patch_formula_ready"),
        "recommendation": summary.get("recommendation"),
        "blockers": summary.get("blockers"),
        "inventory_mean_r2": summary.get("inventory_mean_r2"),
        "inventory_min_r2": summary.get("inventory_min_r2"),
        "summary_json": summary.get("files", {}).get("summary_json"),
        "metrics_csv": summary.get("files", {}).get("metrics_csv"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
