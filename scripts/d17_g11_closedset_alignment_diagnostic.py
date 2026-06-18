from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.d17_g.g11_diagnostics import run_g11_diagnostic


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D17-G1.1 closed-set alignment diagnostic for generator surrogate training.")
    p.add_argument("--config", required=True)
    p.add_argument("--project_root", default=".")
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--g0_profile_semantics_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--single_profile_count", type=int, default=None)
    p.add_argument("--train_profile_count", type=int, default=None)
    p.add_argument("--validation_profile_count", type=int, default=None)
    p.add_argument("--max_time_points", type=int, default=None)
    p.add_argument("--time_window_s", type=float, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    summary = run_g11_diagnostic(
        project_root=args.project_root,
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        out_dir=args.out_dir,
        config=cfg,
        single_profile_count=int(args.single_profile_count if args.single_profile_count is not None else cfg.get("single_profile_count", 1)),
        train_profile_count=int(args.train_profile_count if args.train_profile_count is not None else cfg.get("train_profile_count", 12)),
        validation_profile_count=int(args.validation_profile_count if args.validation_profile_count is not None else cfg.get("validation_profile_count", 3)),
        max_time_points=int(args.max_time_points if args.max_time_points is not None else cfg.get("max_time_points", 512)),
        time_window_s=float(args.time_window_s if args.time_window_s is not None else cfg.get("time_window_s", 40000.0)),
        device_arg=str(args.device if args.device is not None else cfg.get("device", "auto")),
    )
    print(json.dumps({
        "status": summary.get("status"),
        "recommendation": summary.get("recommendation"),
        "reasons": summary.get("reasons"),
        "single_status": summary.get("single_profile_overfit", {}).get("status"),
        "single_mean_r2": summary.get("single_profile_overfit", {}).get("per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "single_min_r2": summary.get("single_profile_overfit", {}).get("per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "closedset_status": summary.get("train_closedset_12profile", {}).get("status"),
        "closedset_mean_r2": summary.get("train_closedset_12profile", {}).get("per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "closedset_min_r2": summary.get("train_closedset_12profile", {}).get("per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "summary_json": summary.get("files", {}).get("summary_json"),
    }, ensure_ascii=False, indent=2))
    # Diagnostic REVIEW is a valid outcome, not a runtime error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
