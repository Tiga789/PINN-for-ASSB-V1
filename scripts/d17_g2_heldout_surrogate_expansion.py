from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g2_trainer import build_and_train_g2


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G2 held-out generator-surrogate expansion")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_profile_count", type=int, default=39)
    ap.add_argument("--validation_profile_count", type=int, default=7)
    ap.add_argument("--internal_heldout_count", type=int, default=None)
    ap.add_argument("--max_time_points", type=int, default=512)
    ap.add_argument("--time_window_s", type=float, default=40000.0)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.internal_heldout_count is not None:
        cfg["internal_heldout_profile_count"] = int(args.internal_heldout_count)
        cfg["internal_heldout_count"] = int(args.internal_heldout_count)

    summary = build_and_train_g2(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        out_dir=args.out_dir,
        config=cfg,
        train_profile_count=args.train_profile_count,
        validation_profile_count=args.validation_profile_count,
        max_time_points=args.max_time_points,
        time_window_s=args.time_window_s,
        device_arg=args.device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    print(json.dumps({
        "status": summary.get("status"),
        "g3_ready": summary.get("g3_ready"),
        "recommendation": summary.get("recommendation"),
        "status_reasons": summary.get("status_reasons"),
        "g3_blockers": summary.get("g3_blockers"),
        "best_epoch": summary.get("best_epoch"),
        "fit_train_mean_r2": summary.get("fit_train_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "internal_heldout_mean_r2": summary.get("internal_heldout_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "internal_heldout_min_r2": summary.get("internal_heldout_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "internal_phie_min_r2": summary.get("internal_heldout_per_target_aggregate", {}).get("phie_r2_min"),
        "validation_mean_r2": summary.get("validation_report_only_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "validation_min_r2": summary.get("validation_report_only_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "validation_phie_min_r2": summary.get("validation_report_only_per_target_aggregate", {}).get("phie_r2_min"),
        "fit_train_protocol_counts": summary.get("dataset", {}).get("fit_train_protocol_counts"),
        "internal_heldout_protocol_counts": summary.get("dataset", {}).get("internal_heldout_protocol_counts"),
        "validation_protocol_counts": summary.get("dataset", {}).get("validation_protocol_counts"),
        "fit_train_semantic_branch_counts": summary.get("dataset", {}).get("fit_train_semantic_branch_counts"),
        "internal_heldout_semantic_branch_counts": summary.get("dataset", {}).get("internal_heldout_semantic_branch_counts"),
        "validation_semantic_branch_counts": summary.get("dataset", {}).get("validation_semantic_branch_counts"),
        "worst_internal_target_profile": summary.get("worst_internal_target_profile"),
        "worst_validation_target_profile": summary.get("worst_validation_target_profile"),
        "summary_json": summary.get("files", {}).get("summary_json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
