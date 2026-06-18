from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g2_trainer import build_and_train_g2
from gv1.d17_g.g21_p4d_branch_tools import build_repair_config, write_json


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G2.1 P4D/random_walk branch coverage repair")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--g2_summary", default="")
    ap.add_argument("--train_profile_count", type=int, default=39)
    ap.add_argument("--validation_profile_count", type=int, default=7)
    ap.add_argument("--internal_heldout_count", type=int, default=6)
    ap.add_argument("--max_time_points", type=int, default=512)
    ap.add_argument("--time_window_s", type=float, default=40000.0)
    ap.add_argument("--epochs", type=int, default=1100)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force_fit_profile_contains", action="append", default=[])
    args = ap.parse_args()
    raw_cfg = load_config(args.config)
    force = ["Batch-4_R3_battery-4", "Batch-5_random_walk_battery-8"] + [x for x in args.force_fit_profile_contains if x]
    cfg = build_repair_config(raw_cfg, force_fit_profile_contains=force)
    cfg["internal_heldout_profile_count"] = int(args.internal_heldout_count)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    effective_cfg_path = out / "D17_G21_EFFECTIVE_REPAIR_CONFIG.json"
    write_json(cfg, effective_cfg_path)
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
    summary = dict(summary)
    summary["protocol"] = "D17-G2.1_P4D_BRANCH_REPAIR"
    summary["source_g2_summary"] = str(args.g2_summary) if args.g2_summary else ""
    summary["effective_repair_config_json"] = str(effective_cfg_path)
    summary["repair_design"] = {
        "force_fit_profile_contains": cfg.get("force_fit_profile_contains"),
        "internal_heldout_profile_count": cfg.get("internal_heldout_profile_count"),
        "min_fit_per_group": cfg.get("min_fit_per_group"),
        "max_internal_per_group": cfg.get("max_internal_per_group"),
        "target_group_weights": cfg.get("target_group_weights"),
        "validation_softlabels_report_only": True,
        "frozen_test_softlabels_used": False,
        "checkpoint_selection": "fit-train + protocol/branch-stratified train-internal heldout only",
    }
    summary["g3_ready"] = bool(summary.get("g3_ready"))
    summary["recommendation"] = "ENTER_D17_G3_FROZEN_TEST_REPORT_ONLY_AUDIT" if summary.get("g3_ready") else "DO_NOT_ENTER_G3_REVIEW_G21_P4D_BRANCH_REPAIR"
    write_json(summary, out / "D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json")
    print(json.dumps({
        "status": summary.get("status"),
        "g3_ready": summary.get("g3_ready"),
        "recommendation": summary.get("recommendation"),
        "g3_blockers": summary.get("g3_blockers") or summary.get("promotion_reasons"),
        "best_epoch": summary.get("best_epoch"),
        "fit_train_mean_r2": summary.get("fit_train_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "internal_heldout_mean_r2": summary.get("internal_heldout_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "internal_heldout_min_r2": summary.get("internal_heldout_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "validation_mean_r2": summary.get("validation_report_only_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "validation_min_r2": summary.get("validation_report_only_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "worst_internal_target_profile": summary.get("worst_internal_target_profile"),
        "worst_validation_target_profile": summary.get("worst_validation_target_profile"),
        "summary_json": str(out / "D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json"),
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
