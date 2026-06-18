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
from gv1.d17_g.g61_tools import build_full_cycle_repair_config, extract_compact_metrics, read_json, utc_now, write_json


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G6.1 full-cycle coverage repair training")
    ap.add_argument("--config", required=True)
    ap.add_argument("--split_manifest", required=True)
    ap.add_argument("--g0_profile_semantics_csv", required=True)
    ap.add_argument("--g6_smoke_scorecard", default="", help="Optional previous G6 smoke scorecard, recorded for traceability only.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_profile_count", type=int, default=39)
    ap.add_argument("--validation_profile_count", type=int, default=7)
    ap.add_argument("--internal_heldout_count", type=int, default=6)
    ap.add_argument("--max_time_points", type=int, default=4096, help="Full-profile training samples per profile. 0 is not recommended for training.")
    ap.add_argument("--time_window_s", type=float, default=0.0, help="Must be 0 for full-profile coverage repair.")
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force_fit_profile_contains", action="append", default=[])
    args = ap.parse_args()

    if int(args.max_time_points) <= 0:
        raise SystemExit("For G6.1 training, --max_time_points must be > 0. Use 2048/4096/8192; do not train on full 100k points by accident.")
    if float(args.time_window_s) != 0.0:
        raise SystemExit("For G6.1 full-cycle coverage repair, --time_window_s must be 0.0 to avoid first-40ks window training.")

    raw_cfg = load_config(args.config)
    force = ["Batch-4_R3_battery-4", "Batch-5_random_walk_battery-8"] + [x for x in args.force_fit_profile_contains if x]
    cfg = build_full_cycle_repair_config(raw_cfg, force_fit_profile_contains=force)
    cfg["internal_heldout_profile_count"] = int(args.internal_heldout_count)
    cfg["full_cycle_coverage_sampling"] = {
        "max_time_points_per_profile": int(args.max_time_points),
        "time_window_s": float(args.time_window_s),
        "sampling_mode": "uniform_over_full_softlabel_time_grid_via_g1_data_linear_sample_indices",
        "why": "G6 all-cycle audit uses the full soft-label time grid; the previous G2.1 model was trained only on an early 40 ks window.",
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    effective_cfg = out / "D17_G61_EFFECTIVE_FULL_CYCLE_COVERAGE_CONFIG.json"
    write_json(cfg, effective_cfg)

    g6_scorecard = read_json(args.g6_smoke_scorecard, default={}) if args.g6_smoke_scorecard else {}

    summary = build_and_train_g2(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        out_dir=args.out_dir,
        config=cfg,
        train_profile_count=int(args.train_profile_count),
        validation_profile_count=int(args.validation_profile_count),
        max_time_points=int(args.max_time_points),
        time_window_s=float(args.time_window_s),
        device_arg=args.device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
    )
    summary = dict(summary)
    summary["protocol"] = "D17-G6.1_FULL_CYCLE_COVERAGE_REPAIR"
    summary["created_at_utc_g61_wrapper"] = utc_now()
    summary["effective_full_cycle_coverage_config_json"] = str(effective_cfg)
    summary["source_g6_smoke_scorecard"] = str(args.g6_smoke_scorecard) if args.g6_smoke_scorecard else ""
    summary["source_g6_smoke_compact"] = {
        "status": g6_scorecard.get("status") if isinstance(g6_scorecard, dict) else None,
        "promotion_status": g6_scorecard.get("promotion_status") if isinstance(g6_scorecard, dict) else None,
        "all_profile_target_r2_mean": g6_scorecard.get("all_profile_target_r2_mean") if isinstance(g6_scorecard, dict) else None,
        "all_profile_target_r2_min": g6_scorecard.get("all_profile_target_r2_min") if isinstance(g6_scorecard, dict) else None,
        "worst_profile_target": g6_scorecard.get("worst_profile_target") if isinstance(g6_scorecard, dict) else None,
    }
    summary["purpose"] = "Repair the evidence gap revealed by G6 smoke: train a generator-distilled surrogate on full-profile time coverage before any all-cell/all-cycle claim."
    summary["policy"] = {
        "train_cell_softlabels_used_for_training": True,
        "validation_softlabels_report_only": True,
        "frozen_test_softlabels_used": False,
        "checkpoint_selection": "fit-train plus protocol/branch-stratified train-internal heldout metrics only; validation/frozen-test labels are not used to select checkpoint",
        "candidate_modified_from_g21": True,
        "why_not_direct_G6": "G21/G4 used max_time_points=512 and time_window_s=40000; G6 needs all-cycle/full-time coverage.",
    }
    summary["g6_ready"] = bool(summary.get("g3_ready")) and str(summary.get("status")) == "PASS"
    summary["recommendation"] = "RUN_D17_G6_FULL_ALLCELL_ALLCYCLE_AUDIT_ON_G61_CANDIDATE" if summary["g6_ready"] else "DO_NOT_RUN_G6_REVIEW_G61_TRAINING"
    summary["compact_metrics"] = extract_compact_metrics(summary)

    out_summary = out / "D17_G61_FULL_CYCLE_COVERAGE_REPAIR_SUMMARY.json"
    write_json(summary, out_summary)

    # Compatibility alias: existing G6 auditor accepts any candidate summary with status PASS and g3_ready=true.
    write_json(summary, out / "D17_G61_CANDIDATE_FOR_G6_SUMMARY.json")

    print(json.dumps({
        "status": summary.get("status"),
        "g6_ready": summary.get("g6_ready"),
        "g3_ready_compat": summary.get("g3_ready"),
        "recommendation": summary.get("recommendation"),
        "best_epoch": summary.get("best_epoch"),
        "max_time_points_per_profile": int(args.max_time_points),
        "time_window_s": float(args.time_window_s),
        **extract_compact_metrics(summary),
        "summary_json": str(out_summary),
        "candidate_for_g6_summary_json": str(out / "D17_G61_CANDIDATE_FOR_G6_SUMMARY.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
