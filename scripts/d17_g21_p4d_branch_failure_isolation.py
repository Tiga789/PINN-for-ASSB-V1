from __future__ import annotations

import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g21_p4d_branch_tools import locate_g2_file, read_csv_rows, read_json, summarize_g2_failure, write_csv_rows, write_json, utc_now


def main() -> int:
    ap = argparse.ArgumentParser(description="D17-G2.1 P4D/random_walk branch failure isolation")
    ap.add_argument("--g2_out_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    g2_out = Path(args.g2_out_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary_path = g2_out / "D17_G2_HELDOUT_SURROGATE_EXPANSION_SUMMARY.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing G2 summary: {summary_path}")
    g2 = read_json(summary_path)
    per_target_path = locate_g2_file(g2, g2_out, "per_target_profile_metrics_csv", "D17_G2_PER_TARGET_PROFILE_METRICS.csv")
    if not per_target_path.exists():
        raise FileNotFoundError(f"Missing G2 per-target metrics CSV: {per_target_path}")
    rows = read_csv_rows(per_target_path)
    triage = summarize_g2_failure(g2, rows)
    triage.update({
        "protocol": "D17-G2.1_P4D_RANDOM_WALK_FAILURE_ISOLATION",
        "created_at_utc": utc_now(),
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "validation_softlabels_used_for_training": False,
        "frozen_test_softlabels_used": False,
        "source_g2_summary": str(summary_path),
        "source_g2_per_target_metrics_csv": str(per_target_path),
        "out_dir": str(out),
    })
    status_reasons = []
    if triage.get("g2_status") != "PASS":
        status_reasons.append("source G2 status is not PASS")
    if not rows:
        status_reasons.append("no per-target metric rows were read")
    if triage.get("recommendation") == "MISSING_G2_PER_TARGET_INTERNAL_ROWS":
        status_reasons.append("no train_internal_heldout per-target rows were found")
    triage["status"] = "PASS" if not status_reasons else "REVIEW"
    triage["status_reasons"] = status_reasons
    triage["g21_repair_recommended"] = bool(triage.get("recommendation") == "RUN_G21_P4D_BRANCH_REPAIR")
    triage["repair_hypothesis"] = (
        "P4D random_walk/GEO current-integral branch needs branch/protocol coverage and inventory-phase robust conditioning; first test by pinning Batch-5_random_walk_battery-8 into fit-train and using stricter protocol+branch internal-heldout stratification."
        if triage.get("recommendation") == "RUN_G21_P4D_BRANCH_REPAIR" else "Review failure mode before running repair."
    )
    write_json(triage, out / "D17_G21_P4D_BRANCH_FAILURE_ISOLATION_SUMMARY.json")
    write_csv_rows(triage.get("internal_by_protocol_branch_target", []), out / "D17_G21_INTERNAL_BY_PROTOCOL_BRANCH_TARGET.csv")
    write_csv_rows(triage.get("internal_by_profile_target", []), out / "D17_G21_INTERNAL_BY_PROFILE_TARGET.csv")
    write_csv_rows(triage.get("p4d_internal_worst_rows", []), out / "D17_G21_P4D_INTERNAL_WORST_ROWS.csv")
    print(json.dumps({
        "status": triage.get("status"),
        "recommendation": triage.get("recommendation"),
        "g21_repair_recommended": triage.get("g21_repair_recommended"),
        "failure_mode": triage.get("failure_mode"),
        "worst_internal_target_profile": triage.get("worst_internal_target_profile"),
        "summary_json": str(out / "D17_G21_P4D_BRANCH_FAILURE_ISOLATION_SUMMARY.json"),
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
