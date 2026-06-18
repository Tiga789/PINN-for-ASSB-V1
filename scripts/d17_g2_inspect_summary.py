from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect D17-G2 summary")
    ap.add_argument("summary_json")
    args = ap.parse_args()
    p = Path(args.summary_json)
    with open(p, "r", encoding="utf-8") as f:
        s = json.load(f)
    keys = {
        "protocol": s.get("protocol"),
        "status": s.get("status"),
        "g3_ready": s.get("g3_ready"),
        "recommendation": s.get("recommendation"),
        "status_reasons": s.get("status_reasons"),
        "g3_blockers": s.get("g3_blockers"),
        "best_epoch": s.get("best_epoch"),
        "fit_train_mean_r2": s.get("fit_train_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "fit_train_min_r2": s.get("fit_train_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "internal_heldout_mean_r2": s.get("internal_heldout_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "internal_heldout_min_r2": s.get("internal_heldout_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "internal_phie_min_r2": s.get("internal_heldout_per_target_aggregate", {}).get("phie_r2_min"),
        "validation_mean_r2": s.get("validation_report_only_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "validation_min_r2": s.get("validation_report_only_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "validation_phie_min_r2": s.get("validation_report_only_per_target_aggregate", {}).get("phie_r2_min"),
        "fit_train_protocol_counts": s.get("dataset", {}).get("fit_train_protocol_counts"),
        "internal_heldout_protocol_counts": s.get("dataset", {}).get("internal_heldout_protocol_counts"),
        "validation_protocol_counts": s.get("dataset", {}).get("validation_protocol_counts"),
        "fit_train_semantic_branch_counts": s.get("dataset", {}).get("fit_train_semantic_branch_counts"),
        "internal_heldout_semantic_branch_counts": s.get("dataset", {}).get("internal_heldout_semantic_branch_counts"),
        "validation_semantic_branch_counts": s.get("dataset", {}).get("validation_semantic_branch_counts"),
        "worst_internal_target_profile": s.get("worst_internal_target_profile"),
        "worst_validation_target_profile": s.get("worst_validation_target_profile"),
    }
    print(json.dumps(keys, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
