from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g61_inspect_summary.py <D17_G61_FULL_CYCLE_COVERAGE_REPAIR_SUMMARY.json>")
        return 2
    d = load(sys.argv[1])
    fit = d.get("fit_train_per_target_aggregate", {}) if isinstance(d.get("fit_train_per_target_aggregate"), dict) else {}
    internal = d.get("internal_heldout_per_target_aggregate", {}) if isinstance(d.get("internal_heldout_per_target_aggregate"), dict) else {}
    val = d.get("validation_report_only_per_target_aggregate", {}) if isinstance(d.get("validation_report_only_per_target_aggregate"), dict) else {}
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "g6_ready": d.get("g6_ready"),
        "g3_ready_compat": d.get("g3_ready"),
        "recommendation": d.get("recommendation"),
        "best_epoch": d.get("best_epoch"),
        "dataset": d.get("dataset"),
        "fit_train_mean_r2": fit.get("all_target_profile_r2_mean"),
        "fit_train_min_r2": fit.get("all_target_profile_r2_min"),
        "internal_heldout_mean_r2": internal.get("all_target_profile_r2_mean"),
        "internal_heldout_min_r2": internal.get("all_target_profile_r2_min"),
        "validation_mean_r2": val.get("all_target_profile_r2_mean"),
        "validation_min_r2": val.get("all_target_profile_r2_min"),
        "validation_phie_min_r2": val.get("phie_r2_min"),
        "worst_internal_target_profile": d.get("worst_internal_target_profile"),
        "worst_validation_target_profile": d.get("worst_validation_target_profile"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
