from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g14_inspect_summary.py <D17_G14_PHIE_VALIDATION_ROBUSTNESS_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    d = json.load(open(p, "r", encoding="utf-8"))
    val = d.get("validation_report_only_per_target_aggregate", {}) or {}
    ih = d.get("internal_heldout_per_target_aggregate", {}) or {}
    print(json.dumps({
        "status": d.get("status"),
        "g2_ready": d.get("g2_ready"),
        "recommendation": d.get("recommendation"),
        "g2_blockers": d.get("g2_blockers"),
        "best_epoch": d.get("best_epoch"),
        "internal_mean_r2": ih.get("all_target_profile_r2_mean"),
        "internal_min_r2": ih.get("all_target_profile_r2_min"),
        "validation_mean_r2": val.get("all_target_profile_r2_mean"),
        "validation_min_r2": val.get("all_target_profile_r2_min"),
        "validation_phie_mean_r2": val.get("phie_r2_mean"),
        "validation_phie_min_r2": val.get("phie_r2_min"),
        "worst_validation_phie_profile": d.get("worst_validation_phie_profile"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
