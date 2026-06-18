from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any, Dict


def load(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect D17-G2.1 summary")
    ap.add_argument("summary_json")
    args = ap.parse_args()
    d = load(args.summary_json)
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "recommendation": d.get("recommendation"),
        "g3_ready": d.get("g3_ready"),
        "g3_blockers": d.get("g3_blockers") or d.get("promotion_reasons"),
        "failure_mode": d.get("failure_mode"),
        "g21_repair_recommended": d.get("g21_repair_recommended"),
        "best_epoch": d.get("best_epoch"),
        "fit_train_mean_r2": (d.get("fit_train_per_target_aggregate") or {}).get("all_target_profile_r2_mean"),
        "internal_heldout_mean_r2": (d.get("internal_heldout_per_target_aggregate") or {}).get("all_target_profile_r2_mean"),
        "internal_heldout_min_r2": (d.get("internal_heldout_per_target_aggregate") or {}).get("all_target_profile_r2_min"),
        "internal_phie_min_r2": (d.get("internal_heldout_per_target_aggregate") or {}).get("phie_r2_min"),
        "validation_mean_r2": (d.get("validation_report_only_per_target_aggregate") or {}).get("all_target_profile_r2_mean"),
        "validation_min_r2": (d.get("validation_report_only_per_target_aggregate") or {}).get("all_target_profile_r2_min"),
        "worst_internal_target_profile": d.get("worst_internal_target_profile"),
        "worst_validation_target_profile": d.get("worst_validation_target_profile"),
    }, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
