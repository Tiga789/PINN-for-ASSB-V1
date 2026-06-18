from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g6_inspect_scorecard.py <D17_G6_SCORECARD.json>")
        return 2
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        d = json.load(f)
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "promotion_status": d.get("promotion_status"),
        "full_cycle_all55_ready": d.get("full_cycle_all55_ready"),
        "blockers": d.get("blockers"),
        "evaluated_profile_count": (d.get("dataset") or {}).get("evaluated_profile_count"),
        "total_time_points_evaluated": (d.get("dataset") or {}).get("total_time_points_evaluated"),
        "max_time_points": (d.get("dataset") or {}).get("max_time_points"),
        "time_window_s": (d.get("dataset") or {}).get("time_window_s"),
        "all_profile_target_r2_mean": d.get("all_profile_target_r2_mean"),
        "all_profile_target_r2_min": d.get("all_profile_target_r2_min"),
        "per_target_profile_r2_summary": d.get("per_target_profile_r2_summary"),
        "worst_profile_target": d.get("worst_profile_target"),
        "worst_cycle_target": d.get("worst_cycle_target"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
