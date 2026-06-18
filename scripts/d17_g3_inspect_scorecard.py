from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect D17-G3 scorecard/summary")
    ap.add_argument("summary_or_scorecard_json")
    args = ap.parse_args()
    d = load(args.summary_or_scorecard_json)
    agg = d.get("frozen_test_per_target_aggregate") or {}
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "promotion_status": d.get("promotion_status"),
        "g4_ready": d.get("g4_ready"),
        "g4_blockers": d.get("g4_blockers"),
        "frozen_mean_r2": agg.get("all_target_profile_r2_mean"),
        "frozen_min_r2": agg.get("all_target_profile_r2_min"),
        "theta_a_min_r2": agg.get("theta_a_r2_min"),
        "theta_c_min_r2": agg.get("theta_c_r2_min"),
        "cs_a_min_r2": agg.get("cs_a_r2_min"),
        "cs_c_min_r2": agg.get("cs_c_r2_min"),
        "phie_min_r2": agg.get("phie_r2_min"),
        "phis_c_min_r2": agg.get("phis_c_r2_min"),
        "worst_frozen_test_target_profile": d.get("worst_frozen_test_target_profile"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
