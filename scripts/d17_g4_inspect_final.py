from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect D17-G4 final scorecard")
    ap.add_argument("final_scorecard_json")
    args = ap.parse_args()
    d = load(args.final_scorecard_json)
    g3 = d.get("g3") or {}
    g21 = d.get("g21") or {}
    speed = d.get("speed_audit") or {}
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "final_candidate_ready": d.get("final_candidate_ready"),
        "recommendation": d.get("recommendation"),
        "blockers": d.get("blockers"),
        "g21_status": g21.get("status"),
        "g21_g3_ready": g21.get("g3_ready"),
        "g3_status": g3.get("status"),
        "g3_promotion_status": g3.get("promotion_status"),
        "g3_g4_ready": g3.get("g4_ready"),
        "frozen_test_mean_r2": g3.get("frozen_test_mean_r2"),
        "frozen_test_min_r2": g3.get("frozen_test_min_r2"),
        "frozen_test_phie_min_r2": g3.get("frozen_test_phie_min_r2"),
        "speed_status": speed.get("status"),
        "speed_device": speed.get("device"),
        "samples_per_second": speed.get("samples_per_second"),
        "latency_us_per_sample": speed.get("latency_us_per_sample"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
