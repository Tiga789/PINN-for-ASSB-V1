from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g62_inspect_summary.py <summary_or_scorecard.json>")
        return 2
    d = load_json(sys.argv[1])
    out = {
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "promotion_status": d.get("promotion_status"),
        "g62_patch_ready": d.get("g62_patch_ready"),
        "g6_streaming_ready": d.get("g6_streaming_ready"),
        "recommendation": d.get("recommendation"),
        "blockers": d.get("blockers"),
        "aggregate": d.get("aggregate"),
        "worst_profile_target": d.get("worst_profile_target"),
        "worst_cycle_target": d.get("worst_cycle_target"),
        "patch_application_counts": d.get("patch_application_counts"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
