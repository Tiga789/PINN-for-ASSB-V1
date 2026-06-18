# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/d17_p4_inspect_scorecard.py <D17_P4_SCORECARD.json>")
        raise SystemExit(2)
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "promotion_status": d.get("promotion_status"),
        "p5_ready": d.get("p5_ready"),
        "normal_frozen_test_profile_count": d.get("normal_frozen_test_profile_count"),
        "normal_frozen_test_state_r2": d.get("normal_frozen_test_state_r2"),
        "promotion_reasons": d.get("promotion_reasons"),
        "outputs": d.get("outputs"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
