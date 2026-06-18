from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g1_inspect_summary.py <D17_G1_SUPERVISED_SURROGATE_SMOKE_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Missing file: {p}")
        return 2
    d = json.load(open(p, "r", encoding="utf-8"))
    print("protocol=", d.get("protocol"))
    print("status=", d.get("status"))
    print("promotion_status=", d.get("promotion_status"))
    print("g2_ready=", d.get("g2_ready"))
    print("reasons=", d.get("reasons", []))
    print("promotion_reasons=", d.get("promotion_reasons", []))
    print("best_epoch=", d.get("best_epoch"))
    print("best_train_loss=", d.get("best_train_loss"))
    print("policy=", d.get("policy"))
    print("dataset=", d.get("dataset"))
    print("train_profile_aggregate=", d.get("train_profile_aggregate"))
    print("validation_profile_aggregate_report_only=", d.get("validation_profile_aggregate_report_only"))
    print("files=", d.get("files"))
    return 0 if d.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
