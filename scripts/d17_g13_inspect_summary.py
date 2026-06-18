from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect D17-G1.3 summary")
    ap.add_argument("summary_json")
    args = ap.parse_args()
    with open(Path(args.summary_json), "r", encoding="utf-8") as f:
        s = json.load(f)
    keys = {
        "status": s.get("status"),
        "g2_ready": s.get("g2_ready"),
        "recommendation": s.get("recommendation"),
        "status_reasons": s.get("status_reasons"),
        "g2_blockers": s.get("g2_blockers"),
        "best_epoch": s.get("best_epoch"),
        "fit_train_per_target_aggregate": s.get("fit_train_per_target_aggregate"),
        "internal_heldout_per_target_aggregate": s.get("internal_heldout_per_target_aggregate"),
        "validation_report_only_per_target_aggregate": s.get("validation_report_only_per_target_aggregate"),
        "files": s.get("files"),
    }
    print(json.dumps(keys, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
