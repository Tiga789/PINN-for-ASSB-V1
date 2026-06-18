from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g6f_inspect_selected_cycle_summary.py <D17_G6F_SELECTED_CYCLE_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding="utf-8"))
    print(json.dumps({
        "protocol": d.get("protocol"),
        "status": d.get("status"),
        "full_training_recommendation": d.get("full_training_recommendation"),
        "candidate_protocol": d.get("candidate_protocol"),
        "candidate_status": d.get("candidate_status"),
        "candidate_g6_ready": d.get("candidate_g6_ready"),
        "candidate_g3_ready": d.get("candidate_g3_ready"),
        "cell": d.get("record", {}).get("canonical_cell_uid"),
        "requested_cycles": d.get("requested_cycles"),
        "evaluated_cycles": d.get("evaluated_cycles"),
        "n_time_points": d.get("n_time_points"),
        "semantic_branch": d.get("semantic_branch"),
        "aggregate_metrics": d.get("aggregate_metrics"),
        "files": d.get("files"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
