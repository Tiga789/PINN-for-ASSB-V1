from __future__ import annotations

import json
import sys
from pathlib import Path


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g62l_inspect_summary.py <summary.json>")
        return 2
    d = read_json(sys.argv[1])
    keys = [
        "protocol", "status", "promotion_status", "g62_patch_formula_ready", "g6c_streaming_smoke_ready",
        "recommendation", "blockers", "selected_profile_count", "evaluated_profile_count",
        "inventory_mean_r2", "inventory_min_r2", "after_patch_mean_r2", "after_patch_min_r2",
        "elapsed_s", "max_time_points", "time_window_s",
    ]
    print(json.dumps({k: d.get(k) for k in keys if k in d}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
