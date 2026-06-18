from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g65_inspect_summary.py <D17_G65_EXACT_PROVENANCE_REPLAY_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding="utf-8"))
    keys = [
        "protocol", "status", "exact_replay_ready", "patch_ready", "recommendation", "blockers",
        "selected_profile_count", "evaluated_profile_count", "failure_count",
        "min_deployable_formula_r2", "min_any_formula_r2", "elapsed_s", "max_time_points",
    ]
    print(json.dumps({k: d.get(k) for k in keys}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
