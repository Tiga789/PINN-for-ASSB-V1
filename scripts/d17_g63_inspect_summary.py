from __future__ import annotations
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/d17_g63_inspect_summary.py <D17_G63_P4D_FORMULA_FORENSICS_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    d = json.loads(p.read_text(encoding="utf-8"))
    keep = [
        "protocol", "status", "patch_ready", "recommendation", "blockers",
        "selected_profile_count", "evaluated_profile_count", "failure_count",
        "min_best_formula_r2", "min_config_formula_r2", "ready_profile_count",
        "elapsed_s", "candidate_metrics_csv", "profile_summaries_json"
    ]
    print(json.dumps({k: d.get(k) for k in keep}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
