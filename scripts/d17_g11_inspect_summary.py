from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/d17_g11_inspect_summary.py <D17_G11_CLOSEDSET_ALIGNMENT_DIAGNOSTIC_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    with open(p, "r", encoding="utf-8") as f:
        s = json.load(f)
    single = s.get("single_profile_overfit", {})
    closed = s.get("train_closedset_12profile", {})
    out = {
        "status": s.get("status"),
        "recommendation": s.get("recommendation"),
        "reasons": s.get("reasons", []),
        "single_status": single.get("status"),
        "single_mean_r2": single.get("per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "single_min_r2": single.get("per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "closedset_status": closed.get("status"),
        "closedset_mean_r2": closed.get("per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "closedset_min_r2": closed.get("per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "target_normalization_audit_csv": s.get("files", {}).get("target_normalization_audit_csv"),
        "time_grid_alignment_audit_csv": s.get("files", {}).get("time_grid_alignment_audit_csv"),
        "closedset_per_target_metrics_csv": s.get("files", {}).get("closedset_per_target_metrics_csv"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
