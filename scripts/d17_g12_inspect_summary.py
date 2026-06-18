from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/d17_g12_inspect_summary.py <D17_G12_PHIE_GAUGE_CLOSEDSET_REPAIR_SUMMARY.json>")
        return 2
    p = Path(sys.argv[1])
    with open(p, "r", encoding="utf-8") as f:
        s = json.load(f)
    agg = s.get("train_closedset_per_target_aggregate", {})
    out = {
        "status": s.get("status"),
        "recommendation": s.get("recommendation"),
        "reasons": s.get("reasons", []),
        "g2_ready": s.get("g2_ready"),
        "train_closedset_mean_r2": agg.get("all_target_profile_r2_mean"),
        "train_closedset_min_r2": agg.get("all_target_profile_r2_min"),
        "theta_a_r2_mean": agg.get("theta_a_r2_mean"),
        "theta_c_r2_mean": agg.get("theta_c_r2_mean"),
        "cs_a_r2_mean": agg.get("cs_a_r2_mean"),
        "cs_c_r2_mean": agg.get("cs_c_r2_mean"),
        "phie_r2_mean": agg.get("phie_r2_mean"),
        "phie_r2_min": agg.get("phie_r2_min"),
        "phis_c_r2_mean": agg.get("phis_c_r2_mean"),
        "per_target_csv": s.get("files", {}).get("per_target_profile_metrics_csv"),
        "target_normalization_audit_csv": s.get("files", {}).get("target_normalization_audit_csv"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
