from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect D17-G1.5R stratified heldout repair summary")
    ap.add_argument("summary_json")
    args = ap.parse_args()
    p = Path(args.summary_json)
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    keys = [
        "status", "recommendation", "g2_ready", "status_reasons", "g2_blockers", "best_epoch"
    ]
    out = {k: d.get(k) for k in keys}
    out["fit_train_protocol_counts"] = d.get("dataset", {}).get("fit_train_protocol_counts")
    out["internal_heldout_protocol_counts"] = d.get("dataset", {}).get("internal_heldout_protocol_counts")
    out["validation_protocol_counts"] = d.get("dataset", {}).get("validation_protocol_counts")
    for split_key in ["fit_train", "internal_heldout", "validation_report_only"]:
        agg_key = {
            "fit_train": "fit_train_per_target_aggregate",
            "internal_heldout": "internal_heldout_per_target_aggregate",
            "validation_report_only": "validation_report_only_per_target_aggregate",
        }[split_key]
        agg = d.get(agg_key, {}) or {}
        out[f"{split_key}_mean_r2"] = agg.get("all_target_profile_r2_mean")
        out[f"{split_key}_min_r2"] = agg.get("all_target_profile_r2_min")
        out[f"{split_key}_phie_mean_r2"] = agg.get("phie_r2_mean")
        out[f"{split_key}_phie_min_r2"] = agg.get("phie_r2_min")
    out["worst_internal_target_profile"] = d.get("worst_internal_target_profile")
    out["worst_validation_target_profile"] = d.get("worst_validation_target_profile")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
