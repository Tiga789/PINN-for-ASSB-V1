from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect D17-G1.5 internal-heldout triage summary")
    ap.add_argument("summary_json")
    args = ap.parse_args()
    p = Path(args.summary_json)
    d: Dict[str, Any] = json.load(open(p, "r", encoding="utf-8"))
    worst = (d.get("worst_internal_heldout_profiles") or [{}])[0]
    cov = (d.get("worst_feature_coverage_profiles") or [{}])[0]
    print(json.dumps({
        "status": d.get("status"),
        "recommendation": d.get("recommendation"),
        "g2_ready": d.get("g2_ready"),
        "g2_blockers": d.get("g2_blockers"),
        "worst_internal_profile": {
            "canonical_cell_uid": worst.get("canonical_cell_uid"),
            "protocol": worst.get("protocol"),
            "semantic_branch": worst.get("semantic_branch"),
            "worst_target": worst.get("worst_target"),
            "worst_target_r2": worst.get("worst_target_r2"),
            "profile_target_r2_mean": worst.get("profile_target_r2_mean"),
            "profile_target_r2_min": worst.get("profile_target_r2_min"),
        },
        "worst_feature_coverage_profile": {
            "canonical_cell_uid": cov.get("canonical_cell_uid"),
            "split": cov.get("split"),
            "protocol": cov.get("protocol"),
            "feature_z_max_abs": cov.get("feature_z_max_abs"),
            "feature_z_max_feature": cov.get("feature_z_max_feature"),
            "features_outside_fit_minmax": cov.get("features_outside_fit_minmax"),
        },
        "recommended_actions": d.get("recommended_actions"),
        "decision_report_md": d.get("files", {}).get("decision_report_md"),
        "recommended_g15r_config_json": d.get("files", {}).get("recommended_g15r_config_json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
