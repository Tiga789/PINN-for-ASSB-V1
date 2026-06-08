#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build D14-P5B 8-cell closed-set manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels_nn.xjtu_p2dlite_closedset_dataset import discover_profiles, write_json, write_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--softlabel_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = discover_profiles(args.softlabel_root, cfg)
    rows_ok = [r for r in rows if r.get("status", "PASS") != "FAIL"]

    checks = []
    expected = int(cfg.get("profile_policy", {}).get("expected_profile_count", 8))
    checks.append({
        "check_id": "P5B-M00",
        "name": "closed-set profile discovery",
        "status": "PASS" if len(rows_ok) >= expected else "FAIL",
        "detail": f"usable_profiles={len(rows_ok)} expected={expected}",
    })
    prior_hashes = sorted(set(r.get("prior_hash", "") for r in rows_ok if r.get("prior_hash", "")))
    checks.append({
        "check_id": "P5B-M01",
        "name": "single prior hash",
        "status": "PASS" if len(prior_hashes) == 1 else "FAIL",
        "detail": f"prior_hashes={prior_hashes}",
    })
    n_r = sorted(set(str(r.get("n_r", "")) for r in rows_ok))
    checks.append({
        "check_id": "P5B-M02",
        "name": "n_r consistency",
        "status": "PASS" if len(n_r) == 1 and n_r[0] == "17" else "FAIL",
        "detail": f"n_r={n_r}",
    })
    batches = sorted(set(r.get("batch", "") for r in rows_ok))
    checks.append({
        "check_id": "P5B-M03",
        "name": "batch coverage",
        "status": "PASS" if all(b in batches for b in ["Batch-1", "Batch-3", "Batch-4"]) else "FAIL",
        "detail": f"batches={batches}",
    })

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    write_csv(out / "D14_P5B_CLOSEDSET_MANIFEST.csv", rows_ok)
    write_json(out / "D14_P5B_CLOSEDSET_MANIFEST.json", {"profiles": rows_ok, "profile_count": len(rows_ok)})
    write_csv(out / "D14_P5B_MANIFEST_CHECKS.csv", checks)
    report = {
        "package": "D14-P5B closed-set manifest",
        "overall_status": overall,
        "profile_count": len(rows_ok),
        "all_discovered_count": len(rows),
        "batches": batches,
        "prior_hashes": prior_hashes,
        "checks": checks,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(out / "D14_P5B_MANIFEST_REPORT.json", report)

    print(f"D14-P5B manifest status={overall} usable_profiles={len(rows_ok)}")
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else 1


if __name__ == "__main__":
    raise SystemExit(main())
