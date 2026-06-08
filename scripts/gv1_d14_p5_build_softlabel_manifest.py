#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build D14-P5 soft-label NN smoke manifest from P4B-v3 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels_nn.xjtu_p2dlite_dataset import (
    discover_softlabel_profiles,
    assign_splits,
    save_manifest,
    write_json,
    write_csv,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--softlabel_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prior_file", default="")
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(Path(args.config))

    rows = discover_softlabel_profiles(args.softlabel_root)
    rows = assign_splits(rows, cfg)

    manifest_csv = output_dir / "D14_P5_SOFTLABEL_NN_MANIFEST.csv"
    manifest_json = output_dir / "D14_P5_SOFTLABEL_NN_MANIFEST.json"
    save_manifest(rows, manifest_csv, manifest_json)

    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    prior_hashes = sorted(set(row.get("prior_hash", "") for row in rows if row.get("prior_hash", "")))
    n_r_pairs = sorted(set((int(row.get("n_r_a", -1)), int(row.get("n_r_c", -1))) for row in rows if row.get("n_r_a", "") != ""))

    checks = []
    min_total = int(cfg.get("profile_split", {}).get("minimum_profiles_total", 4))
    required_splits = cfg.get("profile_split", {}).get("require_nonempty_splits", ["train", "val", "test"])

    checks.append({
        "check_id": "P5-M00",
        "name": "softlabel profile discovery",
        "status": "PASS" if len(rows) >= min_total else "FAIL",
        "detail": f"profile_count={len(rows)} min_total={min_total}",
    })
    missing_splits = [s for s in required_splits if split_counts.get(s, 0) <= 0]
    checks.append({
        "check_id": "P5-M01",
        "name": "train/val/test split availability",
        "status": "PASS" if not missing_splits else "FAIL",
        "detail": f"split_counts={split_counts} missing={missing_splits}",
    })
    checks.append({
        "check_id": "P5-M02",
        "name": "single prior hash",
        "status": "PASS" if len(prior_hashes) == 1 else "FAIL",
        "detail": f"prior_hashes={prior_hashes}",
    })
    checks.append({
        "check_id": "P5-M03",
        "name": "n_r consistency",
        "status": "PASS" if len(n_r_pairs) == 1 else "FAIL",
        "detail": f"n_r_pairs={n_r_pairs}",
    })

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    report = {
        "package": "D14-P5 manifest build",
        "overall_status": overall,
        "softlabel_root": args.softlabel_root,
        "output_dir": str(output_dir),
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "profile_count": len(rows),
        "split_counts": split_counts,
        "prior_hashes": prior_hashes,
        "n_r_pairs": n_r_pairs,
        "checks": checks,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(output_dir / "D14_P5_MANIFEST_REPORT.json", report)
    write_csv(output_dir / "D14_P5_MANIFEST_CHECKS.csv", checks)

    print(f"D14-P5 manifest status={overall}")
    print(f"profile_count={len(rows)} split_counts={split_counts}")
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else 1


if __name__ == "__main__":
    raise SystemExit(main())
