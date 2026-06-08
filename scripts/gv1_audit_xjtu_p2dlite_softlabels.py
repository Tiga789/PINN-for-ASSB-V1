#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone auditor for D14-P4/P4A XJTU P2Dlite soft-label outputs.

This script is intentionally independent from the generator. It re-reads the
same standalone P2Dlite prior file and verifies that generated NPZ files:
  - carry the same prior hash;
  - include batch/protocol/cell_uid metadata;
  - keep phis_c_soft within configured terminal-voltage bounds;
  - do not silently become full-P2D/SOH labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels.p2dlite_prior import load_prior, build_resolved_spec
from gv1.softlabels.xjtu_softlabel_audit import audit_softlabel_npz, write_audit_json


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prior_file", required=True)
    ap.add_argument("--required_keys_json", default="")
    ap.add_argument("--upper_warn_V", type=float, default=4.25)
    ap.add_argument("--upper_fail_V", type=float, default=4.35)
    ap.add_argument("--lower_warn_V", type=float, default=2.45)
    ap.add_argument("--lower_fail_V", type=float, default=2.35)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    prior = load_prior(args.prior_file)
    resolved = build_resolved_spec(prior)

    if args.required_keys_json:
        required = json.loads(Path(args.required_keys_json).read_text(encoding="utf-8"))
    else:
        required = [
            "t_global_s", "I_profile", "voltage_exp", "temperature_C",
            "cycle_id", "step_id", "step_type",
            "r_a", "r_c", "cs_a", "cs_c", "theta_a", "theta_c",
            "phie", "phis_c", "phis_c_base", "phis_c_soft", "phis_c_soft_raw",
            "voltage_bound_correction",
            "batch", "protocol", "cell_uid", "resolved_spec_hash"
        ]

    voltage_bounds = {
        "upper_warn_V": args.upper_warn_V,
        "upper_fail_V": args.upper_fail_V,
        "lower_warn_V": args.lower_warn_V,
        "lower_fail_V": args.lower_fail_V,
    }

    rows = []
    for p in (out / "profiles").rglob("solution_softlabels.npz"):
        row = audit_softlabel_npz(
            p,
            required,
            prior_hash=resolved["prior_hash"],
            voltage_bounds=voltage_bounds,
            require_metadata=True,
        )
        write_audit_json(p, row)
        rows.append(row)
    write_csv(out / "D14_P4A_SOFTLABEL_AUDIT_RERUN.csv", rows)

    fail = [r for r in rows if str(r.get("status", "")).upper() == "FAIL"]
    warn = [r for r in rows if str(r.get("status", "")).upper() == "WARN"]
    print(f"D14-P4A audit rerun: total={len(rows)} warn={len(warn)} fail={len(fail)}")
    for r in warn:
        print(f"WARN {r.get('npz_path')}: {r.get('detail')}")
    for r in fail:
        print(f"FAIL {r.get('npz_path')}: {r.get('detail')}")
    if fail:
        return 1
    if warn and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
