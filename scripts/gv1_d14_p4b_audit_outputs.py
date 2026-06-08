#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D14-P4B-v3 output auditor.

Re-audits all generated P2Dlite soft-label NPZ files using the same standalone
prior file and produces a compact CSV report. It also repeats the source
voltage_exp bound audit so the user can inspect anomalous raw replay points
without regenerating soft labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

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
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def source_voltage_audit_from_npz(npz_path: Path, cfg: dict):
    audit_cfg = cfg.get("source_voltage_audit", {})
    data = np.load(npz_path, allow_pickle=True)
    V = np.asarray(data["voltage_exp"], dtype=float)
    n = max(len(V), 1)
    upper_warn = float(audit_cfg.get("upper_warn_V", 4.25))
    upper_fail = float(audit_cfg.get("upper_fail_V", 4.35))
    lower_warn = float(audit_cfg.get("lower_warn_V", 2.45))
    lower_fail = float(audit_cfg.get("lower_fail_V", 2.35))
    upper_warn_count = int(np.sum(V > upper_warn))
    upper_fail_count = int(np.sum(V > upper_fail))
    lower_warn_count = int(np.sum(V < lower_warn))
    lower_fail_count = int(np.sum(V < lower_fail))
    fail_count = upper_fail_count + lower_fail_count
    fail_fraction = fail_count / n
    hard_fail = (
        fail_fraction > float(audit_cfg.get("fail_if_fail_fraction_gt", 0.001))
        or fail_count > int(audit_cfg.get("fail_if_fail_count_gt", 10))
    )
    warn = (upper_warn_count + lower_warn_count + fail_count) > 0
    status = "FAIL" if hard_fail else ("WARN" if warn else "PASS")
    return {
        "source_voltage_status": status,
        "voltage_exp_min_V": float(np.nanmin(V)) if len(V) else "",
        "voltage_exp_max_V": float(np.nanmax(V)) if len(V) else "",
        "voltage_exp_upper_warn_count": upper_warn_count,
        "voltage_exp_upper_fail_count": upper_fail_count,
        "voltage_exp_lower_warn_count": lower_warn_count,
        "voltage_exp_lower_fail_count": lower_fail_count,
        "voltage_exp_fail_fraction": fail_fraction,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prior_file", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config and Path(args.config).exists() else {}
    required = cfg.get("pass_criteria", {}).get("required_npz_keys", [
        "t_global_s", "I_profile", "voltage_exp", "temperature_C",
        "r_a", "r_c", "cs_a", "cs_c", "theta_a", "theta_c",
        "phie", "phis_c", "phis_c_base", "phis_c_soft", "phis_c_soft_raw",
        "voltage_bound_correction", "batch", "protocol", "cell_uid", "resolved_spec_hash"
    ])
    pc = cfg.get("pass_criteria", {})
    voltage_bounds = {
        "upper_warn_V": float(pc.get("voltage_upper_warn_V", 4.25)),
        "upper_fail_V": float(pc.get("voltage_upper_fail_V", 4.35)),
        "lower_warn_V": float(pc.get("voltage_lower_warn_V", 2.45)),
        "lower_fail_V": float(pc.get("voltage_lower_fail_V", 2.35)),
    }

    prior = load_prior(args.prior_file)
    resolved = build_resolved_spec(prior)

    rows = []
    for p in (output_dir / "profiles").rglob("solution_softlabels.npz"):
        row = audit_softlabel_npz(
            p,
            required,
            prior_hash=resolved["prior_hash"],
            voltage_bounds=voltage_bounds,
            require_metadata=True,
        )
        src = source_voltage_audit_from_npz(p, cfg)
        row.update(src)
        if row.get("status") == "PASS" and src["source_voltage_status"] == "WARN":
            row["status"] = "WARN"
            row["detail"] = "source_voltage_warn_bound"
        elif src["source_voltage_status"] == "FAIL":
            row["status"] = "FAIL"
            row["detail"] = "source_voltage_fail_bound"
        write_audit_json(p, row)
        rows.append(row)

    write_csv(output_dir / "D14_P4B_SOFTLABEL_AUDIT_RERUN.csv", rows)

    fail = [r for r in rows if str(r.get("status", "")).upper() == "FAIL"]
    warn = [r for r in rows if str(r.get("status", "")).upper() == "WARN"]
    print(f"D14-P4B-v3 audit rerun: total={len(rows)} warn={len(warn)} fail={len(fail)}")
    for r in warn[:10]:
        print(f"WARN {r.get('npz_path')}: {r.get('detail')}")
    for r in fail[:10]:
        print(f"FAIL {r.get('npz_path')}: {r.get('detail')}")

    if fail:
        return 1
    if warn and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
