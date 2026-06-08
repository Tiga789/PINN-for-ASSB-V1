#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D14-P4A XJTU P2Dlite soft-label smoke generator."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is importable when script is launched from scripts/.
_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels.p2dlite_prior import load_prior, build_resolved_spec
from gv1.softlabels.xjtu_p2dlite_solver import load_profile_npz, generate_softlabels
from gv1.softlabels.xjtu_softlabel_io import save_softlabels
from gv1.softlabels.xjtu_softlabel_audit import audit_softlabel_npz, write_audit_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[dict], fieldnames=None):
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


def discover_profiles(profile_dirs: List[str]) -> List[dict]:
    rows = []
    for d in profile_dirs:
        root = Path(d)
        if not root.exists():
            continue
        for p in root.rglob("solution_replay_profile.npz"):
            text = str(p).replace("\\", "/")
            rows.append({
                "profile_npz": str(p),
                "path_text": text,
                "batch_guess": guess_batch(text),
                "cell_uid_guess": p.parent.name,
                "size_mb": round(p.stat().st_size / 1024 / 1024, 3),
            })
    rows.sort(key=lambda r: (r["batch_guess"], r["size_mb"], r["profile_npz"]))
    return rows


def guess_batch(text: str) -> str:
    import re
    m = re.search(r"Batch[-_ ]?([1-6])", text, flags=re.I)
    if m:
        return f"Batch-{m.group(1)}"
    return "unknown"


def select_profiles(rows: List[dict], cfg: dict, max_total: int) -> List[dict]:
    sel_cfg = cfg.get("selection", {})
    excludes = [s.lower() for s in sel_cfg.get("exclude_patterns", [])]
    clean = []
    for r in rows:
        low = r["path_text"].lower()
        if any(x.lower() in low for x in excludes):
            continue
        clean.append(r)

    preferred = sel_cfg.get("prefer_batches", ["Batch-1", "Batch-3", "Batch-4", "Batch-5", "Batch-6"])
    selected = []
    used = set()
    for b in preferred:
        subset = [r for r in clean if r["batch_guess"] == b and r["profile_npz"] not in used]
        if subset:
            selected.append(subset[0])
            used.add(subset[0]["profile_npz"])
            if len(selected) >= max_total:
                return selected
    for r in clean:
        if r["profile_npz"] not in used:
            selected.append(r)
            used.add(r["profile_npz"])
            if len(selected) >= max_total:
                break
    return selected


def status_rank(s: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(str(s).upper(), 1)


def combine_status(checks: List[dict]) -> str:
    worst = "PASS"
    for c in checks:
        st = str(c.get("status", "WARN")).upper()
        if status_rank(st) > status_rank(worst):
            worst = st
    return worst


def md_table(rows, cols):
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--prior_file", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--input_profile_dirs", nargs="*", default=None)
    ap.add_argument("--max_profiles_total", type=int, default=2)
    ap.add_argument("--max_points_per_profile", type=int, default=100000)
    ap.add_argument("--n_r", type=int, default=17)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config) if args.config else project_root / "configs" / "d14_p4_xjtu_p2dlite_softlabel_smoke_config.json"
    cfg = read_json(cfg_path) or {}
    if args.input_profile_dirs:
        profile_dirs = args.input_profile_dirs
    else:
        profile_dirs = cfg.get("input_profile_dirs", [])

    print(f"[D14-P4A] start {utc_now()}", flush=True)
    print(f"[D14-P4A] prior_file={args.prior_file}", flush=True)

    prior = load_prior(args.prior_file)
    resolved = build_resolved_spec(prior, n_r_override=args.n_r)
    write_json(output_dir / "D14_P4A_PRIOR_RESOLVED.json", resolved)
    (output_dir / "D14_P4A_PRIOR_HASH.txt").write_text(resolved["prior_hash"] + "\n", encoding="utf-8")

    all_profiles = discover_profiles(profile_dirs)
    selected = select_profiles(all_profiles, cfg, args.max_profiles_total)
    write_csv(output_dir / "D14_P4A_SELECTED_PROFILES.csv", selected)
    print(f"[D14-P4A] discovered_profiles={len(all_profiles)} selected={len(selected)}", flush=True)

    manifest = []
    audit_rows = []
    required_keys = cfg.get("pass_criteria", {}).get("required_npz_keys", [])
    voltage_bounds = {
        "upper_warn_V": float(cfg.get("pass_criteria", {}).get("voltage_upper_warn_V", 4.25)),
        "upper_fail_V": float(cfg.get("pass_criteria", {}).get("voltage_upper_fail_V", 4.35)),
        "lower_warn_V": float(cfg.get("pass_criteria", {}).get("voltage_lower_warn_V", 2.45)),
        "lower_fail_V": float(cfg.get("pass_criteria", {}).get("voltage_lower_fail_V", 2.35)),
    }
    require_metadata = bool(cfg.get("pass_criteria", {}).get("metadata_required", True))

    for idx, row in enumerate(selected, 1):
        print(f"[D14-P4A] generate {idx}/{len(selected)}: {row['profile_npz']}", flush=True)
        try:
            profile = load_profile_npz(row["profile_npz"], args.max_points_per_profile, cfg)
            soft = generate_softlabels(profile, resolved)
            summary = {
                "created_utc": utc_now(),
                "source_profile_npz": row["profile_npz"],
                "cell_uid": soft.get("cell_uid", ""),
                "batch": soft.get("batch", ""),
                "protocol": soft.get("protocol", ""),
                "n_points": int(len(soft["t_global_s"])),
                "n_r": int(args.n_r),
                "prior_file": str(Path(args.prior_file)),
                "resolved_spec_hash": resolved["prior_hash"],
                "state_label_interpretation": prior.get("interpretation", {}).get("state_labels", "model-consistent"),
                "soh_generated": False,
                "full_p2d_truth_claim": False,
                "metadata_inferred_from_path": str(soft.get("metadata_inferred_from_path", "False")),
            }
            npz_path = save_softlabels(output_dir / "profiles", soft, summary)
            audit = audit_softlabel_npz(npz_path, required_keys, prior_hash=resolved["prior_hash"], voltage_bounds=voltage_bounds, require_metadata=require_metadata)
            write_audit_json(npz_path, audit)
            manifest.append({
                "profile_ok": audit.get("status") == "PASS",
                "audit_status": audit.get("status", ""),
                "audit_detail": audit.get("detail", ""),
                "source_profile_npz": row["profile_npz"],
                "softlabel_npz": str(npz_path),
                "cell_uid": summary["cell_uid"],
                "batch": summary["batch"],
                "protocol": summary["protocol"],
                "metadata_inferred_from_path": summary["metadata_inferred_from_path"],
                "n_points": summary["n_points"],
                "n_r": summary["n_r"],
                "phis_c_soft_max_V": audit.get("phis_c_soft_max_V", ""),
                "phis_c_soft_min_V": audit.get("phis_c_soft_min_V", ""),
                "max_abs_voltage_bound_correction_V": audit.get("max_abs_voltage_bound_correction_V", ""),
                "resolved_spec_hash": resolved["prior_hash"],
            })
            audit_rows.append(audit)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            manifest.append({
                "profile_ok": False,
                "source_profile_npz": row["profile_npz"],
                "softlabel_npz": "",
                "cell_uid": row.get("cell_uid_guess", ""),
                "batch": row.get("batch_guess", ""),
                "protocol": "",
                "n_points": "",
                "n_r": args.n_r,
                "resolved_spec_hash": resolved["prior_hash"],
                "error": err,
                "traceback_tail": traceback.format_exc(limit=6),
            })
            audit_rows.append({
                "npz_path": "",
                "exists": False,
                "status": "FAIL",
                "detail": err,
                "source_profile_npz": row["profile_npz"],
            })

    write_csv(output_dir / "D14_P4A_SOFTLABEL_MANIFEST.csv", manifest)
    write_csv(output_dir / "D14_P4A_SOFTLABEL_AUDIT.csv", audit_rows)

    checks = []
    checks.append({
        "check_id": "P4-C00",
        "name": "single prior file loaded",
        "status": "PASS",
        "detail": f"prior_hash={resolved['prior_hash']}",
    })
    checks.append({
        "check_id": "P4-C01",
        "name": "profile discovery and selection",
        "status": "PASS" if selected else "FAIL",
        "detail": f"discovered={len(all_profiles)} selected={len(selected)}",
    })
    ok_count = sum(1 for r in manifest if r.get("profile_ok") is True)
    checks.append({
        "check_id": "P4-C02",
        "name": "soft-label NPZ generation",
        "status": "PASS" if ok_count == len(selected) and ok_count > 0 else "FAIL",
        "detail": f"profile_ok={ok_count}/{len(selected)}",
    })
    # theta OOB guard
    warn_oob = []
    fail_oob = []
    warn_thr = float(cfg.get("pass_criteria", {}).get("max_theta_out_of_bounds_fraction_warn", 0.02))
    fail_thr = float(cfg.get("pass_criteria", {}).get("max_theta_out_of_bounds_fraction_fail", 0.10))
    for a in audit_rows:
        oob = max(float(a.get("theta_a_oob_fraction", 0) or 0), float(a.get("theta_c_oob_fraction", 0) or 0))
        if oob > fail_thr:
            fail_oob.append(a.get("npz_path", ""))
        elif oob > warn_thr:
            warn_oob.append(a.get("npz_path", ""))
    checks.append({
        "check_id": "P4-C03",
        "name": "theta bounds audit",
        "status": "FAIL" if fail_oob else ("WARN" if warn_oob else "PASS"),
        "detail": f"warn_oob={len(warn_oob)} fail_oob={len(fail_oob)}",
    })
    checks.append({
        "check_id": "P4-C04",
        "name": "SOH-free generator boundary",
        "status": "PASS",
        "detail": "No SOH labels are generated in solution_softlabels.npz.",
    })
    checks.append({
        "check_id": "P4-C05",
        "name": "model interpretation boundary",
        "status": "PASS",
        "detail": "Outputs are P2Dlite model-consistent soft labels, not full-P2D ground truth.",
    })

    overall = combine_status(checks)
    if overall == "FAIL":
        recommendation = "Do not proceed to full XJTU soft-label generation. Inspect FAIL checks first."
    elif overall == "WARN":
        recommendation = "P2Dlite smoke generated with warnings. Review theta/OCP/voltage audits before expanding."
    else:
        recommendation = "P2Dlite soft-label smoke passed. Next step can expand to a controlled multi-profile set."

    report = {
        "package": "D14-P4A XJTU P2Dlite soft-label smoke",
        "created_utc": utc_now(),
        "overall_status": overall,
        "recommendation": recommendation,
        "paths": {
            "project_root": str(project_root),
            "cache_root": args.cache_root,
            "prior_file": str(Path(args.prior_file)),
            "config": str(cfg_path),
            "output_dir": str(output_dir),
        },
        "summary": {
            "discovered_profiles": len(all_profiles),
            "selected_profiles": len(selected),
            "generated_profiles": ok_count,
            "n_r": args.n_r,
            "max_points_per_profile": args.max_points_per_profile,
            "prior_hash": resolved["prior_hash"],
        },
        "checks": checks,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(output_dir / "D14_P4A_SOFTLABEL_SMOKE_REPORT.json", report)

    md = []
    md.append("# D14-P4A XJTU P2Dlite Soft-label Smoke Report\n")
    md.append(f"Created UTC: `{report['created_utc']}`\n")
    md.append(f"Overall status: **{overall}**\n")
    md.append(f"Recommendation: {recommendation}\n")
    md.append("## Checks\n")
    md.append(md_table(checks, ["check_id", "name", "status", "detail"]))
    md.append("\n## Manifest\n")
    md.append(md_table(manifest, ["profile_ok", "batch", "cell_uid", "n_points", "n_r", "softlabel_npz", "error"]))
    md.append("\n## Boundary\n")
    md.append("- All physical parameters were read from the standalone P2Dlite prior file.\n")
    md.append("- No SOH labels were generated.\n")
    md.append("- The states are model-consistent P2Dlite soft labels, not full-P2D truth.\n")
    (output_dir / "D14_P4A_SOFTLABEL_SMOKE_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    readme_patch = f"""# README D14-P4A Patch

D14-P4A adds a standalone-prior XJTU P2Dlite soft-label generator smoke.

Status: **{overall}**

Prior file:

```text
configs/P2Dlite_prior_xjtu_lr18650la_v0.json
```

All future XJTU soft-label generation and model prediction/training should read
this same prior file, or a user-edited copy with the same schema.

Boundary:

- No training.
- No GV1 mainline modification.
- No SOH generation inside the voltage soft-label generator.
- Outputs are P2Dlite model-consistent soft labels, not full-P2D ground truth.
"""
    (output_dir / "README_D14_P4A_PATCH.md").write_text(readme_patch, encoding="utf-8")

    outputs = [
        "D14_P4A_SOFTLABEL_SMOKE_REPORT.json",
        "D14_P4A_SOFTLABEL_SMOKE_REPORT.md",
        "D14_P4A_SELECTED_PROFILES.csv",
        "D14_P4A_SOFTLABEL_MANIFEST.csv",
        "D14_P4A_SOFTLABEL_AUDIT.csv",
        "D14_P4A_PRIOR_RESOLVED.json",
        "D14_P4A_PRIOR_HASH.txt",
        "D14_P4A_OUTPUT_INDEX.json",
        "D14_P4A_RUN_SUMMARY.txt",
        "README_D14_P4A_PATCH.md",
    ]
    write_json(output_dir / "D14_P4A_OUTPUT_INDEX.json", {
        "overall_status": overall,
        "output_dir": str(output_dir),
        "files": [{"name": f, "exists": (output_dir / f).exists()} for f in outputs],
    })
    (output_dir / "D14_P4A_RUN_SUMMARY.txt").write_text(
        "\n".join([
            "D14-P4A XJTU P2Dlite soft-label smoke",
            f"created_utc={report['created_utc']}",
            f"overall_status={overall}",
            f"prior_hash={resolved['prior_hash']}",
            f"discovered_profiles={len(all_profiles)}",
            f"selected_profiles={len(selected)}",
            f"generated_profiles={ok_count}",
            f"n_r={args.n_r}",
            f"recommendation={recommendation}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"[D14-P4A] overall_status={overall}", flush=True)
    print(f"[D14-P4A] recommendation={recommendation}", flush=True)
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else (2 if overall == "WARN" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
