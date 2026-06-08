#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D14-P4B-v3 controlled multi-profile XJTU P2Dlite soft-label generator.

Fixes over P4B-v2
-----------------
1. Do NOT infer Batch-1 from the directory token `xjtu_batch134`.
2. If protocol is R2.5, force Batch-3; if protocol is R3, force Batch-4.
3. If path/source_file explicitly contains Batch-3/Batch-4, it takes priority
   over ambiguous aggregate directory names.
4. Require Batch-1/3/4 to be present in the selected set.
5. Add source `voltage_exp` bound audit.
6. Fix output index write order so `D14_P4B_OUTPUT_INDEX.json` and
   `D14_P4B_RUN_SUMMARY.txt` do not appear as missing in their own index.

This script does not train a model, does not generate SOH, and does not claim
full-P2D internal-state truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

import numpy as np

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels.p2dlite_prior import load_prior, build_resolved_spec
from gv1.softlabels.xjtu_p2dlite_solver import load_profile_npz, generate_softlabels
from gv1.softlabels.xjtu_softlabel_io import save_softlabels
from gv1.softlabels.xjtu_softlabel_audit import audit_softlabel_npz, write_audit_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
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


def status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(str(status).upper(), 1)


def combine_status(checks: List[dict]) -> str:
    worst = "PASS"
    for item in checks:
        st = str(item.get("status", "WARN")).upper()
        if status_rank(st) > status_rank(worst):
            worst = st
    return worst


def safe_string(x: Any) -> str:
    try:
        if hasattr(x, "tolist"):
            x = x.tolist()
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        return str(x).strip()
    except Exception:
        return str(x)


def should_skip_path(path: Path, skip_fragments: Iterable[str]) -> bool:
    text = str(path).replace("\\", "/").lower()
    return any(str(frag).lower() in text for frag in skip_fragments)


def scalar_from_npz(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        return ""
    try:
        val = data[key]
        if hasattr(val, "tolist"):
            val = val.tolist()
        if isinstance(val, (list, tuple)) and len(val) == 1:
            val = val[0]
        s = str(val).strip()
        if s in {"", "None", "none", "nan", "NaN", "[]"}:
            return ""
        return s
    except Exception:
        return ""


def npz_metadata_shallow(path: Path) -> Dict[str, str]:
    out = {"npz_batch": "", "npz_protocol": "", "npz_cell_uid": "", "npz_source_file": ""}
    try:
        data = np.load(path, allow_pickle=True)
        out["npz_batch"] = scalar_from_npz(data, "batch")
        out["npz_protocol"] = scalar_from_npz(data, "protocol")
        out["npz_cell_uid"] = scalar_from_npz(data, "cell_uid")
        out["npz_source_file"] = scalar_from_npz(data, "source_file") or scalar_from_npz(data, "source_profile_npz")
    except Exception as exc:
        out["npz_read_error"] = f"{type(exc).__name__}: {exc}"
    return out


def normalize_batch_token(s: str) -> str:
    if not s:
        return ""
    m = re.search(r"batch[-_ ]?([1-6])$", s.strip(), flags=re.I)
    if m:
        return f"Batch-{int(m.group(1))}"
    m = re.search(r"^Batch-([1-6])$", s.strip(), flags=re.I)
    if m:
        return f"Batch-{int(m.group(1))}"
    return s.strip()


def infer_explicit_batch(text: str) -> str:
    """Find explicit `Batch-3` style tokens, but never match aggregate `batch134`.

    The key guard is `(?![0-9])` after the single batch digit, so `batch134`
    does not become Batch-1.
    """
    matches = re.findall(r"(?<![A-Za-z0-9])batch[-_ ]?([1-6])(?![0-9])", text, flags=re.I)
    if not matches:
        return ""
    # If multiple tokens appear, prefer non-aggregate last explicit token from source_file/parent.
    # Example: Batch-5_Batch-5_battery-7 -> Batch-5.
    return f"Batch-{int(matches[-1])}"


def infer_protocol(text: str) -> str:
    lower = text.lower()
    if re.search(r"r2[._-]?5", lower):
        return "R2.5"
    if re.search(r"(?<![a-z0-9])r3(?![a-z0-9])", lower) or "_r3_" in lower or "-r3-" in lower:
        return "R3"
    if re.search(r"(?<![a-z0-9])2c(?![a-z0-9])", lower) or "_2c_" in lower or "-2c-" in lower:
        return "2C"
    if "random" in lower or "random_walk" in lower:
        return "random_walk"
    if "geo" in lower:
        return "GEO"
    return ""


def infer_numbered_batch_protocol(text: str) -> Dict[str, str]:
    """Fallback for the xjtu_batch134 replay-profile numbering convention.

    0001-0008  -> Batch-1 / 2C
    0009-0016  -> Batch-3 / R2.5
    0017-0024  -> Batch-4 / R3
    """
    m = re.search(r"(?<!\d)(00[0-2][0-9]|000[1-9])(?!\d)", text)
    if not m:
        return {"batch": "", "protocol": ""}
    idx = int(m.group(1))
    if 1 <= idx <= 8:
        return {"batch": "Batch-1", "protocol": "2C"}
    if 9 <= idx <= 16:
        return {"batch": "Batch-3", "protocol": "R2.5"}
    if 17 <= idx <= 24:
        return {"batch": "Batch-4", "protocol": "R3"}
    return {"batch": "", "protocol": ""}


def infer_batch_protocol_cell(path: Path, meta: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    meta = meta or {}
    text = " ".join([
        str(path).replace("\\", "/"),
        path.parent.name,
        meta.get("npz_batch", ""),
        meta.get("npz_protocol", ""),
        meta.get("npz_cell_uid", ""),
        meta.get("npz_source_file", ""),
    ])

    npz_batch = normalize_batch_token(meta.get("npz_batch", ""))
    npz_protocol = meta.get("npz_protocol", "").strip()
    explicit_batch = infer_explicit_batch(text)
    protocol = npz_protocol or infer_protocol(text)
    numbered = infer_numbered_batch_protocol(text)

    # Strict protocol->batch rules for XJTU Batch-1/3/4.
    # This deliberately overrides accidental "batch134 -> Batch-1".
    if protocol == "R2.5":
        batch = "Batch-3"
    elif protocol == "R3":
        batch = "Batch-4"
    elif protocol == "2C":
        # Keep explicit Batch-5/6 if source says so, otherwise 2C in batch134 means Batch-1.
        batch = explicit_batch if explicit_batch in {"Batch-5", "Batch-6"} else "Batch-1"
    elif protocol == "random_walk":
        batch = explicit_batch or "Batch-5"
    elif protocol == "GEO":
        batch = explicit_batch or "Batch-6"
    else:
        batch = explicit_batch or npz_batch or numbered.get("batch", "")
        protocol = protocol or numbered.get("protocol", "")

    if not protocol:
        protocol = {
            "Batch-1": "2C",
            "Batch-3": "R2.5",
            "Batch-4": "R3",
            "Batch-5": "random_walk",
            "Batch-6": "GEO",
        }.get(batch, "")

    cell_uid = meta.get("npz_cell_uid", "").strip() or path.parent.name
    return {
        "batch": batch,
        "protocol": protocol,
        "cell_uid": cell_uid,
        "metadata_inference_text": text[:700],
        "explicit_batch_detected": explicit_batch,
        "protocol_detected": protocol,
        "numbered_batch_fallback": numbered.get("batch", ""),
        "numbered_protocol_fallback": numbered.get("protocol", ""),
    }


def is_batch1_battery8_outlier(path_text: str, inferred: dict, exact_patterns: List[str]) -> bool:
    combined = (path_text + " " + " ".join(str(v) for v in inferred.values())).replace("\\", "/").lower()
    for pattern in exact_patterns:
        if pattern.lower() in combined:
            return True
    has_b1 = inferred.get("batch") == "Batch-1" or "batch-1" in combined or "batch_1" in combined or "b1_2c" in combined
    has_2c = inferred.get("protocol") == "2C" or "2c" in combined
    has_battery8 = ("battery-8" in combined or "battery_8" in combined or "battery8" in combined)
    has_other_batch = inferred.get("batch") in {"Batch-3", "Batch-4", "Batch-5", "Batch-6"}
    return bool(has_b1 and has_2c and has_battery8 and not has_other_batch)


def discover_profile_paths(configured_dirs: List[str], cache_root: Path, cfg: dict) -> List[Path]:
    skip = cfg.get("skip_scan_fragments", [])
    candidates: List[Path] = []
    for d in configured_dirs:
        root = Path(d)
        if root.exists():
            for p in root.rglob("solution_replay_profile.npz"):
                if not should_skip_path(p, skip):
                    candidates.append(p)

    if cfg.get("fallback_scan_cache_root", True) and cache_root.exists():
        patterns = cfg.get("fallback_scan_patterns", [])
        for p in cache_root.rglob("solution_replay_profile.npz"):
            text = str(p).replace("\\", "/")
            if should_skip_path(p, skip):
                continue
            if patterns and not any(pattern.lower() in text.lower() for pattern in patterns):
                continue
            candidates.append(p)

    seen = set()
    out = []
    for p in candidates:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def discover_profiles(profile_dirs: List[str], cache_root: Path, cfg: dict) -> List[dict]:
    exclude_patterns = cfg.get("selection", {}).get("exclude_exact_outliers", ["Batch-1_2C_battery-8", "B1_2C_battery-8"])
    paths = discover_profile_paths(profile_dirs, cache_root, cfg)
    rows: List[dict] = []
    for p in paths:
        meta = npz_metadata_shallow(p)
        inferred = infer_batch_protocol_cell(p, meta)
        outlier = is_batch1_battery8_outlier(str(p), inferred, exclude_patterns)
        try:
            size_mb = round(p.stat().st_size / (1024 * 1024), 3)
        except Exception:
            size_mb = ""
        rows.append({
            "profile_npz": str(p),
            "batch": inferred.get("batch", ""),
            "protocol": inferred.get("protocol", ""),
            "cell_uid": inferred.get("cell_uid", p.parent.name),
            "size_mb": size_mb,
            "excluded_by_policy": outlier,
            "exclude_reason": "Batch-1_2C_battery-8_flagged_outlier" if outlier else "",
            "npz_batch": meta.get("npz_batch", ""),
            "npz_protocol": meta.get("npz_protocol", ""),
            "npz_cell_uid": meta.get("npz_cell_uid", ""),
            "npz_source_file": meta.get("npz_source_file", ""),
            "explicit_batch_detected": inferred.get("explicit_batch_detected", ""),
            "protocol_detected": inferred.get("protocol_detected", ""),
            "numbered_batch_fallback": inferred.get("numbered_batch_fallback", ""),
            "metadata_inference_text": inferred.get("metadata_inference_text", ""),
        })
    rows.sort(key=lambda r: (str(r.get("batch", "")), float(r.get("size_mb") or 0), str(r.get("cell_uid", "")), str(r.get("profile_npz", ""))))
    return rows


def select_balanced(rows: List[dict], counts_by_batch: Dict[str, int]) -> List[dict]:
    selected: List[dict] = []
    used = set()
    for batch, count in counts_by_batch.items():
        subset = [r for r in rows if r.get("batch") == batch and not r.get("excluded_by_policy")]
        subset.sort(key=lambda r: (float(r.get("size_mb") or 0), str(r.get("cell_uid", "")), str(r.get("profile_npz", ""))))
        for r in subset[: int(count)]:
            key = r["profile_npz"]
            if key not in used:
                rr = dict(r)
                rr["selection_reason"] = f"balanced_{batch}_quota"
                selected.append(rr)
                used.add(key)
    return selected


def voltage_bounds_from_cfg(cfg: dict) -> Dict[str, float]:
    pc = cfg.get("pass_criteria", {})
    return {
        "upper_warn_V": float(pc.get("voltage_upper_warn_V", 4.25)),
        "upper_fail_V": float(pc.get("voltage_upper_fail_V", 4.35)),
        "lower_warn_V": float(pc.get("voltage_lower_warn_V", 2.45)),
        "lower_fail_V": float(pc.get("voltage_lower_fail_V", 2.35)),
    }


def source_voltage_audit(soft: Dict[str, Any], cfg: dict) -> Dict[str, Any]:
    audit_cfg = cfg.get("source_voltage_audit", {})
    V = np.asarray(soft["voltage_exp"], dtype=float)
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
    if hard_fail:
        status = "FAIL"
    elif warn:
        status = "WARN"
    else:
        status = "PASS"
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


def md_table(rows: List[dict], cols: List[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def aggregate_by_batch_protocol(audit_rows: List[dict], manifest_rows: List[dict]) -> List[dict]:
    by_npz = {r.get("npz_path", ""): r for r in audit_rows}
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for m in manifest_rows:
        a = by_npz.get(m.get("softlabel_npz", ""), {})
        key = (m.get("batch", ""), m.get("protocol", ""))
        row = dict(m)
        row.update({f"audit_{k}": v for k, v in a.items() if k not in row})
        groups[key].append(row)

    out = []
    for (batch, protocol), rows in sorted(groups.items()):
        maes = [float(r.get("audit_phis_c_vs_voltage_mae_V", r.get("phis_c_vs_voltage_mae_V", 0)) or 0) for r in rows]
        corrs = [float(r.get("audit_phis_c_vs_voltage_corr", r.get("phis_c_vs_voltage_corr", 0)) or 0) for r in rows]
        soft_max = [float(r.get("audit_phis_c_soft_max_V", r.get("phis_c_soft_max_V", 0)) or 0) for r in rows]
        bound_corr = [float(r.get("audit_max_abs_voltage_bound_correction_V", r.get("max_abs_voltage_bound_correction_V", 0)) or 0) for r in rows]
        out.append({
            "batch": batch,
            "protocol": protocol,
            "profile_count": len(rows),
            "pass_count": sum(1 for r in rows if r.get("profile_ok") is True),
            "source_voltage_warn_or_fail_count": sum(1 for r in rows if str(r.get("source_voltage_status", "")).upper() in {"WARN", "FAIL"}),
            "mean_phis_c_vs_voltage_mae_V": float(np.mean(maes)) if maes else "",
            "mean_phis_c_vs_voltage_corr": float(np.mean(corrs)) if corrs else "",
            "max_phis_c_soft_max_V": float(np.max(soft_max)) if soft_max else "",
            "max_abs_voltage_bound_correction_V": float(np.max(bound_corr)) if bound_corr else "",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--prior_file", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_profile_dirs", nargs="*", default=None)
    parser.add_argument("--max_points_per_profile", type=int, default=100000)
    parser.add_argument("--n_r", type=int, default=17)
    parser.add_argument("--allow_warn", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config) if args.config else project_root / "configs" / "d14_p4b_xjtu_p2dlite_multicell_softlabel_config.json"
    cfg = read_json(cfg_path) or {}
    profile_dirs = args.input_profile_dirs if args.input_profile_dirs else cfg.get("input_profile_dirs", [])
    selection_cfg = cfg.get("selection", {})
    counts_by_batch = selection_cfg.get("counts_by_batch", {"Batch-1": 2, "Batch-3": 2, "Batch-4": 2, "Batch-5": 1, "Batch-6": 1})
    required_min_by_batch = selection_cfg.get("required_min_by_batch", {"Batch-1": 2, "Batch-3": 2, "Batch-4": 2})
    required_keys = cfg.get("pass_criteria", {}).get("required_npz_keys", [])
    voltage_bounds = voltage_bounds_from_cfg(cfg)
    require_metadata = bool(cfg.get("pass_criteria", {}).get("metadata_required", True))

    print(f"[D14-P4B-v3] start={utc_now()}", flush=True)
    print(f"[D14-P4B-v3] prior_file={args.prior_file}", flush=True)
    print(f"[D14-P4B-v3] output_dir={output_dir}", flush=True)

    prior = load_prior(args.prior_file)
    resolved = build_resolved_spec(prior, n_r_override=args.n_r)
    write_json(output_dir / "D14_P4B_PRIOR_RESOLVED.json", resolved)
    (output_dir / "D14_P4B_PRIOR_HASH.txt").write_text(resolved["prior_hash"] + "\n", encoding="utf-8")

    discovered = discover_profiles(profile_dirs, cache_root, cfg)
    selected = select_balanced(discovered, counts_by_batch)

    write_csv(output_dir / "D14_P4B_DISCOVERED_PROFILES.csv", discovered)
    write_csv(output_dir / "D14_P4B_SELECTED_PROFILES.csv", selected)

    manifest: List[dict] = []
    audit_rows: List[dict] = []

    for i, row in enumerate(selected, 1):
        print(f"[D14-P4B-v3] {i}/{len(selected)} generate {row.get('batch')} {row.get('cell_uid')}", flush=True)
        try:
            profile = load_profile_npz(row["profile_npz"], args.max_points_per_profile, cfg)
            # Override old P4A loader inference if the v3 row metadata is more precise.
            profile["batch"] = row.get("batch", profile.get("batch", ""))
            profile["protocol"] = row.get("protocol", profile.get("protocol", ""))
            profile["cell_uid"] = row.get("cell_uid", profile.get("cell_uid", ""))

            soft = generate_softlabels(profile, resolved)
            src_audit = source_voltage_audit(soft, cfg)

            summary = {
                "created_utc": utc_now(),
                "source_profile_npz": row["profile_npz"],
                "cell_uid": safe_string(soft.get("cell_uid", "")),
                "batch": safe_string(soft.get("batch", "")),
                "protocol": safe_string(soft.get("protocol", "")),
                "n_points": int(len(soft["t_global_s"])),
                "n_r": int(args.n_r),
                "prior_file": str(Path(args.prior_file)),
                "resolved_spec_hash": resolved["prior_hash"],
                "state_label_interpretation": prior.get("interpretation", {}).get("state_labels", "model-consistent"),
                "soh_generated": False,
                "full_p2d_truth_claim": False,
                "stage": "D14-P4B-v3 batch/protocol final fix + source voltage audit",
                "source_voltage_audit": src_audit,
            }

            npz_path = save_softlabels(output_dir / "profiles", soft, summary)
            audit = audit_softlabel_npz(
                npz_path,
                required_keys,
                prior_hash=resolved["prior_hash"],
                voltage_bounds=voltage_bounds,
                require_metadata=require_metadata,
            )
            audit.update(src_audit)
            if audit.get("status") == "PASS" and src_audit["source_voltage_status"] == "WARN":
                audit["status"] = "WARN"
                audit["detail"] = "source_voltage_warn_bound"
            elif src_audit["source_voltage_status"] == "FAIL":
                audit["status"] = "FAIL"
                audit["detail"] = "source_voltage_fail_bound"
            write_audit_json(npz_path, audit)

            manifest.append({
                "profile_ok": audit.get("status") in {"PASS", "WARN"},
                "audit_status": audit.get("status", ""),
                "audit_detail": audit.get("detail", ""),
                "source_profile_npz": row["profile_npz"],
                "softlabel_npz": str(npz_path),
                "batch": summary["batch"],
                "protocol": summary["protocol"],
                "cell_uid": summary["cell_uid"],
                "n_points": summary["n_points"],
                "n_r": summary["n_r"],
                "phis_c_soft_max_V": audit.get("phis_c_soft_max_V", ""),
                "phis_c_soft_min_V": audit.get("phis_c_soft_min_V", ""),
                "phis_c_vs_voltage_mae_V": audit.get("phis_c_vs_voltage_mae_V", ""),
                "phis_c_vs_voltage_corr": audit.get("phis_c_vs_voltage_corr", ""),
                "max_abs_voltage_bound_correction_V": audit.get("max_abs_voltage_bound_correction_V", ""),
                "source_voltage_status": src_audit["source_voltage_status"],
                "voltage_exp_max_V": src_audit["voltage_exp_max_V"],
                "voltage_exp_upper_fail_count": src_audit["voltage_exp_upper_fail_count"],
                "resolved_spec_hash": resolved["prior_hash"],
            })
            audit_rows.append(audit)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            manifest.append({
                "profile_ok": False,
                "audit_status": "FAIL",
                "audit_detail": err,
                "source_profile_npz": row["profile_npz"],
                "softlabel_npz": "",
                "batch": row.get("batch", ""),
                "protocol": row.get("protocol", ""),
                "cell_uid": row.get("cell_uid", ""),
                "n_points": "",
                "n_r": args.n_r,
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

    write_csv(output_dir / "D14_P4B_SOFTLABEL_MANIFEST.csv", manifest)
    write_csv(output_dir / "D14_P4B_SOFTLABEL_AUDIT.csv", audit_rows)

    by_group = aggregate_by_batch_protocol(audit_rows, manifest)
    write_csv(output_dir / "D14_P4B_BY_BATCH_PROTOCOL.csv", by_group)

    selected_by_batch = defaultdict(int)
    discovered_by_batch = defaultdict(int)
    for r in discovered:
        if not r.get("excluded_by_policy"):
            discovered_by_batch[r.get("batch", "")] += 1
    for r in selected:
        selected_by_batch[r.get("batch", "")] += 1

    ok_count = sum(1 for r in manifest if r.get("profile_ok") is True)
    fail_count = sum(1 for r in manifest if r.get("profile_ok") is not True)
    warn_count = sum(1 for r in manifest if str(r.get("audit_status", "")).upper() == "WARN")
    min_profiles = int(cfg.get("pass_criteria", {}).get("min_total_profiles", 6))

    required_batch_failures = []
    for batch, min_n in required_min_by_batch.items():
        if selected_by_batch.get(batch, 0) < int(min_n):
            required_batch_failures.append(f"{batch}:{selected_by_batch.get(batch,0)}/{min_n}")

    checks = []
    checks.append({
        "check_id": "P4B-C00",
        "name": "standalone prior loaded",
        "status": "PASS",
        "detail": f"prior_hash={resolved['prior_hash']}",
    })
    checks.append({
        "check_id": "P4B-C01",
        "name": "profile discovery with batch134-safe inference",
        "status": "PASS" if discovered else "FAIL",
        "detail": f"discovered={len(discovered)} excluded={sum(1 for r in discovered if r.get('excluded_by_policy'))} by_batch={dict(discovered_by_batch)}",
    })
    checks.append({
        "check_id": "P4B-C02",
        "name": "required Batch-1/3/4 coverage",
        "status": "FAIL" if required_batch_failures else "PASS",
        "detail": f"selected_by_batch={dict(selected_by_batch)} required_failures={required_batch_failures}",
    })
    checks.append({
        "check_id": "P4B-C03",
        "name": "controlled total selection",
        "status": "PASS" if len(selected) >= min_profiles else "FAIL",
        "detail": f"selected={len(selected)} min_required={min_profiles} quotas={counts_by_batch}",
    })
    checks.append({
        "check_id": "P4B-C04",
        "name": "soft-label generation",
        "status": "PASS" if ok_count == len(selected) and ok_count >= min_profiles else "FAIL",
        "detail": f"profile_ok={ok_count}/{len(selected)} warnings={warn_count}",
    })
    src_warn = sum(1 for r in manifest if str(r.get("source_voltage_status", "")).upper() == "WARN")
    src_fail = sum(1 for r in manifest if str(r.get("source_voltage_status", "")).upper() == "FAIL")
    checks.append({
        "check_id": "P4B-C05",
        "name": "source voltage_exp audit",
        "status": "FAIL" if src_fail else ("WARN" if src_warn else "PASS"),
        "detail": f"source_voltage_WARN={src_warn} source_voltage_FAIL={src_fail}",
    })
    checks.append({
        "check_id": "P4B-C06",
        "name": "SOH-free boundary",
        "status": "PASS",
        "detail": "No SOH label is generated by P2Dlite voltage soft-label generator.",
    })
    checks.append({
        "check_id": "P4B-C07",
        "name": "Batch-1_2C_battery-8 policy",
        "status": "PASS" if not any(("battery-8" in str(r.get("cell_uid", "")).lower() and r.get("batch") == "Batch-1") for r in selected) else "FAIL",
        "detail": "Only Batch-1_2C_battery-8 is excluded; other battery-8 profiles are not automatically excluded.",
    })

    overall = combine_status(checks)
    if overall == "FAIL":
        recommendation = "Do not expand to full soft-label generation. Inspect required batch coverage, discovery rows, and source-voltage audit."
    elif overall == "WARN":
        recommendation = "P4B-v3 multi-profile generation is usable as a controlled warning-level result; review source-voltage WARN rows before P4C."
    else:
        recommendation = "P4B-v3 controlled multi-profile P2Dlite soft-label expansion passed. Next step can prepare P4C full non-outlier manifest."

    report = {
        "package": "D14-P4B-v3 XJTU P2Dlite batch/protocol final fix + source voltage audit",
        "created_utc": utc_now(),
        "overall_status": overall,
        "recommendation": recommendation,
        "paths": {
            "project_root": str(project_root),
            "cache_root": str(cache_root),
            "prior_file": str(Path(args.prior_file)),
            "config": str(cfg_path),
            "output_dir": str(output_dir),
        },
        "summary": {
            "discovered_profiles": len(discovered),
            "excluded_profiles": sum(1 for r in discovered if r.get("excluded_by_policy")),
            "selected_profiles": len(selected),
            "generated_profiles": ok_count,
            "failed_profiles": fail_count,
            "warning_profiles": warn_count,
            "n_r": args.n_r,
            "max_points_per_profile": args.max_points_per_profile,
            "prior_hash": resolved["prior_hash"],
            "discovered_by_batch": dict(discovered_by_batch),
            "selected_by_batch": dict(selected_by_batch),
        },
        "checks": checks,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(output_dir / "D14_P4B_SOFTLABEL_MULTIPROFILE_REPORT.json", report)

    md = []
    md.append("# D14-P4B-v3 XJTU P2Dlite Controlled Multi-profile Soft-label Report\n")
    md.append(f"Created UTC: `{report['created_utc']}`\n")
    md.append(f"Overall status: **{overall}**\n")
    md.append(f"Recommendation: {recommendation}\n")
    md.append("## Checks\n")
    md.append(md_table(checks, ["check_id", "name", "status", "detail"]))
    md.append("\n## Selected profiles\n")
    md.append(md_table(manifest, ["profile_ok", "audit_status", "batch", "protocol", "cell_uid", "n_points", "phis_c_vs_voltage_mae_V", "phis_c_vs_voltage_corr", "phis_c_soft_max_V", "source_voltage_status", "voltage_exp_max_V"]))
    md.append("\n## By batch/protocol\n")
    md.append(md_table(by_group, ["batch", "protocol", "profile_count", "mean_phis_c_vs_voltage_mae_V", "mean_phis_c_vs_voltage_corr", "max_phis_c_soft_max_V", "source_voltage_warn_or_fail_count"]))
    md.append("\n## Boundary\n")
    md.append("- No training.\n")
    md.append("- No SOH generation.\n")
    md.append("- Uses standalone P2Dlite prior file.\n")
    md.append("- Outputs remain model-consistent soft labels, not full-P2D ground truth.\n")
    (output_dir / "D14_P4B_SOFTLABEL_MULTIPROFILE_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    readme_patch = f"""# README D14-P4B-v3 Patch

D14-P4B-v3 fixes batch/protocol inference and source voltage audit.

Status: **{overall}**

Recommendation: {recommendation}

Prior file:

```text
configs/P2Dlite_prior_xjtu_lr18650la_v0.json
```

Boundary:

- No training.
- No GV1 mainline training-code modification.
- No SOH generation inside the voltage soft-label generator.
- No full-P2D internal-state ground-truth claim.
- Batch-1_2C_battery-8 remains flagged/excluded.
"""
    (output_dir / "README_D14_P4B_PATCH.md").write_text(readme_patch, encoding="utf-8")

    outputs = [
        "D14_P4B_SOFTLABEL_MULTIPROFILE_REPORT.json",
        "D14_P4B_SOFTLABEL_MULTIPROFILE_REPORT.md",
        "D14_P4B_DISCOVERED_PROFILES.csv",
        "D14_P4B_SELECTED_PROFILES.csv",
        "D14_P4B_SOFTLABEL_MANIFEST.csv",
        "D14_P4B_SOFTLABEL_AUDIT.csv",
        "D14_P4B_BY_BATCH_PROTOCOL.csv",
        "D14_P4B_PRIOR_RESOLVED.json",
        "D14_P4B_PRIOR_HASH.txt",
        "D14_P4B_RUN_SUMMARY.txt",
        "README_D14_P4B_PATCH.md",
        "D14_P4B_OUTPUT_INDEX.json",
    ]

    # Write summary before index, then write index last.
    (output_dir / "D14_P4B_RUN_SUMMARY.txt").write_text(
        "\n".join([
            "D14-P4B-v3 XJTU P2Dlite controlled multi-profile soft-label expansion",
            f"created_utc={report['created_utc']}",
            f"overall_status={overall}",
            f"prior_hash={resolved['prior_hash']}",
            f"discovered_profiles={len(discovered)}",
            f"selected_profiles={len(selected)}",
            f"generated_profiles={ok_count}",
            f"failed_profiles={fail_count}",
            f"warning_profiles={warn_count}",
            f"n_r={args.n_r}",
            f"recommendation={recommendation}",
        ]) + "\n",
        encoding="utf-8",
    )
    write_json(output_dir / "D14_P4B_OUTPUT_INDEX.json", {
        "overall_status": overall,
        "output_dir": str(output_dir),
        "files": [
            {"name": name, "exists": True if name == "D14_P4B_OUTPUT_INDEX.json" else (output_dir / name).exists()}
            for name in outputs
        ],
    })

    print(f"[D14-P4B-v3] overall_status={overall}", flush=True)
    print(f"[D14-P4B-v3] recommendation={recommendation}", flush=True)
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else (2 if overall == "WARN" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
