#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D14-P0 freeze/mainline audit for QJW-2 / PINN-for-ASSB-V1.

Purpose
-------
This script does NOT train a model and does NOT generate new soft labels.
It audits that the current local repository and XJTU cache are still aligned
with the accepted D9.6/D9.5.1 + D12-S1K non-outlier voltage mainline and the
ASSB ModelFin_112 deterministic wrapper baseline.

Main checks
-----------
1. Required GV1 mainline source files exist.
2. The core GV1 files are not obviously overwritten by failed branches such as
   hard clamp / metadata_on default / high-safe component guard experiments.
3. ASSB ModelFin_112 wrapper artifacts are present and, if available, their
   audit JSON does not report failure.
4. D10-P1 non-outlier 23-profile 200ks output exists and does not include
   unflagged battery-8 rows.
5. D12-S1K 23x200ks scorecard exists and contains the expected candidates.
6. A sha256 fingerprint snapshot is written for no-regression comparison.

The script is intentionally dependency-light: Python stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}

# D14-P0 battery-8 policy is about the known flagged profile:
# Batch-1 / 2C / battery-8.  Do NOT match generic "battery-8", because
# Batch-3_battery-8 and Batch-4_battery-8 are normal non-outlier cells.
# Also avoid treating folder names such as "exclude_battery8" as data rows.
BATTERY8_PAT = re.compile(
    r"("
    r"batch[-_ ]?1(?:[-_ ]?2c)?(?:.*?battery[-_ ]?8)|"
    r"b1(?:[-_ ]?2c)?(?:.*?battery[-_ ]?8)|"
    r"batch[-_ ]?1[-_ ]?battery[-_ ]?8|"
    r"batch[-_ ]?1[-_ ]?2c[-_ ]?battery[-_ ]?8"
    r")",
    re.IGNORECASE,
)

PATHLIKE_FIELD_PAT = re.compile(r"(path|dir|root|folder|file|csv|json|npz|log|output)", re.IGNORECASE)

def is_pathlike_field(field_name: str) -> bool:
    return bool(PATHLIKE_FIELD_PAT.search(str(field_name or "")))

def row_identity_text_for_battery8_scan(row: Dict[str, Any], fields: List[str]) -> str:
    """Return only profile/cell identity-like text for the flagged B1 battery-8 scan.

    The first D14-P0 audit used all CSV row values, which produced false positives:
    the D10 folder is named "...exclude_battery8", and D12 legitimately contains
    Batch-3_battery-8 / Batch-4_battery-8 profiles.  This function intentionally
    ignores path-like columns and scans only identity columns where available.
    """
    identity_fields: List[str] = []
    for f in fields:
        lf = str(f).lower()
        if is_pathlike_field(f):
            continue
        if (
            lf in {"profile", "profile_id", "profile_name", "cell", "cell_id", "cell_uid", "run", "run_id", "profile_uid"}
            or "profile" in lf
            or "cell" in lf
            or "battery" in lf
            or "batch" in lf
        ):
            identity_fields.append(f)
    if not identity_fields:
        identity_fields = [f for f in fields if not is_pathlike_field(f)]
    return " ".join(str(row.get(f, "")) for f in identity_fields)

EXPECTED_CORE_FILES = [
    "gv1/model.py",
    "gv1/output_transform.py",
    "gv1/profile_adaptive.py",
    "gv1/losses.py",
    "gv1/trainer.py",
    "scripts/gv1_train_conditioned_pinn.py",
]

EXPECTED_ASSB_FILES = [
    "main.py",
    "README.md",
    "util/thermo_assb.py",
    "util/_losses.py",
    "util/_rescale.py",
    "util/init_pinn.py",
    "util/spm_assb_train_discharge.py",
    "integration_spm/spm_int_assb_cycle.py",
]

DEFAULT_D10_P1_DIRNAME = "xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8"
DEFAULT_D12_S1K_DIRNAME = "xjtu_batch134_d12_s1k_two_candidate_23x200ks_scorecard"

EXPECTED_S1K_CANDIDATES = [
    "baseline",
    "low_only_revert_nonlow_to_baseline",
    "low_plus_transition_fade_to_baseline",
]


@dataclasses.dataclass
class Check:
    check_id: str
    title: str
    status: str
    message: str
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_path_str(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_text_best_effort(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        raw = path.read_bytes()[:max_bytes]
    except Exception:
        return ""
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("latin-1", errors="replace")


def run_git(project_root: Path, args: List[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=str(project_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
        return None
    except Exception:
        return None


def json_load_best_effort(path: Path) -> Optional[Any]:
    try:
        return json.loads(read_text_best_effort(path))
    except Exception:
        return None


def flatten_json(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_json(v, key))
    elif isinstance(obj, list):
        # Keep lists compact; flatten first several dict-like objects only.
        out[prefix or "list"] = f"list[{len(obj)}]"
        for i, v in enumerate(obj[:10]):
            out.update(flatten_json(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    else:
        out[prefix] = obj
    return out


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        return float(s)
    except Exception:
        return None


def find_files(root: Path, patterns: Iterable[str], max_files: int = 5000) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file():
                out.append(p)
                if len(out) >= max_files:
                    return out
    return out


def read_csv_rows(path: Path, max_rows: int = 200_000) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for i, row in enumerate(reader):
                rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})
                if i + 1 >= max_rows:
                    break
    except UnicodeDecodeError:
        try:
            with path.open("r", encoding="gbk", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                for i, row in enumerate(reader):
                    rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})
                    if i + 1 >= max_rows:
                        break
        except Exception:
            return [], []
    except Exception:
        return [], []
    return rows, fieldnames


def extract_numeric_summary_from_json_files(root: Path) -> Dict[str, Any]:
    files = find_files(root, ["*.json"], max_files=1000)
    keys_of_interest = [
        "pass", "pass_count", "ok_count", "fail", "fail_count", "borderline", "borderline_count",
        "read_error", "read_error_count", "mean_mae", "mean_MAE", "mae", "MAE", "mean_corr", "corr",
        "metrics_ok", "run_count", "profile_count", "candidate", "mode", "verdict",
    ]
    summary: Dict[str, Any] = {"json_file_count": len(files), "files": []}
    for p in files[:200]:
        obj = json_load_best_effort(p)
        if obj is None:
            continue
        flat = flatten_json(obj)
        hit = {}
        for k, v in flat.items():
            kl = k.lower()
            if any(t.lower() in kl for t in keys_of_interest):
                if isinstance(v, (str, int, float, bool)) or v is None:
                    hit[k] = v
        text = read_text_best_effort(p, max_bytes=500_000)
        if BATTERY8_PAT.search(text):
            hit["contains_battery8_text"] = True
        if hit:
            summary["files"].append({"path": str(p), "hits": hit})
    return summary


def collect_scorecard_csv_index(roots: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in find_files(root, ["*.csv"], max_files=2000):
            rows, fields = read_csv_rows(p, max_rows=300_000)
            if not rows:
                continue
            lower_fields = [f.lower() for f in fields]
            likely = any("mae" in f or "corr" in f or "candidate" in f or "mode" in f or "status" in f for f in lower_fields)
            if not likely:
                continue
            candidate_field = next((f for f in fields if f.lower() in {"candidate", "mode", "variant", "method"}), None)
            status_field = next((f for f in fields if f.lower() in {"status", "run_status", "verdict", "ok", "metrics_ok"}), None)
            profile_field = next((f for f in fields if f.lower() in {"profile", "profile_id", "cell", "cell_id", "cell_uid", "source", "profile_name"}), None)
            mae_field = next((f for f in fields if f.lower() in {"mean_mae", "mae", "mae_v", "global_mae"}), None)
            corr_field = next((f for f in fields if f.lower() in {"mean_corr", "corr", "global_corr"}), None)

            candidates: Dict[str, Dict[str, float]] = {}
            battery8_rows = 0
            unflagged_b8_rows = 0
            for row in rows:
                joined = row_identity_text_for_battery8_scan(row, fields)
                if BATTERY8_PAT.search(joined):
                    battery8_rows += 1
                    status_txt = str(row.get(status_field or "", "")).lower()
                    if not any(w in status_txt for w in ["flag", "exclude", "outlier", "stress", "skip"]):
                        unflagged_b8_rows += 1
                cand = str(row.get(candidate_field or "", "")).strip() or "<none>"
                mae = as_float(row.get(mae_field or "")) if mae_field else None
                corr = as_float(row.get(corr_field or "")) if corr_field else None
                if cand not in candidates:
                    candidates[cand] = {"n": 0, "mae_sum": 0.0, "mae_n": 0, "corr_sum": 0.0, "corr_n": 0}
                candidates[cand]["n"] += 1
                if mae is not None:
                    candidates[cand]["mae_sum"] += mae
                    candidates[cand]["mae_n"] += 1
                if corr is not None:
                    candidates[cand]["corr_sum"] += corr
                    candidates[cand]["corr_n"] += 1

            cand_summ = []
            for cand, d in candidates.items():
                cand_summ.append({
                    "candidate": cand,
                    "n": int(d["n"]),
                    "mean_mae_from_csv": (d["mae_sum"] / d["mae_n"]) if d["mae_n"] else None,
                    "mean_corr_from_csv": (d["corr_sum"] / d["corr_n"]) if d["corr_n"] else None,
                })
            records.append({
                "root": str(root),
                "csv_path": str(p),
                "row_count_scanned": len(rows),
                "field_count": len(fields),
                "candidate_field": candidate_field,
                "status_field": status_field,
                "profile_field": profile_field,
                "mae_field": mae_field,
                "corr_field": corr_field,
                "battery8_rows": battery8_rows,
                "unflagged_battery8_rows": unflagged_b8_rows,
                "candidate_summary": cand_summ,
            })
    return records


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        field_set = []
        for r in rows:
            for k in r.keys():
                if k not in field_set:
                    field_set.append(k)
        fieldnames = field_set
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def fingerprint_files(project_root: Path, relative_files: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rel in relative_files:
        p = project_root / rel
        if p.exists() and p.is_file():
            rows.append({
                "relative_path": rel,
                "exists": True,
                "size_bytes": p.stat().st_size,
                "mtime_iso": _dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "sha256": sha256_file(p),
            })
        else:
            rows.append({"relative_path": rel, "exists": False, "size_bytes": "", "mtime_iso": "", "sha256": ""})
    return rows


def load_baseline_fingerprint(path: Path) -> Dict[str, str]:
    obj = json_load_best_effort(path)
    if not isinstance(obj, dict):
        return {}
    if "files" in obj and isinstance(obj["files"], list):
        return {str(x.get("relative_path")): str(x.get("sha256")) for x in obj["files"] if x.get("relative_path")}
    return {str(k): str(v) for k, v in obj.items()}


def grep_patterns(project_root: Path, rel_files: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    pattern_specs = {
        "hard_clamp_enabled": [
            re.compile(r"enable_voltage_hard_clamp\s*=\s*True", re.IGNORECASE),
            re.compile(r"ENABLE_VOLTAGE_HARD_CLAMP\s*=\s*True", re.IGNORECASE),
        ],
        "metadata_on_default": [
            re.compile(r"metadata[_-]?mode[^\n]{0,80}default\s*=\s*[\'\"]on[\'\"]", re.IGNORECASE),
            re.compile(r"default_metadata[_-]?mode\s*=\s*[\'\"]on[\'\"]", re.IGNORECASE),
            re.compile(r"metadata_on\s*=\s*True", re.IGNORECASE),
        ],
        "failed_branch_markers": [
            re.compile(r"D9\.6\.1|D9_6_1|highsafe|high_safe", re.IGNORECASE),
            re.compile(r"component_guard|component\s+guard|enable_component_guard", re.IGNORECASE),
            re.compile(r"hard\s*clamp|voltage_hard_clamp", re.IGNORECASE),
        ],
        "s1k_wrapper_in_core": [
            re.compile(r"low_plus_transition_fade_to_baseline", re.IGNORECASE),
            re.compile(r"transition[_-]?fade", re.IGNORECASE),
            re.compile(r"low_only_revert_nonlow_to_baseline", re.IGNORECASE),
        ],
        "battery8_unflag_like": [
            re.compile(r"unflag\w*.*battery[-_ ]?8", re.IGNORECASE),
            re.compile(r"include\w*.*battery[-_ ]?8", re.IGNORECASE),
        ],
    }
    results: Dict[str, List[Dict[str, Any]]] = {k: [] for k in pattern_specs}
    for rel in rel_files:
        p = project_root / rel
        if not p.exists() or not p.is_file():
            continue
        text = read_text_best_effort(p)
        lines = text.splitlines()
        for pname, pats in pattern_specs.items():
            for idx, line in enumerate(lines, start=1):
                if any(pat.search(line) for pat in pats):
                    results[pname].append({
                        "file": rel,
                        "line": idx,
                        "text": line.strip()[:300],
                    })
    return results


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    checks = payload.get("checks", [])
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for c in checks:
        counts[c.get("status", "WARN")] = counts.get(c.get("status", "WARN"), 0) + 1
    lines: List[str] = []
    lines.append("# D14-P0 Freeze / Mainline Audit Report")
    lines.append("")
    lines.append(f"Generated: `{payload.get('generated_at')}`")
    lines.append(f"Project root: `{payload.get('project_root')}`")
    lines.append(f"Cache root: `{payload.get('cache_root')}`")
    lines.append(f"Overall status: **{payload.get('overall_status')}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- PASS: {counts.get('PASS', 0)}")
    lines.append(f"- WARN: {counts.get('WARN', 0)}")
    lines.append(f"- FAIL: {counts.get('FAIL', 0)}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for c in checks:
        lines.append(f"### {c.get('check_id')} — {c.get('title')}")
        lines.append("")
        lines.append(f"Status: **{c.get('status')}**")
        lines.append("")
        lines.append(c.get("message", ""))
        details = c.get("details") or {}
        if details:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(details, ensure_ascii=False, indent=2, default=str)[:12000])
            lines.append("```")
        lines.append("")
    lines.append("## Output files")
    lines.append("")
    for k, v in (payload.get("output_files") or {}).items():
        lines.append(f"- {k}: `{v}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D14-P0 freeze/mainline audit")
    ap.add_argument("--project-root", default=".", help="PINN-for-ASSB-V1 repository root")
    ap.add_argument("--cache-root", default=r"E:/XJTU battery dataset/_gv1_cache", help="XJTU GV1 cache root")
    ap.add_argument("--output-dir", default=None, help="Audit output dir; default cache_root/xjtu_d14_p0_freeze_audit")
    ap.add_argument("--d10-p1-dir", default=None, help="D10-P1 23x200ks excluding battery-8 directory")
    ap.add_argument("--d12-s1k-dir", default=None, help="D12-S1K 23x200ks scorecard directory")
    ap.add_argument("--baseline-fingerprint", default=None, help="Optional previous fingerprint JSON to compare against")
    ap.add_argument("--strict-cache", action="store_true", help="Treat missing expected D10/D12 cache dirs as FAIL instead of WARN")
    ap.add_argument("--strict-assb", action="store_true", help="Treat missing ASSB baseline wrapper as FAIL instead of WARN")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else (cache_root / "xjtu_d14_p0_freeze_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    d10_dir = Path(args.d10_p1_dir).expanduser() if args.d10_p1_dir else (cache_root / DEFAULT_D10_P1_DIRNAME)
    d12_dir = Path(args.d12_s1k_dir).expanduser() if args.d12_s1k_dir else (cache_root / DEFAULT_D12_S1K_DIRNAME)

    checks: List[Check] = []

    def add(check_id: str, title: str, status: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        checks.append(Check(check_id, title, status, message, details or {}))

    # C00 project existence
    if project_root.exists() and project_root.is_dir():
        add("C00", "project root exists", "PASS", "Project root is accessible.", {"project_root": str(project_root)})
    else:
        add("C00", "project root exists", "FAIL", "Project root is missing or not a directory.", {"project_root": str(project_root)})

    # Git info
    git_head = run_git(project_root, ["rev-parse", "HEAD"])
    git_status = run_git(project_root, ["status", "--short"])
    add(
        "C01",
        "git status snapshot",
        "PASS" if git_head else "WARN",
        "Git snapshot collected." if git_head else "Git information was not available; this is acceptable for a non-git copy but weakens audit traceability.",
        {"git_head": git_head, "git_status_short": git_status},
    )

    # Required files.
    missing_core = [rel for rel in EXPECTED_CORE_FILES if not (project_root / rel).is_file()]
    add(
        "C02",
        "GV1 D9.6/D9.5.1 mainline files exist",
        "FAIL" if missing_core else "PASS",
        "All required GV1 mainline source files are present." if not missing_core else "Some required GV1 mainline files are missing.",
        {"missing_core_files": missing_core, "expected_core_files": EXPECTED_CORE_FILES},
    )

    missing_assb = [rel for rel in EXPECTED_ASSB_FILES if not (project_root / rel).is_file()]
    add(
        "C03",
        "legacy ASSB files are visible for no-regression fingerprinting",
        "WARN" if missing_assb else "PASS",
        "ASSB legacy files are visible." if not missing_assb else "Some ASSB legacy files were not found; fingerprinting will skip them.",
        {"missing_assb_files": missing_assb},
    )

    # Source pattern audit.
    pattern_hits = grep_patterns(project_root, EXPECTED_CORE_FILES)
    hard_hits = pattern_hits.get("hard_clamp_enabled", [])
    meta_hits = pattern_hits.get("metadata_on_default", [])
    failed_hits = pattern_hits.get("failed_branch_markers", [])
    wrapper_hits = pattern_hits.get("s1k_wrapper_in_core", [])
    b8_unflag_hits = pattern_hits.get("battery8_unflag_like", [])

    add(
        "C04",
        "hard voltage clamp is not enabled in core GV1 files",
        "FAIL" if hard_hits else "PASS",
        "No enabled hard-clamp default was found." if not hard_hits else "Enabled hard-clamp markers were found in core files; D9.6/D9.5.1 mainline may be polluted.",
        {"hits": hard_hits[:50]},
    )
    add(
        "C05",
        "metadata_on is not the core training default",
        "FAIL" if meta_hits else "PASS",
        "No metadata_on default marker was found in core files." if not meta_hits else "metadata_on-like default markers were found; metadata_on should remain ablation/runtime-only, not mainline default.",
        {"hits": meta_hits[:50]},
    )
    # failed branch markers are WARN because code may contain comments or disabled branches.
    add(
        "C06",
        "failed-branch marker scan",
        "WARN" if failed_hits else "PASS",
        "No failed branch marker was found in core files." if not failed_hits else "Markers associated with failed branches were found. Inspect whether they are comments/disabled options or active defaults.",
        {"hits": failed_hits[:80]},
    )
    add(
        "C07",
        "D12-S1K wrapper is not silently baked into core training defaults",
        "WARN" if wrapper_hits else "PASS",
        "No S1K wrapper marker was found in core files." if not wrapper_hits else "S1K wrapper markers were found in core files. This is only acceptable if they are optional and disabled by default.",
        {"hits": wrapper_hits[:80]},
    )
    add(
        "C08",
        "battery-8 remains flagged/excluded, not unflagged in core code",
        "FAIL" if b8_unflag_hits else "PASS",
        "No unflag/include-battery-8 core marker was found." if not b8_unflag_hits else "Core code appears to unflag/include battery-8; this conflicts with D10-D13 policy.",
        {"hits": b8_unflag_hits[:50]},
    )

    # ASSB wrapper checks.
    wrapper_dir = project_root / "ModelFin_112_deterministic_wrapper"
    eval_dir = project_root / "EvalFin_112_deterministic_wrapper"
    assb_status = "PASS" if wrapper_dir.exists() and eval_dir.exists() else ("FAIL" if args.strict_assb else "WARN")
    assb_msg = "ModelFin_112 and EvalFin_112 wrapper directories are present." if assb_status == "PASS" else "ModelFin_112/EvalFin_112 wrapper directories are missing or incomplete."
    assb_details: Dict[str, Any] = {"model_dir": str(wrapper_dir), "model_dir_exists": wrapper_dir.exists(), "eval_dir": str(eval_dir), "eval_dir_exists": eval_dir.exists()}
    # Read audit JSON if available.
    for name in ["unified_eval_audit.json", "build_audit.json", "five_state_scorecard.json"]:
        for base in [eval_dir, wrapper_dir]:
            p = base / name
            if p.exists():
                assb_details[str(p)] = json_load_best_effort(p)
    add("C09", "ASSB ModelFin_112 deterministic wrapper baseline is present", assb_status, assb_msg, assb_details)

    # D10 and D12 directories.
    d10_exists = d10_dir.exists()
    add(
        "C10",
        "D10-P1 non-outlier 23-profile 200ks directory exists",
        "PASS" if d10_exists else ("FAIL" if args.strict_cache else "WARN"),
        "D10-P1 output directory exists." if d10_exists else "D10-P1 output directory was not found at the expected path.",
        {"d10_p1_dir": str(d10_dir), "exists": d10_exists},
    )
    d12_exists = d12_dir.exists()
    add(
        "C11",
        "D12-S1K 23x200ks scorecard directory exists",
        "PASS" if d12_exists else ("FAIL" if args.strict_cache else "WARN"),
        "D12-S1K output directory exists." if d12_exists else "D12-S1K scorecard directory was not found at the expected path.",
        {"d12_s1k_dir": str(d12_dir), "exists": d12_exists},
    )

    # Scorecard CSV index.
    scorecard_index = collect_scorecard_csv_index([d10_dir, d12_dir])
    scorecard_index_path = output_dir / "D14_P0_SCORECARD_INDEX.json"
    scorecard_index_path.write_text(json.dumps(scorecard_index, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # Flatten summary CSV.
    flat_rows: List[Dict[str, Any]] = []
    for item in scorecard_index:
        cand_summary = item.get("candidate_summary") or []
        if not cand_summary:
            flat_rows.append({k: item.get(k) for k in ["root", "csv_path", "row_count_scanned", "battery8_rows", "unflagged_battery8_rows"]})
        else:
            for c in cand_summary:
                flat_rows.append({
                    "root": item.get("root"),
                    "csv_path": item.get("csv_path"),
                    "row_count_scanned": item.get("row_count_scanned"),
                    "battery8_rows": item.get("battery8_rows"),
                    "unflagged_battery8_rows": item.get("unflagged_battery8_rows"),
                    "candidate": c.get("candidate"),
                    "candidate_n": c.get("n"),
                    "candidate_mean_mae_from_csv": c.get("mean_mae_from_csv"),
                    "candidate_mean_corr_from_csv": c.get("mean_corr_from_csv"),
                })
    scorecard_csv_path = output_dir / "D14_P0_SCORECARD_INDEX.csv"
    write_csv(scorecard_csv_path, flat_rows)

    unflagged_b8 = sum(int(item.get("unflagged_battery8_rows") or 0) for item in scorecard_index)
    add(
        "C12",
        "scorecard battery-8 policy scan",
        "FAIL" if unflagged_b8 > 0 else "PASS",
        "No unflagged battery-8 rows were found in scanned scorecard CSV files." if unflagged_b8 == 0 else "Unflagged battery-8 rows were found in scorecard CSV files.",
        {"unflagged_battery8_rows": unflagged_b8, "scorecard_index_json": str(scorecard_index_path), "scorecard_index_csv": str(scorecard_csv_path)},
    )

    # D10 summary details.
    d10_json_summary = extract_numeric_summary_from_json_files(d10_dir) if d10_exists else {"json_file_count": 0, "files": []}
    d10_contains_pass23 = False
    d10_contains_fail0 = False
    d10_text = json.dumps(d10_json_summary, ensure_ascii=False).lower()
    if re.search(r"pass[^0-9]{0,20}23", d10_text) or "23 pass" in d10_text:
        d10_contains_pass23 = True
    if re.search(r"fail[^0-9]{0,20}0", d10_text) or "0 fail" in d10_text:
        d10_contains_fail0 = True
    add(
        "C13",
        "D10-P1 summary indicates 23 non-outlier profiles passed",
        "PASS" if (not d10_exists or (d10_contains_pass23 and d10_contains_fail0)) else "WARN",
        "D10-P1 summary looks consistent with 23 pass / 0 fail." if (d10_contains_pass23 and d10_contains_fail0) else "Could not automatically confirm 23 pass / 0 fail from JSON summaries; inspect the scorecard manually.",
        {"d10_json_summary_sample": d10_json_summary.get("files", [])[:20]},
    )

    # D12 candidate summary.
    d12_text = ""
    if d12_exists:
        for p in find_files(d12_dir, ["*.json", "*.csv", "*.md", "*.txt"], max_files=2000):
            d12_text += "\n" + str(p) + "\n" + read_text_best_effort(p, max_bytes=300_000)
    found_candidates = [cand for cand in EXPECTED_S1K_CANDIDATES if cand.lower() in d12_text.lower()]
    add(
        "C14",
        "D12-S1K expected candidates are visible",
        "PASS" if (not d12_exists or len(found_candidates) >= 2) else "WARN",
        "Expected S1K candidates were found in D12 outputs." if len(found_candidates) >= 2 else "Could not automatically find expected S1K candidate names; inspect D12 scorecard files manually.",
        {"found_candidates": found_candidates, "expected_candidates": EXPECTED_S1K_CANDIDATES},
    )

    # Fingerprints.
    fingerprint_rels = []
    for rel in EXPECTED_CORE_FILES + EXPECTED_ASSB_FILES:
        if rel not in fingerprint_rels:
            fingerprint_rels.append(rel)
    fp_rows = fingerprint_files(project_root, fingerprint_rels)
    fp_csv = output_dir / "D14_P0_FILE_FINGERPRINTS.csv"
    write_csv(fp_csv, fp_rows)
    fp_json = output_dir / "D14_P0_BASELINE_FINGERPRINT.json"
    fp_payload = {
        "generated_at": now_iso(),
        "project_root": str(project_root),
        "git_head": git_head,
        "files": fp_rows,
    }
    fp_json.write_text(json.dumps(fp_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    add(
        "C15",
        "mainline fingerprints written",
        "PASS",
        "File fingerprints were generated for future no-regression comparison.",
        {"fingerprint_csv": str(fp_csv), "baseline_fingerprint_json": str(fp_json)},
    )

    # Optional baseline comparison.
    if args.baseline_fingerprint:
        base_fp_path = Path(args.baseline_fingerprint).expanduser()
        expected = load_baseline_fingerprint(base_fp_path)
        mismatches = []
        for row in fp_rows:
            rel = row["relative_path"]
            old = expected.get(rel)
            new = row.get("sha256")
            if old and new and old != new:
                mismatches.append({"relative_path": rel, "baseline_sha256": old, "current_sha256": new})
            elif old and not new:
                mismatches.append({"relative_path": rel, "baseline_sha256": old, "current_sha256": "<missing>"})
        add(
            "C16",
            "baseline fingerprint comparison",
            "FAIL" if mismatches else "PASS",
            "Current source fingerprints match the supplied baseline." if not mismatches else "Current source fingerprints differ from the supplied baseline.",
            {"baseline_fingerprint": str(base_fp_path), "mismatches": mismatches[:200]},
        )

    # Overall status.
    overall = "PASS"
    for c in checks:
        if STATUS_ORDER[c.status] > STATUS_ORDER[overall]:
            overall = c.status

    audit_json = output_dir / "D14_P0_FREEZE_AUDIT.json"
    audit_md = output_dir / "D14_P0_FREEZE_AUDIT.md"
    run_txt = output_dir / "D14_P0_RUN_SUMMARY.txt"
    payload = {
        "schema_version": "d14_p0_freeze_audit_v1",
        "generated_at": now_iso(),
        "overall_status": overall,
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "d10_p1_dir": str(d10_dir),
        "d12_s1k_dir": str(d12_dir),
        "checks": [dataclasses.asdict(c) for c in checks],
        "git": {"head": git_head, "status_short": git_status},
        "output_files": {
            "audit_json": str(audit_json),
            "audit_md": str(audit_md),
            "run_summary": str(run_txt),
            "file_fingerprints_csv": str(fp_csv),
            "baseline_fingerprint_json": str(fp_json),
            "scorecard_index_json": str(scorecard_index_path),
            "scorecard_index_csv": str(scorecard_csv_path),
        },
    }
    audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(audit_md, payload)
    run_txt.write_text(
        f"D14-P0 freeze audit finished at {payload['generated_at']}\n"
        f"overall_status={overall}\n"
        f"project_root={project_root}\n"
        f"cache_root={cache_root}\n"
        f"audit_json={audit_json}\n"
        f"audit_md={audit_md}\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "overall_status": overall,
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
        "file_fingerprints_csv": str(fp_csv),
        "scorecard_index_csv": str(scorecard_csv_path),
    }, ensure_ascii=False, indent=2))
    return 0 if overall in {"PASS", "WARN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
