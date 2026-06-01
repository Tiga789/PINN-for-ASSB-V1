#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QJW-2 / PINN-for-ASSB-V1 clean-clone reproducibility checker.
This script performs lightweight static checks after a fresh GitHub clone.
It does not modify project files and does not start training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except FileNotFoundError:
        return ""


def run_cmd(cmd: List[str], cwd: Path | None = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return 999, "", f"{type(exc).__name__}: {exc}"


def contains_any(text: str, patterns: Iterable[str]) -> Dict[str, bool]:
    return {p: (p.lower() in text.lower()) for p in patterns}


def find_files(repo: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    skip_dirs = {".git", "__pycache__", ".idea", ".pytest_cache"}
    files: List[Path] = []
    for root, dirs, names in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        root_p = Path(root)
        for name in names:
            if name.lower().endswith(suffixes):
                files.append(root_p / name)
    return files


def rel(repo: Path, path: Path) -> str:
    try:
        return str(path.relative_to(repo)).replace("\\", "/")
    except Exception:
        return str(path)


def check_required_paths(repo: Path) -> Tuple[List[Dict[str, Any]], int]:
    required = [
        "README.md",
        "gv1/model.py",
        "gv1/output_transform.py",
        "gv1/losses.py",
        "gv1/trainer.py",
        "scripts/gv1_train_conditioned_pinn.py",
    ]
    recommended = [
        "gv1/profile_adaptive.py",
        "scripts/gv1_generate_softlabels.py",
        "scripts/gv1_eval.py",
        "scripts/gv1_train.py",
    ]
    rows: List[Dict[str, Any]] = []
    hard_fail = 0
    for p in required:
        ok = (repo / p).exists()
        rows.append({"path": p, "level": "required", "exists": ok})
        if not ok:
            hard_fail += 1
    for p in recommended:
        ok = (repo / p).exists()
        rows.append({"path": p, "level": "recommended", "exists": ok})
    return rows, hard_fail


def check_readme(repo: Path) -> Tuple[Dict[str, Any], int]:
    readme = repo / "README.md"
    text = read_text(readme)
    patterns = [
        "ASSB-D10",
        "ModelFin_112_deterministic_wrapper",
        "D9.6",
        "D9.5.1",
        "battery-8",
        "metadata_on",
        "D12-S2",
        "D12-S3",
    ]
    hits = contains_any(text, patterns)
    missing = [p for p, ok in hits.items() if not ok]
    return {"path": "README.md", "patterns": hits, "missing": missing}, 0 if not missing else 1


def check_mainline_keywords(repo: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    checks = [
        ("scripts/gv1_train_conditioned_pinn.py", ["trend", "rare", "warmup", "metadata"]),
        ("gv1/losses.py", ["rare", "warmup", "tail", "guard"]),
        ("gv1/output_transform.py", ["affine", "voltage"]),
        ("gv1/trainer.py", ["warmup", "metadata", "prediction"]),
    ]
    rows: List[Dict[str, Any]] = []
    warn_count = 0
    for p, pats in checks:
        f = repo / p
        text = read_text(f)
        hits = contains_any(text, pats)
        missing = [k for k, ok in hits.items() if not ok]
        rows.append({"path": p, "patterns": hits, "missing": missing})
        # Missing keywords are warnings only because code names may change.
        if missing:
            warn_count += 1
    return rows, 0, warn_count


def check_for_risky_patterns(repo: Path) -> Tuple[List[Dict[str, Any]], int, int]:
    """Search for patterns that often indicate an accidental regression.
    They are warnings by default, not hard failures, because context matters.
    """
    risky_regexes = [
        ("hard_clamp_true", re.compile(r"enable_voltage_hard_clamp\s*[=:]\s*True", re.IGNORECASE)),
        ("old_d12_long_epochs", re.compile(r"epochs?\s*[=:]\s*40000", re.IGNORECASE)),
        ("old_d12_long_window", re.compile(r"time_window_s\s*[=:]\s*200000", re.IGNORECASE)),
        ("battery8_unflag_hint", re.compile(r"unflag\s*\(?\s*battery-?8", re.IGNORECASE)),
    ]
    files = find_files(repo, (".py", ".ps1", ".md", ".txt", ".json", ".yaml", ".yml"))
    hits: List[Dict[str, Any]] = []
    for f in files:
        text = read_text(f)
        for name, rgx in risky_regexes:
            for m in rgx.finditer(text):
                # Capture a compact context window.
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                ctx = text[start:end].replace("\n", " ")
                hits.append({"risk": name, "path": rel(repo, f), "context": ctx})
    # No hard fail; each hit is warning.
    return hits, 0, len(hits)


def compile_selected(repo: Path, python_exe: str | None) -> Tuple[Dict[str, Any], int]:
    py = python_exe or sys.executable
    targets = [str(p) for p in [repo / "gv1", repo / "scripts"] if p.exists()]
    if not targets:
        return {"python": py, "targets": targets, "returncode": 2, "stdout": "", "stderr": "No gv1/scripts targets found."}, 1
    rc, out, err = run_cmd([py, "-m", "compileall", "-q", *targets], cwd=repo)
    return {"python": py, "targets": targets, "returncode": rc, "stdout": out, "stderr": err}, 0 if rc == 0 else 1


def get_git_info(repo: Path) -> Dict[str, str]:
    rc1, head, err1 = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo)
    rc2, branch, err2 = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    rc3, status, err3 = run_cmd(["git", "status", "--short"], cwd=repo)
    return {
        "head": head if rc1 == 0 else f"ERROR: {err1}",
        "branch": branch if rc2 == 0 else f"ERROR: {err2}",
        "status_short": status if rc3 == 0 else f"ERROR: {err3}",
    }


def check_cache(cache_root: Path | None) -> Dict[str, Any]:
    if cache_root is None:
        return {"enabled": False}
    expected = [
        "xjtu_batch134_replay_profiles",
        "xjtu_batch134_training_ready",
        "xjtu_batch134_soh_labels",
        "xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8",
        "xjtu_batch134_d12_s2_metadata_ablation_scorecard",
    ]
    rows = []
    for name in expected:
        p = cache_root / name
        rows.append({"path": str(p), "exists": p.exists()})
    return {"enabled": True, "cache_root": str(cache_root), "items": rows}


def write_markdown(report: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Clean Clone Reproducibility Check Report")
    lines.append("")
    lines.append(f"Generated at: `{report.get('generated_at')}`")
    lines.append(f"Repository: `{report.get('repo')}`")
    gi = report.get("git", {})
    lines.append(f"Git branch: `{gi.get('branch')}`")
    lines.append(f"Git HEAD: `{gi.get('head')}`")
    lines.append(f"Overall status: **{report.get('overall_status')}**")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Hard failures: `{report.get('hard_failures')}`")
    lines.append(f"- Warnings: `{report.get('warnings')}`")
    lines.append("")
    lines.append("## Required / recommended paths")
    lines.append("")
    for row in report.get("required_paths", []):
        mark = "OK" if row["exists"] else "MISSING"
        lines.append(f"- [{mark}] `{row['path']}` ({row['level']})")
    lines.append("")
    lines.append("## README pattern check")
    lines.append("")
    readme = report.get("readme", {})
    for pat, ok in readme.get("patterns", {}).items():
        lines.append(f"- [{'OK' if ok else 'MISS'}] `{pat}`")
    lines.append("")
    lines.append("## Compileall")
    lines.append("")
    comp = report.get("compileall", {})
    lines.append(f"- Python: `{comp.get('python')}`")
    lines.append(f"- Return code: `{comp.get('returncode')}`")
    if comp.get("stderr"):
        lines.append("- stderr:")
        lines.append("```text")
        lines.append(str(comp.get("stderr"))[:4000])
        lines.append("```")
    lines.append("")
    lines.append("## Risky pattern warnings")
    lines.append("")
    risky = report.get("risky_patterns", [])
    if risky:
        for row in risky[:100]:
            lines.append(f"- `{row['risk']}` in `{row['path']}`: {row['context']}")
        if len(risky) > 100:
            lines.append(f"- ... {len(risky) - 100} more entries omitted in markdown; see JSON.")
    else:
        lines.append("No risky patterns found.")
    lines.append("")
    lines.append("## External cache check")
    lines.append("")
    cache = report.get("external_cache", {})
    if not cache.get("enabled"):
        lines.append("External cache check was not enabled.")
    else:
        lines.append(f"Cache root: `{cache.get('cache_root')}`")
        for row in cache.get("items", []):
            lines.append(f"- [{'OK' if row['exists'] else 'MISS'}] `{row['path']}`")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Path to clean-cloned repository.")
    ap.add_argument("--report-dir", required=True, help="Directory to write reports.")
    ap.add_argument("--python-exe", default=None, help="Python executable used for compileall.")
    ap.add_argument("--cache-root", default=None, help="Optional XJTU cache root path.")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).resolve() if args.cache_root else None

    hard_failures = 0
    warnings = 0

    required_paths, hf = check_required_paths(repo)
    hard_failures += hf

    readme, hf = check_readme(repo)
    hard_failures += hf

    mainline_keywords, hf, warn = check_mainline_keywords(repo)
    hard_failures += hf
    warnings += warn

    risky_patterns, hf, warn = check_for_risky_patterns(repo)
    hard_failures += hf
    warnings += warn

    compileall, hf = compile_selected(repo, args.python_exe)
    hard_failures += hf

    report: Dict[str, Any] = {
        "generated_at": now_iso(),
        "repo": str(repo),
        "git": get_git_info(repo),
        "required_paths": required_paths,
        "readme": readme,
        "mainline_keywords": mainline_keywords,
        "risky_patterns": risky_patterns,
        "compileall": compileall,
        "external_cache": check_cache(cache_root),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "overall_status": "PASS" if hard_failures == 0 else "FAIL",
    }

    (report_dir / "clean_clone_check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, report_dir / "clean_clone_check_report.md")

    print(json.dumps({
        "overall_status": report["overall_status"],
        "hard_failures": hard_failures,
        "warnings": warnings,
        "report_json": str(report_dir / "clean_clone_check_report.json"),
        "report_md": str(report_dir / "clean_clone_check_report.md"),
    }, ensure_ascii=False, indent=2))

    return 0 if hard_failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
