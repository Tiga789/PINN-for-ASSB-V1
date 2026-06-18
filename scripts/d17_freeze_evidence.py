# -*- coding: utf-8 -*-
"""
D17-P0: freeze D16/D17 starting evidence without modifying old project outputs.

This script creates a small, auditable freeze manifest:
- records git revision/status if available;
- records hashes for selected small evidence files;
- inventories selected large result directories without copying them;
- writes a README describing D17-P0 boundaries.

It is safe by default: existing output files are not overwritten unless --force is used.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_GLOBS = [
    "README.md",
    "ASSB-D16_项目进度复盘总结_20260614.docx",
    "D17-PINN重构方案.docx",
    "D16_*.md",
    "D16_*.json",
    "D16_*.csv",
    "D16_P5*.md",
    "D16_P5*.json",
    "D16_P5*.csv",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(project_root: Path, args: List[str]) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            timeout=20,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": repr(exc)}


def find_evidence_files(project_root: Path, patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for root, dirs, files in os.walk(project_root):
        # keep the inventory lightweight and avoid common large/generated dirs
        rel_root = Path(root).relative_to(project_root)
        parts = set(rel_root.parts)
        if parts & {".git", "__pycache__", ".pytest_cache", "ModelFin", "EvalFin"}:
            dirs[:] = []
            continue
        if any(part.startswith("xjtu_") and "cache" in part for part in rel_root.parts):
            dirs[:] = []
            continue
        for name in files:
            for pat in patterns:
                if fnmatch.fnmatch(name, pat):
                    p = Path(root) / name
                    if p not in seen:
                        out.append(p)
                        seen.add(p)
                    break
    return sorted(out)


def inventory_dir(path: Path, max_hash_mb: float) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "type": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
    }
    if not path.exists():
        return info
    if path.is_file():
        size = path.stat().st_size
        info["size_bytes"] = size
        if size <= max_hash_mb * 1024 * 1024:
            info["sha256"] = sha256_file(path)
        else:
            info["sha256"] = None
            info["hash_skipped_reason"] = f"file larger than {max_hash_mb} MB"
        return info

    total_size = 0
    file_count = 0
    top_files = []
    for f in path.rglob("*"):
        if f.is_file():
            try:
                s = f.stat().st_size
            except OSError:
                continue
            total_size += s
            file_count += 1
            if len(top_files) < 50:
                top_files.append({
                    "relative_path": str(f.relative_to(path)),
                    "size_bytes": s,
                    "sha256": sha256_file(f) if s <= max_hash_mb * 1024 * 1024 else None,
                })
    info.update({
        "file_count": file_count,
        "total_size_bytes": total_size,
        "sample_files": top_files,
        "note": "Large directories are inventoried, not copied.",
    })
    return info


def safe_copy(src: Path, dest_root: Path, project_root: Path, max_copy_mb: float) -> Dict[str, Any]:
    rel = src.relative_to(project_root) if src.is_relative_to(project_root) else Path(src.name)
    dest = dest_root / "copied_evidence" / rel
    size = src.stat().st_size
    rec: Dict[str, Any] = {
        "source": str(src),
        "relative_path": str(rel),
        "size_bytes": size,
        "sha256": sha256_file(src),
    }
    if size <= max_copy_mb * 1024 * 1024:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        rec["copied_to"] = str(dest)
    else:
        rec["copied_to"] = None
        rec["copy_skipped_reason"] = f"file larger than {max_copy_mb} MB"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="PINN-for-ASSB-V1 project root")
    ap.add_argument("--out_dir", required=True, help="D17 freeze output directory")
    ap.add_argument("--evidence_glob", action="append", default=[], help="Additional file glob to freeze")
    ap.add_argument("--result_path", action="append", default=[], help="D16/P5K/G4 result file/dir to inventory")
    ap.add_argument("--max_copy_mb", type=float, default=10.0)
    ap.add_argument("--max_hash_mb", type=float, default=50.0)
    ap.add_argument("--force", action="store_true")
    ns = ap.parse_args()

    project_root = Path(ns.project_root).resolve()
    out_dir = Path(ns.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()) and not ns.force:
        raise SystemExit(f"[ABORT] out_dir already exists and is non-empty: {out_dir}\nUse --force only if you intentionally want to overwrite D17-P0 files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = DEFAULT_GLOBS + list(ns.evidence_glob)
    files = find_evidence_files(project_root, patterns)

    copied = []
    for f in files:
        try:
            copied.append(safe_copy(f, out_dir, project_root, ns.max_copy_mb))
        except Exception as exc:
            copied.append({"source": str(f), "error": repr(exc)})

    result_inventory = []
    for p in ns.result_path:
        result_inventory.append(inventory_dir(Path(p), ns.max_hash_mb))

    manifest: Dict[str, Any] = {
        "protocol": "D17-P0_FREEZE",
        "created_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "project_root": str(project_root),
        "out_dir": str(out_dir),
        "purpose": "Freeze D16/P5K-G/G4 evidence before D17-PINN重构. No training is performed.",
        "git": {
            "rev_parse_HEAD": run_git(project_root, ["rev-parse", "HEAD"]),
            "status_short": run_git(project_root, ["status", "--short"]),
            "branch": run_git(project_root, ["branch", "--show-current"]),
        },
        "evidence_patterns": patterns,
        "copied_or_hashed_files": copied,
        "result_path_inventory": result_inventory,
        "do_not_modify_old_outputs": True,
        "next_step": "Run D17-P1 split/no-label audit after this freeze manifest is accepted.",
    }

    manifest_path = out_dir / "d17_p0_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = out_dir / "D17_P0_FREEZE_README.md"
    readme.write_text(
        "# D17-P0 freeze evidence\n\n"
        "This directory freezes the starting evidence for D17-PINN重构.\n\n"
        "- No model is trained here.\n"
        "- Old D16/P5K-G/G4 outputs are treated as read-only.\n"
        "- Large result directories are inventoried rather than copied.\n"
        "- Continue with D17-P1 only after `d17_p0_freeze_manifest.json` is reviewed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "readme": str(readme)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
