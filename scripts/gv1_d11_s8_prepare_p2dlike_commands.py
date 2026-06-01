#!/usr/bin/env python
"""Prepare D11-S8 P2D-like transport-deficit correction commands.

This preparation script does not train.  It finds existing baseline_d951
prediction.npz files, then generates PowerShell scripts that apply a P2D-like
post-transform transport deficit correction for several diagnostic modes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

MODES = [
    "baseline_copy",
    "p2dlike_transport_mild",
    "p2dlike_transport_medium",
    "p2dlike_transport_strong_guarded",
    "p2dlike_transport_discharge_only",
]

DEFAULT_CACHE = r"E:\XJTU battery dataset\_gv1_cache"

CANDIDATE_BASELINE_DIRS = [
    "xjtu_batch134_d11_s7_lowvoltage_escape",
    "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair",
    "xjtu_batch134_d11_s5a_lowtarget_sign_gate_diagnosis",
    "xjtu_batch134_d11_s4_lowtail_correction_smoke",
    "xjtu_batch134_d12_s3_metadata_ablation",
]


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    return s.strip("_") or "unknown"


def _infer_profile(path: Path) -> str:
    parts = list(path.parts)
    # Prefer directory containing battery/profile name, skip mode dirs.
    for p in reversed(parts[:-1]):
        low = p.lower()
        if "battery" in low or "batch" in low:
            if "baseline" not in low and "prediction" not in low:
                return p
    return path.parent.name


def _infer_protocol(profile: str, path: Path) -> str:
    s = (profile + " " + str(path)).lower()
    if "r2.5" in s or "r25" in s or "batch-3" in s or "batch_3" in s:
        return "R2.5"
    if "r3" in s or "batch-4" in s or "batch_4" in s:
        return "R3"
    if "2c" in s or "batch-1" in s or "batch_1" in s:
        return "2C"
    return "unknown"


def find_baseline_predictions(cache_root: Path, explicit_root: Path | None, max_profiles: int):
    roots = []
    if explicit_root is not None:
        roots.append(explicit_root)
    for d in CANDIDATE_BASELINE_DIRS:
        p = cache_root / d
        if p.exists():
            roots.append(p)

    seen = set()
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for pred in root.rglob("prediction.npz"):
            lower = str(pred).lower()
            if "battery-8" in lower or "battery_8" in lower:
                continue
            # Use baseline/off only.  Avoid corrected candidate predictions.
            if not ("baseline_d951" in lower or "metadata_off" in lower or "d10p1" in lower):
                continue
            profile = _infer_profile(pred)
            profile_key = _slug(profile)
            if profile_key in seen:
                continue
            seen.add(profile_key)
            rows.append({
                "profile": profile,
                "profile_slug": profile_key,
                "protocol": _infer_protocol(profile, pred),
                "input_prediction": str(pred),
                "source_root": str(root),
            })
            if len(rows) >= max_profiles:
                return rows
    return rows


def write_ps1(path: Path, commands: list[str]):
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=str(Path.cwd()))
    ap.add_argument("--python_exe", default=r"D:\Anaconda\envs\torchgpu\python.exe")
    ap.add_argument("--cache_root", default=DEFAULT_CACHE)
    ap.add_argument("--baseline_prediction_root", default="")
    ap.add_argument("--out_command_dir", default="")
    ap.add_argument("--prediction_out_root", default="")
    ap.add_argument("--max_profiles", type=int, default=6)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    cache_root = Path(args.cache_root)
    baseline_root = Path(args.baseline_prediction_root) if args.baseline_prediction_root else None
    cmd_dir = Path(args.out_command_dir) if args.out_command_dir else cache_root / "xjtu_batch134_d11_s8_p2dlike_transport_correction_commands"
    pred_root = Path(args.prediction_out_root) if args.prediction_out_root else cache_root / "xjtu_batch134_d11_s8_p2dlike_transport_correction"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    correction_script = project_root / "scripts" / "gv1_d11_s8_p2dlike_transport_correction_from_baseline.py"
    if not correction_script.exists():
        raise FileNotFoundError(correction_script)

    selected = find_baseline_predictions(cache_root, baseline_root, args.max_profiles)
    if len(selected) < args.max_profiles:
        raise RuntimeError(f"Expected {args.max_profiles} baseline predictions, found {len(selected)}. Provide --baseline_prediction_root or run baseline D11-S7/S5C first.")

    manifest_path = cmd_dir / "d11_s8_selected_baseline_predictions.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        w.writeheader()
        w.writerows(selected)

    all_script_lines = [
        "$ErrorActionPreference = 'Stop'",
        "Write-Host 'D11-S8 P2D-like transport correction: post-transform diagnostic only; no training launched.'",
    ]

    for mode in MODES:
        mode_script = cmd_dir / f"run_d11_s8_{mode}.generated.ps1"
        lines = [
            "$ErrorActionPreference = 'Stop'",
            f"Write-Host 'Running D11-S8 mode: {mode}'",
        ]
        for row in selected:
            out_pred = pred_root / mode / row["profile_slug"] / "prediction.npz"
            cmd = (
                f'& "{args.python_exe}" "{correction_script}" '
                f'--input_prediction "{row["input_prediction"]}" '
                f'--output_prediction "{out_pred}" '
                f'--mode "{mode}" '
                f'--profile "{row["profile"]}" '
                f'--protocol "{row["protocol"]}"'
            )
            lines.append(cmd)
            lines.append("if ($LASTEXITCODE -ne 0) { throw 'D11-S8 correction command failed.' }")
        write_ps1(mode_script, lines)
        all_script_lines.append(f'& "{mode_script}"')
        all_script_lines.append("if ($LASTEXITCODE -ne 0) { throw 'D11-S8 mode script failed.' }")

    all_path = cmd_dir / "run_d11_s8_all_modes.generated.ps1"
    write_ps1(all_path, all_script_lines)

    summary = {
        "ok": True,
        "stage": "D11-S8 prepare P2D-like transport correction commands",
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "command_dir": str(cmd_dir),
        "prediction_out_root": str(pred_root),
        "modes": MODES,
        "selected_count": len(selected),
        "manifest": str(manifest_path),
        "all_modes_script": str(all_path),
    }
    (cmd_dir / "d11_s8_command_preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
