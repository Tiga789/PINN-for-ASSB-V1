#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare D12-S1 strict metadata on/off/zero TRUE SMOKE commands.

This script is deliberately conservative. It generates short 3-profile smoke
commands only, never long 200ks/40000-epoch commands. It does not run training.

Default D12-S1 parameters:
  epochs = 100
  time_window_s = 40000
  max_time_points = 1024
  batch_size = 512

The generated run directories use a separate D12-S1 prefix so they cannot be
mistaken for earlier broken D12 runtime outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def pick(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        val = str(row.get(key, "")).strip()
        if val:
            return val
    return ""


def profile_id(row: dict[str, Any]) -> str:
    direct = pick(row, ["profile_id", "cell_uid", "profile_key", "label"])
    if direct:
        return direct
    parts = [pick(row, ["batch_id"]), pick(row, ["protocol"]), pick(row, ["battery_id"])]
    return "_".join([p for p in parts if p])


def source_npz(row: dict[str, Any]) -> str:
    direct = pick(row, ["profile_npz", "solution_npz", "npz_path", "source_npz"])
    if direct:
        return direct
    folder = pick(row, ["prepared_dir", "profile_dir"])
    return str(Path(folder) / "solution_replay_profile.npz") if folder else ""


def is_target_battery8(row: dict[str, Any], target_profile_id: str) -> bool:
    pid = profile_id(row)
    batch = pick(row, ["batch_id"])
    batt = pick(row, ["battery_id"])
    protocol = pick(row, ["protocol"])
    if pid == target_profile_id:
        return True
    return batch == "Batch-1" and batt == "battery-8" and protocol == "2C"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "profile"


def ps_quote(text: str) -> str:
    # Double quoted PowerShell string with backtick-escaped double quotes.
    return '"' + str(text).replace('"', '`"') + '"'


def ensure_smoke_params(args: argparse.Namespace) -> None:
    problems: list[str] = []
    if args.epochs > 200:
        problems.append(f"epochs={args.epochs} > 200")
    if args.time_window_s > 60000:
        problems.append(f"time_window_s={args.time_window_s} > 60000")
    if args.max_time_points > 2048:
        problems.append(f"max_time_points={args.max_time_points} > 2048")
    if args.batch_size > 1024:
        problems.append(f"batch_size={args.batch_size} > 1024")
    if problems:
        raise SystemExit("Refusing to generate non-smoke commands: " + "; ".join(problems))


def build_script_lines(
    *,
    mode: str,
    manifest: Path,
    tag: str,
    selected_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    cache_root: Path,
) -> list[str]:
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location {ps_quote(args.project_root)}",
        f"$Python = {ps_quote(args.python)}",
        "",
    ]
    wrapper = "scripts\\gv1_train_conditioned_pinn_d12_metadata_runtime.py"
    for idx, row in enumerate(selected_rows, start=1):
        pid = profile_id(row)
        sol = source_npz(row)
        if not sol:
            raise ValueError(f"No solution npz found for selected profile {pid}: {row}")
        out_dir = cache_root / f"xjtu_batch134_d12_s1_metadata_{tag}_{safe_name(pid)}_TRUE_SMOKE_40ks_e{args.epochs}"
        lines.extend([
            f"Write-Host 'D12-S1 TRUE SMOKE {tag} {idx}/{len(selected_rows)}: {pid}'",
            f"& $Python {wrapper} `",
            f"  --metadata_mode {mode} `",
            f"  --metadata_manifest {ps_quote(str(manifest))} `",
            f"  --metadata_profile_id {ps_quote(pid)} `",
            "  --metadata_strict_profile_match true `",
            "  --metadata_allow_target_probe false `",
            f"  --solution_npz {ps_quote(sol)} `",
            f"  --output_dir {ps_quote(str(out_dir))} `",
            "  --profile_adaptive_mode auto `",
            f"  --epochs {args.epochs} `",
            f"  --batch_size {args.batch_size} `",
            f"  --seed {args.seed} `",
            f"  --device {args.device} `",
            f"  --max_time_points {args.max_time_points} `",
            f"  --time_window_s {float(args.time_window_s):.1f} `",
            f"  --prediction_time_points {args.prediction_time_points} `",
            f"  --prediction_radial_points {args.prediction_radial_points}",
            "if ($LASTEXITCODE -ne 0) { throw 'D12-S1 TRUE SMOKE runtime command failed.' }",
            "",
        ])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1")
    parser.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    parser.add_argument("--python", default=r"D:\Anaconda\envs\torchgpu\python.exe")
    parser.add_argument("--d12_plan_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--profile_limit", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--time_window_s", type=float, default=40000.0)
    parser.add_argument("--max_time_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--prediction_time_points", type=int, default=1024)
    parser.add_argument("--prediction_radial_points", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target_profile_id", default="Batch-1_2C_battery-8")
    args = parser.parse_args()

    ensure_smoke_params(args)

    cache_root = Path(args.cache_root)
    d12_plan_dir = Path(args.d12_plan_dir) if args.d12_plan_dir else cache_root / "xjtu_batch134_d12_metadata_on_off_ablation_plan"
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d12_s1_metadata_ablation_commands"
    out_dir.mkdir(parents=True, exist_ok=True)

    on_manifest = d12_plan_dir / "d12_metadata_on_23profile_manifest.csv"
    off_manifest = d12_plan_dir / "d12_metadata_off_23profile_manifest.csv"
    on_rows = read_csv(on_manifest)

    non_target_rows = [row for row in on_rows if not is_target_battery8(row, args.target_profile_id)]
    selected_rows = non_target_rows[: max(1, min(args.profile_limit, len(non_target_rows)))]
    selected_ids = [profile_id(row) for row in selected_rows]
    if any(pid == args.target_profile_id for pid in selected_ids):
        raise SystemExit("STOP: selected rows include target battery-8 profile; refusing to prepare D12-S1 commands.")

    n = len(selected_rows)
    scripts = {
        "off": out_dir / f"run_d12_s1_metadata_off_{n}profile.generated.ps1",
        "zero": out_dir / f"run_d12_s1_metadata_zero_{n}profile.generated.ps1",
        "on": out_dir / f"run_d12_s1_metadata_on_{n}profile.generated.ps1",
    }
    scripts["off"].write_text("\n".join(build_script_lines(mode="off", manifest=off_manifest, tag="off", selected_rows=selected_rows, args=args, cache_root=cache_root)), encoding="utf-8")
    scripts["zero"].write_text("\n".join(build_script_lines(mode="zero", manifest=on_manifest, tag="zero", selected_rows=selected_rows, args=args, cache_root=cache_root)), encoding="utf-8")
    scripts["on"].write_text("\n".join(build_script_lines(mode="on", manifest=on_manifest, tag="on", selected_rows=selected_rows, args=args, cache_root=cache_root)), encoding="utf-8")

    summary = {
        "ok": True,
        "stage": "D12-S1 strict metadata ablation command preparation",
        "verdict": "d12_s1_strict_smoke_commands_prepared",
        "out_dir": str(out_dir),
        "profile_limit": n,
        "selected_profile_ids": selected_ids,
        "target_profile_id": args.target_profile_id,
        "target_included": args.target_profile_id in selected_ids,
        "epochs": args.epochs,
        "time_window_s": args.time_window_s,
        "max_time_points": args.max_time_points,
        "batch_size": args.batch_size,
        "prediction_time_points": args.prediction_time_points,
        "seed": args.seed,
        "generated_scripts": {key: str(value) for key, value in scripts.items()},
        "forbidden_params": ["--epochs 40000", "--time_window_s 200000", "--max_time_points 8192", "--batch_size 2048", "_200ks"],
        "note": "Generated only. Run D12-S1 preflight before executing these scripts.",
    }
    (out_dir / "d12_s1_command_preparation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "D12_S1_COMMANDS_README.md").write_text(
        "# D12-S1 strict metadata ablation commands\n\n"
        "Generated commands are intentionally short smoke runs.\n\n"
        f"```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
