#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare D12-S3 clean 23-profile metadata off/zero/on strict 40ks commands.

This script only writes generated PowerShell commands and audit files. It does
not launch training and refuses long-run parameters. It is intended to continue
from the validated D9.6/D9.5.1 mainline while treating B1_2C battery-8 as a
flagged/excluded late-2C discharge outlier.
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
        return [dict(r) for r in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pick(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def profile_id(row: dict[str, Any]) -> str:
    value = pick(row, ["profile_id", "cell_uid", "profile_key", "label"])
    if value:
        return value
    parts = [pick(row, ["batch_id"]), pick(row, ["protocol"]), pick(row, ["battery_id"])]
    return "_".join([part for part in parts if part])


def protocol_of(row: dict[str, Any]) -> str:
    proto = pick(row, ["protocol", "protocol_id", "profile_protocol"])
    if proto:
        return proto
    pid = profile_id(row)
    if "R2.5" in pid or "R25" in pid:
        return "R2.5"
    if "R3" in pid:
        return "R3"
    if "2C" in pid:
        return "2C"
    return "unknown"


def battery_sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    pid = profile_id(row)
    text = pick(row, ["battery_id", "battery", "cell_id"]) or pid
    match = re.search(r"battery[-_]?([0-9]+)", text, flags=re.I)
    number = int(match.group(1)) if match else 10**9
    return (protocol_of(row), number, pid)


def source_npz(row: dict[str, Any]) -> str:
    value = pick(row, ["profile_npz", "solution_npz", "npz_path", "source_npz"])
    if value:
        return value
    folder = pick(row, ["prepared_dir", "profile_dir"])
    return str(Path(folder) / "solution_replay_profile.npz") if folder else ""


def is_target(row: dict[str, Any], target_profile_id: str) -> bool:
    pid = profile_id(row)
    if pid == target_profile_id:
        return True
    return (
        pick(row, ["batch_id"]) == "Batch-1"
        and pick(row, ["battery_id"]) == "battery-8"
        and protocol_of(row) == "2C"
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "profile"


def ps_quote(value: str | Path) -> str:
    return '"' + str(value).replace('"', '`"') + '"'


def ensure_strict_smoke(args: argparse.Namespace) -> None:
    bad: list[str] = []
    if args.epochs > 200:
        bad.append(f"epochs={args.epochs} > 200")
    if args.time_window_s > 60000:
        bad.append(f"time_window_s={args.time_window_s} > 60000")
    if args.max_time_points > 2048:
        bad.append(f"max_time_points={args.max_time_points} > 2048")
    if args.batch_size > 1024:
        bad.append(f"batch_size={args.batch_size} > 1024")
    if args.prediction_time_points > 2048:
        bad.append(f"prediction_time_points={args.prediction_time_points} > 2048")
    if bad:
        raise SystemExit("Refusing to generate non-smoke commands: " + "; ".join(bad))


def select_23(rows: list[dict[str, Any]], *, target_profile_id: str, expected_count: int) -> list[dict[str, Any]]:
    selected = sorted([row for row in rows if not is_target(row, target_profile_id)], key=battery_sort_key)
    ids = [profile_id(row) for row in selected]
    if target_profile_id in ids:
        raise SystemExit("STOP: selected rows include target battery-8")
    if len(selected) != expected_count:
        raise SystemExit(
            f"STOP: expected {expected_count} non-target profiles, got {len(selected)}. ids={ids}"
        )
    return selected


def build_script(*, mode: str, manifest: Path, tag: str, rows: list[dict[str, Any]], args: argparse.Namespace, cache_root: Path) -> str:
    wrapper = Path(args.project_root) / "scripts" / "gv1_train_conditioned_pinn_d12_metadata_runtime.py"
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location {ps_quote(args.project_root)}",
        f"$Python = {ps_quote(args.python)}",
        f"$Wrapper = {ps_quote(wrapper)}",
        "if (-not (Test-Path $Wrapper)) { throw \"Missing metadata runtime wrapper: $Wrapper\" }",
        "",
    ]
    for idx, row in enumerate(rows, 1):
        pid = profile_id(row)
        proto = protocol_of(row)
        sol = source_npz(row)
        if not sol:
            raise ValueError(f"Missing solution npz for {pid}")
        out_dir = cache_root / f"xjtu_batch134_d12_s3_metadata_{tag}_{safe_name(pid)}_STRICT_40ks_e{args.epochs}"
        lines += [
            f"Write-Host 'D12-S3 STRICT {tag} {idx}/{len(rows)} [{proto}]: {pid}'",
            f"& $Python $Wrapper `",
            f"  --metadata_mode {mode} `",
            f"  --metadata_manifest {ps_quote(manifest)} `",
            f"  --metadata_profile_id {ps_quote(pid)} `",
            "  --metadata_strict_profile_match true `",
            "  --metadata_allow_target_probe false `",
            f"  --solution_npz {ps_quote(sol)} `",
            f"  --output_dir {ps_quote(out_dir)} `",
            "  --profile_adaptive_mode auto `",
            f"  --epochs {args.epochs} `",
            f"  --batch_size {args.batch_size} `",
            f"  --seed {args.seed} `",
            f"  --device {args.device} `",
            f"  --max_time_points {args.max_time_points} `",
            f"  --time_window_s {float(args.time_window_s):.1f} `",
            f"  --prediction_time_points {args.prediction_time_points} `",
            f"  --prediction_radial_points {args.prediction_radial_points}",
            "if ($LASTEXITCODE -ne 0) { throw 'D12-S3 STRICT runtime command failed.' }",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1")
    parser.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    parser.add_argument("--python", default=r"D:\Anaconda\envs\torchgpu\python.exe")
    parser.add_argument("--d12_plan_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--expected_profile_count", type=int, default=23)
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

    ensure_strict_smoke(args)
    project_root = Path(args.project_root)
    wrapper = project_root / "scripts" / "gv1_train_conditioned_pinn_d12_metadata_runtime.py"
    if not wrapper.exists():
        raise SystemExit(f"STOP: missing metadata runtime wrapper: {wrapper}")

    cache_root = Path(args.cache_root)
    d12_plan_dir = Path(args.d12_plan_dir) if args.d12_plan_dir else cache_root / "xjtu_batch134_d12_metadata_on_off_ablation_plan"
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d12_s3_metadata_ablation_commands"
    out_dir.mkdir(parents=True, exist_ok=True)

    on_manifest = d12_plan_dir / "d12_metadata_on_23profile_manifest.csv"
    off_manifest = d12_plan_dir / "d12_metadata_off_23profile_manifest.csv"
    on_rows = read_csv(on_manifest)
    selected = select_23(on_rows, target_profile_id=args.target_profile_id, expected_count=args.expected_profile_count)

    export_rows: list[dict[str, Any]] = []
    missing_npz: list[str] = []
    by_protocol: dict[str, list[str]] = {}
    for row in selected:
        pid = profile_id(row)
        proto = protocol_of(row)
        sol = source_npz(row)
        by_protocol.setdefault(proto, []).append(pid)
        if not sol or not Path(sol).exists():
            missing_npz.append(f"{pid}: {sol}")
        export_rows.append({
            "profile_id": pid,
            "protocol": proto,
            "batch_id": pick(row, ["batch_id"]),
            "battery_id": pick(row, ["battery_id"]),
            "profile_npz": sol,
        })
    if missing_npz:
        raise SystemExit("STOP: missing profile npz files:\n" + "\n".join(missing_npz[:20]))

    scripts = {
        "off": out_dir / "run_d12_s3_metadata_off_23profile.generated.ps1",
        "zero": out_dir / "run_d12_s3_metadata_zero_23profile.generated.ps1",
        "on": out_dir / "run_d12_s3_metadata_on_23profile.generated.ps1",
    }
    scripts["off"].write_text(build_script(mode="off", manifest=off_manifest, tag="off", rows=selected, args=args, cache_root=cache_root), encoding="utf-8")
    scripts["zero"].write_text(build_script(mode="zero", manifest=on_manifest, tag="zero", rows=selected, args=args, cache_root=cache_root), encoding="utf-8")
    scripts["on"].write_text(build_script(mode="on", manifest=on_manifest, tag="on", rows=selected, args=args, cache_root=cache_root), encoding="utf-8")

    run_all = out_dir / "run_d12_s3_all_modes_23profile.generated.ps1"
    run_all.write_text(
        "\n".join([
            "$ErrorActionPreference = 'Stop'",
            f"& {ps_quote(scripts['off'])}",
            f"& {ps_quote(scripts['zero'])}",
            f"& {ps_quote(scripts['on'])}",
        ]),
        encoding="utf-8",
    )

    selected_manifest = out_dir / "d12_s3_selected_23profile_manifest.csv"
    write_csv(selected_manifest, export_rows)
    summary = {
        "ok": True,
        "stage": "D12-S3 clean 23-profile strict metadata ablation command preparation",
        "verdict": "d12_s3_clean_23profile_strict_commands_prepared",
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "d12_plan_dir": str(d12_plan_dir),
        "out_dir": str(out_dir),
        "profile_count": len(selected),
        "expected_profile_count": args.expected_profile_count,
        "selected_by_protocol": by_protocol,
        "target_profile_id": args.target_profile_id,
        "target_included": args.target_profile_id in [profile_id(r) for r in selected],
        "epochs": args.epochs,
        "time_window_s": args.time_window_s,
        "max_time_points": args.max_time_points,
        "batch_size": args.batch_size,
        "prediction_time_points": args.prediction_time_points,
        "prediction_radial_points": args.prediction_radial_points,
        "generated_scripts": {key: str(path) for key, path in scripts.items()},
        "run_all_script": str(run_all),
        "selected_manifest": str(selected_manifest),
        "forbidden_params": ["--epochs 40000", "--time_window_s 200000", "--max_time_points 8192", "--batch_size 2048", "_200ks"],
        "note": "Generated only. Inspect scripts before running. This does not alter D9.6/D9.5.1 source files.",
    }
    write_json(out_dir / "d12_s3_command_preparation_summary.json", summary)
    (out_dir / "D12_S3_COMMANDS_README.md").write_text(
        "# D12-S3 clean 23-profile strict metadata ablation commands\n\n```json\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
