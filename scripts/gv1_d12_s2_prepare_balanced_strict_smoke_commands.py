#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare D12-S2 balanced 6-profile metadata on/off/zero strict smoke commands.

Balanced selection: 2 non-target profiles from each protocol: 2C, R2.5, R3.
This script only writes generated PowerShell commands and audits. It refuses
long-run parameters and does not launch training.
"""
from __future__ import annotations

import argparse, csv, json, re
from pathlib import Path
from typing import Any

PROTOCOLS = ["2C", "R2.5", "R3"]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def pick(row: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = str(row.get(k, "")).strip()
        if v:
            return v
    return ""


def profile_id(row: dict[str, Any]) -> str:
    v = pick(row, ["profile_id", "cell_uid", "profile_key", "label"])
    if v:
        return v
    parts = [pick(row, ["batch_id"]), pick(row, ["protocol"]), pick(row, ["battery_id"])]
    return "_".join([p for p in parts if p])


def protocol_of(row: dict[str, Any]) -> str:
    p = pick(row, ["protocol", "protocol_id", "profile_protocol"])
    if p:
        return p
    pid = profile_id(row)
    if "R2.5" in pid or "R25" in pid:
        return "R2.5"
    if "R3" in pid:
        return "R3"
    if "2C" in pid:
        return "2C"
    return "unknown"


def battery_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    text = pick(row, ["battery_id", "battery", "cell_id"]) or profile_id(row)
    m = re.search(r"battery[-_]?([0-9]+)", text, flags=re.I)
    return (int(m.group(1)) if m else 10**9, profile_id(row))


def source_npz(row: dict[str, Any]) -> str:
    v = pick(row, ["profile_npz", "solution_npz", "npz_path", "source_npz"])
    if v:
        return v
    folder = pick(row, ["prepared_dir", "profile_dir"])
    return str(Path(folder) / "solution_replay_profile.npz") if folder else ""


def is_target(row: dict[str, Any], target_profile_id: str) -> bool:
    pid = profile_id(row)
    if pid == target_profile_id:
        return True
    return pick(row,["batch_id"]) == "Batch-1" and pick(row,["battery_id"]) == "battery-8" and protocol_of(row) == "2C"


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_") or "profile"


def ps_quote(x: str) -> str:
    return '"' + str(x).replace('"', '`"') + '"'


def ensure_smoke(args: argparse.Namespace) -> None:
    bad = []
    if args.epochs > 200: bad.append(f"epochs={args.epochs} > 200")
    if args.time_window_s > 60000: bad.append(f"time_window_s={args.time_window_s} > 60000")
    if args.max_time_points > 2048: bad.append(f"max_time_points={args.max_time_points} > 2048")
    if args.batch_size > 1024: bad.append(f"batch_size={args.batch_size} > 1024")
    if args.profile_per_protocol > 3: bad.append(f"profile_per_protocol={args.profile_per_protocol} > 3")
    if bad:
        raise SystemExit("Refusing to generate non-smoke commands: " + "; ".join(bad))


def choose(rows: list[dict[str, Any]], *, per_protocol: int, target: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    selected: list[dict[str, Any]] = []
    by_proto: dict[str, list[str]] = {}
    non_target = [r for r in rows if not is_target(r, target)]
    for proto in PROTOCOLS:
        cands = sorted([r for r in non_target if protocol_of(r) == proto], key=battery_sort_key)
        if len(cands) < per_protocol:
            raise SystemExit(f"STOP: protocol {proto} needs {per_protocol}, found {len(cands)}: {[profile_id(r) for r in cands]}")
        chosen = cands[:per_protocol]
        selected.extend(chosen)
        by_proto[proto] = [profile_id(r) for r in chosen]
    ids = [profile_id(r) for r in selected]
    if target in ids:
        raise SystemExit("STOP: selected rows include target battery-8")
    return selected, by_proto


def build_script(*, mode: str, manifest: Path, tag: str, rows: list[dict[str, Any]], args: argparse.Namespace, cache_root: Path) -> str:
    lines = ["$ErrorActionPreference = 'Stop'", f"Set-Location {ps_quote(args.project_root)}", f"$Python = {ps_quote(args.python)}", ""]
    wrapper = "scripts\\gv1_train_conditioned_pinn_d12_metadata_runtime.py"
    for i, row in enumerate(rows, 1):
        pid, proto, sol = profile_id(row), protocol_of(row), source_npz(row)
        if not sol:
            raise ValueError(f"Missing solution npz for {pid}")
        out_dir = cache_root / f"xjtu_batch134_d12_s2_metadata_{tag}_{safe_name(pid)}_TRUE_SMOKE_40ks_e{args.epochs}"
        lines += [
            f"Write-Host 'D12-S2 TRUE SMOKE {tag} {i}/{len(rows)} [{proto}]: {pid}'",
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
            "if ($LASTEXITCODE -ne 0) { throw 'D12-S2 TRUE SMOKE runtime command failed.' }",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1")
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--python", default=r"D:\Anaconda\envs\torchgpu\python.exe")
    ap.add_argument("--d12_plan_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--profile_per_protocol", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--time_window_s", type=float, default=40000.0)
    ap.add_argument("--max_time_points", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--prediction_time_points", type=int, default=1024)
    ap.add_argument("--prediction_radial_points", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--target_profile_id", default="Batch-1_2C_battery-8")
    args = ap.parse_args(); ensure_smoke(args)

    cache_root = Path(args.cache_root)
    d12_plan_dir = Path(args.d12_plan_dir) if args.d12_plan_dir else cache_root / "xjtu_batch134_d12_metadata_on_off_ablation_plan"
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d12_s2_metadata_ablation_commands"
    out_dir.mkdir(parents=True, exist_ok=True)

    on_manifest = d12_plan_dir / "d12_metadata_on_23profile_manifest.csv"
    off_manifest = d12_plan_dir / "d12_metadata_off_23profile_manifest.csv"
    rows = read_csv(on_manifest)
    selected, by_proto = choose(rows, per_protocol=args.profile_per_protocol, target=args.target_profile_id)
    ids = [profile_id(r) for r in selected]
    n = len(selected)
    scripts = {
        "off": out_dir / f"run_d12_s2_metadata_off_{n}profile.generated.ps1",
        "zero": out_dir / f"run_d12_s2_metadata_zero_{n}profile.generated.ps1",
        "on": out_dir / f"run_d12_s2_metadata_on_{n}profile.generated.ps1",
    }
    scripts["off"].write_text(build_script(mode="off", manifest=off_manifest, tag="off", rows=selected, args=args, cache_root=cache_root), encoding="utf-8")
    scripts["zero"].write_text(build_script(mode="zero", manifest=on_manifest, tag="zero", rows=selected, args=args, cache_root=cache_root), encoding="utf-8")
    scripts["on"].write_text(build_script(mode="on", manifest=on_manifest, tag="on", rows=selected, args=args, cache_root=cache_root), encoding="utf-8")

    selected_export = []
    for r in selected:
        selected_export.append({"profile_id": profile_id(r), "protocol": protocol_of(r), "batch_id": pick(r,["batch_id"]), "battery_id": pick(r,["battery_id"]), "profile_npz": source_npz(r)})
    write_csv(out_dir / "d12_s2_selected_balanced_6profile_manifest.csv", selected_export)

    summary = {
        "ok": True,
        "stage": "D12-S2 balanced strict metadata ablation command preparation",
        "verdict": "d12_s2_balanced_strict_smoke_commands_prepared",
        "out_dir": str(out_dir),
        "profile_count": n,
        "profile_per_protocol": args.profile_per_protocol,
        "protocol_order": PROTOCOLS,
        "selected_by_protocol": by_proto,
        "selected_profile_ids": ids,
        "target_profile_id": args.target_profile_id,
        "target_included": args.target_profile_id in ids,
        "epochs": args.epochs,
        "time_window_s": args.time_window_s,
        "max_time_points": args.max_time_points,
        "batch_size": args.batch_size,
        "prediction_time_points": args.prediction_time_points,
        "seed": args.seed,
        "generated_scripts": {k: str(v) for k, v in scripts.items()},
        "selected_manifest": str(out_dir / "d12_s2_selected_balanced_6profile_manifest.csv"),
        "forbidden_params": ["--epochs 40000", "--time_window_s 200000", "--max_time_points 8192", "--batch_size 2048", "_200ks"],
        "note": "Generated only. Run D12-S2 preflight before executing these scripts.",
    }
    (out_dir / "d12_s2_command_preparation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "D12_S2_COMMANDS_README.md").write_text("# D12-S2 balanced strict metadata ablation commands\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
