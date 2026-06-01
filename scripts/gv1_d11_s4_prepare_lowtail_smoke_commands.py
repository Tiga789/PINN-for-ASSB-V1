#!/usr/bin/env python
"""Prepare D11-S4 low-voltage tail correction smoke commands.

This script generates PowerShell run scripts only. It does not launch training.
It keeps B1_2C battery-8 excluded and avoids hard voltage clamps.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

FOCUS_PROFILE_ORDER = [
    "Batch-1_2C_battery-6",
    "Batch-1_2C_battery-7",
    "Batch-3_R2.5_battery-7",
    "Batch-3_R2.5_battery-8",
    "Batch-4_R3_battery-6",
    "Batch-4_R3_battery-7",
]

MODES = {
    "baseline_d951": {
        "profile_adaptive_mode": "d951",
        "rare_loss_warmup_start_frac": 0.30,
        "rare_loss_warmup_full_frac": 0.85,
        "rare_loss_start_scale": 0.05,
        "rare_loss_final_scale": 1.00,
        "rare_sample_start_scale": 0.30,
        "rare_sample_final_scale": 0.80,
        "description": "D9.5.1 reference settings; trend-first warmup rare-regime mainline.",
    },
    "lowtail_mild": {
        "profile_adaptive_mode": "trend_tail_hybrid",
        "rare_loss_warmup_start_frac": 0.42,
        "rare_loss_warmup_full_frac": 0.95,
        "rare_loss_start_scale": 0.02,
        "rare_loss_final_scale": 0.85,
        "rare_sample_start_scale": 0.20,
        "rare_sample_final_scale": 0.75,
        "description": "Delayed low-tail emphasis; conservative candidate to protect global trend.",
    },
    "lowtail_strong_safe": {
        "profile_adaptive_mode": "trend_tail_hybrid",
        "rare_loss_warmup_start_frac": 0.55,
        "rare_loss_warmup_full_frac": 0.98,
        "rare_loss_start_scale": 0.01,
        "rare_loss_final_scale": 1.10,
        "rare_sample_start_scale": 0.15,
        "rare_sample_final_scale": 0.90,
        "description": "Stronger but late low-tail emphasis; still no hard clamp.",
    },
}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_manifest(cache_root: Path) -> Path:
    candidates = [
        cache_root / "xjtu_batch134_training_ready" / "xjtu_batch134_profile_manifest.csv",
        cache_root / "xjtu_batch134_training_ready" / "profile_manifest.csv",
        cache_root / "xjtu_batch134_training_ready" / "xjtu_batch134_profile_manifest_labeled_only.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No profile manifest found under xjtu_batch134_training_ready")


def col_first(row: Dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return str(row[name]).strip()
    return default


def normalize_profile_name(s: str) -> str:
    s = str(s).replace("\\", "/")
    # Common XJTU names already contain Batch-1_2C_battery-1 etc.
    m = re.search(r"Batch[-_]?([134])[_/\- ]*([A-Za-z0-9.]+)[_/\- ]*battery[-_ ]?([0-9]+)", s, re.IGNORECASE)
    if m:
        batch = m.group(1)
        protocol = m.group(2).replace("2.5", "2.5").replace("R25", "R2.5")
        batt = int(m.group(3))
        if protocol.lower() in {"r2.5", "r25", "r2_5"}:
            protocol = "R2.5"
        elif protocol.lower() == "r3":
            protocol = "R3"
        elif protocol.lower() == "2c":
            protocol = "2C"
        return f"Batch-{batch}_{protocol}_battery-{batt}"
    # Try from file path pattern.
    m2 = re.search(r"(Batch[-_]?[134]).*?(2C|R2\.5|R25|R3).*?battery[-_ ]?([0-9]+)", s, re.IGNORECASE)
    if m2:
        batch = m2.group(1).replace("_", "-")
        protocol = m2.group(2).replace("R25", "R2.5")
        batt = int(m2.group(3))
        return f"{batch}_{protocol}_battery-{batt}"
    return Path(s).stem


def infer_protocol(profile_name: str, row: Dict[str, str]) -> str:
    p = col_first(row, ["protocol", "protocol_id", "protocol_name", "experiment_protocol"], "")
    if p:
        p = p.strip()
        if p.lower() in {"r25", "r2_5", "r2.5"}:
            return "R2.5"
        if p.lower() == "r3":
            return "R3"
        if p.lower() == "2c":
            return "2C"
        return p
    if "R2.5" in profile_name or "R25" in profile_name:
        return "R2.5"
    if "R3" in profile_name:
        return "R3"
    if "2C" in profile_name:
        return "2C"
    return "unknown"


def find_profile_npz(row: Dict[str, str], cache_root: Path) -> Optional[Path]:
    val = col_first(row, ["profile_npz", "solution_npz", "solution_path", "npz_path", "profile_path", "replay_profile_npz"], "")
    if val:
        p = Path(val)
        if not p.is_absolute():
            p = cache_root / val
        if p.exists():
            return p
    # If manifest row contains a profile/output directory, search one level.
    dir_val = col_first(row, ["profile_dir", "output_dir", "profile_output_dir"], "")
    if dir_val:
        d = Path(dir_val)
        if not d.is_absolute():
            d = cache_root / dir_val
        if d.exists():
            for name in ["solution_replay_profile.npz", "solution.npz", "profile.npz"]:
                p = d / name
                if p.exists():
                    return p
    return None


def load_profiles(cache_root: Path) -> List[Dict[str, str]]:
    manifest = find_manifest(cache_root)
    rows = read_csv_rows(manifest)
    profiles: List[Dict[str, str]] = []
    for row in rows:
        npz = find_profile_npz(row, cache_root)
        if npz is None:
            continue
        raw_name = col_first(row, ["profile", "profile_id", "cell_uid", "cell_id", "source_file", "file", "profile_npz"], str(npz))
        profile = normalize_profile_name(raw_name)
        protocol = infer_protocol(profile, row)
        profiles.append({
            "profile": profile,
            "protocol": protocol,
            "profile_npz": str(npz),
            "manifest": str(manifest),
        })
    # De-duplicate by profile.
    seen = set()
    dedup = []
    for p in profiles:
        key = p["profile"]
        if key not in seen:
            seen.add(key)
            dedup.append(p)
    return dedup


def is_b1_battery8(profile: str) -> bool:
    return ("Batch-1" in profile or "Batch_1" in profile) and ("battery-8" in profile or "battery_8" in profile)


def select_profiles(profiles: List[Dict[str, str]], n: int = 6) -> List[Dict[str, str]]:
    by_name = {p["profile"]: p for p in profiles if not is_b1_battery8(p["profile"])}
    selected: List[Dict[str, str]] = []
    for name in FOCUS_PROFILE_ORDER:
        if name in by_name and name not in {x["profile"] for x in selected}:
            selected.append(by_name[name])
    if len(selected) >= n:
        return selected[:n]
    # Fallback: balanced by protocol.
    for protocol in ["2C", "R2.5", "R3"]:
        candidates = [p for p in profiles if p["protocol"] == protocol and not is_b1_battery8(p["profile"])]
        candidates = sorted(candidates, key=lambda x: x["profile"])
        for c in candidates:
            if c["profile"] not in {x["profile"] for x in selected}:
                selected.append(c)
            if len([x for x in selected if x["protocol"] == protocol]) >= 2:
                break
    if len(selected) < n:
        for c in sorted([p for p in profiles if not is_b1_battery8(p["profile"])], key=lambda x: x["profile"]):
            if c["profile"] not in {x["profile"] for x in selected}:
                selected.append(c)
            if len(selected) >= n:
                break
    return selected[:n]


def detect_arg(train_script_text: str, preferred: str, alternatives: List[str]) -> str:
    if preferred in train_script_text:
        return preferred
    for alt in alternatives:
        if alt in train_script_text:
            return alt
    return preferred


def ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_train_command(
    python_exe: str,
    project_root: Path,
    train_script: Path,
    args_map: Dict[str, str],
    profile_npz: str,
    output_dir: str,
    mode_name: str,
    mode_cfg: Dict[str, object],
    profile: str,
    seed: int,
    epochs: int,
    time_window_s: int,
    max_time_points: int,
    batch_size: int,
) -> str:
    pieces = [
        "&", ps_quote(python_exe), ps_quote(str(train_script)),
        args_map["profile_npz"], ps_quote(profile_npz),
        args_map["output_dir"], ps_quote(output_dir),
        "--epochs", str(epochs),
        "--time_window_s", str(time_window_s),
        "--max_time_points", str(max_time_points),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--profile_adaptive_mode", str(mode_cfg["profile_adaptive_mode"]),
        "--rare_loss_warmup_start_frac", str(mode_cfg["rare_loss_warmup_start_frac"]),
        "--rare_loss_warmup_full_frac", str(mode_cfg["rare_loss_warmup_full_frac"]),
        "--rare_loss_start_scale", str(mode_cfg["rare_loss_start_scale"]),
        "--rare_loss_final_scale", str(mode_cfg["rare_loss_final_scale"]),
        "--rare_sample_start_scale", str(mode_cfg["rare_sample_start_scale"]),
        "--rare_sample_final_scale", str(mode_cfg["rare_sample_final_scale"]),
    ]
    # These optional args are added only if present in training script.
    if args_map.get("enable_hard_clamp"):
        pieces += [args_map["enable_hard_clamp"], "False"]
    if args_map.get("run_tag"):
        pieces += [args_map["run_tag"], f"D11-S4_{mode_name}_{profile}"]
    return " ".join(pieces)


def write_ps1(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1")
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--python_exe", default=r"D:\Anaconda\envs\torchgpu\python.exe")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--time_window_s", type=int, default=40000)
    ap.add_argument("--max_time_points", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    cache_root = Path(args.cache_root)
    command_dir = cache_root / "xjtu_batch134_d11_s4_lowtail_correction_smoke_commands"
    output_root = cache_root / "xjtu_batch134_d11_s4_lowtail_correction_smoke"
    command_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    train_script = project_root / "scripts" / "gv1_train_conditioned_pinn.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    txt = train_script.read_text(encoding="utf-8", errors="ignore")
    args_map = {
        "profile_npz": detect_arg(txt, "--solution_npz", ["--profile_npz", "--input_npz", "--npz"]),
        "output_dir": detect_arg(txt, "--output_dir", ["--out_dir", "--model_dir", "--run_dir"]),
        "enable_hard_clamp": "--enable_voltage_hard_clamp" if "--enable_voltage_hard_clamp" in txt else "",
        "run_tag": "--run_tag" if "--run_tag" in txt else "",
    }

    profiles = load_profiles(cache_root)
    if not profiles:
        raise RuntimeError("No usable replay profiles found from training-ready manifest.")
    selected = select_profiles(profiles, n=6)
    if len(selected) < 3:
        raise RuntimeError(f"Too few selected profiles: {len(selected)}")

    # Persist selected manifest.
    manifest_path = command_dir / "d11_s4_selected_lowtail_focus_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["profile", "protocol", "profile_npz"])
        w.writeheader()
        for p in selected:
            w.writerow({"profile": p["profile"], "protocol": p["protocol"], "profile_npz": p["profile_npz"]})

    all_scripts = []
    for mode_name, mode_cfg in MODES.items():
        lines = []
        lines.append("$ErrorActionPreference = 'Stop'")
        lines.append(f"Set-Location {ps_quote(str(project_root))}")
        lines.append(f"Write-Host '==== D11-S4 mode: {mode_name} ===='")
        for p in selected:
            profile = p["profile"]
            out_dir = output_root / mode_name / profile
            cmd = build_train_command(
                args.python_exe,
                project_root,
                train_script,
                args_map,
                p["profile_npz"],
                str(out_dir),
                mode_name,
                mode_cfg,
                profile,
                args.seed,
                args.epochs,
                args.time_window_s,
                args.max_time_points,
                args.batch_size,
            )
            lines.append("")
            lines.append(f"Write-Host '--- {profile} ---'")
            lines.append(cmd)
            lines.append("if ($LASTEXITCODE -ne 0) { throw 'Training command failed.' }")
        ps1 = command_dir / f"run_d11_s4_{mode_name}.generated.ps1"
        write_ps1(ps1, "\n".join(lines) + "\n")
        all_scripts.append(ps1)

    all_lines = ["$ErrorActionPreference = 'Stop'", f"Set-Location {ps_quote(str(project_root))}"]
    for ps1 in all_scripts:
        all_lines.append(f"& {ps_quote(str(ps1))}")
        all_lines.append("if ($LASTEXITCODE -ne 0) { throw 'D11-S4 generated mode script failed.' }")
    all_ps1 = command_dir / "run_d11_s4_all_modes.generated.ps1"
    write_ps1(all_ps1, "\n".join(all_lines) + "\n")

    summary = {
        "ok": True,
        "stage": "D11-S4 low-voltage tail correction smoke command preparation",
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "command_dir": str(command_dir),
        "output_root": str(output_root),
        "train_script": str(train_script),
        "detected_args": args_map,
        "epochs": args.epochs,
        "time_window_s": args.time_window_s,
        "max_time_points": args.max_time_points,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "selected_profiles": selected,
        "modes": MODES,
        "expected_runs": len(selected) * len(MODES),
        "generated_scripts": [str(x) for x in all_scripts] + [str(all_ps1)],
        "battery8_policy": "B1_2C battery-8 remains excluded from D11-S4 smoke.",
        "mainline_policy": "Do not promote a low-tail mode unless global trend and low-target segments both improve.",
    }
    (command_dir / "d11_s4_command_preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
