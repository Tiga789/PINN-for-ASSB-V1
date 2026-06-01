#!/usr/bin/env python
"""Prepare D11-S7 low-target correction amplitude repair commands.

This script only generates PowerShell scripts.  It does not launch training.
D11-S7 is a 6-profile / 40 ks repair smoke after D11-S6 showed a structural low-voltage floor / output-transform capacity barrier. The goal is to test an explicit low-voltage escape branch in 6-profile / 40 ks smoke runs before any 200 ks confirmation.
"""
from __future__ import annotations

import argparse
import csv
import json
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

# Optional CLI names are detected dynamically.  If the current train script does
# not expose a specific option, that option is skipped and recorded in the
# preparation summary.  This keeps the package compatible with the current GV1
# mainline without forcing a mainline code overwrite.
OPTIONAL_ARGS = {
    "enable_low_voltage_escape": ["--enable_low_voltage_escape"],
    "low_voltage_escape_scale_V": ["--low_voltage_escape_scale_V"],
    "low_voltage_escape_gate_center_V": ["--low_voltage_escape_gate_center_V"],
    "low_voltage_escape_gate_width_V": ["--low_voltage_escape_gate_width_V"],
    "low_voltage_escape_pred_center_V": ["--low_voltage_escape_pred_center_V"],
    "low_voltage_escape_pred_width_V": ["--low_voltage_escape_pred_width_V"],
    "voltage_tail": ["--voltage_tail_weight", "--w_voltage_tail", "--loss_voltage_tail"],
    "voltage_tail_balance": ["--voltage_tail_balance_weight", "--w_voltage_tail_balance"],
    "tail_fraction": ["--tail_fraction", "--low_tail_fraction", "--voltage_tail_fraction"],
    "tail_weight_gain": ["--tail_weight_gain", "--voltage_tail_weight_gain"],
    "low_tail_extra_gain": ["--low_tail_extra_gain", "--low_target_extra_gain", "--low_voltage_tail_gain"],
    "high_tail_extra_gain": ["--high_tail_extra_gain"],
    "low_voltage_gate_center_V": ["--low_voltage_gate_center_V", "--low_gate_center_V"],
    "low_voltage_gate_width_V": ["--low_voltage_gate_width_V", "--low_gate_width_V"],
    "direct_voltage_mix": ["--direct_voltage_mix"],
    "low_tail_correction_scale": ["--low_tail_correction_scale", "--voltage_low_tail_scale", "--low_voltage_tail_scale"],
    "low_tail_downward_gain": ["--low_tail_downward_gain", "--lowtarget_downward_gain", "--voltage_low_tail_downward_gain"],
}

MODES: Dict[str, Dict[str, object]] = {
    "baseline_d951": {
        "profile_adaptive_mode": "d951",
        "rare_loss_warmup_start_frac": 0.30,
        "rare_loss_warmup_full_frac": 0.85,
        "rare_loss_start_scale": 0.05,
        "rare_loss_final_scale": 1.00,
        "rare_sample_start_scale": 0.30,
        "rare_sample_final_scale": 0.80,
        "optional": {
            "enable_low_voltage_escape": "False",
        },
        "description": "D9.5.1 reference; low-voltage escape disabled.",
    },
    "lowvoltage_escape_mild": {
        "profile_adaptive_mode": "d951",
        "rare_loss_warmup_start_frac": 0.30,
        "rare_loss_warmup_full_frac": 0.85,
        "rare_loss_start_scale": 0.05,
        "rare_loss_final_scale": 1.00,
        "rare_sample_start_scale": 0.30,
        "rare_sample_final_scale": 0.80,
        "optional": {
            "enable_low_voltage_escape": "True",
            "low_voltage_escape_scale_V": 0.35,
            "low_voltage_escape_gate_center_V": 3.08,
            "low_voltage_escape_gate_width_V": 0.20,
            "low_voltage_escape_pred_center_V": 3.55,
            "low_voltage_escape_pred_width_V": 0.20,
            "direct_voltage_mix": 0.80,
            "voltage_tail": 0.38,
            "voltage_tail_balance": 0.06,
            "tail_fraction": 0.24,
            "tail_weight_gain": 2.2,
            "low_tail_extra_gain": 3.5,
            "high_tail_extra_gain": 0.35,
        },
        "description": "Mild explicit escape: subtract up to ~0.35 V in activated low-voltage region.",
    },
    "lowvoltage_escape_medium": {
        "profile_adaptive_mode": "d951",
        "rare_loss_warmup_start_frac": 0.35,
        "rare_loss_warmup_full_frac": 0.90,
        "rare_loss_start_scale": 0.03,
        "rare_loss_final_scale": 1.10,
        "rare_sample_start_scale": 0.20,
        "rare_sample_final_scale": 0.90,
        "optional": {
            "enable_low_voltage_escape": "True",
            "low_voltage_escape_scale_V": 0.55,
            "low_voltage_escape_gate_center_V": 3.10,
            "low_voltage_escape_gate_width_V": 0.22,
            "low_voltage_escape_pred_center_V": 3.58,
            "low_voltage_escape_pred_width_V": 0.22,
            "direct_voltage_mix": 0.78,
            "voltage_tail": 0.45,
            "voltage_tail_balance": 0.08,
            "tail_fraction": 0.25,
            "tail_weight_gain": 2.6,
            "low_tail_extra_gain": 4.2,
            "high_tail_extra_gain": 0.30,
        },
        "description": "Medium explicit escape: designed to break the ~3.4 V low-target prediction floor.",
    },
    "lowvoltage_escape_strong_guarded": {
        "profile_adaptive_mode": "d951",
        "rare_loss_warmup_start_frac": 0.40,
        "rare_loss_warmup_full_frac": 0.95,
        "rare_loss_start_scale": 0.02,
        "rare_loss_final_scale": 1.20,
        "rare_sample_start_scale": 0.15,
        "rare_sample_final_scale": 1.00,
        "optional": {
            "enable_low_voltage_escape": "True",
            "low_voltage_escape_scale_V": 0.75,
            "low_voltage_escape_gate_center_V": 3.12,
            "low_voltage_escape_gate_width_V": 0.24,
            "low_voltage_escape_pred_center_V": 3.60,
            "low_voltage_escape_pred_width_V": 0.24,
            "direct_voltage_mix": 0.76,
            "voltage_tail": 0.52,
            "voltage_tail_balance": 0.10,
            "tail_fraction": 0.26,
            "tail_weight_gain": 3.0,
            "low_tail_extra_gain": 5.0,
            "high_tail_extra_gain": 0.25,
        },
        "description": "Strong guarded escape.  Promotion requires low_target improvement and no rest/high-tail damage.",
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
    m = re.search(r"Batch[-_]?([134])[_/\- ]*([A-Za-z0-9.]+)[_/\- ]*battery[-_ ]?([0-9]+)", s, re.IGNORECASE)
    if m:
        batch = m.group(1)
        protocol = m.group(2).replace("R25", "R2.5").replace("R2_5", "R2.5")
        batt = int(m.group(3))
        if protocol.lower() in {"r2.5", "r25", "r2_5"}:
            protocol = "R2.5"
        elif protocol.lower() == "r3":
            protocol = "R3"
        elif protocol.lower() == "2c":
            protocol = "2C"
        return f"Batch-{batch}_{protocol}_battery-{batt}"
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
        profiles.append({"profile": profile, "protocol": protocol, "profile_npz": str(npz), "manifest": str(manifest)})
    seen = set(); dedup = []
    for p in profiles:
        if p["profile"] not in seen:
            seen.add(p["profile"]); dedup.append(p)
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
    for protocol in ["2C", "R2.5", "R3"]:
        candidates = sorted([p for p in profiles if p["protocol"] == protocol and not is_b1_battery8(p["profile"])], key=lambda x: x["profile"])
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


def detect_optional_args(train_script_text: str) -> Dict[str, str]:
    accepted: Dict[str, str] = {}
    for semantic, names in OPTIONAL_ARGS.items():
        for name in names:
            if name in train_script_text:
                accepted[semantic] = name
                break
    return accepted


def ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def build_train_command(
    python_exe: str,
    train_script: Path,
    args_map: Dict[str, str],
    optional_cli: Dict[str, str],
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
) -> Tuple[str, Dict[str, object], Dict[str, object]]:
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
    if args_map.get("enable_hard_clamp"):
        pieces += [args_map["enable_hard_clamp"], "False"]
    if args_map.get("run_tag"):
        pieces += [args_map["run_tag"], f"D11-S7_{mode_name}_{profile}"]

    accepted_values: Dict[str, object] = {}
    skipped_values: Dict[str, object] = {}
    for semantic, value in dict(mode_cfg.get("optional", {})).items():
        cli = optional_cli.get(semantic)
        if cli:
            pieces += [cli, str(value)]
            accepted_values[semantic] = {"cli": cli, "value": value}
        else:
            skipped_values[semantic] = value
    return " ".join(pieces), accepted_values, skipped_values


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
    command_dir = cache_root / "xjtu_batch134_d11_s7_lowvoltage_escape_commands"
    output_root = cache_root / "xjtu_batch134_d11_s7_lowvoltage_escape"
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
    optional_cli = detect_optional_args(txt)

    profiles = load_profiles(cache_root)
    selected = select_profiles(profiles, n=6)
    if len(selected) < 6:
        raise RuntimeError(f"Expected 6 selected profiles, got {len(selected)}")

    manifest_path = command_dir / "d11_s7_selected_lowvoltage_escape_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["profile", "protocol", "profile_npz", "manifest"], extrasaction="ignore")
        w.writeheader()
        for p in selected:
            w.writerow(p)

    all_scripts = []
    accepted_optional: Dict[str, Dict[str, object]] = {}
    skipped_optional: Dict[str, Dict[str, object]] = {}
    for mode_name, mode_cfg in MODES.items():
        lines = [
            "$ErrorActionPreference = 'Stop'",
            f"Set-Location {ps_quote(str(project_root))}",
            f"Write-Host '==== D11-S7 mode: {mode_name} ===='",
            "Write-Host 'Policy: 6 profiles, 40ks, battery-8 excluded, metadata_on disabled, hard clamp disabled; low-voltage escape diagnostic.'",
        ]
        accepted_optional[mode_name] = {}
        skipped_optional[mode_name] = {}
        for p in selected:
            profile = p["profile"]
            out_dir = output_root / mode_name / profile
            cmd, acc, skip = build_train_command(
                args.python_exe, train_script, args_map, optional_cli,
                p["profile_npz"], str(out_dir), mode_name, mode_cfg, profile,
                args.seed, args.epochs, args.time_window_s, args.max_time_points, args.batch_size,
            )
            accepted_optional[mode_name] = acc
            skipped_optional[mode_name] = skip
            lines += ["", f"Write-Host '--- {profile} ---'", cmd, "if ($LASTEXITCODE -ne 0) { throw 'Training command failed.' }"]
        ps1 = command_dir / f"run_d11_s7_{mode_name}.generated.ps1"
        write_ps1(ps1, "\n".join(lines) + "\n")
        all_scripts.append(ps1)

    all_lines = ["$ErrorActionPreference = 'Stop'", f"Set-Location {ps_quote(str(project_root))}"]
    for ps1 in all_scripts:
        all_lines += [f"& {ps_quote(str(ps1))}", "if ($LASTEXITCODE -ne 0) { throw 'D11-S7 generated mode script failed.' }"]
    all_ps1 = command_dir / "run_d11_s7_all_modes.generated.ps1"
    write_ps1(all_ps1, "\n".join(all_lines) + "\n")

    summary = {
        "ok": True,
        "stage": "D11-S7 low-target correction amplitude repair command preparation",
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "command_dir": str(command_dir),
        "output_root": str(output_root),
        "train_script": str(train_script),
        "detected_args": args_map,
        "detected_optional_cli_args": optional_cli,
        "accepted_optional_args_by_mode": accepted_optional,
        "skipped_optional_args_by_mode": skipped_optional,
        "epochs": args.epochs,
        "time_window_s": args.time_window_s,
        "max_time_points": args.max_time_points,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "selected_profiles": selected,
        "modes": MODES,
        "expected_runs": len(selected) * len(MODES),
        "generated_scripts": [str(x) for x in all_scripts] + [str(all_ps1)],
        "battery8_policy": "B1_2C battery-8 remains excluded.",
        "promotion_policy": "Do not expand unless low_target and low_target_le_2p75 improve versus baseline and no global/rest/high-tail damage is introduced.",
    }
    (command_dir / "d11_s7_command_preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
