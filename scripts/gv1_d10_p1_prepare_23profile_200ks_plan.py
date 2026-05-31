#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a 23-profile 200ks plan excluding/flagging B1_2C battery-8.

This script does not start training. It creates:
  - d10_p1_23profile_manifest_excluding_battery8.csv
  - run_d10_p1_23profile_200ks_excluding_battery8.generated.ps1

Run the generated ps1 only after D10-P0 judgement confirms that battery-8 should
be flagged/excluded for the next medium-window verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_CACHE_ROOT = Path(r"E:/XJTU battery dataset/_gv1_cache")
DEFAULT_PROJECT_ROOT = Path(r"C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/PINN-for-ASSB-V1")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pick(row: dict[str, Any], names: list[str], default: str = "") -> str:
    lower = {str(k).lower(): k for k in row}
    for name in names:
        k = lower.get(name.lower())
        if k is not None and str(row.get(k, "")).strip():
            return str(row.get(k, "")).strip()
    return default


def _safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:160] or "profile"


def _is_excluded(label: str, exclude_regex: str) -> bool:
    return re.search(exclude_regex, label, flags=re.IGNORECASE) is not None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--project_root", default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--profile_manifest", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--output_root", default=None)
    ap.add_argument("--python_exe", default=r"D:/Anaconda/envs/torchgpu/python.exe")
    ap.add_argument("--exclude_regex", default=r"(B1.*2C.*battery[-_ ]?8|battery[-_ ]?8.*2C|0008.*battery[-_ ]?8)")
    ap.add_argument("--time_window_s", type=float, default=200000.0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--max_time_points", type=int, default=12000)
    ap.add_argument("--prediction_time_points", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile_adaptive_mode", default="auto")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    project_root = Path(args.project_root)
    profile_manifest = Path(args.profile_manifest) if args.profile_manifest else (
        cache_root / "xjtu_batch134_training_ready" / "xjtu_batch134_profile_manifest.csv"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (cache_root / "xjtu_batch134_d10_p1_23profile_200ks_plan")
    output_root = Path(args.output_root) if args.output_root else (cache_root / "xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8")
    if not profile_manifest.exists():
        raise FileNotFoundError(profile_manifest)
    rows = _read_csv(profile_manifest)
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        profile_npz = _pick(row, ["profile_npz", "solution_npz", "npz_path", "profile_path", "solution_path"])
        label_parts = [
            _pick(row, ["profile_id", "profile_uid", "cell_uid", "cell_id", "source_file"], default=f"profile_{i:02d}"),
            profile_npz,
            _pick(row, ["protocol", "batch", "split"], default=""),
        ]
        label = " | ".join([x for x in label_parts if x])
        profile_run_id = _safe_id(_pick(row, ["profile_id", "cell_uid", "cell_id"], default=Path(profile_npz).parent.name or f"profile_{i:02d}"))
        new = dict(row)
        new.update({
            "profile_npz": profile_npz,
            "d10_label": label,
            "profile_run_id": profile_run_id,
            "d10_output_dir": str(output_root / profile_run_id),
        })
        if _is_excluded(label, args.exclude_regex):
            new["d10_exclude_reason"] = "matched_battery8_exclude_regex"
            excluded.append(new)
        else:
            kept.append(new)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "d10_p1_23profile_manifest_excluding_battery8.csv"
    excluded_path = out_dir / "d10_p1_excluded_profiles.csv"
    _write_csv(manifest_path, kept)
    _write_csv(excluded_path, excluded)

    ps_path = out_dir / "run_d10_p1_23profile_200ks_excluding_battery8.generated.ps1"
    # PS script reads manifest and runs profiles sequentially to avoid hidden parallel failures.
    # Keep path normalization outside the f-string for Python < 3.12 compatibility.
    python_exe_norm = str(args.python_exe).replace("\\", "/")
    ps = f'''param(
  [string]$ProjectRoot = "{project_root.as_posix()}",
  [string]$PythonExe = "{python_exe_norm}",
  [string]$ManifestCsv = "{manifest_path.as_posix()}",
  [string]$OutputRoot = "{output_root.as_posix()}",
  [int]$Epochs = {int(args.epochs)},
  [int]$BatchSize = {int(args.batch_size)},
  [int]$MaxTimePoints = {int(args.max_time_points)},
  [int]$PredictionTimePoints = {int(args.prediction_time_points)},
  [double]$TimeWindowS = {float(args.time_window_s)},
  [int]$Seed = {int(args.seed)},
  [string]$ProfileAdaptiveMode = "{args.profile_adaptive_mode}"
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$rows = Import-Csv $ManifestCsv
Write-Host "D10-P1 23-profile 200ks plan: profiles=" $rows.Count
foreach ($row in $rows) {{
  $out = $row.d10_output_dir
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  Write-Host "=== D10-P1 training" $row.profile_run_id "==="
  & $PythonExe "scripts/gv1_train_conditioned_pinn.py" `
    --solution_npz $row.profile_npz `
    --output_dir $out `
    --profile_adaptive_mode $ProfileAdaptiveMode `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --max_time_points $MaxTimePoints `
    --prediction_time_points $PredictionTimePoints `
    --time_window_s $TimeWindowS `
    --seed $Seed `
    --device auto
  if ($LASTEXITCODE -ne 0) {{ throw "Training failed for $($row.profile_run_id)" }}
  & $PythonExe "scripts/gv1_d10_metrics_from_prediction.py" `
    --prediction_npz (Join-Path $out "prediction.npz") `
    --output_json (Join-Path $out "d10_voltage_metrics.json") `
    --output_csv (Join-Path $out "d10_voltage_metrics_by_segment.csv")
  if ($LASTEXITCODE -ne 0) {{ throw "Metrics failed for $($row.profile_run_id)" }}
}}
& $PythonExe "scripts/gv1_d10_p1_collect_scorecard.py" --runs_root $OutputRoot --out_dir $OutputRoot
'''
    ps_path.write_text(ps, encoding="utf-8")
    report = {
        "ok": True,
        "profile_manifest": str(profile_manifest),
        "kept_count": len(kept),
        "excluded_count": len(excluded),
        "excluded_labels": [r.get("d10_label") for r in excluded],
        "manifest_csv": str(manifest_path),
        "excluded_csv": str(excluded_path),
        "generated_ps1": str(ps_path),
        "output_root": str(output_root),
        "note": "Run the generated PS1 only after D10-P0 judgement confirms battery-8 should be excluded/flagged.",
    }
    (out_dir / "d10_p1_prepare_23profile_plan_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
