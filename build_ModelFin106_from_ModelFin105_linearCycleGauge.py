#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Materialize ModelFin_106 as a calibrated wrapper of ModelFin_105.

Expected input: an output directory produced by
  calibrate_apply_common_mode_potential_offset.py --method linear_cycle_mean

This script copies ModelFin_105 checkpoint/config files into ModelFin_106 and
stores the linear-cycle gauge metadata in ModelFin_106/gauge_linear_cycle_mean.json.
It does not retrain neural-network weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _copy_tree_non_destructive(src: Path, dst: Path, overwrite: bool = False) -> List[str]:
    if not src.exists():
        raise FileNotFoundError(f"Source model dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
            copied.append(str(target))
        else:
            if target.exists() and not overwrite:
                continue
            shutil.copy2(item, target)
            copied.append(str(target))
    return copied


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ModelFin_106 wrapper from ModelFin_105 + linear-cycle gauge correction.")
    p.add_argument("--source_model_dir", default="ModelFin_105")
    p.add_argument("--target_model_dir", default="ModelFin_106")
    p.add_argument("--gauge_eval_dir", default="EvalFin_106_cycles5_100_v2_massclosed_candidate_linearCycleGauge_softlabel_only")
    p.add_argument("--overwrite_model_files", action="store_true")
    p.add_argument("--no_copy_model", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    src = Path(args.source_model_dir)
    dst = Path(args.target_model_dir)
    gauge_eval_dir = Path(args.gauge_eval_dir)
    summary_path = gauge_eval_dir / "gauge_calibration_summary.json"
    summary = _read_json(summary_path)

    if summary.get("method") != "linear_cycle_mean":
        method = summary.get("method", "<missing>")
        raise ValueError(f"Expected linear_cycle_mean gauge summary, got method={method}")

    calib = summary.get("calibration", {})
    required = ["linear_bias_slope_V_per_cycle", "linear_bias_intercept_V"]
    missing = [k for k in required if k not in calib]
    if missing:
        raise KeyError(f"Gauge calibration summary missing fields: {missing}")

    copied: List[str] = []
    if not args.no_copy_model:
        copied = _copy_tree_non_destructive(src, dst, overwrite=args.overwrite_model_files)
    else:
        dst.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_id": 106,
        "model_name": "ModelFin_106",
        "model_definition": "ModelFin_105 checkpoint plus linear-cycle common-mode potential gauge correction.",
        "base_model_dir": str(src),
        "target_model_dir": str(dst),
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "gauge_eval_dir": str(gauge_eval_dir),
        "gauge_summary_source": str(summary_path),
        "gauge_mode": "linear_cycle_mean_common_mode",
        "calibration": calib,
        "metrics_global_before": summary.get("metrics_global_before"),
        "metrics_global_after": summary.get("metrics_global_after"),
        "common_mode_diagnostic": summary.get("common_mode_diagnostic"),
        "offset_formula": "offset_to_add_V = -(linear_bias_slope_V_per_cycle * cycle_id + linear_bias_intercept_V)",
        "important_note": "This is a calibrated wrapper, not a newly trained neural-network checkpoint. Use the ModelFin_106 linearGauge evaluation script to apply the gauge.",
        "copied_files_or_dirs": copied,
    }
    _write_json(dst / "gauge_linear_cycle_mean.json", meta)
    _write_json(dst / "model106_manifest.json", meta)
    (dst / "MODELFIN_106_README.txt").write_text(
        "ModelFin_106 = ModelFin_105 weights + linear-cycle common-mode potential gauge correction.\n"
        "Gauge metadata: gauge_linear_cycle_mean.json\n"
        "Final cycle5-100 eval directory: EvalFin_106_cycles5_100_v2_massclosed_candidate_linearCycleGauge_softlabel_only\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok",
        "target_model_dir": str(dst),
        "gauge_file": str(dst / "gauge_linear_cycle_mean.json"),
        "linear_bias_slope_V_per_cycle": calib.get("linear_bias_slope_V_per_cycle"),
        "linear_bias_intercept_V": calib.get("linear_bias_intercept_V"),
        "copied_count": len(copied),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
