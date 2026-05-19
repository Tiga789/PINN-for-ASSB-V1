# -*- coding: utf-8 -*-
"""Build ModelFin_112 deterministic five-target wrapper.

This script packages a frozen ModelFin_107A state source and the deterministic
ridge SOH head into one auditable model directory.  It does not retrain either
component and it does not use held-out test metrics for selection.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb112_deterministic_wrapper import (
    default_state_npz_candidates,
    default_state_scorecard_candidates,
    load_json,
    save_json,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ModelFin_112 deterministic wrapper")
    p.add_argument("--state_core_dir", default="ModelFin_107A", help="Frozen state model directory; referenced for provenance")
    p.add_argument("--state_eval_dir", default="", help="Optional state EvalFin directory")
    p.add_argument("--state_eval_npz", default="", help="Optional paired state eval NPZ for evaluator")
    p.add_argument("--state_scorecard_csv", default="", help="Optional CSV containing cs_a/cs_c/phie/phis_c metrics")
    p.add_argument("--soh_model_dir", "--soh_head_dir", dest="soh_model_dir", default="ModelFin_112_deterministicSOH_ridge_g4")
    p.add_argument("--dataset_csv", default=r"Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv")
    p.add_argument("--split_manifest_json", default=r"Data\assb111_seed42locked_repro_c00\split_manifest.json")
    p.add_argument("--output_model_dir", default="ModelFin_112_deterministic_wrapper")
    p.add_argument("--model_name", default="ModelFin_112_deterministic_wrapper")
    p.add_argument("--copy_state_eval_npz", action="store_true", help="Copy state eval NPZ if provided/found; otherwise reference by path")
    p.add_argument("--copy_state_scorecard", action="store_true", help="Copy state scorecard if provided/found; otherwise reference by path")
    p.add_argument("--clean", action="store_true", help="Remove output_model_dir before rebuilding")
    return p.parse_args(argv)


def _maybe_copy(src: Path, dst: Path, enabled: bool) -> str:
    if not src or not src.exists():
        return ""
    if enabled:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst.name if dst.parent.name == dst.parent.parent.name else dst.relative_to(dst.parent.parent)) if False else str(dst)
    return str(src)


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_model_dir)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    soh_dir = Path(args.soh_model_dir)
    if not soh_dir.exists():
        raise FileNotFoundError(f"SOH deterministic model directory not found: {soh_dir}")
    required = ["deterministic_soh_model.json", "feature_scaler.json", "feature_schema.json", "train_summary.json", "selected_checkpoint_audit.json", "metrics_soh_by_split_final_report.json"]
    missing = [name for name in required if not (soh_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"SOH model directory is missing required deterministic files: {missing}")

    train_summary = load_json(soh_dir / "train_summary.json")
    audit = load_json(soh_dir / "selected_checkpoint_audit.json")
    if train_summary.get("model_variant") != "deterministic_ridge_soh_head":
        raise RuntimeError(f"Expected deterministic_ridge_soh_head, got {train_summary.get('model_variant')}")
    if bool(train_summary.get("test_metrics_used_for_selection", True)):
        raise RuntimeError("SOH train_summary says test_metrics_used_for_selection=true; refusing to package")
    if not bool(train_summary.get("no_test_metrics_in_training_history", False)):
        raise RuntimeError("SOH train_summary does not confirm no_test_metrics_in_training_history=true")
    if bool(audit.get("test_metrics_used_for_selection", True)):
        raise RuntimeError("selected_checkpoint_audit says test_metrics_used_for_selection=true; refusing to package")

    # Copy deterministic SOH artifacts into a subdirectory for one-dir operation.
    soh_sub = out / "soh_deterministic"
    soh_sub.mkdir(exist_ok=True)
    for name in [
        "deterministic_soh_model.json", "ridge_model.json", "feature_scaler.json", "feature_schema.json",
        "train_summary.json", "training_summary.json", "selected_checkpoint_audit.json",
        "metrics_soh_by_split_final_report.json", "metrics_soh_by_split_train_eval.json",
        "deterministic_soh_scorecard.csv", "feature_importance.csv", "soh_pred_by_cycle.csv",
        "alpha_selection_visible_only.csv",
    ]:
        _copy_if_exists(soh_dir / name, soh_sub / name)

    state_scorecard = Path(args.state_scorecard_csv) if args.state_scorecard_csv else None
    if not state_scorecard or not state_scorecard.exists():
        candidates = []
        if args.state_eval_dir:
            candidates += [Path(args.state_eval_dir) / "five_state_scorecard.csv", Path(args.state_eval_dir) / "metrics_global.csv"]
        candidates += default_state_scorecard_candidates(ROOT)
        state_scorecard = _first_existing(candidates)

    state_eval_npz = Path(args.state_eval_npz) if args.state_eval_npz else None
    if not state_eval_npz or not state_eval_npz.exists():
        candidates = []
        if args.state_eval_dir:
            candidates += [Path(args.state_eval_dir) / "evaluation_paired.npz", Path(args.state_eval_dir) / "eval_paired.npz"]
        candidates += default_state_npz_candidates(ROOT)
        state_eval_npz = _first_existing(candidates)

    state_scorecard_ref = ""
    if state_scorecard and state_scorecard.exists():
        if args.copy_state_scorecard:
            dst = out / "state_scorecard.csv"
            shutil.copy2(state_scorecard, dst)
            state_scorecard_ref = "state_scorecard.csv"
        else:
            state_scorecard_ref = str(state_scorecard)

    state_eval_npz_ref = ""
    if state_eval_npz and state_eval_npz.exists():
        if args.copy_state_eval_npz:
            dst = out / "state_eval_paired.npz"
            shutil.copy2(state_eval_npz, dst)
            state_eval_npz_ref = "state_eval_paired.npz"
        else:
            state_eval_npz_ref = str(state_eval_npz)

    # Optional metadata copies from state core.
    state_meta = out / "state_core_metadata"
    state_core_dir = Path(args.state_core_dir)
    if state_core_dir.exists():
        for name in ["config.json", "gauge_config.json", "csA_correction_config.json", "train_summary.json", "training_summary.json"]:
            _copy_if_exists(state_core_dir / name, state_meta / name)

    split_ref = str(args.split_manifest_json)
    if args.split_manifest_json and Path(args.split_manifest_json).exists():
        shutil.copy2(args.split_manifest_json, out / "split_manifest.json")
        split_ref = "split_manifest.json"

    cfg: Dict[str, Any] = {
        "model_name": args.model_name,
        "model_level": "L1_engineering_single_directory_wrapper",
        "state_core_type": "frozen_ModelFin_107A_state_source",
        "state_core_dir": str(args.state_core_dir),
        "state_eval_npz": state_eval_npz_ref,
        "state_scorecard_csv": state_scorecard_ref,
        "soh_model_type": "deterministic_ridge_soh_head",
        "soh_model_dir": "soh_deterministic",
        "dataset_csv": str(args.dataset_csv),
        "split_manifest_json": split_ref,
        "feature_schema_json": "soh_deterministic/feature_schema.json",
        "feature_scaler_json": "soh_deterministic/feature_scaler.json",
        "deterministic_soh_model_json": "soh_deterministic/deterministic_soh_model.json",
        "boundary_note": "This is one auditable wrapper directory: frozen 107A state source + deterministic ridge SOH. It is not an end-to-end jointly trained single neural network.",
        "no_test_metrics_in_training_history": True,
        "test_metrics_used_for_selection": False,
    }
    save_json(cfg, out / "unified_config.json")

    build_audit = {
        "ok": True,
        "output_model_dir": str(out),
        "model_name": args.model_name,
        "soh_model_dir": str(soh_dir),
        "soh_model_variant": train_summary.get("model_variant"),
        "soh_selected_alpha": train_summary.get("selected_alpha"),
        "soh_no_test_metrics_in_training_history": train_summary.get("no_test_metrics_in_training_history"),
        "soh_test_metrics_used_for_selection": train_summary.get("test_metrics_used_for_selection"),
        "state_core_dir": str(args.state_core_dir),
        "state_scorecard_csv": state_scorecard_ref,
        "state_eval_npz": state_eval_npz_ref,
        "warning_if_state_source_empty": "Evaluator can still run SOH if state_scorecard_csv/state_eval_npz is empty, but five-target scorecard requires one of them.",
    }
    save_json(build_audit, out / "build_audit.json")
    save_json({"ok": True, "selected_model": "unified_config.json", "test_metrics_used_for_selection": False}, out / "selected_checkpoint_audit.json")

    print(f"[OK] deterministic wrapper built: {out}")
    print(f"     SOH model      : {soh_dir}")
    print(f"     state scorecard: {state_scorecard_ref or '[not found; evaluator will try NPZ]'}")
    print(f"     state eval NPZ : {state_eval_npz_ref or '[not found]'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
