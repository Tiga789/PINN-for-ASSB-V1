# -*- coding: utf-8 -*-
"""Build the ASSB-111 strict30 SOH dataset.

This script merges non-label cycle features with capacity/SOH targets, attaches
split flags from split_manifest.json, fits a train-only feature scaler, and
runs leakage audit before training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from util.assb111_feature_schema import select_feature_columns, write_schema_json, write_scaler_json
from util.assb111_leakage_guard import attach_split_from_manifest, audit_assb111_dataset, fit_scaler_train_only, save_dataset_with_split, write_audit_json
from util.assb111_split import load_capacity_targets_strict30, load_manifest


def _json_clean(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return [_json_clean(v) for v in x.tolist()]
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    return x


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build ASSB111 strict30 dataset")
    p.add_argument("--features_csv", default=r"Data\assb111\features_107A_cycle.csv")
    p.add_argument("--capacity_target_csv", default=r"Data\assb_capacity_soh_targets\capacity_soh_targets.csv")
    p.add_argument("--split_manifest_json", default=r"Data\assb111\split_manifest.json")
    p.add_argument("--output_dir", default=r"Data\assb111")
    p.add_argument("--dataset_csv", "--output_csv", dest="dataset_csv", default="", help="Default: <output_dir>/dataset.csv")
    p.add_argument("--masked_train_dataset_csv", "--masked_train_csv", dest="masked_train_dataset_csv", default="", help="Default: <output_dir>/masked_train_dataset.csv")
    p.add_argument("--scaler_json", "--feature_scaler_json", dest="scaler_json", default="", help="Default: <output_dir>/feature_scaler.json")
    p.add_argument("--schema_json", "--feature_schema_json", dest="schema_json", default="", help="Default: <output_dir>/feature_schema.json")
    p.add_argument("--audit_json", default="", help="Default: <output_dir>/leakage_audit_dataset.json")
    p.add_argument("--feature_mode", default="p1_107a_strict")
    p.add_argument("--allow_upper_bound", action="store_true")
    p.add_argument("--allow_missing_features", action="store_true")
    p.add_argument("--scaler_scope", default="train", choices=["train", "visible"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = Path(args.dataset_csv) if args.dataset_csv else out_dir / "dataset.csv"
    train_csv = Path(args.masked_train_dataset_csv) if args.masked_train_dataset_csv else out_dir / "masked_train_dataset.csv"
    scaler_json = Path(args.scaler_json) if args.scaler_json else out_dir / "feature_scaler.json"
    schema_json = Path(args.schema_json) if args.schema_json else out_dir / "feature_schema.json"
    audit_json = Path(args.audit_json) if args.audit_json else out_dir / "leakage_audit_dataset.json"

    features = pd.read_csv(args.features_csv)
    if "cycle_id" not in features.columns:
        raise KeyError("features_csv must contain cycle_id")
    features["cycle_id"] = features["cycle_id"].astype(int)
    targets = load_capacity_targets_strict30(args.capacity_target_csv, cycle_from=5, cycle_to=522)
    target_cols = [c for c in ["cycle_id", "Q_obs_Ah", "Q_obs_mAh", "SOH_obs", "complete_cycle", "q_ref_Ah", "q_ref_mAh"] if c in targets.columns]
    data = features.merge(targets[target_cols], on="cycle_id", how="left")
    manifest = load_manifest(args.split_manifest_json)
    data = attach_split_from_manifest(data, manifest)
    feature_columns = select_feature_columns(
        data, args.feature_mode, allow_upper_bound=bool(args.allow_upper_bound), allow_missing=bool(args.allow_missing_features)
    )
    if not feature_columns:
        raise RuntimeError("No feature columns selected. Check feature extraction and feature_mode.")
    scaler = fit_scaler_train_only(data, feature_columns, manifest, scope=args.scaler_scope)
    audit = audit_assb111_dataset(
        data,
        manifest=manifest,
        feature_columns=feature_columns,
        feature_mode=args.feature_mode,
        allow_upper_bound=bool(args.allow_upper_bound),
        scaler=scaler,
        fit_splits=("train",),
    )
    write_audit_json(audit, audit_json)
    if not audit.ok:
        raise RuntimeError("ASSB111 leakage audit failed before training: " + "; ".join(audit.failures))
    save_dataset_with_split(data, manifest, dataset_csv)
    masked = data.copy()
    heldout = masked["split"].astype(str).str.lower().isin(["test", "partial"])
    for col in ["SOH_obs", "Q_obs_Ah", "Q_obs_mAh"]:
        if col in masked.columns:
            masked.loc[heldout, col] = float("nan")
    masked.to_csv(train_csv, index=False, encoding="utf-8-sig")
    write_scaler_json(scaler, scaler_json)
    write_schema_json(schema_json, args.feature_mode, allow_upper_bound=bool(args.allow_upper_bound))
    summary = {
        "dataset_csv": str(dataset_csv),
        "masked_train_dataset_csv": str(train_csv),
        "scaler_json": str(scaler_json),
        "schema_json": str(schema_json),
        "audit_json": str(audit_json),
        "feature_mode": args.feature_mode,
        "n_rows": int(len(data)),
        "n_features": int(len(feature_columns)),
        "feature_columns": list(feature_columns),
        "split_counts": {str(k): int(v) for k, v in data["split"].value_counts().sort_index().to_dict().items()},
        "audit_ok": bool(audit.ok),
        "audit_warnings": list(audit.warnings),
    }
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_clean(summary), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
