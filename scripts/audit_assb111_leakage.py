# -*- coding: utf-8 -*-
r"""Run ASSB111 strict-30 leakage audits.

The audit is deliberately conservative: it verifies split safety, feature names,
scaler provenance, and training history column names. It does not forbid storing
held-out SOH labels in the final evaluation dataset; it forbids using them in
loss, scaler fitting, or model selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb111_split import load_manifest
from util.assb111_feature_schema import select_feature_columns
from util.assb111_leakage_guard import audit_assb111_dataset, write_audit_json


def _load_json_if_exists(path: str):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_clean(x: Any) -> Any:
    import math
    if isinstance(x, dict):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Audit ASSB111 leakage controls")
    p.add_argument("--dataset_csv", default="Data/assb111/dataset_strict30.csv")
    p.add_argument("--split_manifest_json", default="Data/assb111/split_manifest.json")
    p.add_argument("--feature_mode", default="p1_107a_strict")
    p.add_argument("--allow_upper_bound", action="store_true")
    p.add_argument("--feature_columns", default="", help="optional comma-separated feature list; otherwise schema selection is used")
    p.add_argument("--scaler_json", default="ModelFin_111/feature_scaler.json")
    p.add_argument("--train_history_csv", default="ModelFin_111/train_history.csv")
    p.add_argument("--output_json", default="ModelFin_111/leakage_audit.json")
    p.add_argument("--soft_fail", action="store_true", help="write audit but do not return nonzero on failure")
    args = p.parse_args(argv)

    frame = pd.read_csv(args.dataset_csv)
    manifest = load_manifest(args.split_manifest_json)
    if args.feature_columns.strip():
        feature_columns = [c.strip() for c in args.feature_columns.split(",") if c.strip()]
    else:
        feature_columns = select_feature_columns(
            frame,
            args.feature_mode,
            allow_upper_bound=bool(args.allow_upper_bound),
            allow_missing=False,
        )
    scaler = _load_json_if_exists(args.scaler_json)
    history = None
    if str(args.train_history_csv).strip():
        hp = Path(args.train_history_csv)
        if hp.exists() and hp.is_file():
            history = pd.read_csv(hp)
    result = audit_assb111_dataset(
        frame,
        manifest=manifest,
        feature_columns=feature_columns,
        feature_mode=args.feature_mode,
        allow_upper_bound=bool(args.allow_upper_bound),
        scaler=scaler,
        fit_splits=("train",),
        train_history=history,
    )
    write_audit_json(result, args.output_json)
    print("[audit_assb111_leakage] wrote", args.output_json)
    print(json.dumps(_json_clean({
        "ok": result.ok,
        "failures": result.failures,
        "warnings": result.warnings,
        "details": result.details,
    }), ensure_ascii=False, indent=2))
    if not result.ok and not args.soft_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
