# -*- coding: utf-8 -*-
"""Leakage guard for ASSB ModelFin_111 strict-30 SOH prediction.

The central rule is simple: test cycles 160-521 and partial cycle 522 must never
enter training loss, scaler fitting, early stopping, or model selection. This
module turns that rule into reusable checks for dataset construction and model
training scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np
import pandas as pd

try:
    from util.assb111_feature_schema import forbidden_columns, select_feature_columns, fit_standard_scaler, transform_with_scaler
    from util.assb111_split import load_manifest, validate_split_manifest, split_for_cycles, assert_fit_splits_safe, mask_for_split
except Exception:  # pragma: no cover
    from assb111_feature_schema import forbidden_columns, select_feature_columns, fit_standard_scaler, transform_with_scaler  # type: ignore
    from assb111_split import load_manifest, validate_split_manifest, split_for_cycles, assert_fit_splits_safe, mask_for_split  # type: ignore

PathLike = Union[str, Path]


@dataclass
class LeakageAuditResult:
    ok: bool
    failures: List[str]
    warnings: List[str]
    details: Dict[str, Any]

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise RuntimeError("ASSB111 leakage audit failed:\n" + "\n".join(f"- {x}" for x in self.failures))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def write_audit_json(result: LeakageAuditResult, path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(result.to_dict()), f, ensure_ascii=False, indent=2, sort_keys=True)


def _cycles_from_frame(frame: pd.DataFrame) -> np.ndarray:
    if "cycle_id" not in frame.columns:
        raise KeyError("dataset must contain cycle_id for ASSB111 leakage audit")
    return frame["cycle_id"].astype(int).to_numpy()


def attach_split_from_manifest(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    validate_split_manifest(manifest)
    out = frame.copy()
    out["split"] = split_for_cycles(_cycles_from_frame(out), manifest)
    out["is_train"] = out["split"].astype(str).str.lower().eq("train")
    out["is_val"] = out["split"].astype(str).str.lower().eq("val")
    out["is_test"] = out["split"].astype(str).str.lower().eq("test")
    out["is_partial"] = out["split"].astype(str).str.lower().eq("partial")
    out["label_visible"] = out["split"].astype(str).str.lower().isin(["train", "val"])
    return out


def make_supervised_masks(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    assigned = split_for_cycles(_cycles_from_frame(frame), manifest).astype(str)
    return {
        "train": np.char.lower(assigned) == "train",
        "val": np.char.lower(assigned) == "val",
        "test": np.char.lower(assigned) == "test",
        "partial": np.char.lower(assigned) == "partial",
        "visible": np.isin(np.char.lower(assigned), ["train", "val"]),
        "fit": np.char.lower(assigned) == "train",
    }


def check_feature_columns(
    feature_columns: Sequence[str],
    *,
    allow_upper_bound: bool = False,
) -> Tuple[List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []
    bad = forbidden_columns(feature_columns, strict=True)
    if bad and not allow_upper_bound:
        failures.append(f"Forbidden target/capacity feature columns in feature_columns: {bad}")
    elif bad:
        warnings.append(f"Upper-bound diagnostic mode includes forbidden/target-like columns: {bad}")
    return failures, warnings


def check_scaler_fit_cycles(scaler: Optional[Mapping[str, Any]], manifest: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []
    if scaler is None:
        warnings.append("No scaler metadata provided; cannot verify scaler fit cycles")
        return failures, warnings
    cycles = scaler.get("fit_cycles", None)
    if cycles is None:
        failures.append("Scaler metadata has no fit_cycles; strict30 requires scaler provenance")
        return failures, warnings
    cycles_i = np.asarray([int(c) for c in cycles], dtype=int)
    if cycles_i.size == 0:
        failures.append("Scaler fit_cycles is empty")
        return failures, warnings
    assigned = split_for_cycles(cycles_i, manifest).astype(str)
    bad_mask = ~np.isin(np.char.lower(assigned), ["train", "val"])
    # Default input uses train scaler fit; allowing val inside visible 30% is
    # less strict but still non-test. Warn instead of fail when val is present.
    if np.any(np.char.lower(assigned) == "val"):
        warnings.append("Scaler was fit using validation cycles; prefer train-only scaler for final strict result")
    if np.any(bad_mask):
        bad_cycles = sorted(set(int(c) for c in cycles_i[bad_mask]))
        failures.append(f"Scaler fit cycles include forbidden test/partial/out-of-scope cycles: {bad_cycles[:20]}")
    return failures, warnings


def check_train_history(train_history: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]]) -> Tuple[List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []
    if train_history is None:
        return failures, warnings
    hist = pd.DataFrame(train_history)
    if hist.empty:
        return failures, warnings
    cols_lower = {str(c).lower(): c for c in hist.columns}
    bad_tokens = ["test", "heldout", "holdout"]
    metric_tokens = ["loss", "r2", "mae", "rmse", "score", "metric", "corr", "nrmse", "nmae", "bias"]
    for key, original in cols_lower.items():
        if any(tok in key for tok in bad_tokens) and any(tok in key for tok in metric_tokens):
            failures.append(f"Training history contains held-out/test metric column used during training/model selection: {original}")
    # Best epoch/model selection should not be based on all/test metrics. The
    # train script should only log train_* and val_* metrics plus train-only loss
    # components. Warn on ambiguous all_* metric columns.
    for key, original in cols_lower.items():
        if key.startswith("all_") and any(tok in key for tok in metric_tokens):
            warnings.append(f"Training history contains ambiguous all-split metric column; avoid using it for model selection: {original}")
    return failures, warnings

def audit_assb111_dataset(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    feature_columns: Optional[Sequence[str]] = None,
    feature_mode: str = "p1_107a_strict",
    allow_upper_bound: bool = False,
    scaler: Optional[Mapping[str, Any]] = None,
    fit_splits: Sequence[str] = ("train",),
    train_history: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]] = None,
) -> LeakageAuditResult:
    """Run all strict-30 leakage checks and return a structured result."""
    failures: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    try:
        validate_split_manifest(manifest)
    except Exception as exc:
        failures.append(f"Invalid split manifest: {exc}")
        return LeakageAuditResult(False, failures, warnings, details)

    try:
        assert_fit_splits_safe(fit_splits, manifest)
    except Exception as exc:
        failures.append(str(exc))

    if "cycle_id" not in frame.columns:
        failures.append("dataset has no cycle_id column")
        return LeakageAuditResult(False, failures, warnings, details)

    cycles = _cycles_from_frame(frame)
    split = split_for_cycles(cycles, manifest).astype(str)
    details["split_counts_in_frame"] = {str(k): int(v) for k, v in pd.Series(split).value_counts().sort_index().to_dict().items()}
    if np.any(split == "out_of_scope"):
        bad = sorted(set(int(c) for c in cycles[split == "out_of_scope"]))
        warnings.append(f"Dataset contains out-of-scope cycles: {bad[:20]}")

    if feature_columns is None:
        try:
            feature_columns = select_feature_columns(frame, feature_mode, allow_upper_bound=allow_upper_bound, allow_missing=False)
        except Exception as exc:
            failures.append(f"Cannot select feature columns for {feature_mode}: {exc}")
            feature_columns = []
    details["feature_columns"] = list(feature_columns)
    f_fail, f_warn = check_feature_columns(feature_columns, allow_upper_bound=allow_upper_bound)
    failures.extend(f_fail)
    warnings.extend(f_warn)

    label_cols = [c for c in ["SOH_obs", "Q_obs_Ah", "Q_obs_mAh"] if c in frame.columns]
    details["label_columns_present"] = label_cols
    for col in label_cols:
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        has_test_label = np.isfinite(vals) & np.isin(np.char.lower(split), ["test", "partial"])
        if has_test_label.any():
            # Labels may exist in the stored full dataset; that is acceptable only
            # if training code masks them. Warn here, fail only if feature columns
            # include labels or fit_splits are unsafe.
            warnings.append(f"{col} is present for held-out cycles; training script must mask it from loss")

    s_fail, s_warn = check_scaler_fit_cycles(scaler, manifest)
    failures.extend(s_fail)
    warnings.extend(s_warn)

    h_fail, h_warn = check_train_history(train_history)
    failures.extend(h_fail)
    warnings.extend(h_warn)

    # Confirm that supervised fit mask would be train-only.
    masks = make_supervised_masks(frame, manifest)
    if np.any(masks["fit"] & masks["test"]):
        failures.append("fit mask overlaps test mask")
    if np.any(masks["fit"] & masks["partial"]):
        failures.append("fit mask overlaps partial mask")
    details["n_fit_rows"] = int(np.sum(masks["fit"]))
    details["n_test_rows"] = int(np.sum(masks["test"]))
    details["n_partial_rows"] = int(np.sum(masks["partial"]))

    ok = len(failures) == 0
    return LeakageAuditResult(ok=ok, failures=failures, warnings=warnings, details=details)


def fit_scaler_train_only(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    manifest: Mapping[str, Any],
    *,
    scope: str = "train",
) -> Dict[str, Any]:
    """Fit a scaler using only allowed visible rows."""
    scope_l = str(scope).strip().lower()
    if scope_l == "train":
        mask = mask_for_split(_cycles_from_frame(frame), manifest, "train")
    elif scope_l in {"visible", "train_val", "train+val"}:
        mask = mask_for_split(_cycles_from_frame(frame), manifest, ["train", "val"])
    else:
        raise ValueError("scaler scope must be 'train' or 'visible'; never use test")
    scaler = fit_standard_scaler(frame, feature_columns, fit_mask=mask)
    # Enforce no test cycles in scaler fit.
    failures, warnings = check_scaler_fit_cycles(scaler, manifest)
    if failures:
        raise RuntimeError("Scaler leakage check failed: " + "; ".join(failures))
    return scaler


def transform_features_checked(
    frame: pd.DataFrame,
    scaler: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> np.ndarray:
    failures, _warnings = check_scaler_fit_cycles(scaler, manifest)
    if failures:
        raise RuntimeError("Refusing to transform with leaking scaler: " + "; ".join(failures))
    return transform_with_scaler(frame, scaler)


def save_dataset_with_split(frame: pd.DataFrame, manifest: Mapping[str, Any], output_csv: PathLike) -> pd.DataFrame:
    out = attach_split_from_manifest(frame, manifest)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def audit_from_paths(
    dataset_csv: PathLike,
    split_manifest_json: PathLike,
    *,
    feature_mode: str = "p1_107a_strict",
    allow_upper_bound: bool = False,
    scaler_json: Optional[PathLike] = None,
    output_json: Optional[PathLike] = None,
) -> LeakageAuditResult:
    frame = pd.read_csv(dataset_csv)
    manifest = load_manifest(split_manifest_json)
    scaler = None
    if scaler_json is not None and Path(scaler_json).exists():
        with Path(scaler_json).open("r", encoding="utf-8") as f:
            scaler = json.load(f)
    result = audit_assb111_dataset(
        frame,
        manifest=manifest,
        feature_mode=feature_mode,
        allow_upper_bound=allow_upper_bound,
        scaler=scaler,
        fit_splits=("train",),
    )
    if output_json is not None:
        write_audit_json(result, output_json)
    return result


def audit_seed42locked_protocol_metadata(
    *,
    seed: int,
    fit_splits: Sequence[str] = ("train",),
    select_splits: Sequence[str] = ("val",),
    required_seed: int = 42,
    train_history: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]] = None,
) -> LeakageAuditResult:
    """Protocol-level audit for ASSB-111 seed42-locked engineering runs.

    This complements dataset leakage checks. It does not inspect final test
    metrics; it only verifies that training/checkpoint-selection metadata is
    compatible with the seed42-locked train/val-only protocol.
    """
    failures: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {
        "protocol": "ASSB111_seed42_locked_trainval_only_small_optimization",
        "seed": int(seed),
        "required_seed": int(required_seed),
        "fit_splits": list(fit_splits),
        "select_splits": list(select_splits),
    }
    if int(seed) != int(required_seed):
        failures.append(f"seed42-locked protocol requires seed={required_seed}, got {seed}")
    fit_lower = [str(s).strip().lower() for s in fit_splits]
    select_lower = [str(s).strip().lower() for s in select_splits]
    if fit_lower != ["train"]:
        failures.append(f"fit_splits must be train-only, got {fit_lower}")
    forbidden = {"test", "partial", "heldout", "held_out", "all"}
    bad_select = sorted(set(select_lower) & forbidden)
    if bad_select:
        failures.append(f"selection splits include forbidden held-out/all splits: {bad_select}")
    h_fail, h_warn = check_train_history(train_history)
    failures.extend(h_fail)
    warnings.extend(h_warn)
    return LeakageAuditResult(ok=(len(failures) == 0), failures=failures, warnings=warnings, details=details)


__all__ = [
    "LeakageAuditResult",
    "write_audit_json",
    "attach_split_from_manifest",
    "make_supervised_masks",
    "check_feature_columns",
    "check_scaler_fit_cycles",
    "check_train_history",
    "audit_assb111_dataset",
    "fit_scaler_train_only",
    "transform_features_checked",
    "save_dataset_with_split",
    "audit_from_paths",
    "audit_seed42locked_protocol_metadata",
]
