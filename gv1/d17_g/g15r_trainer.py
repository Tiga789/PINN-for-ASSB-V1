from __future__ import annotations

import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .g1_data import G1Dataset, ProfilePack, build_g1_dataset, json_dump, save_profile_predictions
from .g1_metrics import aggregate_profile_rows, profile_metrics
from .g13_trainer import (
    G13PreparedData,
    prepare_g13_data,
    _norm_X,
    _norm_Y,
    _predict_profiles,
    _normalization_audit,
)
from .g14_model import ValidationRobustObservedProfileSurrogate
from .g14_trainer import (
    _aggregate_or_empty,
    _device_from_arg,
    _group_balanced_loss,
    _phie_profile_rows,
    _target_aggregate,
    _train_loop,
    _write_csv,
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _profile_uid(p: ProfilePack) -> str:
    return str(p.canonical_cell_uid or p.cell_uid)


def _contains_any_uid(uid: str, needles: Sequence[str]) -> bool:
    u = str(uid)
    return any(str(n) and str(n) in u for n in needles)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _concat_XY(profiles: Sequence[ProfilePack], per_profile_X: Mapping[str, np.ndarray], target_dim: int, feature_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if not profiles:
        return np.zeros((0, feature_dim), dtype=np.float32), np.zeros((0, target_dim), dtype=np.float32)
    X = np.concatenate([per_profile_X[_profile_uid(p)] for p in profiles], axis=0).astype(np.float32)
    Y = np.concatenate([p.targets for p in profiles], axis=0).astype(np.float32)
    return X, Y


def _recompute_norms(X_fit: np.ndarray, Y_fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = np.nanmean(X_fit, axis=0).astype(np.float32)
    x_std = np.nanstd(X_fit, axis=0).astype(np.float32)
    x_std[~np.isfinite(x_std) | (x_std < 1e-8)] = 1.0
    y_mean = np.nanmean(Y_fit, axis=0).astype(np.float32)
    y_std = np.nanstd(Y_fit, axis=0).astype(np.float32)
    y_std[~np.isfinite(y_std) | (y_std < 1e-8)] = 1.0
    return x_mean, x_std, y_mean, y_std


def select_stratified_internal_heldout(
    train_profiles: Sequence[ProfilePack],
    internal_heldout_count: int,
    force_fit_profile_contains: Sequence[str] = (),
    min_fit_per_protocol: int = 1,
    max_internal_per_protocol: int = 1,
    seed: int = 20260615,
) -> Tuple[List[ProfilePack], List[ProfilePack], List[Dict[str, Any]]]:
    """Build a deterministic protocol-stratified internal-heldout split.

    G1.4 used the last N train profiles as internal heldout.  In the observed
    run, this accidentally placed R3 profiles almost exclusively in heldout,
    while fit-train had no/poor R3 coverage.  This splitter holds out profiles
    only when the corresponding protocol keeps enough fit-train coverage.  Known
    problematic profiles can be pinned to fit-train to test whether the failure
    is coverage-related rather than a model-form issue.
    """
    profiles = list(train_profiles)
    rng = np.random.default_rng(int(seed))
    by_protocol: Dict[str, List[ProfilePack]] = {}
    for p in profiles:
        by_protocol.setdefault(str(p.protocol), []).append(p)

    pinned_fit = {_profile_uid(p) for p in profiles if _contains_any_uid(_profile_uid(p), force_fit_profile_contains)}
    heldout: List[ProfilePack] = []
    heldout_ids = set()
    audit: List[Dict[str, Any]] = []

    # First pass: at most one per protocol, while leaving min fit coverage.
    for proto in sorted(by_protocol):
        group = by_protocol[proto]
        candidates = [p for p in group if _profile_uid(p) not in pinned_fit]
        max_allowed = max(0, len(group) - int(min_fit_per_protocol))
        if max_allowed <= 0 or not candidates or len(heldout) >= int(internal_heldout_count):
            continue
        # Prefer a deterministic profile near the middle of the group rather than always last.
        # A tiny seeded shuffle prevents systematic file-order bias while keeping reproducibility.
        order = list(range(len(candidates)))
        rng.shuffle(order)
        selected = candidates[order[0]]
        if _profile_uid(selected) not in heldout_ids:
            heldout.append(selected)
            heldout_ids.add(_profile_uid(selected))
            audit.append({
                "stage": "round_robin_protocol",
                "protocol": proto,
                "canonical_cell_uid": _profile_uid(selected),
                "group_size": len(group),
                "pinned_fit_in_protocol": sum(1 for p in group if _profile_uid(p) in pinned_fit),
            })
        if len(heldout) >= int(internal_heldout_count):
            break

    # Second pass: fill remaining capacity from protocols with abundant coverage.
    while len(heldout) < int(internal_heldout_count):
        best_proto = None
        best_candidates: List[ProfilePack] = []
        best_surplus = -1
        protocol_counts_held = {proto: sum(1 for p in heldout if str(p.protocol) == proto) for proto in by_protocol}
        for proto, group in sorted(by_protocol.items()):
            if protocol_counts_held.get(proto, 0) >= int(max_internal_per_protocol):
                continue
            current_fit_count = len([p for p in group if _profile_uid(p) not in heldout_ids])
            surplus = current_fit_count - int(min_fit_per_protocol)
            candidates = [p for p in group if _profile_uid(p) not in heldout_ids and _profile_uid(p) not in pinned_fit]
            if surplus > best_surplus and surplus > 0 and candidates:
                best_proto = proto
                best_candidates = candidates
                best_surplus = surplus
        if best_proto is None or not best_candidates:
            break
        order = list(range(len(best_candidates)))
        rng.shuffle(order)
        selected = best_candidates[order[0]]
        heldout.append(selected)
        heldout_ids.add(_profile_uid(selected))
        audit.append({
            "stage": "fill_abundant_protocol",
            "protocol": best_proto,
            "canonical_cell_uid": _profile_uid(selected),
            "surplus_before_select": int(best_surplus),
        })

    fit = [p for p in profiles if _profile_uid(p) not in heldout_ids]

    # Final audit rows for every protocol.
    for proto, group in sorted(by_protocol.items()):
        audit.append({
            "stage": "protocol_final_counts",
            "protocol": proto,
            "total": len(group),
            "fit_count": sum(1 for p in fit if str(p.protocol) == proto),
            "internal_heldout_count": sum(1 for p in heldout if str(p.protocol) == proto),
            "pinned_fit_count": sum(1 for p in group if _profile_uid(p) in pinned_fit),
        })
    if pinned_fit:
        for uid in sorted(pinned_fit):
            audit.append({"stage": "pinned_fit", "canonical_cell_uid": uid, "reason": "force_fit_profile_contains"})
    return fit, heldout, audit


def prepare_g15r_data(base: G1Dataset, config: Mapping[str, Any]) -> Tuple[G13PreparedData, List[Dict[str, Any]]]:
    seed = int(config.get("seed", 20260615))
    internal_count = int(config.get("internal_heldout_profile_count", config.get("internal_heldout_count", 4)))
    force_fit = list(config.get("force_fit_profile_contains", ["Batch-4_R3_battery-4"]))
    min_fit_per_protocol = int(config.get("min_fit_per_protocol", 1))
    max_internal_per_protocol = int(config.get("max_internal_per_protocol", 1))

    # Build all observed/profile-summary features once, but with no internal split.
    all_data = prepare_g13_data(base, internal_heldout_count=0)
    fit_profiles, internal_profiles, split_audit = select_stratified_internal_heldout(
        base.train_profiles,
        internal_heldout_count=internal_count,
        force_fit_profile_contains=force_fit,
        min_fit_per_protocol=min_fit_per_protocol,
        max_internal_per_protocol=max_internal_per_protocol,
        seed=seed,
    )
    feat_dim = int(all_data.X_fit.shape[1])
    target_dim = int(all_data.Y_fit.shape[1])
    X_fit, Y_fit = _concat_XY(fit_profiles, all_data.per_profile_X, target_dim, feat_dim)
    X_internal, Y_internal = _concat_XY(internal_profiles, all_data.per_profile_X, target_dim, feat_dim)
    X_all, Y_all = _concat_XY(base.train_profiles, all_data.per_profile_X, target_dim, feat_dim)
    X_val, Y_val = _concat_XY(base.validation_profiles, all_data.per_profile_X, target_dim, feat_dim)
    x_mean, x_std, y_mean, y_std = _recompute_norms(X_fit, Y_fit)

    data = replace(
        all_data,
        fit_profiles=fit_profiles,
        internal_profiles=internal_profiles,
        validation_profiles=base.validation_profiles,
        X_fit=X_fit,
        Y_fit=Y_fit,
        X_train_all=X_all,
        Y_train_all=Y_all,
        X_internal=X_internal,
        Y_internal=Y_internal,
        X_validation=X_val,
        Y_validation=Y_val,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    return data, split_audit


def _protocol_counts(profiles: Sequence[ProfilePack]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in profiles:
        out[str(p.protocol)] = out.get(str(p.protocol), 0) + 1
    return out


def _semantic_counts(profiles: Sequence[ProfilePack]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in profiles:
        out[str(p.branch)] = out.get(str(p.branch), 0) + 1
    return out


def train_g15r_stratified_repair(
    base: G1Dataset,
    out_dir: str | Path,
    config: Mapping[str, Any],
    device_arg: str = "auto",
    epochs: int = 900,
    lr: float = 6e-4,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(device_arg)
    seed = int(config.get("seed", 20260615))
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(int(config.get("torch_num_threads", 2)))
    except Exception:
        pass

    data, split_audit = prepare_g15r_data(base, config)
    Xn = _norm_X(data.X_fit, data)
    Yn = _norm_Y(data.Y_fit, data)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(Xn), torch.as_tensor(Yn)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    model_cfg = dict(config.get("model", {}))
    model = ValidationRobustObservedProfileSurrogate(
        local_input_dim=data.local_input_dim,
        profile_input_dim=data.profile_input_dim,
        target_slices=data.base.target_slices,
        width=int(model_cfg.get("width", 896)),
        depth=int(model_cfg.get("depth", 8)),
        profile_width=int(model_cfg.get("profile_width", 256)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        phie_direct_width=int(model_cfg.get("phie_direct_width", 320)),
    ).to(device)

    best, history = _train_loop(model, loader, data, config, device, epochs=int(epochs), lr=float(lr))
    if best.get("state_dict") is not None:
        model.load_state_dict(best["state_dict"])

    fit_preds = _predict_profiles(model, data.fit_profiles, data, device)
    internal_preds = _predict_profiles(model, data.internal_profiles, data, device) if data.internal_profiles else []
    all_train_preds = _predict_profiles(model, data.base.train_profiles, data, device)
    validation_preds = _predict_profiles(model, data.validation_profiles, data, device) if data.validation_profiles else []

    fit_rows, fit_trows, fit_agg, fit_tagg = _aggregate_or_empty(data.fit_profiles, fit_preds, "train_fit")
    int_rows, int_trows, int_agg, int_tagg = _aggregate_or_empty(data.internal_profiles, internal_preds, "train_internal_heldout")
    all_rows, all_trows, all_agg, all_tagg = _aggregate_or_empty(data.base.train_profiles, all_train_preds, "train_all_report")
    val_rows, val_trows, val_agg, val_tagg = _aggregate_or_empty(data.validation_profiles, validation_preds, "validation_report_only")

    phie_rows: List[Dict[str, Any]] = []
    phie_rows.extend(_phie_profile_rows(data.fit_profiles, fit_preds, "train_fit"))
    phie_rows.extend(_phie_profile_rows(data.internal_profiles, internal_preds, "train_internal_heldout"))
    phie_rows.extend(_phie_profile_rows(data.validation_profiles, validation_preds, "validation_report_only"))

    pred_manifest: List[Dict[str, Any]] = []
    pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_fit", "G15R_train_fit", data.fit_profiles, fit_preds))
    if data.internal_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_internal_heldout", "G15R_train_internal_heldout", data.internal_profiles, internal_preds))
    if data.validation_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "validation_report_only", "G15R_validation_report_only", data.validation_profiles, validation_preds))

    _write_csv(history, out / "D17_G15R_training_history.csv")
    _write_csv(split_audit, out / "D17_G15R_STRATIFIED_SPLIT_AUDIT.csv")
    _write_csv(fit_rows + int_rows + all_rows + val_rows, out / "D17_G15R_PROFILE_METRICS.csv")
    _write_csv(fit_trows + int_trows + all_trows + val_trows, out / "D17_G15R_PER_TARGET_PROFILE_METRICS.csv")
    _write_csv([fit_tagg, int_tagg, all_tagg, val_tagg], out / "D17_G15R_PER_TARGET_AGGREGATE.csv")
    _write_csv(phie_rows, out / "D17_G15R_PHIE_AUDIT.csv")
    _write_csv(_normalization_audit(data), out / "D17_G15R_TARGET_NORMALIZATION_AUDIT.csv")
    _write_csv(data.feature_audit_rows, out / "D17_G15R_PROFILE_ENCODER_FEATURE_AUDIT.csv")
    _write_csv(pred_manifest, out / "D17_G15R_PREDICTION_MANIFEST.csv")

    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": data.x_mean,
        "x_std": data.x_std,
        "y_mean": data.y_mean,
        "y_std": data.y_std,
        "feature_names": data.feature_names,
        "target_names": data.base.target_names,
        "target_slices": data.base.target_slices,
        "local_input_dim": data.local_input_dim,
        "profile_input_dim": data.profile_input_dim,
        "profile_feature_names": data.profile_feature_names,
        "config": dict(config),
        "best_epoch": int(best.get("epoch", 0)),
        "profile_conditioning": "observed_profile_encoder_no_train_profile_id_stratified_internal_heldout",
        "stratified_split_audit": split_audit,
    }, model_dir / "best_model.pt")

    fit_mean = _safe_float(fit_tagg.get("all_target_profile_r2_mean"))
    fit_min = _safe_float(fit_tagg.get("all_target_profile_r2_min"))
    int_mean = _safe_float(int_tagg.get("all_target_profile_r2_mean")) if data.internal_profiles else float("nan")
    int_min = _safe_float(int_tagg.get("all_target_profile_r2_min")) if data.internal_profiles else float("nan")
    val_mean = _safe_float(val_tagg.get("all_target_profile_r2_mean")) if data.validation_profiles else float("nan")
    val_min = _safe_float(val_tagg.get("all_target_profile_r2_min")) if data.validation_profiles else float("nan")
    val_phie_mean = _safe_float(val_tagg.get("phie_r2_mean"))
    val_phie_min = _safe_float(val_tagg.get("phie_r2_min"))
    int_phie_mean = _safe_float(int_tagg.get("phie_r2_mean"))
    int_phie_min = _safe_float(int_tagg.get("phie_r2_min"))

    status_reasons: List[str] = []
    if fit_mean < float(config.get("fit_status_r2_mean_threshold", 0.98)) or fit_min < float(config.get("fit_status_r2_min_threshold", 0.95)):
        status_reasons.append(f"fit-train target/profile R2 below status threshold: mean={fit_mean:.6g}, min={fit_min:.6g}")
    if not data.internal_profiles:
        status_reasons.append("no internal-heldout profiles were selected")
    # Internal split coverage sanity: every protocol in selected train should remain present in fit_train.
    train_protocols = set(_protocol_counts(data.base.train_profiles))
    fit_protocols = set(_protocol_counts(data.fit_profiles))
    missing_fit_protocols = sorted(train_protocols - fit_protocols)
    if missing_fit_protocols:
        status_reasons.append(f"fit-train missing protocols present in train set: {missing_fit_protocols}")
    status = "PASS" if not status_reasons else "REVIEW"

    g2_reasons: List[str] = []
    if fit_mean < float(config.get("fit_train_r2_mean_threshold", 0.99)) or fit_min < float(config.get("fit_train_r2_min_threshold", 0.97)):
        g2_reasons.append(f"fit train target/profile R2 below G2 gate: mean={fit_mean:.6g}, min={fit_min:.6g}")
    if data.internal_profiles and (int_mean < float(config.get("internal_heldout_r2_mean_threshold", 0.95)) or int_min < float(config.get("internal_heldout_r2_min_threshold", 0.90))):
        g2_reasons.append(f"internal heldout target/profile R2 below gate: mean={int_mean:.6g}, min={int_min:.6g}")
    if data.internal_profiles and (int_phie_mean < float(config.get("internal_phie_r2_mean_threshold", 0.90)) or int_phie_min < float(config.get("internal_phie_r2_min_threshold", 0.85))):
        g2_reasons.append(f"internal heldout phie R2 below gate: mean={int_phie_mean:.6g}, min={int_phie_min:.6g}")
    if data.validation_profiles and (val_mean < float(config.get("validation_r2_mean_threshold", 0.95)) or val_min < float(config.get("validation_r2_min_threshold", 0.90))):
        g2_reasons.append(f"validation report-only target/profile R2 below gate: mean={val_mean:.6g}, min={val_min:.6g}")
    if data.validation_profiles and (val_phie_mean < float(config.get("validation_phie_r2_mean_threshold", 0.93)) or val_phie_min < float(config.get("validation_phie_r2_min_threshold", 0.90))):
        g2_reasons.append(f"validation phie R2 below gate: mean={val_phie_mean:.6g}, min={val_phie_min:.6g}")
    g2_ready = len(status_reasons) == 0 and len(g2_reasons) == 0
    recommendation = "ENTER_D17_G2_HELDOUT_SURROGATE_EXPANSION" if g2_ready else "DO_NOT_ENTER_G2_REVIEW_G15R_STRATIFIED_REPAIR"

    # Worst rows for quick diagnosis.
    def _worst(trows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any] | None:
        rows = [dict(r) for r in trows if str(r.get("split", split)) == split or split in str(r.get("split", ""))]
        if not rows:
            rows = [dict(r) for r in trows]
        if not rows:
            return None
        return min(rows, key=lambda r: _safe_float(r.get("r2"), 1e99))

    summary: Dict[str, Any] = {
        "protocol": "D17-G1.5R_STRATIFIED_INTERNAL_HELDOUT_COVERAGE_REPAIR",
        "created_at_utc": _utc_now(),
        "status": status,
        "status_reasons": status_reasons,
        "recommendation": recommendation,
        "g2_ready": bool(g2_ready),
        "g2_blockers": g2_reasons,
        "purpose": "Rerun G1.4-style supervised generator surrogate with a stratified train-internal heldout split so R3 is represented in fit-train and known bad internal profile is not accidentally isolated by file order.",
        "policy": {
            "train_cell_softlabels_used_for_training": True,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_used": False,
            "checkpoint_selection": "fit-train plus stratified train-internal heldout metrics only; validation report-only metrics are not used to select checkpoint",
            "not_a_G2_run": True,
        },
        "source_diagnosis": {
            "G15_reason": "G1.5 triage located worst internal-heldout profile Batch-4_R3_battery-4 / phie and protocol coverage mismatch.",
            "coverage_repair": "Use protocol-stratified internal heldout and pin Batch-4_R3_battery-4 into fit-train unless user changes force_fit_profile_contains.",
        },
        "device": str(device),
        "seed": seed,
        "epochs_requested": int(epochs),
        "best_epoch": int(best.get("epoch", 0)),
        "dataset": {
            **dict(base.manifest_summary),
            "fit_train_profile_count": len(data.fit_profiles),
            "internal_heldout_profile_count": len(data.internal_profiles),
            "validation_profile_count": len(data.validation_profiles),
            "fit_train_protocol_counts": _protocol_counts(data.fit_profiles),
            "internal_heldout_protocol_counts": _protocol_counts(data.internal_profiles),
            "validation_protocol_counts": _protocol_counts(data.validation_profiles),
            "fit_train_semantic_branch_counts": _semantic_counts(data.fit_profiles),
            "internal_heldout_semantic_branch_counts": _semantic_counts(data.internal_profiles),
            "augmented_feature_dim": int(data.X_fit.shape[1]),
            "local_input_dim": int(data.local_input_dim),
            "profile_input_dim": int(data.profile_input_dim),
        },
        "model": {
            "class": "ValidationRobustObservedProfileSurrogate",
            "profile_conditioning": "observed profile summary encoder; no train profile-id embedding",
            "phie_head": "G1.4 phie convention head reused; only internal-heldout split construction is changed",
            "target_group_weights": dict(config.get("target_group_weights", {})),
        },
        "fit_train_per_target_aggregate": fit_tagg,
        "internal_heldout_per_target_aggregate": int_tagg,
        "train_all_report_per_target_aggregate": all_tagg,
        "validation_report_only_per_target_aggregate": val_tagg,
        "fit_train_profile_aggregate": fit_agg,
        "internal_heldout_profile_aggregate": int_agg,
        "validation_report_only_profile_aggregate": val_agg,
        "worst_internal_target_profile": _worst(int_trows, "train_internal_heldout"),
        "worst_validation_target_profile": _worst(val_trows, "validation_report_only"),
        "stratified_split_audit_preview": split_audit[:40],
        "files": {
            "summary_json": str(out / "D17_G15R_STRATIFIED_HELDOUT_REPAIR_SUMMARY.json"),
            "stratified_split_audit_csv": str(out / "D17_G15R_STRATIFIED_SPLIT_AUDIT.csv"),
            "profile_metrics_csv": str(out / "D17_G15R_PROFILE_METRICS.csv"),
            "per_target_profile_metrics_csv": str(out / "D17_G15R_PER_TARGET_PROFILE_METRICS.csv"),
            "per_target_aggregate_csv": str(out / "D17_G15R_PER_TARGET_AGGREGATE.csv"),
            "phie_audit_csv": str(out / "D17_G15R_PHIE_AUDIT.csv"),
            "profile_encoder_feature_audit_csv": str(out / "D17_G15R_PROFILE_ENCODER_FEATURE_AUDIT.csv"),
            "target_normalization_audit_csv": str(out / "D17_G15R_TARGET_NORMALIZATION_AUDIT.csv"),
            "prediction_manifest_csv": str(out / "D17_G15R_PREDICTION_MANIFEST.csv"),
            "training_history_csv": str(out / "D17_G15R_training_history.csv"),
            "best_model_pt": str(model_dir / "best_model.pt"),
        },
    }
    json_dump(summary, out / "D17_G15R_STRATIFIED_HELDOUT_REPAIR_SUMMARY.json")
    return summary


def build_and_train_g15r(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    out_dir: str | Path,
    config: Mapping[str, Any],
    train_profile_count: int,
    validation_profile_count: int,
    max_time_points: int,
    time_window_s: float,
    device_arg: str,
    epochs: int,
    lr: float,
    batch_size: int,
) -> Dict[str, Any]:
    ds = build_g1_dataset(
        split_manifest=split_manifest,
        g0_profile_semantics_csv=g0_profile_semantics_csv,
        train_profile_count=int(train_profile_count),
        validation_profile_count=int(validation_profile_count),
        max_time_points=int(max_time_points),
        time_window_s=float(time_window_s),
    )
    return train_g15r_stratified_repair(ds, out_dir, config, device_arg=device_arg, epochs=int(epochs), lr=float(lr), batch_size=int(batch_size))
