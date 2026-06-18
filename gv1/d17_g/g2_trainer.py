from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .g1_data import G1Dataset, ProfilePack, build_g1_dataset, json_dump, save_profile_predictions
from .g13_trainer import (
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
    _phie_profile_rows,
    _train_loop,
    _write_csv,
)
from .g15r_trainer import (
    _profile_uid,
    _contains_any_uid,
    _safe_float,
    _concat_XY,
    _recompute_norms,
    _protocol_counts,
    _semantic_counts,
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _branch(p: ProfilePack) -> str:
    return str(getattr(p, "branch", "UNKNOWN") or "UNKNOWN")


def _protocol(p: ProfilePack) -> str:
    return str(getattr(p, "protocol", "UNKNOWN") or "UNKNOWN")


def _group_counts(profiles: Sequence[ProfilePack], key_fn) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for p in profiles:
        k = str(key_fn(p))
        out[k] = out.get(k, 0) + 1
    return out


def _combo_counts(profiles: Sequence[ProfilePack]) -> Dict[str, int]:
    return _group_counts(profiles, lambda p: f"{_protocol(p)}::{_branch(p)}")


def _candidate_ok(
    candidate: ProfilePack,
    all_profiles: Sequence[ProfilePack],
    heldout_ids_after: set[str],
    min_fit_per_protocol: int,
    min_fit_per_semantic_branch: int,
    min_fit_per_protocol_branch: int,
) -> bool:
    fit_after = [p for p in all_profiles if _profile_uid(p) not in heldout_ids_after]
    proto_counts = _protocol_counts(fit_after)
    branch_counts = _semantic_counts(fit_after)
    combo_counts = _combo_counts(fit_after)
    if proto_counts.get(_protocol(candidate), 0) < int(min_fit_per_protocol):
        return False
    if branch_counts.get(_branch(candidate), 0) < int(min_fit_per_semantic_branch):
        return False
    combo = f"{_protocol(candidate)}::{_branch(candidate)}"
    # Only enforce protocol-branch coverage when the original group has enough examples.
    orig_combo_count = _combo_counts(all_profiles).get(combo, 0)
    if orig_combo_count > int(min_fit_per_protocol_branch) and combo_counts.get(combo, 0) < int(min_fit_per_protocol_branch):
        return False
    return True


def select_g2_stratified_internal_heldout(
    train_profiles: Sequence[ProfilePack],
    internal_heldout_count: int,
    force_fit_profile_contains: Sequence[str] = (),
    min_fit_per_protocol: int = 2,
    min_fit_per_semantic_branch: int = 2,
    min_fit_per_protocol_branch: int = 1,
    max_internal_per_protocol: int = 3,
    max_internal_per_semantic_branch: int = 6,
    seed: int = 20260615,
) -> Tuple[List[ProfilePack], List[ProfilePack], List[Dict[str, Any]]]:
    """Protocol + generator-branch stratified train-internal heldout for G2.

    G1.5R fixed the accidental R3 coverage failure.  G2 expands to the full
    train split, where the ALL55 D15 generator contains two semantic branches:
    RG repair from source softlabels and P4D full replay current-integral labels.
    This splitter keeps fit-train coverage for both protocol and generator
    branch, so internal-heldout failure is not caused by a trivial coverage hole.
    """
    profiles = list(train_profiles)
    rng = np.random.default_rng(int(seed))
    pinned_fit = {_profile_uid(p) for p in profiles if _contains_any_uid(_profile_uid(p), force_fit_profile_contains)}
    heldout: List[ProfilePack] = []
    heldout_ids: set[str] = set()
    audit: List[Dict[str, Any]] = []

    def eligible_candidates() -> List[ProfilePack]:
        out: List[ProfilePack] = []
        for p in profiles:
            uid = _profile_uid(p)
            if uid in heldout_ids or uid in pinned_fit:
                continue
            ids_after = set(heldout_ids)
            ids_after.add(uid)
            if _candidate_ok(
                p,
                profiles,
                ids_after,
                min_fit_per_protocol=min_fit_per_protocol,
                min_fit_per_semantic_branch=min_fit_per_semantic_branch,
                min_fit_per_protocol_branch=min_fit_per_protocol_branch,
            ):
                out.append(p)
        return out

    def add_selected(p: ProfilePack, stage: str, detail: Dict[str, Any] | None = None) -> None:
        uid = _profile_uid(p)
        if uid in heldout_ids:
            return
        heldout.append(p)
        heldout_ids.add(uid)
        row = {
            "stage": stage,
            "canonical_cell_uid": uid,
            "protocol": _protocol(p),
            "semantic_branch": _branch(p),
        }
        if detail:
            row.update(detail)
        audit.append(row)

    # Pass 1: one candidate from each protocol when feasible.
    for proto in sorted(_protocol_counts(profiles)):
        if len(heldout) >= int(internal_heldout_count):
            break
        candidates = [p for p in eligible_candidates() if _protocol(p) == proto]
        if not candidates:
            continue
        order = np.arange(len(candidates))
        rng.shuffle(order)
        add_selected(candidates[int(order[0])], "protocol_round_robin")

    # Pass 2: one candidate from each semantic branch when feasible.
    for branch in sorted(_semantic_counts(profiles)):
        if len(heldout) >= int(internal_heldout_count):
            break
        if sum(1 for p in heldout if _branch(p) == branch) > 0:
            continue
        candidates = [p for p in eligible_candidates() if _branch(p) == branch]
        if not candidates:
            continue
        order = np.arange(len(candidates))
        rng.shuffle(order)
        add_selected(candidates[int(order[0])], "semantic_branch_round_robin")

    # Pass 3: fill remaining capacity from the most covered protocol/branch cells,
    # while respecting max internal counts.
    while len(heldout) < int(internal_heldout_count):
        candidates = eligible_candidates()
        if not candidates:
            break
        proto_internal = _protocol_counts(heldout)
        branch_internal = _semantic_counts(heldout)
        candidates = [
            p for p in candidates
            if proto_internal.get(_protocol(p), 0) < int(max_internal_per_protocol)
            and branch_internal.get(_branch(p), 0) < int(max_internal_per_semantic_branch)
        ]
        if not candidates:
            break
        # Prefer profiles from over-covered fit groups to keep heldout diverse.
        current_fit = [p for p in profiles if _profile_uid(p) not in heldout_ids]
        proto_fit = _protocol_counts(current_fit)
        branch_fit = _semantic_counts(current_fit)
        combo_fit = _combo_counts(current_fit)

        def score(p: ProfilePack) -> Tuple[int, int, int, float]:
            combo = f"{_protocol(p)}::{_branch(p)}"
            return (
                int(proto_fit.get(_protocol(p), 0)),
                int(branch_fit.get(_branch(p), 0)),
                int(combo_fit.get(combo, 0)),
                float(rng.random()),
            )

        selected = max(candidates, key=score)
        add_selected(selected, "coverage_balanced_fill", {
            "proto_fit_before": proto_fit.get(_protocol(selected), 0),
            "branch_fit_before": branch_fit.get(_branch(selected), 0),
        })

    fit = [p for p in profiles if _profile_uid(p) not in heldout_ids]

    for proto, count in sorted(_protocol_counts(profiles).items()):
        audit.append({
            "stage": "protocol_final_counts",
            "protocol": proto,
            "total": count,
            "fit_count": sum(1 for p in fit if _protocol(p) == proto),
            "internal_heldout_count": sum(1 for p in heldout if _protocol(p) == proto),
        })
    for branch, count in sorted(_semantic_counts(profiles).items()):
        audit.append({
            "stage": "semantic_branch_final_counts",
            "semantic_branch": branch,
            "total": count,
            "fit_count": sum(1 for p in fit if _branch(p) == branch),
            "internal_heldout_count": sum(1 for p in heldout if _branch(p) == branch),
        })
    for combo, count in sorted(_combo_counts(profiles).items()):
        audit.append({
            "stage": "protocol_branch_final_counts",
            "protocol_branch": combo,
            "total": count,
            "fit_count": sum(1 for p in fit if f"{_protocol(p)}::{_branch(p)}" == combo),
            "internal_heldout_count": sum(1 for p in heldout if f"{_protocol(p)}::{_branch(p)}" == combo),
        })
    if pinned_fit:
        for uid in sorted(pinned_fit):
            audit.append({"stage": "pinned_fit", "canonical_cell_uid": uid, "reason": "force_fit_profile_contains"})
    return fit, heldout, audit


def prepare_g2_data(base: G1Dataset, config: Mapping[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
    seed = int(config.get("seed", 20260615))
    internal_count = int(config.get("internal_heldout_profile_count", config.get("internal_heldout_count", 8)))
    force_fit = list(config.get("force_fit_profile_contains", ["Batch-4_R3_battery-4"]))
    min_fit_per_protocol = int(config.get("min_fit_per_protocol", 2))
    min_fit_per_semantic_branch = int(config.get("min_fit_per_semantic_branch", 2))
    min_fit_per_protocol_branch = int(config.get("min_fit_per_protocol_branch", 1))
    max_internal_per_protocol = int(config.get("max_internal_per_protocol", 3))
    max_internal_per_semantic_branch = int(config.get("max_internal_per_semantic_branch", 6))

    all_data = prepare_g13_data(base, internal_heldout_count=0)
    fit_profiles, internal_profiles, split_audit = select_g2_stratified_internal_heldout(
        base.train_profiles,
        internal_heldout_count=internal_count,
        force_fit_profile_contains=force_fit,
        min_fit_per_protocol=min_fit_per_protocol,
        min_fit_per_semantic_branch=min_fit_per_semantic_branch,
        min_fit_per_protocol_branch=min_fit_per_protocol_branch,
        max_internal_per_protocol=max_internal_per_protocol,
        max_internal_per_semantic_branch=max_internal_per_semantic_branch,
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


def _write_model_checkpoint(model, data, config, best: Mapping[str, Any], model_path: Path, split_audit: Sequence[Mapping[str, Any]]) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
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
        "profile_conditioning": "observed_profile_encoder_no_train_profile_id_g2_full_train_expansion",
        "g2_stratified_split_audit": list(split_audit),
    }, model_path)


def _worst(trows: Sequence[Mapping[str, Any]], split_hint: str) -> Dict[str, Any] | None:
    rows = [dict(r) for r in trows if split_hint in str(r.get("split", ""))]
    if not rows:
        rows = [dict(r) for r in trows]
    if not rows:
        return None
    return min(rows, key=lambda r: _safe_float(r.get("r2"), 1e99))


def train_g2_heldout_surrogate_expansion(
    base: G1Dataset,
    out_dir: str | Path,
    config: Mapping[str, Any],
    device_arg: str = "auto",
    epochs: int = 1200,
    lr: float = 5e-4,
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

    data, split_audit = prepare_g2_data(base, config)
    Xn = _norm_X(data.X_fit, data)
    Yn = _norm_Y(data.Y_fit, data)
    loader = DataLoader(TensorDataset(torch.as_tensor(Xn), torch.as_tensor(Yn)), batch_size=int(batch_size), shuffle=True, drop_last=False)

    model_cfg = dict(config.get("model", {}))
    model = ValidationRobustObservedProfileSurrogate(
        local_input_dim=data.local_input_dim,
        profile_input_dim=data.profile_input_dim,
        target_slices=data.base.target_slices,
        width=int(model_cfg.get("width", 960)),
        depth=int(model_cfg.get("depth", 8)),
        profile_width=int(model_cfg.get("profile_width", 320)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        phie_direct_width=int(model_cfg.get("phie_direct_width", 384)),
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
    pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_fit", "G2_train_fit", data.fit_profiles, fit_preds))
    if data.internal_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_internal_heldout", "G2_train_internal_heldout", data.internal_profiles, internal_preds))
    if data.validation_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "validation_report_only", "G2_validation_report_only", data.validation_profiles, validation_preds))

    _write_csv(history, out / "D17_G2_training_history.csv")
    _write_csv(split_audit, out / "D17_G2_STRATIFIED_SPLIT_AUDIT.csv")
    _write_csv(fit_rows + int_rows + all_rows + val_rows, out / "D17_G2_PROFILE_METRICS.csv")
    _write_csv(fit_trows + int_trows + all_trows + val_trows, out / "D17_G2_PER_TARGET_PROFILE_METRICS.csv")
    _write_csv([fit_tagg, int_tagg, all_tagg, val_tagg], out / "D17_G2_PER_TARGET_AGGREGATE.csv")
    _write_csv(phie_rows, out / "D17_G2_PHIE_AUDIT.csv")
    _write_csv(_normalization_audit(data), out / "D17_G2_TARGET_NORMALIZATION_AUDIT.csv")
    _write_csv(data.feature_audit_rows, out / "D17_G2_PROFILE_ENCODER_FEATURE_AUDIT.csv")
    _write_csv(pred_manifest, out / "D17_G2_PREDICTION_MANIFEST.csv")

    model_dir = out / "model"
    _write_model_checkpoint(model, data, config, best, model_dir / "best_model.pt", split_audit)

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
    train_protocols = set(_protocol_counts(data.base.train_profiles))
    fit_protocols = set(_protocol_counts(data.fit_profiles))
    missing_fit_protocols = sorted(train_protocols - fit_protocols)
    if missing_fit_protocols:
        status_reasons.append(f"fit-train missing protocols present in train set: {missing_fit_protocols}")
    train_branches = set(_semantic_counts(data.base.train_profiles))
    fit_branches = set(_semantic_counts(data.fit_profiles))
    missing_fit_branches = sorted(train_branches - fit_branches)
    if missing_fit_branches:
        status_reasons.append(f"fit-train missing generator branches present in train set: {missing_fit_branches}")
    status = "PASS" if not status_reasons else "REVIEW"

    g3_reasons: List[str] = []
    if fit_mean < float(config.get("fit_train_r2_mean_threshold", 0.99)) or fit_min < float(config.get("fit_train_r2_min_threshold", 0.97)):
        g3_reasons.append(f"fit train target/profile R2 below G3 gate: mean={fit_mean:.6g}, min={fit_min:.6g}")
    if data.internal_profiles and (int_mean < float(config.get("internal_heldout_r2_mean_threshold", 0.95)) or int_min < float(config.get("internal_heldout_r2_min_threshold", 0.90))):
        g3_reasons.append(f"internal heldout target/profile R2 below gate: mean={int_mean:.6g}, min={int_min:.6g}")
    if data.internal_profiles and (int_phie_mean < float(config.get("internal_phie_r2_mean_threshold", 0.90)) or int_phie_min < float(config.get("internal_phie_r2_min_threshold", 0.85))):
        g3_reasons.append(f"internal heldout phie R2 below gate: mean={int_phie_mean:.6g}, min={int_phie_min:.6g}")
    if data.validation_profiles and (val_mean < float(config.get("validation_r2_mean_threshold", 0.95)) or val_min < float(config.get("validation_r2_min_threshold", 0.90))):
        g3_reasons.append(f"validation report-only target/profile R2 below gate: mean={val_mean:.6g}, min={val_min:.6g}")
    if data.validation_profiles and (val_phie_mean < float(config.get("validation_phie_r2_mean_threshold", 0.93)) or val_phie_min < float(config.get("validation_phie_r2_min_threshold", 0.90))):
        g3_reasons.append(f"validation phie R2 below gate: mean={val_phie_mean:.6g}, min={val_phie_min:.6g}")
    g3_ready = len(status_reasons) == 0 and len(g3_reasons) == 0
    recommendation = "ENTER_D17_G3_FROZEN_TEST_REPORT_ONLY_AUDIT" if g3_ready else "DO_NOT_ENTER_G3_REVIEW_G2_EXPANSION"

    summary: Dict[str, Any] = {
        "protocol": "D17-G2_HELDOUT_GENERATOR_SURROGATE_EXPANSION",
        "created_at_utc": _utc_now(),
        "status": status,
        "status_reasons": status_reasons,
        "recommendation": recommendation,
        "g3_ready": bool(g3_ready),
        "g3_blockers": g3_reasons,
        "purpose": "Expand G1.5R supervised generator-surrogate from a repaired 24-train-profile smoke to the full D17 train split, with protocol/branch-stratified train-internal heldout and validation report-only audit. Frozen-test soft labels remain unused.",
        "policy": {
            "train_cell_softlabels_used_for_training": True,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_used": False,
            "checkpoint_selection": "fit-train plus protocol/branch-stratified train-internal heldout metrics only; validation report-only metrics are not used to select checkpoint",
            "not_a_G3_or_frozen_test_run": True,
        },
        "source_prerequisite": {
            "G15R_required": True,
            "G15R_expected_status": "PASS / g2_ready=true",
            "G15R_reason": "G1.5R fixed protocol coverage mismatch and recommended ENTER_D17_G2_HELDOUT_SURROGATE_EXPANSION.",
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
            "validation_semantic_branch_counts": _semantic_counts(data.validation_profiles),
            "fit_train_protocol_branch_counts": _combo_counts(data.fit_profiles),
            "internal_heldout_protocol_branch_counts": _combo_counts(data.internal_profiles),
            "validation_protocol_branch_counts": _combo_counts(data.validation_profiles),
            "augmented_feature_dim": int(data.X_fit.shape[1]),
            "local_input_dim": int(data.local_input_dim),
            "profile_input_dim": int(data.profile_input_dim),
        },
        "model": {
            "class": "ValidationRobustObservedProfileSurrogate",
            "profile_conditioning": "observed profile summary encoder; no train profile-id embedding",
            "phie_head": "G1.4/G1.5R phie convention head reused",
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
        "stratified_split_audit_preview": split_audit[:60],
        "files": {
            "summary_json": str(out / "D17_G2_HELDOUT_SURROGATE_EXPANSION_SUMMARY.json"),
            "stratified_split_audit_csv": str(out / "D17_G2_STRATIFIED_SPLIT_AUDIT.csv"),
            "profile_metrics_csv": str(out / "D17_G2_PROFILE_METRICS.csv"),
            "per_target_profile_metrics_csv": str(out / "D17_G2_PER_TARGET_PROFILE_METRICS.csv"),
            "per_target_aggregate_csv": str(out / "D17_G2_PER_TARGET_AGGREGATE.csv"),
            "phie_audit_csv": str(out / "D17_G2_PHIE_AUDIT.csv"),
            "profile_encoder_feature_audit_csv": str(out / "D17_G2_PROFILE_ENCODER_FEATURE_AUDIT.csv"),
            "target_normalization_audit_csv": str(out / "D17_G2_TARGET_NORMALIZATION_AUDIT.csv"),
            "prediction_manifest_csv": str(out / "D17_G2_PREDICTION_MANIFEST.csv"),
            "training_history_csv": str(out / "D17_G2_training_history.csv"),
            "best_model_pt": str(model_dir / "best_model.pt"),
        },
    }
    json_dump(summary, out / "D17_G2_HELDOUT_SURROGATE_EXPANSION_SUMMARY.json")
    return summary


def build_and_train_g2(
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
    return train_g2_heldout_surrogate_expansion(ds, out_dir, config, device_arg=device_arg, epochs=int(epochs), lr=float(lr), batch_size=int(batch_size))
