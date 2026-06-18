from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .g1_data import G1Dataset, ProfilePack, build_g1_dataset, json_dump, save_profile_predictions
from .g1_metrics import aggregate_profile_rows, group_metrics, profile_metrics, r2_score
from .g13_trainer import prepare_g13_data, _norm_X, _norm_Y, _predict_np, _predict_profiles, _normalization_audit
from .g14_model import ValidationRobustObservedProfileSurrogate


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _device_from_arg(arg: str) -> torch.device:
    if str(arg) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(arg))


def _write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    if not fields:
        fields = ["empty"]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def _group_balanced_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_slices: Mapping[str, Tuple[int, int]],
    weights: Mapping[str, float],
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    wsum = 0.0
    for key, (a, b) in target_slices.items():
        weight = float(weights.get(key, 1.0))
        if weight <= 0:
            continue
        losses.append(torch.mean((pred[:, a:b] - target[:, a:b]) ** 2) * weight)
        wsum += weight
    if not losses:
        return torch.mean((pred - target) ** 2)
    return torch.stack(losses).sum() / max(wsum, 1e-12)


def _per_target_rows(profiles: Sequence[ProfilePack], preds: Sequence[np.ndarray], split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prof, pred in zip(profiles, preds):
        for key, (a, b) in prof.target_slices.items():
            yt = prof.targets[:, a:b]
            yp = pred[:, a:b]
            rows.append({
                "split": split_name,
                "canonical_cell_uid": prof.canonical_cell_uid,
                "protocol": prof.protocol,
                "semantic_branch": prof.branch,
                "target": key,
                "mae": float(np.nanmean(np.abs(yp - yt))),
                "rmse": float(np.sqrt(np.nanmean((yp - yt) ** 2))),
                "r2": r2_score(yt, yp),
                "n_points": int(yt.size),
                "target_range": float(np.nanmax(yt) - np.nanmin(yt)) if yt.size else float("nan"),
                "target_std": float(np.nanstd(yt)) if yt.size else float("nan"),
                "bias": float(np.nanmean(yp - yt)) if yt.size else float("nan"),
            })
    return rows


def _target_aggregate(rows: Sequence[Mapping[str, Any]], split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split_name}
    vals: List[float] = []
    targets = sorted({str(r.get("target")) for r in rows})
    for t in targets:
        rs: List[float] = []
        for r in rows:
            if str(r.get("target")) != t:
                continue
            try:
                v = float(r.get("r2", float("nan")))
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                rs.append(v)
        if rs:
            out[f"{t}_r2_mean"] = float(np.mean(rs))
            out[f"{t}_r2_min"] = float(np.min(rs))
            vals.extend(rs)
    out["all_target_profile_r2_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["all_target_profile_r2_min"] = float(np.min(vals)) if vals else float("nan")
    return out


def _aggregate_or_empty(
    profiles: Sequence[ProfilePack],
    preds: Sequence[np.ndarray],
    split_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if not profiles:
        return [], [], {"split": split_name}, {"split": split_name}
    rows = profile_metrics(profiles, preds)["rows"]
    trows = _per_target_rows(profiles, preds, split_name)
    return rows, trows, aggregate_profile_rows(rows), _target_aggregate(trows, split_name)


def _phie_profile_rows(profiles: Sequence[ProfilePack], preds: Sequence[np.ndarray], split_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prof, pred in zip(profiles, preds):
        if "phie" not in prof.target_slices:
            continue
        a, b = prof.target_slices["phie"]
        yt = prof.targets[:, a:b]
        yp = pred[:, a:b]
        rows.append({
            "split": split_name,
            "canonical_cell_uid": prof.canonical_cell_uid,
            "protocol": prof.protocol,
            "semantic_branch": prof.branch,
            "phie_r2": r2_score(yt, yp),
            "phie_mae": float(np.nanmean(np.abs(yp - yt))),
            "phie_rmse": float(np.sqrt(np.nanmean((yp - yt) ** 2))),
            "phie_bias": float(np.nanmean(yp - yt)),
            "phie_true_range": float(np.nanmax(yt) - np.nanmin(yt)),
            "phie_true_std": float(np.nanstd(yt)),
            "phie_pred_range": float(np.nanmax(yp) - np.nanmin(yp)),
            "phie_pred_std": float(np.nanstd(yp)),
            "n_time": int(yt.shape[0]),
        })
    return rows


def _train_loop(
    model: torch.nn.Module,
    loader: DataLoader,
    data: Any,
    config: Mapping[str, Any],
    device: torch.device,
    epochs: int,
    lr: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model_cfg = dict(config.get("model", {}))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(model_cfg.get("weight_decay", 2e-6)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(model_cfg.get("min_lr", 1e-5)))
    base_weights = dict(config.get("target_group_weights", {
        "theta_a": 1.5,
        "theta_c": 1.5,
        "cs_a": 1.0,
        "cs_c": 1.0,
        "phie": 16.0,
        "phis_c": 3.0,
    }))
    phie_focus_epochs = int(config.get("phie_focus_epochs", 220))
    phie_focus_multiplier = float(config.get("phie_focus_multiplier", 1.75))
    eval_every = int(config.get("eval_every", 50))
    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": 0, "score": -1e99, "fit_loss": float("inf"), "state_dict": None}

    for ep in range(1, int(epochs) + 1):
        weights = dict(base_weights)
        phase = "phie_focus" if ep <= phie_focus_epochs else "balanced"
        if phase == "phie_focus":
            weights["phie"] = float(weights.get("phie", 1.0)) * phie_focus_multiplier
            # Keep the other heads learning, but prevent phie from being drowned by cs/theta dimensions.
            for k in ["theta_a", "theta_c", "cs_a", "cs_c"]:
                weights[k] = float(weights.get(k, 1.0)) * float(config.get("non_phie_focus_scale", 0.75))
        model.train()
        batch_losses: List[float] = []
        for xb, yb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            pred = model(xb)
            loss = _group_balanced_loss(pred, yb, data.base.target_slices, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg.get("grad_clip_norm", 5.0)))
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        fit_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        row: Dict[str, Any] = {"epoch": ep, "phase": phase, "fit_train_loss": fit_loss, "lr": float(opt.param_groups[0]["lr"])}
        do_eval = ep == 1 or ep == int(epochs) or ep % eval_every == 0
        if do_eval:
            pred_fit = _predict_np(model, data.X_fit, data, device)
            fit_gm = group_metrics(data.Y_fit, pred_fit, data.base.target_slices)
            row["fit_train_r2_mean"] = float(fit_gm["__aggregate__"]["r2_mean"])
            row["fit_train_r2_min"] = float(fit_gm["__aggregate__"]["r2_min"])
            row["fit_phie_r2"] = float(fit_gm.get("phie", {}).get("r2", float("nan")))
            if data.X_internal.shape[0] > 0:
                pred_int = _predict_np(model, data.X_internal, data, device)
                int_gm = group_metrics(data.Y_internal, pred_int, data.base.target_slices)
                row["internal_heldout_r2_mean"] = float(int_gm["__aggregate__"]["r2_mean"])
                row["internal_heldout_r2_min"] = float(int_gm["__aggregate__"]["r2_min"])
                row["internal_phie_r2"] = float(int_gm.get("phie", {}).get("r2", float("nan")))
            else:
                row["internal_heldout_r2_mean"] = float("nan")
                row["internal_heldout_r2_min"] = float("nan")
                row["internal_phie_r2"] = float("nan")
            score = row["fit_train_r2_mean"] + 0.2 * row["fit_train_r2_min"] + 0.3 * row["fit_phie_r2"]
            if np.isfinite(row.get("internal_heldout_r2_mean", float("nan"))):
                score += 0.7 * row["internal_heldout_r2_mean"] + 0.2 * row["internal_heldout_r2_min"] + 0.7 * row["internal_phie_r2"]
            score -= 0.005 * fit_loss
            if np.isfinite(score) and score > best["score"]:
                best = {"epoch": ep, "score": float(score), "fit_loss": fit_loss, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        elif math.isfinite(fit_loss) and fit_loss < best["fit_loss"] and best.get("state_dict") is None:
            best = {"epoch": ep, "score": best.get("score", -1e99), "fit_loss": fit_loss, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        history.append(row)
    return best, history


def train_g14_phie_validation_robust(
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

    internal_count = int(config.get("internal_heldout_profile_count", 4))
    data = prepare_g13_data(base, internal_heldout_count=internal_count)
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

    phie_rows = []
    phie_rows.extend(_phie_profile_rows(data.fit_profiles, fit_preds, "train_fit"))
    phie_rows.extend(_phie_profile_rows(data.internal_profiles, internal_preds, "train_internal_heldout"))
    phie_rows.extend(_phie_profile_rows(data.validation_profiles, validation_preds, "validation_report_only"))

    pred_manifest: List[Dict[str, Any]] = []
    pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_fit", "G14_train_fit", data.fit_profiles, fit_preds))
    if data.internal_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_internal_heldout", "G14_train_internal_heldout", data.internal_profiles, internal_preds))
    if data.validation_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "validation_report_only", "G14_validation_report_only", data.validation_profiles, validation_preds))

    _write_csv(history, out / "D17_G14_training_history.csv")
    _write_csv(fit_rows + int_rows + all_rows + val_rows, out / "D17_G14_PROFILE_METRICS.csv")
    _write_csv(fit_trows + int_trows + all_trows + val_trows, out / "D17_G14_PER_TARGET_PROFILE_METRICS.csv")
    _write_csv([fit_tagg, int_tagg, all_tagg, val_tagg], out / "D17_G14_PER_TARGET_AGGREGATE.csv")
    _write_csv(phie_rows, out / "D17_G14_PHIE_ROBUSTNESS_AUDIT.csv")
    _write_csv(_normalization_audit(data), out / "D17_G14_TARGET_NORMALIZATION_AUDIT.csv")
    _write_csv(data.feature_audit_rows, out / "D17_G14_PROFILE_ENCODER_FEATURE_AUDIT.csv")
    _write_csv(pred_manifest, out / "D17_G14_PREDICTION_MANIFEST.csv")

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
        "profile_conditioning": "observed_profile_encoder_no_train_profile_id_phie_convention_head",
    }, model_dir / "best_model.pt")

    fit_mean = float(fit_tagg.get("all_target_profile_r2_mean", float("nan")))
    fit_min = float(fit_tagg.get("all_target_profile_r2_min", float("nan")))
    int_mean = float(int_tagg.get("all_target_profile_r2_mean", float("nan"))) if data.internal_profiles else float("nan")
    int_min = float(int_tagg.get("all_target_profile_r2_min", float("nan"))) if data.internal_profiles else float("nan")
    val_mean = float(val_tagg.get("all_target_profile_r2_mean", float("nan"))) if data.validation_profiles else float("nan")
    val_min = float(val_tagg.get("all_target_profile_r2_min", float("nan"))) if data.validation_profiles else float("nan")
    val_phie_mean = float(val_tagg.get("phie_r2_mean", float("nan")))
    val_phie_min = float(val_tagg.get("phie_r2_min", float("nan")))
    val_phis = float(val_tagg.get("phis_c_r2_mean", float("nan")))

    status_reasons: List[str] = []
    if fit_mean < float(config.get("fit_status_r2_mean_threshold", 0.98)) or fit_min < float(config.get("fit_status_r2_min_threshold", 0.95)):
        status_reasons.append(f"fit-train target/profile R2 below status threshold: mean={fit_mean:.6g}, min={fit_min:.6g}")
    status = "PASS" if not status_reasons else "REVIEW"

    g2_reasons: List[str] = []
    if fit_mean < float(config.get("fit_train_r2_mean_threshold", 0.99)) or fit_min < float(config.get("fit_train_r2_min_threshold", 0.97)):
        g2_reasons.append(f"fit train target/profile R2 below G2 gate: mean={fit_mean:.6g}, min={fit_min:.6g}")
    if data.internal_profiles and (int_mean < float(config.get("internal_heldout_r2_mean_threshold", 0.95)) or int_min < float(config.get("internal_heldout_r2_min_threshold", 0.90))):
        g2_reasons.append(f"internal heldout target/profile R2 below gate: mean={int_mean:.6g}, min={int_min:.6g}")
    if data.validation_profiles and (val_mean < float(config.get("validation_r2_mean_threshold", 0.95)) or val_min < float(config.get("validation_r2_min_threshold", 0.90))):
        g2_reasons.append(f"validation report-only target/profile R2 below gate: mean={val_mean:.6g}, min={val_min:.6g}")
    if data.validation_profiles and (val_phie_mean < float(config.get("validation_phie_r2_mean_threshold", 0.93)) or val_phie_min < float(config.get("validation_phie_r2_min_threshold", 0.90))):
        g2_reasons.append(f"validation phie R2 below gate: mean={val_phie_mean:.6g}, min={val_phie_min:.6g}")
    if data.validation_profiles and (val_phis < float(config.get("validation_phis_c_r2_mean_threshold", 0.90))):
        g2_reasons.append(f"validation phis_c R2 below gate: mean={val_phis:.6g}")
    g2_ready = len(g2_reasons) == 0
    recommendation = "ENTER_D17_G2_HELDOUT_SURROGATE_EXPANSION" if g2_ready else "DO_NOT_ENTER_G2_FIX_PHIE_OR_TRAIN_COVERAGE"

    worst_val_phie = None
    val_phie_rows = [r for r in phie_rows if r.get("split") == "validation_report_only"]
    if val_phie_rows:
        worst_val_phie = min(val_phie_rows, key=lambda r: float(r.get("phie_r2", float("inf"))))

    summary: Dict[str, Any] = {
        "protocol": "D17-G1.4_PHIE_VALIDATION_ROBUSTNESS_REPAIR",
        "created_at_utc": _utc_now(),
        "status": status,
        "status_reasons": status_reasons,
        "recommendation": recommendation,
        "g2_ready": bool(g2_ready),
        "g2_blockers": g2_reasons,
        "purpose": "Repair G1.3 validation phie low-profile failure without using validation soft labels for training or checkpoint selection.",
        "policy": {
            "train_cell_softlabels_used_for_training": True,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_used": False,
            "checkpoint_selection": "fit-train plus train-internal heldout metrics only; validation report-only metrics are not used to select checkpoint",
            "not_a_G2_run": True,
        },
        "generator_semantics_used": {
            "G0_required": True,
            "D15_RG_branch_note": "RG repair branch preserves source voltage/phi labels; phie handled as generator convention/gauge target.",
            "P4D_branch_note": "If selected by train_profile_count, branch embeddings remain observed metadata; validation labels are still report-only.",
            "phie_repair_note": "phie head uses observed I/V/T/profile basis plus nonlinear residual; no validation target is used in loss.",
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
            "augmented_feature_dim": int(data.X_fit.shape[1]),
            "local_input_dim": int(data.local_input_dim),
            "profile_input_dim": int(data.profile_input_dim),
        },
        "model": {
            "class": "ValidationRobustObservedProfileSurrogate",
            "profile_conditioning": "observed profile summary encoder; no train profile-id embedding",
            "phie_head": "observed-feature basis + gated nonlinear residual convention head",
            "target_group_weights": dict(config.get("target_group_weights", {})),
        },
        "fit_train_per_target_aggregate": fit_tagg,
        "internal_heldout_per_target_aggregate": int_tagg,
        "train_all_report_per_target_aggregate": all_tagg,
        "validation_report_only_per_target_aggregate": val_tagg,
        "fit_train_profile_aggregate": fit_agg,
        "internal_heldout_profile_aggregate": int_agg,
        "validation_report_only_profile_aggregate": val_agg,
        "worst_validation_phie_profile": worst_val_phie,
        "files": {
            "summary_json": str(out / "D17_G14_PHIE_VALIDATION_ROBUSTNESS_SUMMARY.json"),
            "profile_metrics_csv": str(out / "D17_G14_PROFILE_METRICS.csv"),
            "per_target_profile_metrics_csv": str(out / "D17_G14_PER_TARGET_PROFILE_METRICS.csv"),
            "per_target_aggregate_csv": str(out / "D17_G14_PER_TARGET_AGGREGATE.csv"),
            "phie_robustness_audit_csv": str(out / "D17_G14_PHIE_ROBUSTNESS_AUDIT.csv"),
            "profile_encoder_feature_audit_csv": str(out / "D17_G14_PROFILE_ENCODER_FEATURE_AUDIT.csv"),
            "target_normalization_audit_csv": str(out / "D17_G14_TARGET_NORMALIZATION_AUDIT.csv"),
            "prediction_manifest_csv": str(out / "D17_G14_PREDICTION_MANIFEST.csv"),
            "training_history_csv": str(out / "D17_G14_training_history.csv"),
            "best_model_pt": str(model_dir / "best_model.pt"),
        },
    }
    json_dump(summary, out / "D17_G14_PHIE_VALIDATION_ROBUSTNESS_SUMMARY.json")
    return summary


def build_and_train_g14(
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
    return train_g14_phie_validation_robust(ds, out_dir, config, device_arg=device_arg, epochs=int(epochs), lr=float(lr), batch_size=int(batch_size))
