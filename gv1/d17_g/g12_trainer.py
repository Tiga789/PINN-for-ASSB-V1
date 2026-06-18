from __future__ import annotations

import csv
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .g1_data import G1Dataset, ProfilePack, build_g1_dataset, json_dump, save_profile_predictions
from .g1_metrics import aggregate_profile_rows, group_metrics, profile_metrics, r2_score
from .g12_model import ProfileConditionedMultiHeadSurrogate


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


def _profile_row_ids(profiles: Sequence[ProfilePack], offset: int = 1) -> Tuple[np.ndarray, Dict[str, int]]:
    ids: List[np.ndarray] = []
    mapping: Dict[str, int] = {}
    for i, p in enumerate(profiles, start=offset):
        mapping[p.canonical_cell_uid] = i
        ids.append(np.full(p.features.shape[0], i, dtype=np.int64))
    return (np.concatenate(ids, axis=0) if ids else np.zeros((0,), dtype=np.int64)), mapping


def _normalized_targets(dataset: G1Dataset) -> np.ndarray:
    y = ((dataset.Y_train - dataset.y_mean[None, :]) / dataset.y_std[None, :]).astype(np.float32)
    y[~np.isfinite(y)] = 0.0
    return y


def _normalized_features(x: np.ndarray, dataset: G1Dataset) -> np.ndarray:
    out = ((x - dataset.x_mean[None, :]) / dataset.x_std[None, :]).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _group_balanced_loss(pred: torch.Tensor, target: torch.Tensor, target_slices: Mapping[str, Tuple[int, int]], weights: Mapping[str, float]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    wsum = 0.0
    for key, (a, b) in target_slices.items():
        weight = float(weights.get(key, 1.0))
        if weight <= 0:
            continue
        group_mse = torch.mean((pred[:, a:b] - target[:, a:b]) ** 2)
        losses.append(group_mse * weight)
        wsum += weight
    if not losses:
        return torch.mean((pred - target) ** 2)
    return torch.stack(losses).sum() / max(wsum, 1e-12)


def _predict_np(
    model: torch.nn.Module,
    X: np.ndarray,
    profile_ids: np.ndarray,
    dataset: G1Dataset,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    Xn = _normalized_features(X, dataset)
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch_size):
            xb = torch.as_tensor(Xn[i:i + batch_size], dtype=torch.float32, device=device)
            pid = torch.as_tensor(profile_ids[i:i + batch_size], dtype=torch.long, device=device)
            yp = model(xb, pid).detach().cpu().numpy()
            outs.append(yp)
    yn = np.concatenate(outs, axis=0) if outs else np.zeros((0, dataset.y_mean.size), dtype=np.float32)
    return (yn * dataset.y_std[None, :] + dataset.y_mean[None, :]).astype(np.float32)


def _predict_profiles(
    model: torch.nn.Module,
    profiles: Sequence[ProfilePack],
    profile_id_map: Mapping[str, int],
    dataset: G1Dataset,
    device: torch.device,
    unknown_profile_id: int = 0,
) -> List[np.ndarray]:
    preds: List[np.ndarray] = []
    for p in profiles:
        pid_val = int(profile_id_map.get(p.canonical_cell_uid, unknown_profile_id))
        pids = np.full(p.features.shape[0], pid_val, dtype=np.int64)
        preds.append(_predict_np(model, p.features, pids, dataset, device))
    return preds


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
            })
    return rows


def _target_aggregate(rows: Sequence[Mapping[str, Any]], split_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"split": split_name}
    targets = sorted({str(r.get("target")) for r in rows})
    vals = []
    for t in targets:
        rs = [float(r.get("r2", float("nan"))) for r in rows if str(r.get("target")) == t and np.isfinite(float(r.get("r2", float("nan"))))]
        if rs:
            out[f"{t}_r2_mean"] = float(np.mean(rs))
            out[f"{t}_r2_min"] = float(np.min(rs))
            vals.extend(rs)
    out["all_target_profile_r2_mean"] = float(np.mean(vals)) if vals else float("nan")
    out["all_target_profile_r2_min"] = float(np.min(vals)) if vals else float("nan")
    return out


def _normalization_audit(dataset: G1Dataset) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, (a, b) in dataset.target_slices.items():
        arr = dataset.Y_train[:, a:b]
        rows.append({
            "target": key,
            "dim": int(b - a),
            "mean_abs": float(np.nanmean(np.abs(arr))),
            "std_mean": float(np.nanmean(np.nanstd(arr, axis=0))),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "y_std_min": float(np.nanmin(dataset.y_std[a:b])),
            "y_std_max": float(np.nanmax(dataset.y_std[a:b])),
        })
    return rows


def train_g12_closedset(
    dataset: G1Dataset,
    out_dir: str | Path,
    config: Mapping[str, Any],
    device_arg: str = "auto",
    epochs: int = 700,
    lr: float = 8e-4,
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

    Xn = _normalized_features(dataset.X_train, dataset)
    Yn = _normalized_targets(dataset)
    profile_ids, profile_id_map = _profile_row_ids(dataset.train_profiles, offset=1)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(Xn), torch.as_tensor(Yn), torch.as_tensor(profile_ids)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )

    model_cfg = dict(config.get("model", {}))
    model = ProfileConditionedMultiHeadSurrogate(
        input_dim=dataset.X_train.shape[1],
        target_slices=dataset.target_slices,
        profile_count=len(dataset.train_profiles),
        width=int(model_cfg.get("width", 640)),
        depth=int(model_cfg.get("depth", 7)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        profile_embedding_dim=int(model_cfg.get("profile_embedding_dim", 24)),
        phie_direct_width=int(model_cfg.get("phie_direct_width", 192)),
    ).to(device)
    group_weights = dict(config.get("target_group_weights", {
        "theta_a": 1.5,
        "theta_c": 1.5,
        "cs_a": 1.0,
        "cs_c": 1.0,
        "phie": 10.0,
        "phis_c": 2.5,
    }))
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(model_cfg.get("weight_decay", 1e-6)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(model_cfg.get("min_lr", 1e-5)))
    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"epoch": 0, "score": -1e99, "train_loss": float("inf"), "state_dict": None}
    eval_every = int(config.get("eval_every", 50))
    min_epochs_before_stop = int(config.get("min_epochs_before_early_stop", 200))
    pass_mean = float(config.get("closedset_r2_mean_threshold", 0.98))
    pass_min = float(config.get("closedset_r2_min_threshold", 0.95))

    train_all_pids = profile_ids.astype(np.int64)
    for ep in range(1, int(epochs) + 1):
        model.train()
        batch_losses = []
        for xb, yb, pb in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            pb = pb.to(device=device, dtype=torch.long)
            pred = model(xb, pb)
            loss = _group_balanced_loss(pred, yb, dataset.target_slices, group_weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_cfg.get("grad_clip_norm", 5.0)))
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        row: Dict[str, Any] = {"epoch": ep, "train_loss": train_loss, "lr": float(opt.param_groups[0]["lr"])}
        do_eval = ep == 1 or ep == int(epochs) or ep % eval_every == 0
        if do_eval:
            pred_train = _predict_np(model, dataset.X_train, train_all_pids, dataset, device, batch_size=8192)
            train_gm = group_metrics(dataset.Y_train, pred_train, dataset.target_slices)
            row["train_r2_mean"] = float(train_gm["__aggregate__"]["r2_mean"])
            row["train_r2_min"] = float(train_gm["__aggregate__"]["r2_min"])
            for key in ["theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c"]:
                if key in train_gm:
                    row[f"{key}_r2"] = float(train_gm[key]["r2"])
                    row[f"{key}_mae"] = float(train_gm[key]["mae"])
            score = row["train_r2_mean"] + 0.2 * row["train_r2_min"] - 0.01 * train_loss
            if np.isfinite(score) and score > best["score"]:
                best = {"epoch": ep, "score": float(score), "train_loss": train_loss, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
            if bool(config.get("early_stop_on_pass", True)) and ep >= min_epochs_before_stop:
                if row["train_r2_mean"] >= pass_mean and row["train_r2_min"] >= pass_min:
                    history.append(row)
                    break
        elif math.isfinite(train_loss) and train_loss < best["train_loss"]:
            best = {"epoch": ep, "score": best.get("score", -1e99), "train_loss": train_loss, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        history.append(row)

    if best.get("state_dict") is not None:
        model.load_state_dict(best["state_dict"])

    train_preds = _predict_profiles(model, dataset.train_profiles, profile_id_map, dataset, device, unknown_profile_id=0)
    val_preds = _predict_profiles(model, dataset.validation_profiles, {}, dataset, device, unknown_profile_id=0) if dataset.validation_profiles else []
    train_rows = profile_metrics(dataset.train_profiles, train_preds)["rows"]
    val_rows = profile_metrics(dataset.validation_profiles, val_preds)["rows"] if dataset.validation_profiles else []
    train_target_rows = _per_target_rows(dataset.train_profiles, train_preds, "train_closedset")
    val_target_rows = _per_target_rows(dataset.validation_profiles, val_preds, "validation_report_only_unknown_profile_id") if dataset.validation_profiles else []
    train_agg = aggregate_profile_rows(train_rows)
    val_agg = aggregate_profile_rows(val_rows)
    train_target_agg = _target_aggregate(train_target_rows, "train_closedset")
    val_target_agg = _target_aggregate(val_target_rows, "validation_report_only_unknown_profile_id") if val_target_rows else {}

    pred_manifest: List[Dict[str, Any]] = []
    pred_manifest.extend(save_profile_predictions(out / "predictions" / "train_closedset", "train_closedset", dataset.train_profiles, train_preds))
    if dataset.validation_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "validation_report_only", "validation_report_only", dataset.validation_profiles, val_preds))
    _write_csv(history, out / "D17_G12_training_history.csv")
    _write_csv(train_rows + val_rows, out / "D17_G12_PROFILE_METRICS.csv")
    _write_csv(train_target_rows + val_target_rows, out / "D17_G12_PER_TARGET_PROFILE_METRICS.csv")
    _write_csv([train_target_agg] + ([val_target_agg] if val_target_agg else []), out / "D17_G12_PER_TARGET_AGGREGATE.csv")
    _write_csv(_normalization_audit(dataset), out / "D17_G12_TARGET_NORMALIZATION_AUDIT.csv")
    _write_csv(pred_manifest, out / "D17_G12_PREDICTION_MANIFEST.csv")

    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_mean": dataset.x_mean,
        "x_std": dataset.x_std,
        "y_mean": dataset.y_mean,
        "y_std": dataset.y_std,
        "feature_names": dataset.feature_names,
        "target_names": dataset.target_names,
        "target_slices": dataset.target_slices,
        "profile_id_map": profile_id_map,
        "config": dict(config),
        "best_epoch": int(best.get("epoch", 0)),
    }, model_dir / "best_model.pt")

    mean_r2 = float(train_target_agg.get("all_target_profile_r2_mean", float("nan")))
    min_r2 = float(train_target_agg.get("all_target_profile_r2_min", float("nan")))
    reasons: List[str] = []
    if not math.isfinite(mean_r2) or mean_r2 < pass_mean:
        reasons.append(f"train closed-set mean target/profile R2 below {pass_mean}: {mean_r2:.6g}")
    if not math.isfinite(min_r2) or min_r2 < pass_min:
        reasons.append(f"train closed-set min target/profile R2 below {pass_min}: {min_r2:.6g}")
    status = "PASS" if not reasons else "REVIEW"
    recommendation = "G1_2_CLOSEDSET_REPAIRED_RERUN_G1_WITH_VALIDATION" if status == "PASS" else "DO_NOT_ENTER_G2_FIX_G1_2_REMAINING_TARGETS"
    summary: Dict[str, Any] = {
        "protocol": "D17-G1.2_PHIE_GAUGE_TARGET_SCALING_CLOSEDSET_REPAIR",
        "created_at_utc": _utc_now(),
        "status": status,
        "reasons": reasons,
        "recommendation": recommendation,
        "g2_ready": False,
        "purpose": "Repair G1 train closed-set generator-surrogate reproduction before any G2 expansion.",
        "policy": {
            "train_cell_softlabels_used_for_training": True,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_used": False,
            "checkpoint_selection": "train closed-set target/profile R2 and train loss only",
            "not_a_promotion_run": True,
        },
        "generator_semantics_used": {
            "G0_required": True,
            "D15_RG_branch_note": "RG repair branch preserves source voltage/phi labels; phie is handled with a dedicated gauge/current-aware head.",
            "P4D_branch_note": "P4D branch can define phis_c from voltage_exp and phie from an ohmic-current-like source; branch embeddings remain in features.",
        },
        "device": str(device),
        "seed": seed,
        "epochs_requested": int(epochs),
        "best_epoch": int(best.get("epoch", 0)),
        "dataset": dataset.manifest_summary,
        "model": {
            "class": "ProfileConditionedMultiHeadSurrogate",
            "target_group_weights": group_weights,
            "profile_conditioning": "train profile ids 1..N; validation report-only profiles use unknown id 0",
        },
        "train_closedset_profile_aggregate": train_agg,
        "train_closedset_per_target_aggregate": train_target_agg,
        "validation_report_only_profile_aggregate": val_agg,
        "validation_report_only_per_target_aggregate": val_target_agg,
        "train_profile_metrics_sample": train_rows[:5],
        "files": {
            "summary_json": str(out / "D17_G12_PHIE_GAUGE_CLOSEDSET_REPAIR_SUMMARY.json"),
            "profile_metrics_csv": str(out / "D17_G12_PROFILE_METRICS.csv"),
            "per_target_profile_metrics_csv": str(out / "D17_G12_PER_TARGET_PROFILE_METRICS.csv"),
            "per_target_aggregate_csv": str(out / "D17_G12_PER_TARGET_AGGREGATE.csv"),
            "target_normalization_audit_csv": str(out / "D17_G12_TARGET_NORMALIZATION_AUDIT.csv"),
            "prediction_manifest_csv": str(out / "D17_G12_PREDICTION_MANIFEST.csv"),
            "training_history_csv": str(out / "D17_G12_training_history.csv"),
            "best_model_pt": str(model_dir / "best_model.pt"),
        },
    }
    json_dump(summary, out / "D17_G12_PHIE_GAUGE_CLOSEDSET_REPAIR_SUMMARY.json")
    return summary


def build_and_train_g12(
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
    return train_g12_closedset(ds, out_dir, config, device_arg=device_arg, epochs=int(epochs), lr=float(lr), batch_size=int(batch_size))
