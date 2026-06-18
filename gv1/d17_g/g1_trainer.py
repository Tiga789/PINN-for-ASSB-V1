from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .g1_data import G1Dataset, json_dump, save_profile_predictions
from .g1_metrics import aggregate_profile_rows, group_metrics, profile_metrics
from .g1_model import GeneratorSurrogateMLP, make_group_weights


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _predict_np(model: torch.nn.Module, Xn: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch_size):
            xb = torch.as_tensor(Xn[i:i + batch_size], dtype=torch.float32, device=device)
            yp = model(xb).detach().cpu().numpy()
            outs.append(yp)
    yn = np.concatenate(outs, axis=0) if outs else np.zeros((0, y_mean.size), dtype=np.float32)
    return (yn * y_std[None, :] + y_mean[None, :]).astype(np.float32)


def _split_profile_predictions(model: torch.nn.Module, profiles: Sequence[Any], x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device: torch.device) -> List[np.ndarray]:
    out = []
    for p in profiles:
        Xn = ((p.features - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
        out.append(_predict_np(model, Xn, y_mean, y_std, device))
    return out


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def train_g1_smoke(
    dataset: G1Dataset,
    out_dir: str | Path,
    config: Mapping[str, Any],
    device_arg: str = "auto",
    epochs: int = 180,
    lr: float = 1e-3,
    batch_size: int = 2048,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(device_arg)
    seed = int(config.get("seed", 20260615))
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        torch.set_num_threads(int(config.get("torch_num_threads", 2)))
    except Exception:
        pass
    Xn = ((dataset.X_train - dataset.x_mean[None, :]) / dataset.x_std[None, :]).astype(np.float32)
    Yn = ((dataset.Y_train - dataset.y_mean[None, :]) / dataset.y_std[None, :]).astype(np.float32)
    train_loader = DataLoader(TensorDataset(torch.as_tensor(Xn), torch.as_tensor(Yn)), batch_size=int(batch_size), shuffle=True, drop_last=False)
    model_cfg = dict(config.get("model", {}))
    model = GeneratorSurrogateMLP(
        input_dim=dataset.X_train.shape[1],
        output_dim=dataset.Y_train.shape[1],
        width=int(model_cfg.get("width", 256)),
        depth=int(model_cfg.get("depth", 4)),
        dropout=float(model_cfg.get("dropout", 0.02)),
    ).to(device)
    group_weights = make_group_weights(dataset.target_slices, dict(config.get("target_group_weights", {})), dataset.Y_train.shape[1], device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(config.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)), eta_min=float(config.get("min_lr", 1e-5)))
    history: List[Dict[str, Any]] = []
    best = {"epoch": 0, "train_loss": float("inf"), "state_dict": None}
    for ep in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            sq = (pred - yb) ** 2
            loss = (sq * group_weights[None, :]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 5.0)))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        train_loss = float(np.mean(losses)) if losses else float("nan")
        row: Dict[str, Any] = {"epoch": ep, "train_loss": train_loss, "lr": float(opt.param_groups[0]["lr"])}
        if ep == 1 or ep == int(epochs) or ep % int(config.get("eval_every", 10)) == 0:
            train_pred = _predict_np(model, Xn, dataset.y_mean, dataset.y_std, device)
            train_gm = group_metrics(dataset.Y_train, train_pred, dataset.target_slices)
            row["train_r2_mean"] = train_gm["__aggregate__"]["r2_mean"]
            row["train_r2_min"] = train_gm["__aggregate__"]["r2_min"]
            if dataset.X_validation.shape[0] > 0:
                Xvn = ((dataset.X_validation - dataset.x_mean[None, :]) / dataset.x_std[None, :]).astype(np.float32)
                val_pred = _predict_np(model, Xvn, dataset.y_mean, dataset.y_std, device)
                val_gm = group_metrics(dataset.Y_validation, val_pred, dataset.target_slices)
                row["validation_r2_mean_report_only"] = val_gm["__aggregate__"]["r2_mean"]
                row["validation_r2_min_report_only"] = val_gm["__aggregate__"]["r2_min"]
        if math.isfinite(train_loss) and train_loss < best["train_loss"]:
            best = {"epoch": ep, "train_loss": train_loss, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        history.append(row)
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    train_preds_by_profile = _split_profile_predictions(model, dataset.train_profiles, dataset.x_mean, dataset.x_std, dataset.y_mean, dataset.y_std, device)
    val_preds_by_profile = _split_profile_predictions(model, dataset.validation_profiles, dataset.x_mean, dataset.x_std, dataset.y_mean, dataset.y_std, device) if dataset.validation_profiles else []
    train_pm = profile_metrics(dataset.train_profiles, train_preds_by_profile)["rows"]
    val_pm = profile_metrics(dataset.validation_profiles, val_preds_by_profile)["rows"] if dataset.validation_profiles else []
    pred_manifest = []
    pred_manifest.extend(save_profile_predictions(out / "predictions" / "train", "train", dataset.train_profiles, train_preds_by_profile))
    if dataset.validation_profiles:
        pred_manifest.extend(save_profile_predictions(out / "predictions" / "validation_report_only", "validation", dataset.validation_profiles, val_preds_by_profile))
    _write_csv(history, out / "training_history.csv")
    _write_csv(train_pm + val_pm, out / "D17_G1_PROFILE_METRICS.csv")
    _write_csv(pred_manifest, out / "D17_G1_PREDICTION_MANIFEST.csv")
    train_agg = aggregate_profile_rows(train_pm)
    val_agg = aggregate_profile_rows(val_pm)
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
        "config": dict(config),
        "best_epoch": int(best["epoch"]),
    }, model_dir / "best_model.pt")
    train_mean = float(train_agg.get("r2_mean_mean", float("nan")))
    val_mean = float(val_agg.get("r2_mean_mean", float("nan"))) if val_agg else float("nan")
    train_min = float(train_agg.get("r2_min_min", float("nan")))
    val_min = float(val_agg.get("r2_min_min", float("nan"))) if val_agg else float("nan")
    smoke_train_threshold = float(config.get("smoke_train_r2_mean_threshold", 0.80))
    g2_val_threshold = float(config.get("g2_validation_r2_mean_threshold", 0.95))
    reasons: List[str] = []
    if not math.isfinite(train_mean) or train_mean < smoke_train_threshold:
        reasons.append(f"train profile mean R2 below smoke threshold {smoke_train_threshold}: {train_mean:.6g}")
    status = "PASS" if not reasons else "REVIEW"
    promotion_reasons: List[str] = []
    if not math.isfinite(val_mean) or val_mean < g2_val_threshold:
        promotion_reasons.append(f"validation report-only mean R2 below G2 threshold {g2_val_threshold}: {val_mean:.6g}")
    promotion_status = "PASS" if status == "PASS" and not promotion_reasons else "REVIEW"
    summary: Dict[str, Any] = {
        "protocol": "D17-G1_SUPERVISED_GENERATOR_SURROGATE_SMOKE",
        "created_at_utc": _utc_now(),
        "status": status,
        "reasons": reasons,
        "promotion_status": promotion_status,
        "promotion_reasons": promotion_reasons,
        "g2_ready": bool(promotion_status == "PASS"),
        "device": str(device),
        "seed": seed,
        "epochs": int(epochs),
        "best_epoch": int(best["epoch"]),
        "best_train_loss": float(best["train_loss"]),
        "policy": {
            "train_cell_softlabels_used_for_training": True,
            "validation_softlabels_used_for_training": False,
            "validation_softlabels_report_only": True,
            "frozen_test_softlabels_used": False,
            "checkpoint_selection": "train_loss_only_for_G1_smoke",
            "purpose": "supervised generator surrogate, not strict no-state-label inverse PINN",
        },
        "dataset": dataset.manifest_summary,
        "train_profile_aggregate": train_agg,
        "validation_profile_aggregate_report_only": val_agg,
        "train_profile_metrics_sample": train_pm[:5],
        "validation_profile_metrics_sample": val_pm[:5],
        "files": {
            "summary_json": str(out / "D17_G1_SUPERVISED_SURROGATE_SMOKE_SUMMARY.json"),
            "profile_metrics_csv": str(out / "D17_G1_PROFILE_METRICS.csv"),
            "prediction_manifest_csv": str(out / "D17_G1_PREDICTION_MANIFEST.csv"),
            "training_history_csv": str(out / "training_history.csv"),
            "best_model_pt": str(model_dir / "best_model.pt"),
        },
    }
    json_dump(summary, out / "D17_G1_SUPERVISED_SURROGATE_SMOKE_SUMMARY.json")
    return summary
