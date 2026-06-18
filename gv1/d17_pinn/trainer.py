# -*- coding: utf-8 -*-
"""D17-P2 smoke trainer utilities."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from .config import cfg_get
from .dataset import D17ProfileDataset, load_observed_profile
from .losses import audit_numbers, total_d17_loss
from .model import D17MechanisticPINN, make_batch_from_profile
from .p2dlite_prior import load_p2dlite_prior, prior_to_jsonable


FORBIDDEN_PROFILE_KEYS = {
    "cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c",
    "cs_a_soft", "cs_c_soft", "theta_a_soft", "theta_c_soft", "phie_soft", "phis_c_soft",
    "theta0_oracle", "oracle_shift",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def crop_time_window(profile: Dict[str, Any], time_window_s: Optional[float], max_time_points: Optional[int]) -> Dict[str, Any]:
    t_key = "t_global_s" if "t_global_s" in profile else "time_s"
    t = np.asarray(profile[t_key], dtype=np.float64).reshape(-1)
    if t.size < 2:
        return profile
    t0 = float(t[0])
    mask = np.ones_like(t, dtype=bool)
    if time_window_s is not None and time_window_s > 0:
        mask &= (t - t0) <= float(time_window_s)
    idx = np.where(mask)[0]
    if idx.size < 8:
        idx = np.arange(min(len(t), 8))
    if max_time_points is not None and idx.size > int(max_time_points):
        idx = np.linspace(idx[0], idx[-1], int(max_time_points)).round().astype(int)
    out: Dict[str, Any] = {}
    for k, v in profile.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == len(t):
            out[k] = v[idx]
        else:
            out[k] = v
    out["_crop_indices_count"] = int(len(idx))
    out["_time_window_s"] = float(time_window_s or 0.0)
    return out


def assert_no_state_profile_keys(profile: Mapping[str, Any]) -> None:
    present = sorted(FORBIDDEN_PROFILE_KEYS.intersection(profile.keys()))
    if present:
        raise RuntimeError(f"D17-P2 observed profile contains forbidden state-answer keys: {present}")
    ignored = profile.get("_ignored_state_keys", [])
    if ignored:
        # Ignored keys may appear in emergency files, but formal replay_npz should not contain them.
        raise RuntimeError(f"D17-P2 refused profile because source contained state-answer keys: {ignored}")


def load_one_profile_from_manifest(
    split_manifest: str | Path,
    split: str = "train",
    profile_index: int = 0,
    time_window_s: float = 40000.0,
    max_time_points: int = 4096,
    allow_softlabel_npz_profile_source: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ds = D17ProfileDataset(
        split_manifest=split_manifest,
        split=split,
        allow_softlabel_npz_profile_source=allow_softlabel_npz_profile_source,
    )
    if len(ds) == 0:
        raise RuntimeError(f"No records found for split={split}")
    idx = int(profile_index) % len(ds)
    rec = ds.records[idx]
    profile = ds[idx]
    assert_no_state_profile_keys(profile)
    profile = crop_time_window(profile, time_window_s=time_window_s, max_time_points=max_time_points)
    return profile, rec


def train_smoke(cfg: Mapping[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg_get(cfg, "seed", 20260615))
    set_seed(seed)
    device = choose_device(str(cfg_get(cfg, "train.device", "auto")))
    split_manifest = cfg_get(cfg, "paths.split_manifest")
    resolved_spec = cfg_get(cfg, "paths.resolved_spec")
    profile, rec = load_one_profile_from_manifest(
        split_manifest=split_manifest,
        split=str(cfg_get(cfg, "train.split", "train")),
        profile_index=int(cfg_get(cfg, "train.profile_index", 0)),
        time_window_s=float(cfg_get(cfg, "train.time_window_s", 40000.0)),
        max_time_points=int(cfg_get(cfg, "train.max_time_points", 4096)),
        allow_softlabel_npz_profile_source=bool(cfg_get(cfg, "train.allow_softlabel_npz_profile_source", False)),
    )
    prior = load_p2dlite_prior(resolved_spec, allow_smoke_defaults=True)
    n_r = int(cfg_get(cfg, "train.n_r", 17))
    batch = make_batch_from_profile(profile, n_r=n_r, device=device)
    model = D17MechanisticPINN(
        prior=prior,
        feature_dim=int(batch["features"].shape[-1]),
        n_r=n_r,
        hidden_dim=int(cfg_get(cfg, "model.hidden_dim", 64)),
        latent_hidden_dim=int(cfg_get(cfg, "model.latent_hidden_dim", 64)),
        delta_layers=int(cfg_get(cfg, "model.delta_layers", 3)),
        delta_amp_fraction=float(cfg_get(cfg, "model.delta_amp_fraction", 0.018)),
        enable_low_transition_residual=bool(cfg_get(cfg, "model.enable_low_transition_residual", False)),
        use_observed_voltage_for_gate=bool(cfg_get(cfg, "model.use_observed_voltage_for_gate", True)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg_get(cfg, "train.lr", 1.0e-3)),
        weight_decay=float(cfg_get(cfg, "train.weight_decay", 0.0)),
    )
    epochs = int(cfg_get(cfg, "train.epochs", 100))
    grad_clip = float(cfg_get(cfg, "train.gradient_clip_norm", 10.0))
    weights = cfg.get("loss_weights", {}) if isinstance(cfg.get("loss_weights", {}), Mapping) else {}
    history = []
    best = {"loss": float("inf"), "epoch": -1, "state_dict": None, "metrics": None}
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss, parts = total_d17_loss(out, batch, prior, weights=weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) if grad_clip > 0 else torch.tensor(0.0)
        opt.step()
        with torch.no_grad():
            out_eval = model(batch)
            metrics = audit_numbers(out_eval, batch)
            row = {"epoch": epoch, "total_loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu())}
            for k, v in parts.items():
                row[f"loss_{k}"] = float(v.detach().cpu())
            row.update(metrics)
            history.append(row)
            if row["total_loss"] < best["loss"]:
                best = {
                    "loss": row["total_loss"],
                    "epoch": epoch,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "metrics": metrics,
                }
    hist_path = out_dir / "training_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    if best["state_dict"] is not None:
        torch.save(best["state_dict"], model_dir / "best_model.pt")
    torch.save(model.state_dict(), model_dir / "last_model.pt")
    final_metrics = history[-1]
    status = "PASS"
    reasons = []
    if not np.isfinite(final_metrics["total_loss"]):
        status = "FAIL"; reasons.append("non-finite total loss")
    if not (1.5 <= final_metrics["V_pred_min"] <= 5.5 and 1.5 <= final_metrics["V_pred_max"] <= 5.5):
        status = "REVIEW"; reasons.append("V_pred range outside broad smoke bounds")
    if max(final_metrics["zero_mean_max_abs_a_mol_m3"], final_metrics["zero_mean_max_abs_c_mol_m3"]) > 1.0e-2:
        status = "REVIEW"; reasons.append("zero-volume-mean audit larger than smoke threshold")
    summary = {
        "protocol": "D17-P2_FORWARD_BACKWARD_SMOKE",
        "status": status,
        "reasons": reasons,
        "seed": seed,
        "device": str(device),
        "split_manifest": str(split_manifest),
        "resolved_spec": str(resolved_spec),
        "manifest_record": rec,
        "profile_source_npz": profile.get("_source_npz"),
        "n_time_points": int(batch["t_s"].numel()),
        "n_r": n_r,
        "epochs": epochs,
        "best_epoch": int(best["epoch"]),
        "best_loss": float(best["loss"]),
        "initial_metrics": history[0],
        "final_metrics": final_metrics,
        "best_metrics": best["metrics"],
        "no_state_label_policy": {
            "training_uses_state_softlabels": False,
            "profile_loader": "replay_npz observed-only",
            "forbidden_state_keys": sorted(FORBIDDEN_PROFILE_KEYS),
            "softlabel_paths_only_for_future_frozen_audit": True,
        },
        "prior_snapshot": prior_to_jsonable(prior),
        "outputs": {
            "training_history_csv": str(hist_path),
            "best_model_pt": str(model_dir / "best_model.pt"),
            "last_model_pt": str(model_dir / "last_model.pt"),
        },
    }
    (out_dir / "D17_P2_SMOKE_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Save lightweight prediction sample for plotting/debugging, no state labels.
    with torch.no_grad():
        pred = model(batch)
    np.savez_compressed(
        out_dir / "D17_P2_SMOKE_PRED_OBS_ONLY.npz",
        t_s=batch["t_s"].detach().cpu().numpy(),
        I_profile=batch["current_A"].detach().cpu().numpy(),
        voltage_exp=batch["voltage_exp"].detach().cpu().numpy(),
        V_pred=pred["V_pred"].detach().cpu().numpy(),
        cbar_a=pred["cbar_a"].detach().cpu().numpy(),
        cbar_c=pred["cbar_c"].detach().cpu().numpy(),
        theta_a_surface=pred["theta_a_surface"].detach().cpu().numpy(),
        theta_c_surface=pred["theta_c_surface"].detach().cpu().numpy(),
        phie=pred["phie"].detach().cpu().numpy(),
        phis_c=pred["phis_c"].detach().cpu().numpy(),
        r_norm=batch["r_norm"].detach().cpu().numpy(),
    )
    return summary
