# -*- coding: utf-8 -*-
"""Minimal D17-P2 evaluator for voltage/physics smoke outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from .config import cfg_get
from .losses import audit_numbers, total_d17_loss
from .model import D17MechanisticPINN, make_batch_from_profile
from .p2dlite_prior import load_p2dlite_prior
from .trainer import crop_time_window, load_one_profile_from_manifest, set_seed, choose_device


def eval_one_profile(cfg: Mapping[str, Any], model_path: str | Path, out_dir: str | Path, split: str = "validation", profile_index: int = 0) -> Dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg_get(cfg, "seed", 20260615)))
    device = choose_device(str(cfg_get(cfg, "train.device", "auto")))
    profile, rec = load_one_profile_from_manifest(
        split_manifest=cfg_get(cfg, "paths.split_manifest"),
        split=split,
        profile_index=profile_index,
        time_window_s=float(cfg_get(cfg, "train.time_window_s", 40000.0)),
        max_time_points=int(cfg_get(cfg, "train.max_time_points", 4096)),
        allow_softlabel_npz_profile_source=False,
    )
    prior = load_p2dlite_prior(cfg_get(cfg, "paths.resolved_spec"), allow_smoke_defaults=True)
    batch = make_batch_from_profile(profile, n_r=int(cfg_get(cfg, "train.n_r", 17)), device=device)
    model = D17MechanisticPINN(
        prior=prior,
        feature_dim=int(batch["features"].shape[-1]),
        n_r=int(cfg_get(cfg, "train.n_r", 17)),
        hidden_dim=int(cfg_get(cfg, "model.hidden_dim", 64)),
        latent_hidden_dim=int(cfg_get(cfg, "model.latent_hidden_dim", 64)),
        delta_layers=int(cfg_get(cfg, "model.delta_layers", 3)),
        delta_amp_fraction=float(cfg_get(cfg, "model.delta_amp_fraction", 0.018)),
        enable_low_transition_residual=bool(cfg_get(cfg, "model.enable_low_transition_residual", False)),
        use_observed_voltage_for_gate=bool(cfg_get(cfg, "model.use_observed_voltage_for_gate", True)),
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    with torch.no_grad():
        pred = model(batch)
        _, parts = total_d17_loss(pred, batch, prior, weights=cfg.get("loss_weights", {}))
        metrics = audit_numbers(pred, batch)
    summary = {
        "protocol": "D17-P2_EVAL_OBSERVED_ONLY",
        "status": "PASS" if np.isfinite(metrics.get("voltage_mae_V", np.nan)) else "FAIL",
        "split": split,
        "profile_index": profile_index,
        "manifest_record": rec,
        "model_path": str(model_path),
        "metrics": metrics,
        "losses": {k: float(v.detach().cpu()) for k, v in parts.items()},
        "no_state_label_policy": "observed I/V/T only; soft labels are not loaded by this evaluator",
    }
    (out_dir / "D17_P2_EVAL_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "D17_P2_EVAL_PRED_OBS_ONLY.npz",
        t_s=batch["t_s"].detach().cpu().numpy(),
        I_profile=batch["current_A"].detach().cpu().numpy(),
        voltage_exp=batch["voltage_exp"].detach().cpu().numpy(),
        V_pred=pred["V_pred"].detach().cpu().numpy(),
        phie=pred["phie"].detach().cpu().numpy(),
        phis_c=pred["phis_c"].detach().cpu().numpy(),
    )
    return summary
