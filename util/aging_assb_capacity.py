# -*- coding: utf-8 -*-
"""
Capacity/SOH aging head for ASSB ModelFin_108.

This file does not turn on the original soft-label data loss. It defines a
cycle-level capacity branch and the scalar physics-style regularization terms
needed to fit Q_dis(k)/SOH(k).

The head is intentionally independent from the existing SPM state output. In the
first landing step it can be trained/evaluated as a wrapper. Later it can be
registered inside util/myNN.py and called from util/_losses.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CapacityHeadConfig:
    n_features: int
    hidden: int = 32
    soh_min: float = 0.45
    soh_max: float = 1.02
    initial_rate_bias: float = -6.0
    smooth_eps: float = 1.0e-8


class AgingCapacityHead(nn.Module):
    """Monotone cycle-level capacity/SOH head.

    Parameters
    ----------
    features:
        Protocol/cycle features, shape (N_cycle, n_features). Capacity labels
        must not be included in features.
    d_tau:
        Non-negative cycle increments, shape (N_cycle,). Usually normalized
        cycle increments or throughput increments.
    q_ref_Ah:
        Scalar reference capacity in Ah.

    Returns
    -------
    dict with Q_pred_Ah, SOH_pred, Q_loss_frac, dQ_loss_frac, raw_rate.
    """

    def __init__(self, n_features: int, hidden: int = 32, soh_min: float = 0.45, soh_max: float = 1.02, initial_rate_bias: float = -6.0):
        super().__init__()
        self.config = CapacityHeadConfig(
            n_features=int(n_features),
            hidden=int(hidden),
            soh_min=float(soh_min),
            soh_max=float(soh_max),
            initial_rate_bias=float(initial_rate_bias),
        )
        self.net = nn.Sequential(
            nn.Linear(int(n_features), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), 1),
        )
        # A trainable global scale lets the head fit amplitude without relying on
        # unstable large raw network outputs.
        self.log_rate_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                nn.init.zeros_(mod.bias)
        # Start close to no degradation; training can increase the rate.
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, self.config.initial_rate_bias)

    @property
    def n_features(self) -> int:
        return self.config.n_features

    def forward(self, features: torch.Tensor, d_tau: torch.Tensor, q_ref_Ah: torch.Tensor | float) -> Dict[str, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(f"features must be 2D, got shape={tuple(features.shape)}")
        if features.shape[-1] != self.n_features:
            raise ValueError(f"features last dim {features.shape[-1]} != n_features {self.n_features}")
        dtype = features.dtype
        device = features.device
        d_tau = torch.as_tensor(d_tau, dtype=dtype, device=device).reshape(-1)
        q_ref = torch.as_tensor(q_ref_Ah, dtype=dtype, device=device)
        if d_tau.shape[0] != features.shape[0]:
            raise ValueError("d_tau length must match number of cycle features")

        raw_rate = self.net(features).squeeze(-1)
        # Non-negative incremental loss. The bias keeps initialization stable.
        rate_scale = torch.exp(self.log_rate_scale.to(dtype=dtype, device=device))
        d_loss = F.softplus(raw_rate) * d_tau.clamp_min(self.config.smooth_eps) * rate_scale
        q_loss_frac = torch.cumsum(d_loss, dim=0)
        soh = (1.0 - q_loss_frac).clamp(min=self.config.soh_min, max=self.config.soh_max)
        q_pred = q_ref * soh
        return {
            "Q_pred_Ah": q_pred,
            "SOH_pred": soh,
            "Q_loss_frac": q_loss_frac,
            "dQ_loss_frac": d_loss,
            "raw_rate": raw_rate,
        }


def capacity_physics_loss(
    head: AgingCapacityHead,
    batch: Dict[str, torch.Tensor],
    *,
    w_capacity: float = 1.0,
    w_monotone: float = 0.1,
    w_smooth: float = 0.05,
    w_prior: float = 0.0,
    huber_beta: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute scalar capacity regularization losses.

    This is a physics-style cycle-level residual. It is not the original
    pointwise soft-label state data loss.
    """
    pred = head(batch["features"], batch["d_tau"], batch["Q_ref_Ah"])
    q_pred = pred["Q_pred_Ah"]
    soh_pred = pred["SOH_pred"]
    q_obs = batch["Q_dis_Ah"].to(dtype=q_pred.dtype, device=q_pred.device)
    soh_obs = batch.get("SOH")
    if soh_obs is None:
        soh_obs = q_obs / batch["Q_ref_Ah"].to(dtype=q_pred.dtype, device=q_pred.device).clamp_min(1.0e-12)
    else:
        soh_obs = soh_obs.to(dtype=q_pred.dtype, device=q_pred.device)
    mask = batch.get("train_mask")
    if mask is None:
        mask = torch.ones_like(q_obs, dtype=torch.bool)
    else:
        mask = mask.to(dtype=torch.bool, device=q_pred.device)
    scale = batch["Q_ref_Ah"].to(dtype=q_pred.dtype, device=q_pred.device).clamp_min(1.0e-12)

    # Robust residual in normalized capacity units.
    l_cap = F.smooth_l1_loss(q_pred[mask] / scale, q_obs[mask] / scale, beta=float(huber_beta))

    # Q should not increase materially with cycle. This is a soft penalty because
    # raw experimental capacity can have tiny early-cycle rebound/noise.
    if q_pred.numel() >= 2:
        dq = q_pred[1:] - q_pred[:-1]
        l_mon = torch.mean(torch.relu(dq / scale) ** 2)
    else:
        l_mon = torch.zeros((), dtype=q_pred.dtype, device=q_pred.device)

    # Smooth second difference to prevent saw-tooth per-cycle overfit.
    if q_pred.numel() >= 3:
        ddq = q_pred[2:] - 2.0 * q_pred[1:-1] + q_pred[:-2]
        l_smooth = torch.mean((ddq / scale) ** 2)
    else:
        l_smooth = torch.zeros((), dtype=q_pred.dtype, device=q_pred.device)

    # Optional weak prior discouraging unrealistic total loss above the lower SOH bound.
    qloss = pred["Q_loss_frac"]
    l_prior = torch.mean(torch.relu(qloss - (1.0 - head.config.soh_min)) ** 2) if w_prior else torch.zeros((), dtype=q_pred.dtype, device=q_pred.device)

    total = float(w_capacity) * l_cap + float(w_monotone) * l_mon + float(w_smooth) * l_smooth + float(w_prior) * l_prior
    with torch.no_grad():
        q_mae_mAh = torch.mean(torch.abs(q_pred[mask] - q_obs[mask])) * 1000.0
        soh_mae = torch.mean(torch.abs(soh_pred[mask] - soh_obs[mask]))
        monotone_violation = torch.mean(torch.relu(q_pred[1:] - q_pred[:-1])) * 1000.0 if q_pred.numel() >= 2 else torch.zeros((), dtype=q_pred.dtype, device=q_pred.device)
    metrics = {
        "cap_loss_total": total.detach(),
        "cap_loss": l_cap.detach(),
        "cap_monotone_loss": l_mon.detach(),
        "cap_smooth_loss": l_smooth.detach(),
        "cap_prior_loss": l_prior.detach(),
        "cap_mae_mAh": q_mae_mAh.detach(),
        "soh_mae": soh_mae.detach(),
        "monotone_violation_mAh": monotone_violation.detach(),
        **{k: v for k, v in pred.items()},
    }
    return total, metrics


def save_capacity_head(head: AgingCapacityHead, model_dir: str | Path, *, filename: str = "capacity_head.pt") -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), model_path / filename)
    (model_path / "capacity_head_config.json").write_text(json.dumps(asdict(head.config), indent=2), encoding="utf-8")


def load_capacity_head(model_dir: str | Path, *, filename: str = "capacity_head.pt", map_location=None) -> AgingCapacityHead:
    model_path = Path(model_dir)
    cfg_path = model_path / "capacity_head_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing capacity head config: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    head = AgingCapacityHead(
        n_features=int(cfg["n_features"]),
        hidden=int(cfg.get("hidden", 32)),
        soh_min=float(cfg.get("soh_min", 0.45)),
        soh_max=float(cfg.get("soh_max", 1.02)),
        initial_rate_bias=float(cfg.get("initial_rate_bias", -6.0)),
    )
    state_path = model_path / filename
    if not state_path.exists():
        raise FileNotFoundError(f"Missing capacity head weights: {state_path}")
    state = torch.load(state_path, map_location=map_location or "cpu")
    head.load_state_dict(state)
    return head
