# -*- coding: utf-8 -*-
"""Closed-set precision losses for D14-P5B."""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F


def closedset_loss(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    weights = cfg.get("training", {}).get("loss_weights", {})
    w_ta = float(weights.get("theta_a", 1.0))
    w_tc = float(weights.get("theta_c", 1.0))
    w_phie = float(weights.get("phie", 0.45))
    w_phis = float(weights.get("phis_c", 0.85))
    w_surface = float(weights.get("surface_theta", 0.20))
    w_shape = float(weights.get("radial_shape", 0.15))

    loss_ta = F.mse_loss(pred["theta_a"], batch["theta_a"])
    loss_tc = F.mse_loss(pred["theta_c"], batch["theta_c"])
    loss_phie = F.mse_loss(pred["phie_norm"], batch["phie"])
    loss_phis = F.mse_loss(pred["phis_c_norm"], batch["phis_c"])

    # Surface is especially important for voltage/OCP consistency.
    loss_surface = F.mse_loss(pred["theta_a"][:, -1], batch["theta_a"][:, -1]) + F.mse_loss(pred["theta_c"][:, -1], batch["theta_c"][:, -1])

    # Radial shape loss: compare surface-minus-mean and center-minus-mean.
    pa_mean = pred["theta_a"].mean(dim=1)
    ta_mean = batch["theta_a"].mean(dim=1)
    pc_mean = pred["theta_c"].mean(dim=1)
    tc_mean = batch["theta_c"].mean(dim=1)
    loss_shape = (
        F.mse_loss(pred["theta_a"][:, -1] - pa_mean, batch["theta_a"][:, -1] - ta_mean)
        + F.mse_loss(pred["theta_c"][:, -1] - pc_mean, batch["theta_c"][:, -1] - tc_mean)
        + F.mse_loss(pred["theta_a"][:, 0] - pa_mean, batch["theta_a"][:, 0] - ta_mean)
        + F.mse_loss(pred["theta_c"][:, 0] - pc_mean, batch["theta_c"][:, 0] - tc_mean)
    )

    loss = w_ta * loss_ta + w_tc * loss_tc + w_phie * loss_phie + w_phis * loss_phis + w_surface * loss_surface + w_shape * loss_shape
    return {
        "loss": loss,
        "loss_theta_a": loss_ta.detach(),
        "loss_theta_c": loss_tc.detach(),
        "loss_phie": loss_phie.detach(),
        "loss_phis_c": loss_phis.detach(),
        "loss_surface": loss_surface.detach(),
        "loss_shape": loss_shape.detach(),
    }
