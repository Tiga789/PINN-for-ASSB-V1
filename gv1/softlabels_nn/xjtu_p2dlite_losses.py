# -*- coding: utf-8 -*-
"""Loss functions for D14-P5 XJTU P2Dlite soft-label NN smoke."""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F


def supervised_loss(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    weights = cfg.get("training", {}).get("loss_weights", {})
    w_ta = float(weights.get("theta_a", 1.0))
    w_tc = float(weights.get("theta_c", 1.0))
    w_phie = float(weights.get("phie", 0.35))
    w_phis = float(weights.get("phis_c", 0.60))

    loss_theta_a = F.mse_loss(pred["theta_a"], batch["theta_a"])
    loss_theta_c = F.mse_loss(pred["theta_c"], batch["theta_c"])
    loss_phie = F.mse_loss(pred["phie_norm"], batch["phie"])
    loss_phis = F.mse_loss(pred["phis_c_norm"], batch["phis_c"])

    total = w_ta * loss_theta_a + w_tc * loss_theta_c + w_phie * loss_phie + w_phis * loss_phis
    return {
        "loss": total,
        "loss_theta_a": loss_theta_a.detach(),
        "loss_theta_c": loss_theta_c.detach(),
        "loss_phie": loss_phie.detach(),
        "loss_phis_c": loss_phis.detach(),
    }
