# -*- coding: utf-8 -*-
"""High-throughput MLP for D14-P5B closed-set precision benchmark."""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    return nn.GELU()


class ClosedSetPrecisionMLP(nn.Module):
    """Larger MLP for closed-set calibration.

    This model is deliberately larger than the P5 smoke network and is paired
    with large batches/AMP/GPU-resident tensors to better utilize GPU compute.
    """

    def __init__(self, input_dim: int, n_r: int = 17, hidden_dim: int = 768, num_layers: int = 6, dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_r = int(n_r)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        layers = []
        in_dim = self.input_dim
        for _ in range(max(1, self.num_layers)):
            layers.append(nn.Linear(in_dim, self.hidden_dim))
            layers.append(make_activation(activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.theta_a_head = nn.Linear(self.hidden_dim, self.n_r)
        self.theta_c_head = nn.Linear(self.hidden_dim, self.n_r)
        self.scalar_head = nn.Linear(self.hidden_dim, 2)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(X)
        scalars = self.scalar_head(h)
        return {
            "theta_a": torch.sigmoid(self.theta_a_head(h)),
            "theta_c": torch.sigmoid(self.theta_c_head(h)),
            "phie_norm": scalars[:, 0:1],
            "phis_c_norm": scalars[:, 1:2],
        }

    def config_dict(self):
        return {
            "model_class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "n_r": self.n_r,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def build_model(input_dim: int, n_r: int, cfg: Dict[str, Any]) -> ClosedSetPrecisionMLP:
    mc = cfg.get("model", {})
    return ClosedSetPrecisionMLP(
        input_dim=input_dim,
        n_r=n_r,
        hidden_dim=int(mc.get("hidden_dim", 768)),
        num_layers=int(mc.get("num_layers", 6)),
        dropout=float(mc.get("dropout", 0.0)),
        activation=str(mc.get("activation", "gelu")),
    )
