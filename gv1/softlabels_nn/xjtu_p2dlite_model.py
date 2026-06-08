# -*- coding: utf-8 -*-
"""Neural network for D14-P5 XJTU P2Dlite soft-label smoke."""

from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    return nn.GELU()


class P2DliteSoftlabelMLP(nn.Module):
    """Small supervised network for P2Dlite soft-label smoke.

    Inputs are sampled measured signals and metadata features. Outputs are:
      theta_a: (N, n_r), sigmoid bounded
      theta_c: (N, n_r), sigmoid bounded
      phie_norm: (N, 1), normalized scalar
      phis_c_norm: (N, 1), normalized scalar
    """

    def __init__(
        self,
        input_dim: int,
        n_r: int = 17,
        hidden_dim: int = 160,
        num_layers: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(X)
        theta_a = torch.sigmoid(self.theta_a_head(h))
        theta_c = torch.sigmoid(self.theta_c_head(h))
        scalars = self.scalar_head(h)
        return {
            "theta_a": theta_a,
            "theta_c": theta_c,
            "phie_norm": scalars[:, 0:1],
            "phis_c_norm": scalars[:, 1:2],
        }

    def config_dict(self) -> dict:
        return {
            "model_class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "n_r": self.n_r,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def build_model_from_config(input_dim: int, n_r: int, cfg: Dict[str, Any]) -> P2DliteSoftlabelMLP:
    mc = cfg.get("model", {})
    return P2DliteSoftlabelMLP(
        input_dim=input_dim,
        n_r=n_r,
        hidden_dim=int(mc.get("hidden_dim", 160)),
        num_layers=int(mc.get("num_layers", 4)),
        dropout=float(mc.get("dropout", 0.0)),
        activation=str(mc.get("activation", "gelu")),
    )
