from __future__ import annotations

from typing import Dict, Any

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, activation: str = 'silu', dropout: float = 0.0):
        super().__init__()
        act = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            act,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
        )
        self.out_act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(x + self.net(x))


def _activation(name: str) -> nn.Module:
    n = str(name).lower()
    if n in {'silu', 'swish'}:
        return nn.SiLU()
    if n in {'gelu'}:
        return nn.GELU()
    if n in {'relu'}:
        return nn.ReLU()
    if n in {'tanh'}:
        return nn.Tanh()
    raise ValueError(f'Unsupported activation: {name}')


class ClosedSetRGMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_hidden_layers: int = 4, activation: str = 'silu', dropout: float = 0.0, residual_blocks: bool = True):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), _activation(activation)]
        if residual_blocks:
            for _ in range(max(1, num_hidden_layers)):
                layers.append(ResidualMLPBlock(hidden_dim, activation=activation, dropout=dropout))
        else:
            for _ in range(max(1, num_hidden_layers)):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), _activation(activation)])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, output_dim: int, cfg: Dict[str, Any]) -> ClosedSetRGMLP:
    return ClosedSetRGMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(cfg.get('hidden_dim', 256)),
        num_hidden_layers=int(cfg.get('num_hidden_layers', 4)),
        activation=str(cfg.get('activation', 'silu')),
        dropout=float(cfg.get('dropout', 0.0)),
        residual_blocks=bool(cfg.get('residual_blocks', True)),
    )
