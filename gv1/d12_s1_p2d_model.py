"""D12-S1G high-safe train-inside protocol/P2D-like localized correction model.

This module intentionally does **not** overwrite the D9.6/D9.5.1 GV1
mainline.  It extends the compact conditioned effective-SPM PINN with one
extra potential raw channel, ``raw_p2d_deficit``, used only by the D12-S1G
output transform.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass
class D12S1ModelConfig:
    """Architecture options for :class:`D12S1P2DLocalPINN`.

    The default condition vector keeps the D8/D9 8-dimensional profile
    statistics.  The potential branch returns five raw channels:

    - raw_phie
    - raw_phis_c
    - raw_voltage_low
    - raw_voltage_event
    - raw_p2d_deficit
    """

    condition_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 3
    activation: str = "tanh"
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "D12S1ModelConfig":
        if not data:
            return cls()
        valid = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def _activation(name: str) -> nn.Module:
    key = str(name).lower().strip()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name!r}")


class MLP(nn.Module):
    """Small deterministic MLP matching the D9.x GV1 style."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 64,
        num_layers: int = 3,
        activation: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(num_layers)):
            linear = nn.Linear(in_dim, int(hidden_dim))
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(_activation(activation))
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        out = nn.Linear(in_dim, int(output_dim))
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class D12S1P2DLocalPINN(nn.Module):
    """Conditioned effective-SPM PINN with a train-inside P2D-like channel."""

    def __init__(self, config: D12S1ModelConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, D12S1ModelConfig) else D12S1ModelConfig.from_mapping(config)
        cdim = int(self.config.condition_dim)
        radial_in = 4 + cdim  # t, r, I, T, condition...
        potential_in = 3 + cdim  # t, I, T, condition...
        kwargs = dict(
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            activation=self.config.activation,
            dropout=self.config.dropout,
        )
        self.anode_branch = MLP(radial_in, 1, **kwargs)
        self.cathode_branch = MLP(radial_in, 1, **kwargs)
        self.potential_branch = MLP(potential_in, 5, **kwargs)

    @staticmethod
    def _col(x: torch.Tensor | float, *, like: torch.Tensor | None = None) -> torch.Tensor:
        if not torch.is_tensor(x):
            if like is None:
                raise TypeError("Scalar inputs require a reference tensor via like=")
            x = torch.full_like(like, float(x))
        if x.ndim == 0:
            if like is None:
                x = x.reshape(1, 1)
            else:
                x = torch.full_like(like, float(x.item()))
        elif x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def _condition_matrix(self, condition: torch.Tensor, n: int, ref: torch.Tensor) -> torch.Tensor:
        if condition.ndim == 1:
            condition = condition.reshape(1, -1).expand(n, -1)
        elif condition.ndim == 2 and condition.shape[0] == 1:
            condition = condition.expand(n, -1)
        elif condition.ndim != 2 or condition.shape[0] != n:
            raise ValueError(
                f"condition must have shape ({n}, C), (1, C), or (C,), got {tuple(condition.shape)}"
            )
        condition = condition.to(device=ref.device, dtype=ref.dtype)
        if condition.shape[1] != int(self.config.condition_dim):
            raise ValueError(
                f"condition_dim mismatch: model expects {self.config.condition_dim}, got {condition.shape[1]}"
            )
        return condition

    def forward(
        self,
        t_norm: torch.Tensor,
        r_norm: torch.Tensor,
        current_norm: torch.Tensor,
        temperature_norm: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t = self._col(t_norm)
        r = self._col(r_norm, like=t)
        cur = self._col(current_norm, like=t).to(device=t.device, dtype=t.dtype)
        temp = self._col(temperature_norm, like=t).to(device=t.device, dtype=t.dtype)
        n = t.shape[0]
        cond = self._condition_matrix(condition, n, t)
        radial_x = torch.cat([t, r, cur, temp, cond], dim=1)
        pot_x = torch.cat([t, cur, temp, cond], dim=1)
        pot = self.potential_branch(pot_x)
        return {
            "raw_a": self.anode_branch(radial_x),
            "raw_c": self.cathode_branch(radial_x),
            "raw_phie": pot[:, 0:1],
            "raw_phis_c": pot[:, 1:2],
            "raw_voltage_low": pot[:, 2:3],
            "raw_voltage_event": pot[:, 3:4],
            "raw_p2d_deficit": pot[:, 4:5],
        }


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
