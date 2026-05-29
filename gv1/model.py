"""GV1 D9.5.1 conditioned effective-SPM PINN model.

This module is independent from the old ASSB ``main.py`` / ``util/*`` stack.
D9.5.1 keeps the compact D9.3 conditioned PINN and its voltage regime channels.
The adaptive/hybrid behavior is implemented in :mod:`gv1.output_transform` and
:mod:`gv1.profile_adaptive`, not by changing the neural-network architecture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass
class ModelConfig:
    """Architecture options for :class:`ConditionedEffectiveSPMPINN`.

    The default condition vector still has eight entries, matching D8/D9
    profile-level statistics.  The potential branch now returns four raw
    channels: ``raw_phie``, ``raw_phis_c``, ``raw_voltage_low`` and
    ``raw_voltage_event``.  The extra two channels are consumed by the D9.3
    output transform as regime-aware corrections.
    """

    condition_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 3
    activation: str = "tanh"
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ModelConfig":
        if not data:
            return cls()
        valid = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def _activation(name: str) -> nn.Module:
    key = name.lower().strip()
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
    """Simple fully connected network with deterministic Xavier initialization."""

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
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(num_layers)):
            linear = nn.Linear(in_dim, int(hidden_dim))
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            in_dim = int(hidden_dim)
        out = nn.Linear(in_dim, int(output_dim))
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class ConditionedEffectiveSPMPINN(nn.Module):
    """Conditioned effective-SPM PINN core.

    Inputs are column tensors ``(N, 1)`` for normalized time, radius, current and
    temperature, plus a condition matrix ``(N, C)`` or vector ``(C,)``.  Outputs
    are raw channels.  All physical structure, voltage mixing and regime gates
    are applied in :mod:`gv1.output_transform`.
    """

    def __init__(self, config: ModelConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config if isinstance(config, ModelConfig) else ModelConfig.from_mapping(config)
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
        # D9.5.1/D9.3: phie, terminal head, low-tail correction, event correction.
        self.potential_branch = MLP(potential_in, 4, **kwargs)

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
        }

    @torch.no_grad()
    def predict_raw(
        self,
        t_norm: torch.Tensor,
        r_norm: torch.Tensor,
        current_norm: torch.Tensor,
        temperature_norm: torch.Tensor,
        condition: torch.Tensor,
        *,
        batch_size: int = 65536,
    ) -> dict[str, torch.Tensor]:
        """Batched raw inference helper for large profile prediction grids."""
        self.eval()
        n = int(t_norm.reshape(-1).shape[0])
        keys = ["raw_a", "raw_c", "raw_phie", "raw_phis_c", "raw_voltage_low", "raw_voltage_event"]
        chunks: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
        for start in range(0, n, int(batch_size)):
            sl = slice(start, min(start + int(batch_size), n))
            out = self(
                t_norm.reshape(-1, 1)[sl],
                r_norm.reshape(-1, 1)[sl],
                current_norm.reshape(-1, 1)[sl],
                temperature_norm.reshape(-1, 1)[sl],
                condition if condition.ndim == 1 else condition[sl],
            )
            for k, v in out.items():
                chunks[k].append(v.detach().cpu())
        return {k: torch.cat(v, dim=0) for k, v in chunks.items()}


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
