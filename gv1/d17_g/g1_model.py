from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GeneratorSurrogateMLP(nn.Module):
    """Train-cell supervised generator surrogate.

    The network maps observed replay features (time/current/voltage/temp/protocol/branch)
    to D15 P2Dlite-RG soft-label state arrays. It intentionally does not solve a
    new inverse problem; it distills the D15 generator output convention on train cells.
    """

    def __init__(self, input_dim: int, output_dim: int, width: int = 256, depth: int = 4, dropout: float = 0.02):
        super().__init__()
        width = int(width)
        depth = int(depth)
        layers: List[nn.Module] = [nn.Linear(int(input_dim), width), nn.SiLU()]
        for _ in range(max(0, depth)):
            layers.append(ResidualBlock(width, dropout=dropout))
        layers.extend([nn.LayerNorm(width), nn.Linear(width, int(output_dim))])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_group_weights(target_slices: Dict[str, Tuple[int, int]], weights: Dict[str, float], output_dim: int, device: torch.device) -> torch.Tensor:
    w = torch.ones(int(output_dim), dtype=torch.float32, device=device)
    for key, val in weights.items():
        if key in target_slices:
            a, b = target_slices[key]
            w[a:b] = float(val)
    return w
