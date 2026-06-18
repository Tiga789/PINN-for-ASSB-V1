from __future__ import annotations

from collections import OrderedDict
from typing import List, Mapping, Tuple

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


class ObservedProfileConditionedMultiHeadSurrogate(nn.Module):
    """Validation-aware D17-G1.3 generator surrogate.

    Unlike G1.2, this model does not use train-profile ID embeddings.  Its input
    is `[local per-time observed features, profile-level observed summary]`.
    The profile summary is derived from I/V/T/protocol/branch/replay features,
    not from state soft labels.  Target groups remain split and phie keeps a
    direct observed-feature path because G0/G1.1/G1.2 showed that phie is a
    generator convention/gauge target that should not share only cs/theta heads.
    """

    def __init__(
        self,
        local_input_dim: int,
        profile_input_dim: int,
        target_slices: Mapping[str, Tuple[int, int]],
        width: int = 768,
        depth: int = 7,
        profile_width: int = 192,
        dropout: float = 0.03,
        phie_direct_width: int = 192,
    ):
        super().__init__()
        self.local_input_dim = int(local_input_dim)
        self.profile_input_dim = int(profile_input_dim)
        self.target_slices = OrderedDict((str(k), (int(v[0]), int(v[1]))) for k, v in target_slices.items())
        self.target_order = list(self.target_slices.keys())
        self.output_dim = max(b for _, b in self.target_slices.values())
        w = int(width)
        pw = int(profile_width)

        self.local_encoder = nn.Sequential(
            nn.Linear(self.local_input_dim, w),
            nn.SiLU(),
            ResidualBlock(w, dropout=float(dropout)),
            nn.LayerNorm(w),
        )
        if self.profile_input_dim > 0:
            self.profile_encoder = nn.Sequential(
                nn.Linear(self.profile_input_dim, pw),
                nn.SiLU(),
                ResidualBlock(pw, dropout=float(dropout)),
                nn.LayerNorm(pw),
            )
            fused_in = w + pw
        else:
            self.profile_encoder = None
            fused_in = w

        fusion_layers: List[nn.Module] = [nn.Linear(fused_in, w), nn.SiLU()]
        for _ in range(max(0, int(depth))):
            fusion_layers.append(ResidualBlock(w, dropout=float(dropout)))
        fusion_layers.append(nn.LayerNorm(w))
        self.fusion = nn.Sequential(*fusion_layers)

        self.heads = nn.ModuleDict()
        for key, (a, b) in self.target_slices.items():
            if key == "phie":
                continue
            out_dim = int(b - a)
            half = max(64, w // 2)
            self.heads[key] = nn.Sequential(
                nn.Linear(w, half),
                nn.SiLU(),
                nn.LayerNorm(half),
                nn.Linear(half, out_dim),
            )

        phie_dim = max(1, int(self.target_slices.get("phie", (0, 1))[1] - self.target_slices.get("phie", (0, 1))[0]))
        self.phie_head = nn.Sequential(
            nn.Linear(w + self.local_input_dim + max(0, self.profile_input_dim), int(phie_direct_width)),
            nn.SiLU(),
            nn.LayerNorm(int(phie_direct_width)),
            nn.Linear(int(phie_direct_width), phie_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = x[:, : self.local_input_dim]
        lz = self.local_encoder(local)
        if self.profile_input_dim > 0:
            profile = x[:, self.local_input_dim : self.local_input_dim + self.profile_input_dim]
            pz = self.profile_encoder(profile) if self.profile_encoder is not None else None
            z = self.fusion(torch.cat([lz, pz], dim=-1))
            phie_direct = torch.cat([z, local, profile], dim=-1)
        else:
            z = self.fusion(lz)
            phie_direct = torch.cat([z, local], dim=-1)

        chunks: List[torch.Tensor] = []
        for key in self.target_order:
            if key == "phie":
                chunks.append(self.phie_head(phie_direct))
            else:
                chunks.append(self.heads[key](z))
        return torch.cat(chunks, dim=-1)
