from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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


class ProfileConditionedMultiHeadSurrogate(nn.Module):
    """D17-G1.2 generator-surrogate repair model.

    The model remains a neural surrogate, but it separates generator-output
    groups into dedicated heads.  The phie head gets a direct current/voltage
    feature path and a train-profile gauge embedding because G0/G1.1 show that
    D15 RG branch preserves source phi labels rather than recomputing a unique
    electrochemical phie gauge.

    Notes
    -----
    * profile_id=0 is reserved for unknown/validation profiles.
    * train closed-set profiles use profile_id=1..N.
    * The model predicts normalized targets.  Denormalization is handled by the
      trainer with the dataset's y_mean/y_std.
    """

    def __init__(
        self,
        input_dim: int,
        target_slices: Mapping[str, Tuple[int, int]],
        profile_count: int,
        width: int = 512,
        depth: int = 6,
        dropout: float = 0.0,
        profile_embedding_dim: int = 16,
        phie_direct_width: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.profile_count = int(profile_count)
        self.profile_embedding_dim = int(profile_embedding_dim)
        self.target_slices = OrderedDict((str(k), (int(v[0]), int(v[1]))) for k, v in target_slices.items())
        self.target_order = list(self.target_slices.keys())
        self.output_dim = max(b for _, b in self.target_slices.values())

        self.profile_embedding = nn.Embedding(self.profile_count + 1, self.profile_embedding_dim)
        shared_in = self.input_dim + self.profile_embedding_dim
        layers: List[nn.Module] = [nn.Linear(shared_in, int(width)), nn.SiLU()]
        for _ in range(max(0, int(depth))):
            layers.append(ResidualBlock(int(width), dropout=float(dropout)))
        layers.append(nn.LayerNorm(int(width)))
        self.shared = nn.Sequential(*layers)

        # Most state groups benefit from shared nonlinear representation. phie
        # is intentionally special-cased below.
        self.heads = nn.ModuleDict()
        for key, (a, b) in self.target_slices.items():
            out_dim = int(b - a)
            if key == "phie":
                continue
            self.heads[key] = nn.Sequential(
                nn.Linear(int(width), int(width // 2)),
                nn.SiLU(),
                nn.LayerNorm(int(width // 2)),
                nn.Linear(int(width // 2), out_dim),
            )

        # phie source labels in the RG branch are preserved from source P2Dlite
        # labels; in P4D they can be ohmic-current-like.  Give phie a direct
        # observed-feature path and profile-gauge bias, rather than forcing it
        # through the same inventory/radial representation as cs/theta.
        phie_in = int(width) + self.input_dim + self.profile_embedding_dim
        self.phie_head = nn.Sequential(
            nn.Linear(phie_in, int(phie_direct_width)),
            nn.SiLU(),
            nn.LayerNorm(int(phie_direct_width)),
            nn.Linear(int(phie_direct_width), max(1, self.target_slices.get("phie", (0, 1))[1] - self.target_slices.get("phie", (0, 1))[0])),
        )
        self.phie_profile_bias = nn.Embedding(self.profile_count + 1, max(1, self.target_slices.get("phie", (0, 1))[1] - self.target_slices.get("phie", (0, 1))[0]))
        nn.init.zeros_(self.phie_profile_bias.weight)

    def forward(self, x: torch.Tensor, profile_id: torch.Tensor | None = None) -> torch.Tensor:
        if profile_id is None:
            profile_id = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        profile_id = profile_id.long().clamp(min=0, max=self.profile_count)
        pe = self.profile_embedding(profile_id)
        z = self.shared(torch.cat([x, pe], dim=-1))
        chunks: List[torch.Tensor] = []
        for key in self.target_order:
            if key == "phie":
                phie = self.phie_head(torch.cat([z, x, pe], dim=-1)) + self.phie_profile_bias(profile_id)
                chunks.append(phie)
            else:
                chunks.append(self.heads[key](z))
        return torch.cat(chunks, dim=-1)
