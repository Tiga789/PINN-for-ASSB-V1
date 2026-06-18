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


class PhieConventionHead(nn.Module):
    """Dedicated phie/gauge head for D17-G1.4.

    G0 showed that D15-RG soft labels preserve source phi labels, while P4D can
    use an ohmic-current-like phie convention.  Therefore phie should be learned
    as a generator-output convention from observed I/V/T/profile features, not
    only as a shared electrochemical latent.  The head combines:

    - a low-dimensional affine/basis path from local + profile observed features;
    - a nonlinear residual path from fused latent + observed features;
    - a small gate that lets the model choose how much residual correction to use.

    All outputs are in normalized target space.
    """

    def __init__(self, fused_dim: int, local_dim: int, profile_dim: int, out_dim: int = 1, width: int = 320, dropout: float = 0.04):
        super().__init__()
        self.local_dim = int(local_dim)
        self.profile_dim = int(profile_dim)
        self.out_dim = int(out_dim)
        obs_dim = self.local_dim + max(0, self.profile_dim)
        self.obs_basis = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, max(64, int(width) // 2)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(64, int(width) // 2), out_dim),
        )
        self.residual = nn.Sequential(
            nn.Linear(int(fused_dim) + obs_dim, int(width)),
            nn.SiLU(),
            ResidualBlock(int(width), dropout=float(dropout)),
            nn.LayerNorm(int(width)),
            nn.Linear(int(width), out_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(obs_dim, max(32, int(width) // 4)),
            nn.SiLU(),
            nn.Linear(max(32, int(width) // 4), out_dim),
            nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor, local: torch.Tensor, profile: torch.Tensor | None) -> torch.Tensor:
        obs = local if profile is None else torch.cat([local, profile], dim=-1)
        basis = self.obs_basis(obs)
        residual = self.residual(torch.cat([fused, obs], dim=-1))
        gate = self.gate(obs)
        return basis + gate * residual


class ValidationRobustObservedProfileSurrogate(nn.Module):
    """D17-G1.4 validation-aware generator surrogate.

    This is a conservative extension of G1.3:
      * no train-profile ID embedding;
      * same observed profile encoder logic;
      * split heads for theta/cs/phis_c;
      * a stronger phie convention/gauge head with observed-feature basis.
    """

    def __init__(
        self,
        local_input_dim: int,
        profile_input_dim: int,
        target_slices: Mapping[str, Tuple[int, int]],
        width: int = 896,
        depth: int = 8,
        profile_width: int = 256,
        dropout: float = 0.05,
        phie_direct_width: int = 320,
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
            ResidualBlock(w, dropout=float(dropout)),
            nn.LayerNorm(w),
        )
        if self.profile_input_dim > 0:
            self.profile_encoder = nn.Sequential(
                nn.Linear(self.profile_input_dim, pw),
                nn.SiLU(),
                ResidualBlock(pw, dropout=float(dropout)),
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
            half = max(96, w // 2)
            self.heads[key] = nn.Sequential(
                nn.Linear(w, half),
                nn.SiLU(),
                ResidualBlock(half, dropout=float(dropout)),
                nn.LayerNorm(half),
                nn.Linear(half, out_dim),
            )

        phie_a, phie_b = self.target_slices.get("phie", (0, 1))
        phie_dim = max(1, int(phie_b - phie_a))
        self.phie_head = PhieConventionHead(
            fused_dim=w,
            local_dim=self.local_input_dim,
            profile_dim=self.profile_input_dim,
            out_dim=phie_dim,
            width=int(phie_direct_width),
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = x[:, : self.local_input_dim]
        lz = self.local_encoder(local)
        if self.profile_input_dim > 0:
            profile = x[:, self.local_input_dim : self.local_input_dim + self.profile_input_dim]
            pz = self.profile_encoder(profile) if self.profile_encoder is not None else None
            z = self.fusion(torch.cat([lz, pz], dim=-1))
        else:
            profile = None
            z = self.fusion(lz)

        chunks: List[torch.Tensor] = []
        for key in self.target_order:
            if key == "phie":
                chunks.append(self.phie_head(z, local, profile))
            else:
                chunks.append(self.heads[key](z))
        return torch.cat(chunks, dim=-1)
