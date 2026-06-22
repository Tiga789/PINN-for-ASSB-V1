from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class S2ModelConfig:
    cycle_feature_dim: int = 20
    local_feature_dim: int = 14
    cycle_hidden_dim: int = 64
    local_hidden_dim: int = 64
    fused_hidden_dim: int = 96
    branch_embed_dim: int = 8
    branch_count: int = 2
    radial_basis_count: int = 6
    n_radial_a: int = 17
    n_radial_c: int = 17
    dropout: float = 0.0
    max_radial_fraction_a: float = 0.18
    max_radial_fraction_c: float = 0.18
    max_inventory_residual_fraction_a: float = 0.20
    max_inventory_residual_fraction_c: float = 0.20
    potential_residual_scale: float = 0.35
    gauge_scale: float = 0.25
    theta_min: float = 0.0
    theta_max: float = 1.0
    physical_eps: float = 1.0e-6
    min_inventory_margin: float = 1.0e-8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "S2ModelConfig":
        if not value:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: value[k] for k in value if k in allowed})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZeroMeanRadialBasis(nn.Module):
    """Zero-volume-mean, pointwise-bounded radial basis on a spherical particle."""

    def __init__(self, n_radial: int, basis_count: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        if n_radial < 2 or basis_count < 1:
            raise ValueError("Invalid radial basis dimensions")
        self.eps = float(eps)
        rho = torch.linspace(0.0, 1.0, n_radial, dtype=torch.float64)
        edges = torch.empty(n_radial + 1, dtype=torch.float64)
        edges[0], edges[-1] = 0.0, 1.0
        edges[1:-1] = 0.5 * (rho[:-1] + rho[1:])
        weights = torch.diff(edges**3)
        weights = weights / weights.sum()
        basis: list[torch.Tensor] = []
        for order in range(1, basis_count + 12):
            raw = rho**order
            raw = raw - torch.sum(raw * weights)
            for prior in basis:
                denom = torch.sum(prior * prior * weights)
                raw = raw - torch.sum(raw * prior * weights) / torch.clamp(denom, min=1e-18) * prior
            norm = torch.sqrt(torch.sum(raw * raw * weights))
            if float(norm) <= 1e-10:
                continue
            raw = raw / norm
            raw = raw - torch.sum(raw * weights)
            raw = raw / torch.clamp(torch.max(torch.abs(raw)), min=1e-12)
            basis.append(raw)
            if len(basis) == basis_count:
                break
        if len(basis) != basis_count:
            raise RuntimeError("Could not construct radial basis")
        self.register_buffer("rho", rho.to(torch.float32))
        self.register_buffer("weights", weights.to(torch.float32))
        self.register_buffer("basis", torch.stack(basis).to(torch.float32))

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        raw = torch.einsum("...k,kr->...r", coefficients, self.basis)
        raw = raw - torch.sum(raw * self.weights, dim=-1, keepdim=True)
        peak = torch.amax(torch.abs(raw), dim=-1, keepdim=True)
        shape = torch.where(peak > self.eps, raw / torch.clamp(peak, min=self.eps), torch.zeros_like(raw))
        shape = shape - torch.sum(shape * self.weights, dim=-1, keepdim=True)
        peak2 = torch.amax(torch.abs(shape), dim=-1, keepdim=True)
        return torch.where(peak2 > self.eps, shape / torch.clamp(peak2, min=self.eps), torch.zeros_like(shape))

    def weighted_mean(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * self.weights, dim=-1)


class BranchAdapterBank(nn.Module):
    def __init__(self, hidden_dim: int, branch_count: int, dropout: float) -> None:
        super().__init__()
        self.branch_count = int(branch_count)
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(branch_count)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, branch_id: torch.Tensor) -> torch.Tensor:
        if branch_id.ndim != 1 or branch_id.shape[0] != x.shape[0]:
            raise ValueError("branch_id must be [B]")
        outputs = torch.stack([adapter(x) for adapter in self.adapters], dim=1)
        gather = branch_id[:, None, None, None].expand(-1, 1, x.shape[1], x.shape[2])
        selected = torch.gather(outputs, 1, gather).squeeze(1)
        return self.norm(x + selected)


class SegmentedLocalGRU(nn.Module):
    """Run a shared GRU independently inside each selected cycle, resetting hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or cycle_index.ndim != 2 or x.shape[:2] != cycle_index.shape:
            raise ValueError("local features/cycle_index shape mismatch")
        batches: list[torch.Tensor] = []
        for b in range(x.shape[0]):
            parts: list[torch.Tensor] = []
            ids = cycle_index[b]
            # Input casepack is contiguous by cycle. Enforce this instead of silently reordering.
            unique = torch.unique_consecutive(ids)
            reconstructed: list[torch.Tensor] = []
            for cid in unique:
                pos = torch.nonzero(ids == cid, as_tuple=False).reshape(-1)
                if pos.numel() == 0:
                    continue
                if pos[-1] - pos[0] + 1 != pos.numel():
                    raise ValueError("cycle_index is not contiguous")
                segment = x[b : b + 1, pos[0] : pos[-1] + 1]
                encoded, _ = self.gru(segment)
                parts.append(encoded.squeeze(0))
                reconstructed.append(pos)
            if not parts:
                raise ValueError("Profile has no cycle segments")
            output = torch.cat(parts, dim=0)
            if output.shape[0] != x.shape[1]:
                raise ValueError("Segmented local encoder did not preserve time length")
            batches.append(output)
        return torch.stack(batches, dim=0)


class CycleAwareS2Operator(nn.Module):
    """
    D18-S2 cycle-aware micro-smoke operator.

    It adds a bounded cycle-level inventory correction to the S0 scaffold because S1
    showed cycle/history-dependent inventory bias. The correction is explicit and
    regularized; it does not replace the current-integral baseline.
    """

    def __init__(self, config: S2ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cycle_encoder = nn.GRU(config.cycle_feature_dim, config.cycle_hidden_dim, batch_first=True)
        self.local_encoder = SegmentedLocalGRU(config.local_feature_dim, config.local_hidden_dim)
        self.branch_embedding = nn.Embedding(config.branch_count, config.branch_embed_dim)
        fusion_in = config.cycle_hidden_dim + config.local_hidden_dim + config.branch_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, config.fused_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fused_hidden_dim, config.fused_hidden_dim),
            nn.SiLU(),
        )
        self.branch_adapters = BranchAdapterBank(config.fused_hidden_dim, config.branch_count, config.dropout)
        self.inventory_cycle_head = nn.Sequential(
            nn.Linear(config.cycle_hidden_dim + config.branch_embed_dim, config.fused_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.fused_hidden_dim // 2, 2),
        )
        self.radial_coeff_head = nn.Linear(config.fused_hidden_dim, 2 * config.radial_basis_count)
        self.radial_amplitude_head = nn.Linear(config.fused_hidden_dim, 2)
        self.gauge_head = nn.Linear(config.fused_hidden_dim, 1)
        self.potential_diff_head = nn.Linear(config.fused_hidden_dim, 2)
        self.basis_a = ZeroMeanRadialBasis(config.n_radial_a, config.radial_basis_count)
        self.basis_c = ZeroMeanRadialBasis(config.n_radial_c, config.radial_basis_count)

    @staticmethod
    def _gather_cycle(cycle_values: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        b, c, h = cycle_values.shape
        idx = cycle_index.to(torch.long).clamp(0, c - 1)
        return torch.gather(cycle_values, 1, idx[..., None].expand(-1, -1, h))

    def _bounds(self, offset: torch.Tensor, scale: torch.Tensor, electrode: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        o = offset[:, electrode : electrode + 1]
        s = scale[:, electrode : electrode + 1]
        if torch.any(s <= 0):
            raise ValueError("theta_scale must be positive")
        lower = ((cfg.theta_min - o) / s)[:, None, :]
        upper = ((cfg.theta_max - o) / s)[:, None, :]
        return lower, upper

    def _safe_cbar(
        self,
        baseline: torch.Tensor,
        cycle_correction_unit: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        max_fraction: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        eps = cfg.physical_eps
        lo = lower + eps
        hi = upper - eps
        safe_base = torch.maximum(torch.minimum(baseline, hi), lo)
        span = torch.clamp(hi - lo, min=cfg.min_inventory_margin)
        correction = max_fraction * span * torch.tanh(cycle_correction_unit)
        corrected = torch.maximum(torch.minimum(safe_base + correction, hi), lo)
        return corrected, corrected - safe_base, safe_base - baseline

    def _radial_state(
        self,
        cbar: torch.Tensor,
        shape: torch.Tensor,
        amplitude_fraction: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        margin = torch.minimum(cbar - lower, upper - cbar)
        margin = torch.clamp(margin, min=cfg.min_inventory_margin)
        delta = amplitude_fraction * margin * shape
        cs = cbar + delta
        cs = torch.maximum(torch.minimum(cs, upper), lower)
        return margin, delta, cs

    def forward(
        self,
        *,
        cycle_features: torch.Tensor,
        local_features: torch.Tensor,
        cycle_index: torch.Tensor,
        cbar_baseline: torch.Tensor,
        potential_baseline: torch.Tensor,
        branch_id: torch.Tensor,
        theta_offset: torch.Tensor,
        theta_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if cycle_features.ndim != 3 or cycle_features.shape[-1] != cfg.cycle_feature_dim:
            raise ValueError("cycle_features shape invalid")
        if local_features.ndim != 3 or local_features.shape[-1] != cfg.local_feature_dim:
            raise ValueError("local_features shape invalid")
        if cycle_index.shape != local_features.shape[:2]:
            raise ValueError("cycle_index shape invalid")
        if cbar_baseline.shape != (*local_features.shape[:2], 2):
            raise ValueError("cbar_baseline shape invalid")
        if potential_baseline.shape != (*local_features.shape[:2], 2):
            raise ValueError("potential_baseline shape invalid")

        cycle_latent, _ = self.cycle_encoder(cycle_features)
        local_latent = self.local_encoder(local_features, cycle_index)
        branch_vec = self.branch_embedding(branch_id.to(torch.long))
        branch_time = branch_vec[:, None, :].expand(-1, local_features.shape[1], -1)
        cycle_time = self._gather_cycle(cycle_latent, cycle_index)
        fused = self.fusion(torch.cat([cycle_time, local_latent, branch_time], dim=-1))
        fused = self.branch_adapters(fused, branch_id.to(torch.long))

        branch_cycle = branch_vec[:, None, :].expand(-1, cycle_features.shape[1], -1)
        inventory_cycle_unit = self.inventory_cycle_head(torch.cat([cycle_latent, branch_cycle], dim=-1))
        inventory_time_unit = self._gather_cycle(inventory_cycle_unit, cycle_index)

        lower_a, upper_a = self._bounds(theta_offset, theta_scale, 0)
        lower_c, upper_c = self._bounds(theta_offset, theta_scale, 1)
        cbar_a, inventory_delta_a, cbar_clip_a = self._safe_cbar(
            cbar_baseline[..., 0:1], inventory_time_unit[..., 0:1], lower_a, upper_a,
            cfg.max_inventory_residual_fraction_a,
        )
        cbar_c, inventory_delta_c, cbar_clip_c = self._safe_cbar(
            cbar_baseline[..., 1:2], inventory_time_unit[..., 1:2], lower_c, upper_c,
            cfg.max_inventory_residual_fraction_c,
        )

        coeff = torch.tanh(self.radial_coeff_head(fused))
        coeff_a, coeff_c = torch.chunk(coeff, 2, dim=-1)
        shape_a = self.basis_a(coeff_a)
        shape_c = self.basis_c(coeff_c)
        amplitude = torch.sigmoid(self.radial_amplitude_head(fused))
        amp_a = cfg.max_radial_fraction_a * amplitude[..., 0:1]
        amp_c = cfg.max_radial_fraction_c * amplitude[..., 1:2]
        margin_a, delta_a, cs_a = self._radial_state(cbar_a, shape_a, amp_a, lower_a, upper_a)
        margin_c, delta_c, cs_c = self._radial_state(cbar_c, shape_c, amp_c, lower_c, upper_c)

        theta_a = theta_offset[:, None, 0:1] + theta_scale[:, None, 0:1] * cs_a
        theta_c = theta_offset[:, None, 1:2] + theta_scale[:, None, 1:2] * cs_c
        gauge = cfg.gauge_scale * torch.tanh(self.gauge_head(fused))
        differential = cfg.potential_residual_scale * torch.tanh(self.potential_diff_head(fused))
        phie = potential_baseline[..., 0:1] + gauge + differential[..., 0:1]
        phis_c = potential_baseline[..., 1:2] + gauge + differential[..., 1:2]
        return {
            "cycle_latent": cycle_latent,
            "local_latent": local_latent,
            "fused_latent": fused,
            "inventory_cycle_unit": inventory_cycle_unit,
            "inventory_delta_a": inventory_delta_a,
            "inventory_delta_c": inventory_delta_c,
            "cbar_clip_a": cbar_clip_a,
            "cbar_clip_c": cbar_clip_c,
            "cbar_a": cbar_a,
            "cbar_c": cbar_c,
            "inventory_margin_a": margin_a,
            "inventory_margin_c": margin_c,
            "radial_shape_a": shape_a,
            "radial_shape_c": shape_c,
            "delta_cs_a": delta_a,
            "delta_cs_c": delta_c,
            "cs_a": cs_a,
            "cs_c": cs_c,
            "theta_a": theta_a,
            "theta_c": theta_c,
            "potential_gauge": gauge,
            "phie": phie,
            "phis_c": phis_c,
            "cs_lower_a": lower_a,
            "cs_upper_a": upper_a,
            "cs_lower_c": lower_c,
            "cs_upper_c": upper_c,
        }


def architecture_contract(config: S2ModelConfig) -> dict[str, Any]:
    return {
        "stage": "D18-S2-MICRO-SMOKE",
        "formal_training_enabled": False,
        "architecture": "cycle_gru_plus_segmented_within_cycle_gru_branch_adapters_low_rank_radial_dynamic_inventory_v1",
        "sequence_mechanisms": [
            "causal cycle-level GRU",
            "within-cycle GRU reset at each cycle boundary",
            "cycle-level bounded inventory correction",
        ],
        "branch_mechanisms": ["shared history encoder", "separate RG/P4D residual adapters"],
        "hard_constraints": [
            "theta is derived from cs",
            "radial residual has zero spherical-volume mean",
            "radial shape is pointwise bounded",
            "cs/theta stay inside physical bounds",
            "inventory correction is bounded and explicitly regularized",
        ],
        "inputs": {
            "cycle_features": ["B", "C", config.cycle_feature_dim],
            "local_features": ["B", "T", config.local_feature_dim],
            "cycle_index": ["B", "T"],
            "cbar_baseline": ["B", "T", 2],
            "potential_baseline": ["B", "T", 2],
            "branch_id": ["B"],
            "theta_offset": ["B", 2],
            "theta_scale": ["B", 2],
        },
        "outputs": {
            "cs_a": ["B", "T", config.n_radial_a],
            "cs_c": ["B", "T", config.n_radial_c],
            "theta_a": ["B", "T", config.n_radial_a],
            "theta_c": ["B", "T", config.n_radial_c],
            "phie": ["B", "T", 1],
            "phis_c": ["B", "T", 1],
        },
        "config": config.as_dict(),
    }


def synthetic_forward_check(config: S2ModelConfig, seed: int = 1802) -> dict[str, Any]:
    torch.manual_seed(seed)
    b, c, points = 4, 6, 32
    t = c * points
    model = CycleAwareS2Operator(config).eval()
    cycle_features = torch.randn(b, c, config.cycle_feature_dim)
    local_features = torch.randn(b, t, config.local_feature_dim)
    cycle_index = torch.arange(c).repeat_interleave(points)[None, :].repeat(b, 1)
    cbar = torch.rand(b, t, 2) * 18000.0 + torch.tensor([9000.0, 18000.0])
    potential = torch.randn(b, t, 2) * 0.03 + torch.tensor([0.0, 3.7])
    branch = torch.tensor([0, 1, 0, 1])
    theta_offset = torch.zeros(b, 2)
    theta_scale = torch.tensor([[1 / 32000.0, 1 / 51000.0]]).repeat(b, 1)
    with torch.no_grad():
        out = model(
            cycle_features=cycle_features,
            local_features=local_features,
            cycle_index=cycle_index,
            cbar_baseline=cbar,
            potential_baseline=potential,
            branch_id=branch,
            theta_offset=theta_offset,
            theta_scale=theta_scale,
        )
    zero_a = float(torch.max(torch.abs(model.basis_a.weighted_mean(out["delta_cs_a"]))))
    zero_c = float(torch.max(torch.abs(model.basis_c.weighted_mean(out["delta_cs_c"]))))
    relation_a = theta_offset[:, None, 0:1] + theta_scale[:, None, 0:1] * out["cs_a"]
    relation_c = theta_offset[:, None, 1:2] + theta_scale[:, None, 1:2] * out["cs_c"]
    checks = {
        "finite": all(bool(torch.isfinite(out[k]).all()) for k in ("cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c")),
        "zero_mean": zero_a < 1e-4 and zero_c < 1e-4,
        "theta_relation": float(torch.max(torch.abs(out["theta_a"] - relation_a))) < 1e-6
        and float(torch.max(torch.abs(out["theta_c"] - relation_c))) < 1e-6,
        "theta_bounds": bool(((out["theta_a"] >= -1e-6) & (out["theta_a"] <= 1 + 1e-6)).all())
        and bool(((out["theta_c"] >= -1e-6) & (out["theta_c"] <= 1 + 1e-6)).all()),
        "both_branches_present": set(branch.tolist()) == {0, 1},
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "zero_mean_max_a": zero_a,
        "zero_mean_max_c": zero_c,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
