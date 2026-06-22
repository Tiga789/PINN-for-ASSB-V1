from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class D18ModelConfig:
    """Architecture-only configuration for the D18 cycle-aware operator scaffold."""

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
    potential_residual_scale: float = 0.25
    gauge_scale: float = 0.25
    theta_min: float = 0.0
    theta_max: float = 1.0
    physical_eps: float = 1.0e-6
    min_inventory_margin: float = 1.0e-8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "D18ModelConfig":
        if not value:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        kwargs = {k: value[k] for k in value if k in allowed}
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ZeroMeanRadialBasis(nn.Module):
    """
    Weighted zero-volume-mean polynomial basis on a spherical particle.

    The old S0 scaffold normalized each basis by weighted L2 norm only. A bounded
    coefficient vector could therefore produce a pointwise shape much larger than
    one. The fixed implementation performs two protections:

    1. each stored basis is zero-volume-mean and unit peak;
    2. every *combined* radial shape is projected to zero mean again and normalized
       by its own pointwise maximum absolute value.

    Consequently, ``abs(forward(coefficients)) <= 1`` (up to floating error).
    """

    def __init__(self, n_radial: int, basis_count: int, eps: float = 1.0e-8) -> None:
        super().__init__()
        if n_radial < 2:
            raise ValueError("n_radial must be >= 2")
        if basis_count < 1:
            raise ValueError("basis_count must be >= 1")
        self.eps = float(eps)
        rho = torch.linspace(0.0, 1.0, n_radial, dtype=torch.float64)
        edges = torch.empty(n_radial + 1, dtype=torch.float64)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges[1:-1] = 0.5 * (rho[:-1] + rho[1:])
        weights = torch.diff(edges**3)
        weights = weights / weights.sum()

        candidates: list[torch.Tensor] = []
        for order in range(1, basis_count + 8):
            raw = rho**order
            raw = raw - torch.sum(raw * weights)
            for prior in candidates:
                denom = torch.sum(prior * prior * weights)
                raw = raw - (torch.sum(raw * prior * weights) / torch.clamp(denom, min=1e-18)) * prior
            norm = torch.sqrt(torch.sum(raw * raw * weights))
            if float(norm) > 1e-10:
                raw = raw / norm
                peak = torch.max(torch.abs(raw))
                raw = raw / torch.clamp(peak, min=1e-12)
                # Numerical zero-mean cleanup after peak normalization.
                raw = raw - torch.sum(raw * weights)
                raw = raw / torch.clamp(torch.max(torch.abs(raw)), min=1e-12)
                candidates.append(raw)
            if len(candidates) == basis_count:
                break
        if len(candidates) != basis_count:
            raise RuntimeError("Could not construct the requested radial basis")
        basis = torch.stack(candidates, dim=0).to(torch.float32)
        self.register_buffer("rho", rho.to(torch.float32))
        self.register_buffer("weights", weights.to(torch.float32))
        self.register_buffer("basis", basis)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[-1] != self.basis.shape[0]:
            raise ValueError(
                f"coefficient dimension {coefficients.shape[-1]} != basis count {self.basis.shape[0]}"
            )
        raw = torch.einsum("...k,kr->...r", coefficients, self.basis)
        raw = raw - torch.sum(raw * self.weights, dim=-1, keepdim=True)
        peak = torch.amax(torch.abs(raw), dim=-1, keepdim=True)
        shape = torch.where(peak > self.eps, raw / torch.clamp(peak, min=self.eps), torch.zeros_like(raw))
        # A final projection keeps the weighted mean at machine precision. The
        # second peak normalization preserves the pointwise bound after projection.
        shape = shape - torch.sum(shape * self.weights, dim=-1, keepdim=True)
        peak2 = torch.amax(torch.abs(shape), dim=-1, keepdim=True)
        return torch.where(peak2 > self.eps, shape / torch.clamp(peak2, min=self.eps), torch.zeros_like(shape))

    def weighted_mean(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sum(values * self.weights, dim=-1)

    @property
    def basis_peak_max_abs(self) -> float:
        return float(torch.max(torch.abs(self.basis)).detach().cpu())


class BranchAdapterBank(nn.Module):
    """Separate RG/P4D adapters without duplicating the shared history encoder."""

    def __init__(self, hidden_dim: int, branch_count: int, dropout: float) -> None:
        super().__init__()
        self.branch_count = branch_count
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
            raise ValueError("branch_id must have shape [batch]")
        if torch.any(branch_id < 0) or torch.any(branch_id >= self.branch_count):
            raise ValueError("branch_id contains an out-of-range value")
        all_outputs = torch.stack([adapter(x) for adapter in self.adapters], dim=1)
        gather_index = branch_id[:, None, None, None].expand(-1, 1, x.shape[1], x.shape[2])
        selected = torch.gather(all_outputs, dim=1, index=gather_index).squeeze(1)
        return self.norm(x + selected)


class CycleAwareOperator(nn.Module):
    """D18-S0 cycle-aware architecture scaffold; this module does not train a model."""

    def __init__(self, config: D18ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cycle_encoder = nn.GRU(
            input_size=config.cycle_feature_dim,
            hidden_size=config.cycle_hidden_dim,
            batch_first=True,
        )
        self.local_encoder = nn.GRU(
            input_size=config.local_feature_dim,
            hidden_size=config.local_hidden_dim,
            batch_first=True,
        )
        self.branch_embedding = nn.Embedding(config.branch_count, config.branch_embed_dim)
        fusion_in = config.cycle_hidden_dim + config.local_hidden_dim + config.branch_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, config.fused_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fused_hidden_dim, config.fused_hidden_dim),
            nn.SiLU(),
        )
        self.branch_adapters = BranchAdapterBank(
            config.fused_hidden_dim, config.branch_count, config.dropout
        )
        self.radial_coeff_head = nn.Linear(config.fused_hidden_dim, 2 * config.radial_basis_count)
        self.radial_amplitude_head = nn.Linear(config.fused_hidden_dim, 2)
        self.gauge_head = nn.Linear(config.fused_hidden_dim, 1)
        self.potential_diff_head = nn.Linear(config.fused_hidden_dim, 2)
        self.basis_a = ZeroMeanRadialBasis(config.n_radial_a, config.radial_basis_count)
        self.basis_c = ZeroMeanRadialBasis(config.n_radial_c, config.radial_basis_count)

    @staticmethod
    def _gather_cycle_latent(cycle_latent: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        if cycle_index.ndim != 2:
            raise ValueError("cycle_index must have shape [batch, time]")
        b, c, h = cycle_latent.shape
        if cycle_index.shape[0] != b:
            raise ValueError("cycle_index batch size mismatch")
        idx = cycle_index.to(torch.long).clamp(0, c - 1)
        return torch.gather(cycle_latent, 1, idx[..., None].expand(-1, -1, h))

    @staticmethod
    def _require_shape(name: str, value: torch.Tensor, ndim: int, trailing: int | None = None) -> None:
        if value.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D, got {tuple(value.shape)}")
        if trailing is not None and value.shape[-1] != trailing:
            raise ValueError(f"{name} last dimension must be {trailing}, got {value.shape[-1]}")

    def _concentration_bounds(
        self,
        theta_offset: torch.Tensor,
        theta_scale: torch.Tensor,
        electrode_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        offset = theta_offset[:, electrode_index : electrode_index + 1]
        scale = theta_scale[:, electrode_index : electrode_index + 1]
        if torch.any(torch.abs(scale) < cfg.physical_eps):
            raise ValueError("theta_scale contains a near-zero value; concentration bounds are undefined")
        cs_at_min = (cfg.theta_min - offset) / scale
        cs_at_max = (cfg.theta_max - offset) / scale
        lower = torch.minimum(cs_at_min, cs_at_max)[:, None, :]
        upper = torch.maximum(cs_at_min, cs_at_max)[:, None, :]
        return lower, upper

    def _bounded_radial_state(
        self,
        *,
        cbar_input: torch.Tensor,
        radial_shape: torch.Tensor,
        amplitude_fraction: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        safe_lower = lower + cfg.physical_eps
        safe_upper = upper - cfg.physical_eps
        if torch.any(safe_upper <= safe_lower):
            raise ValueError("theta mapping produces an empty admissible concentration interval")
        cbar_safe = torch.maximum(torch.minimum(cbar_input, safe_upper), safe_lower)
        lower_margin = cbar_safe - lower
        upper_margin = upper - cbar_safe
        margin = torch.clamp(torch.minimum(lower_margin, upper_margin), min=cfg.min_inventory_margin)
        delta = amplitude_fraction * margin * radial_shape
        cs = cbar_safe + delta
        return cbar_safe, margin, delta, cs

    def forward(
        self,
        *,
        cycle_features: torch.Tensor,
        local_features: torch.Tensor,
        cycle_index: torch.Tensor,
        cbar: torch.Tensor,
        potential_baseline: torch.Tensor,
        branch_id: torch.Tensor,
        theta_offset: torch.Tensor,
        theta_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        self._require_shape("cycle_features", cycle_features, 3, cfg.cycle_feature_dim)
        self._require_shape("local_features", local_features, 3, cfg.local_feature_dim)
        self._require_shape("cbar", cbar, 3, 2)
        self._require_shape("potential_baseline", potential_baseline, 3, 2)
        self._require_shape("theta_offset", theta_offset, 2, 2)
        self._require_shape("theta_scale", theta_scale, 2, 2)
        if local_features.shape[:2] != cbar.shape[:2] or cbar.shape != potential_baseline.shape:
            raise ValueError("local_features, cbar and potential_baseline time axes must match")

        cycle_sequence, _ = self.cycle_encoder(cycle_features)
        local_sequence, _ = self.local_encoder(local_features)
        cycle_at_time = self._gather_cycle_latent(cycle_sequence, cycle_index)
        branch = self.branch_embedding(branch_id.to(torch.long))[:, None, :]
        branch = branch.expand(-1, local_features.shape[1], -1)
        fused = self.fusion(torch.cat([cycle_at_time, local_sequence, branch], dim=-1))
        fused = self.branch_adapters(fused, branch_id.to(torch.long))

        coeff = torch.tanh(self.radial_coeff_head(fused))
        coeff_a, coeff_c = torch.chunk(coeff, 2, dim=-1)
        shape_a = self.basis_a(coeff_a)
        shape_c = self.basis_c(coeff_c)
        amp = torch.sigmoid(self.radial_amplitude_head(fused))
        amp_a = cfg.max_radial_fraction_a * amp[..., 0:1]
        amp_c = cfg.max_radial_fraction_c * amp[..., 1:2]

        lower_a, upper_a = self._concentration_bounds(theta_offset, theta_scale, 0)
        lower_c, upper_c = self._concentration_bounds(theta_offset, theta_scale, 1)
        cbar_a_safe, margin_a, delta_a, cs_a = self._bounded_radial_state(
            cbar_input=cbar[..., 0:1],
            radial_shape=shape_a,
            amplitude_fraction=amp_a,
            lower=lower_a,
            upper=upper_a,
        )
        cbar_c_safe, margin_c, delta_c, cs_c = self._bounded_radial_state(
            cbar_input=cbar[..., 1:2],
            radial_shape=shape_c,
            amplitude_fraction=amp_c,
            lower=lower_c,
            upper=upper_c,
        )

        theta_a = theta_offset[:, None, 0:1] + theta_scale[:, None, 0:1] * cs_a
        theta_c = theta_offset[:, None, 1:2] + theta_scale[:, None, 1:2] * cs_c

        gauge = cfg.gauge_scale * torch.tanh(self.gauge_head(fused))
        differential = cfg.potential_residual_scale * torch.tanh(self.potential_diff_head(fused))
        phie = potential_baseline[..., 0:1] + gauge + differential[..., 0:1]
        phis_c = potential_baseline[..., 1:2] + gauge + differential[..., 1:2]

        return {
            "cycle_latent": cycle_sequence,
            "local_latent": local_sequence,
            "fused_latent": fused,
            "cbar_a_input": cbar[..., 0:1],
            "cbar_c_input": cbar[..., 1:2],
            "cbar_a": cbar_a_safe,
            "cbar_c": cbar_c_safe,
            "cbar_clamp_abs_a": torch.abs(cbar_a_safe - cbar[..., 0:1]),
            "cbar_clamp_abs_c": torch.abs(cbar_c_safe - cbar[..., 1:2]),
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
            "cs_lower_a": lower_a,
            "cs_upper_a": upper_a,
            "cs_lower_c": lower_c,
            "cs_upper_c": upper_c,
            "potential_gauge": gauge,
            "phie": phie,
            "phis_c": phis_c,
        }


def architecture_contract(config: D18ModelConfig) -> dict[str, Any]:
    return {
        "stage": "D18-S0-FIX",
        "training_enabled": False,
        "architecture": "hierarchical_cycle_aware_operator_scaffold_pointwise_bounded_v2",
        "inputs": {
            "cycle_features": ["B", "C", config.cycle_feature_dim],
            "local_features": ["B", "T", config.local_feature_dim],
            "cycle_index": ["B", "T"],
            "cbar": ["B", "T", 2],
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
        "hard_constraints": [
            "combined radial shape is zero-volume-mean and pointwise bounded to [-1,1]",
            "delta_cs is bounded by a fraction of the admissible inventory margin",
            "cs remains inside concentration bounds implied by theta_min/theta_max",
            "theta is derived from cs and remains inside [theta_min, theta_max]",
            "RG and P4D use separate residual adapters after a shared history encoder",
            "cycle history is causal through a cycle-level recurrent encoder",
            "phie/phis_c share a bounded profile/time gauge component",
        ],
        "config": config.as_dict(),
    }


def _fraction_outside(x: torch.Tensor, lower: float, upper: float, tol: float = 1.0e-7) -> float:
    mask = (x < lower - tol) | (x > upper + tol)
    return float(mask.to(torch.float32).mean().detach().cpu())


def synthetic_architecture_check(
    config: D18ModelConfig,
    *,
    seed: int = 1801,
    batch_size: int = 3,
    cycle_count: int = 7,
    time_count: int = 257,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = CycleAwareOperator(config).eval()
    cycle_features = torch.randn(batch_size, cycle_count, config.cycle_feature_dim)
    local_features = torch.randn(batch_size, time_count, config.local_feature_dim)
    cycle_index = torch.arange(time_count)[None, :].repeat(batch_size, 1)
    cycle_index = torch.div(cycle_index * cycle_count, time_count, rounding_mode="floor").clamp_max(cycle_count - 1)
    cbar = torch.rand(batch_size, time_count, 2) * 0.50 + 0.25
    potential_baseline = torch.randn(batch_size, time_count, 2) * 0.05 + torch.tensor([0.1, 3.7])
    branch_id = torch.tensor([i % config.branch_count for i in range(batch_size)], dtype=torch.long)
    theta_offset = torch.zeros(batch_size, 2)
    theta_scale = torch.ones(batch_size, 2)
    with torch.no_grad():
        out = model(
            cycle_features=cycle_features,
            local_features=local_features,
            cycle_index=cycle_index,
            cbar=cbar,
            potential_baseline=potential_baseline,
            branch_id=branch_id,
            theta_offset=theta_offset,
            theta_scale=theta_scale,
        )

    mean_a = model.basis_a.weighted_mean(out["delta_cs_a"])
    mean_c = model.basis_c.weighted_mean(out["delta_cs_c"])
    expected_shapes = architecture_contract(config)["outputs"]
    actual_shapes = {k: list(out[k].shape) for k in expected_shapes}
    shape_ok = all(
        actual_shapes[k]
        == [
            batch_size,
            time_count,
            1 if k in {"phie", "phis_c"} else config.n_radial_a if k.endswith("_a") else config.n_radial_c,
        ]
        for k in expected_shapes
    )
    finite_ok = all(bool(torch.isfinite(out[k]).all()) for k in expected_shapes)
    zero_mean_max_a = float(torch.max(torch.abs(mean_a)))
    zero_mean_max_c = float(torch.max(torch.abs(mean_c)))
    relation_a = theta_offset[:, None, 0:1] + theta_scale[:, None, 0:1] * out["cs_a"]
    relation_c = theta_offset[:, None, 1:2] + theta_scale[:, None, 1:2] * out["cs_c"]
    theta_relation_a = float(torch.max(torch.abs(out["theta_a"] - relation_a)))
    theta_relation_c = float(torch.max(torch.abs(out["theta_c"] - relation_c)))
    radial_shape_peak_a = float(torch.max(torch.abs(out["radial_shape_a"])))
    radial_shape_peak_c = float(torch.max(torch.abs(out["radial_shape_c"])))
    theta_oob_a = _fraction_outside(out["theta_a"], config.theta_min, config.theta_max)
    theta_oob_c = _fraction_outside(out["theta_c"], config.theta_min, config.theta_max)
    cs_oob_a = float(
        ((out["cs_a"] < out["cs_lower_a"] - 1e-7) | (out["cs_a"] > out["cs_upper_a"] + 1e-7))
        .to(torch.float32)
        .mean()
        .detach()
        .cpu()
    )
    cs_oob_c = float(
        ((out["cs_c"] < out["cs_lower_c"] - 1e-7) | (out["cs_c"] > out["cs_upper_c"] + 1e-7))
        .to(torch.float32)
        .mean()
        .detach()
        .cpu()
    )
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    checks = {
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "zero_mean_ok": zero_mean_max_a < 1e-5 and zero_mean_max_c < 1e-5,
        "radial_shape_pointwise_bound_ok": radial_shape_peak_a <= 1.00001 and radial_shape_peak_c <= 1.00001,
        "theta_relation_ok": theta_relation_a < 1e-7 and theta_relation_c < 1e-7,
        "theta_bounds_ok": theta_oob_a == 0.0 and theta_oob_c == 0.0,
        "concentration_bounds_ok": cs_oob_a == 0.0 and cs_oob_c == 0.0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        **checks,
        "actual_shapes": actual_shapes,
        "basis_peak_max_abs_a": model.basis_a.basis_peak_max_abs,
        "basis_peak_max_abs_c": model.basis_c.basis_peak_max_abs,
        "radial_shape_peak_max_abs_a": radial_shape_peak_a,
        "radial_shape_peak_max_abs_c": radial_shape_peak_c,
        "zero_volume_mean_max_abs_a": zero_mean_max_a,
        "zero_volume_mean_max_abs_c": zero_mean_max_c,
        "theta_from_cs_max_abs_error_a": theta_relation_a,
        "theta_from_cs_max_abs_error_c": theta_relation_c,
        "theta_outside_fraction_a": theta_oob_a,
        "theta_outside_fraction_c": theta_oob_c,
        "concentration_outside_fraction_a": cs_oob_a,
        "concentration_outside_fraction_c": cs_oob_c,
        "cs_a_min": float(torch.min(out["cs_a"])),
        "cs_a_max": float(torch.max(out["cs_a"])),
        "cs_c_min": float(torch.min(out["cs_c"])),
        "cs_c_max": float(torch.max(out["cs_c"])),
        "theta_a_min": float(torch.min(out["theta_a"])),
        "theta_a_max": float(torch.max(out["theta_a"])),
        "theta_c_min": float(torch.min(out["theta_c"])),
        "theta_c_max": float(torch.max(out["theta_c"])),
        "cbar_clamped_fraction_a": float((out["cbar_clamp_abs_a"] > 0).to(torch.float32).mean()),
        "cbar_clamped_fraction_c": float((out["cbar_clamp_abs_c"] > 0).to(torch.float32).mean()),
        "parameter_count": parameter_count,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
