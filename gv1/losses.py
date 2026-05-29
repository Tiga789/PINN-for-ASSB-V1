"""Losses for the GV1 D9.5.1 trend-first warmup rare-regime PINN trainer.

D9.5.1 responds to the D9.5 failure: the rare-regime terms were useful
diagnostics, but they were applied too strongly from epoch 1 and weakened B3/B4
trend quality.  D9.5.1 therefore returns to a D9.3 trend-first objective and
adds only **scheduled** rare-regime terms.  The trainer injects two scalar
schedules into each batch:

* ``rare_sample_weight_scale`` controls how much tail/event oversampling affects
  the robust voltage data term;
* ``rare_loss_scale`` controls explicit tail/coverage/ultra-quantile/event
  losses.

Correlation, range and ordinary voltage losses remain active from the beginning;
rare-regime losses warm up gradually so they do not destroy the global V(t)
shape while still nudging low-voltage, high-voltage, current-event and
temperature-event regimes later in training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .model import ConditionedEffectiveSPMPINN
from .output_transform import GV1OutputTransform


@dataclass
class LossWeights:
    voltage_data: float = 1.0
    voltage_tail: float = 0.30
    voltage_bias: float = 0.10
    voltage_range: float = 0.08
    voltage_quantile: float = 0.22
    voltage_asymmetry: float = 0.10
    voltage_event: float = 0.14
    voltage_corr: float = 0.45
    voltage_ultra_quantile: float = 0.10
    voltage_low_coverage: float = 0.06
    voltage_tail_balance: float = 0.04
    voltage_guardrail: float = 0.01
    center_symmetry: float = 0.05
    surface_flux: float = 0.05
    radial_smooth: float = 0.002
    potential_smooth: float = 0.0005
    cbar_anchor: float = 0.0
    tail_fraction: float = 0.22
    tail_weight_gain: float = 1.7
    low_tail_extra_gain: float = 2.4
    high_tail_extra_gain: float = 0.8
    event_weight_gain: float = 0.9
    huber_delta_V: float = 0.08
    low_coverage_threshold_V: float = 2.75
    high_coverage_threshold_V: float = 4.10
    coverage_temperature_V: float = 0.035

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "LossWeights":
        if not data:
            return cls()
        valid = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def mse(x: torch.Tensor, y: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(y):
        y = torch.full_like(x, float(y))
    return torch.mean((x - y.to(device=x.device, dtype=x.dtype)) ** 2)


def safe_grad(y: torch.Tensor, x: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    """Compute dy/dx and return zeros when autograd has no dependency."""
    g = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros_like(x)
    return g


def _safe_std(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return x.sum() * 0.0
    return torch.std(x, unbiased=False)


def _weighted_mean(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return torch.mean(x)
    w = weight.to(device=x.device, dtype=x.dtype)
    return torch.sum(w * x) / torch.clamp(torch.sum(w), min=1.0)


def pseudo_huber(err: torch.Tensor, delta: float) -> torch.Tensor:
    """Smooth robust error in V^2-like units."""
    d = max(float(delta), 1e-6)
    z = err / d
    return d * d * (torch.sqrt(1.0 + z * z) - 1.0)

def _scheduled_scalar(batch: Mapping[str, torch.Tensor], key: str, ref: torch.Tensor, default: float = 1.0) -> torch.Tensor:
    """Return a differentiable scalar tensor for trainer-injected schedules."""
    value = batch.get(key)
    if value is None:
        return torch.as_tensor(float(default), device=ref.device, dtype=ref.dtype)
    if torch.is_tensor(value):
        return value.to(device=ref.device, dtype=ref.dtype).reshape(()).clamp(0.0, 1.0)
    return torch.as_tensor(float(value), device=ref.device, dtype=ref.dtype).reshape(()).clamp(0.0, 1.0)


RARE_LOSS_NAMES = {
    "voltage_tail",
    "voltage_quantile",
    "voltage_asymmetry",
    "voltage_event",
    "voltage_ultra_quantile",
    "voltage_low_coverage",
    "voltage_tail_balance",
}


class GV1LossComputer:
    """Compute D9.5.1 trend-first warmup rare-regime losses for a batch dictionary."""

    def __init__(self, weights: LossWeights | Mapping[str, Any] | None = None) -> None:
        self.weights = weights if isinstance(weights, LossWeights) else LossWeights.from_mapping(weights)

    def to_dict(self) -> dict[str, Any]:
        return self.weights.to_dict()

    @staticmethod
    def _batch_with_r(batch: Mapping[str, torch.Tensor], r_value: float) -> dict[str, torch.Tensor]:
        out = dict(batch)
        r = torch.full_like(batch["r_norm"], float(r_value), requires_grad=True)
        out["r_norm"] = r
        return out

    @staticmethod
    def _forward_transformed(
        model: ConditionedEffectiveSPMPINN,
        transform: GV1OutputTransform,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        raw = model(
            batch["t_norm"],
            batch["r_norm"],
            batch["current_norm"],
            batch["temperature_norm"],
            batch["condition"],
        )
        return transform(
            raw,
            r_norm=batch["r_norm"],
            current_A=batch["current_A"],
            current_norm=batch["current_norm"],
            cbar_a=batch["cbar_a"],
            cbar_c=batch["cbar_c"],
            temperature_norm=batch.get("temperature_norm"),
            condition=batch.get("condition"),
        )

    def _voltage_losses(
        self,
        pred_v: torch.Tensor,
        true_v: torch.Tensor,
        transform: GV1OutputTransform,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cfg = transform.config
        y = true_v.to(device=pred_v.device, dtype=pred_v.dtype)
        err = pred_v - y
        losses: dict[str, torch.Tensor] = {}
        span = max(float(cfg.voltage_max_V) - float(cfg.voltage_min_V), 0.5)
        low_thr = float(cfg.voltage_min_V) + float(self.weights.tail_fraction) * span
        high_thr = float(cfg.voltage_max_V) - float(self.weights.tail_fraction) * span
        low_mask = (y <= low_thr).to(dtype=pred_v.dtype)
        high_mask = (y >= high_thr).to(dtype=pred_v.dtype)
        tail_mask = torch.clamp(low_mask + high_mask, 0.0, 1.0)
        event_marker = batch.get("event_marker")
        if event_marker is None:
            event_marker = torch.zeros_like(pred_v)
        else:
            event_marker = event_marker.to(device=pred_v.device, dtype=pred_v.dtype)
        sample_weight = batch.get("sample_weight")
        if sample_weight is None:
            sample_weight = torch.ones_like(pred_v)
        else:
            sample_weight = sample_weight.to(device=pred_v.device, dtype=pred_v.dtype)

        rare_sample_scale = _scheduled_scalar(batch, "rare_sample_weight_scale", pred_v, default=1.0)
        # D9.5.1: do not let rare-regime weighting dominate the main voltage
        # fit from epoch 1.  The sampler may over-represent rare points, but the
        # loss itself sees only a scheduled fraction of the tail/event boost.
        sample_weight_eff = 1.0 + rare_sample_scale * torch.relu(sample_weight - 1.0)
        regime_boost = (
            float(self.weights.tail_weight_gain) * tail_mask
            + float(self.weights.low_tail_extra_gain) * low_mask
            + float(self.weights.high_tail_extra_gain) * high_mask
            + float(self.weights.event_weight_gain) * event_marker
        )
        point_weight = sample_weight_eff * (1.0 + rare_sample_scale * regime_boost)
        losses["voltage_data"] = _weighted_mean(pseudo_huber(err, self.weights.huber_delta_V), point_weight)

        if self.weights.voltage_tail > 0:
            tail_weight = point_weight * torch.clamp(tail_mask, 0.0, 1.0)
            losses["voltage_tail"] = _weighted_mean(pseudo_huber(err, self.weights.huber_delta_V), tail_weight)
        if self.weights.voltage_event > 0:
            event_weight = point_weight * torch.clamp(event_marker, 0.0, 1.0)
            losses["voltage_event"] = _weighted_mean(pseudo_huber(err, self.weights.huber_delta_V), event_weight)
        if self.weights.voltage_bias > 0:
            losses["voltage_bias"] = _weighted_mean(err, sample_weight).pow(2)
        if self.weights.voltage_range > 0:
            pred_std = _safe_std(pred_v)
            true_std = _safe_std(y)
            pred_amp = torch.mean(torch.abs(pred_v - torch.mean(pred_v)))
            true_amp = torch.mean(torch.abs(y - torch.mean(y)))
            losses["voltage_range"] = (pred_std - true_std).pow(2) + 0.5 * (pred_amp - true_amp).pow(2)
        if self.weights.voltage_quantile > 0 and pred_v.numel() >= 16:
            q = torch.tensor([0.02, 0.05, 0.10, 0.50, 0.90, 0.95, 0.98], device=pred_v.device, dtype=pred_v.dtype)
            pred_q = torch.quantile(pred_v.reshape(-1), q)
            true_q = torch.quantile(y.reshape(-1), q)
            q_weight = torch.tensor([2.0, 1.7, 1.3, 0.4, 1.0, 1.2, 1.2], device=pred_v.device, dtype=pred_v.dtype)
            losses["voltage_quantile"] = torch.mean(q_weight * (pred_q - true_q).pow(2))
        if self.weights.voltage_ultra_quantile > 0 and pred_v.numel() >= 32:
            # D9.5.1: this term is explicitly warm-scheduled in __call__.
            # It remains weak by default and only nudges extreme quantiles after
            # the ordinary voltage/correlation losses have established shape.
            q2 = torch.tensor([0.005, 0.01, 0.02, 0.05, 0.95, 0.98, 0.99, 0.995], device=pred_v.device, dtype=pred_v.dtype)
            pred_q2 = torch.quantile(pred_v.reshape(-1), q2)
            true_q2 = torch.quantile(y.reshape(-1), q2)
            q2_weight = torch.tensor([3.2, 2.8, 2.2, 1.5, 0.8, 1.0, 1.1, 1.2], device=pred_v.device, dtype=pred_v.dtype)
            losses["voltage_ultra_quantile"] = torch.mean(q2_weight * (pred_q2 - true_q2).pow(2))
        if self.weights.voltage_corr > 0 and pred_v.numel() >= 8:
            # Correlation/shape loss prevents low-tail fixes from destroying the
            # global V(t) trend, which was the main D9.4/D9.4.1 failure mode for B1.
            wp = pred_v.reshape(-1)
            wy = y.reshape(-1)
            wp = wp - torch.mean(wp)
            wy = wy - torch.mean(wy)
            denom = torch.sqrt(torch.mean(wp.pow(2)) * torch.mean(wy.pow(2)) + 1.0e-12)
            corr = torch.mean(wp * wy) / denom
            losses["voltage_corr"] = (1.0 - torch.clamp(corr, -1.0, 1.0)).pow(2)
        if self.weights.voltage_low_coverage > 0:
            # Differentiable coverage matching.  It matches the fraction of points
            # below/above physical thresholds, but uses soft sigmoids so gradients
            # remain usable even when the predicted fraction is initially zero.
            tau = max(float(self.weights.coverage_temperature_V), 1e-4)
            low_cov_thr = float(self.weights.low_coverage_threshold_V)
            high_cov_thr = float(self.weights.high_coverage_threshold_V)
            pred_low_frac = torch.mean(torch.sigmoid((low_cov_thr - pred_v) / tau))
            true_low_frac = torch.mean((y <= low_cov_thr).to(dtype=pred_v.dtype))
            pred_high_frac = torch.mean(torch.sigmoid((pred_v - high_cov_thr) / tau))
            true_high_frac = torch.mean((y >= high_cov_thr).to(dtype=pred_v.dtype))
            losses["voltage_low_coverage"] = (pred_low_frac - true_low_frac).pow(2) + 0.5 * (pred_high_frac - true_high_frac).pow(2)
        if self.weights.voltage_tail_balance > 0 and pred_v.numel() >= 16:
            # Match low/high tail means.  This is less brittle than only matching
            # global min/max, and it generalizes to high-current and temperature
            # regimes where tails may be sparse.
            low_true_mask = (y <= torch.quantile(y.reshape(-1), torch.tensor(0.10, device=y.device, dtype=y.dtype))).to(dtype=pred_v.dtype)
            high_true_mask = (y >= torch.quantile(y.reshape(-1), torch.tensor(0.90, device=y.device, dtype=y.dtype))).to(dtype=pred_v.dtype)
            low_den = torch.clamp(torch.sum(low_true_mask), min=1.0)
            high_den = torch.clamp(torch.sum(high_true_mask), min=1.0)
            low_mean_pred = torch.sum(low_true_mask * pred_v) / low_den
            low_mean_true = torch.sum(low_true_mask * y) / low_den
            high_mean_pred = torch.sum(high_true_mask * pred_v) / high_den
            high_mean_true = torch.sum(high_true_mask * y) / high_den
            losses["voltage_tail_balance"] = (low_mean_pred - low_mean_true).pow(2) + 0.5 * (high_mean_pred - high_mean_true).pow(2)
        if self.weights.voltage_asymmetry > 0:
            # Low-voltage tail was systematically over-predicted in D9.2; the
            # symmetric high-tail term guards against solving one tail by
            # breaking the other.
            low_over = low_mask * torch.relu(err).pow(2)
            high_under = high_mask * torch.relu(-err).pow(2)
            denom = torch.clamp(torch.sum(low_mask + high_mask), min=1.0)
            losses["voltage_asymmetry"] = torch.sum(low_over + 0.7 * high_under) / denom
        if self.weights.voltage_guardrail > 0:
            low = float(cfg.voltage_guard_low_V)
            high = float(cfg.voltage_guard_high_V)
            losses["voltage_guardrail"] = torch.mean(torch.relu(low - pred_v).pow(2) + torch.relu(pred_v - high).pow(2))
        return losses

    def __call__(
        self,
        model: ConditionedEffectiveSPMPINN,
        transform: GV1OutputTransform,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        work = dict(batch)
        work["t_norm"] = batch["t_norm"].detach().clone().requires_grad_(True)
        work["r_norm"] = batch["r_norm"].detach().clone().requires_grad_(True)
        pred = self._forward_transformed(model, transform, work)
        losses: dict[str, torch.Tensor] = {}
        if "voltage_exp" in work and self.weights.voltage_data > 0:
            losses.update(self._voltage_losses(pred["phis_c"], work["voltage_exp"], transform, work))
        if self.weights.cbar_anchor > 0:
            losses["cbar_anchor"] = mse(pred["theta_a"], work["cbar_a"]) + mse(pred["theta_c"], work["cbar_c"])
        if self.weights.radial_smooth > 0:
            dtheta_a_dr = safe_grad(pred["theta_a"], work["r_norm"])
            dtheta_c_dr = safe_grad(pred["theta_c"], work["r_norm"])
            losses["radial_smooth"] = torch.mean(dtheta_a_dr.pow(2) + dtheta_c_dr.pow(2))
        if self.weights.potential_smooth > 0:
            dphis_dt = safe_grad(pred["phis_c"], work["t_norm"])
            dphie_dt = safe_grad(pred["phie"], work["t_norm"])
            losses["potential_smooth"] = torch.mean(dphis_dt.pow(2) + dphie_dt.pow(2))
        if self.weights.center_symmetry > 0:
            center_batch = self._batch_with_r(work, 0.0)
            center = self._forward_transformed(model, transform, center_batch)
            da = safe_grad(center["theta_a"], center_batch["r_norm"])
            dc = safe_grad(center["theta_c"], center_batch["r_norm"])
            losses["center_symmetry"] = torch.mean(da.pow(2) + dc.pow(2))
        if self.weights.surface_flux > 0:
            surface_batch = self._batch_with_r(work, 1.0)
            surf = self._forward_transformed(model, transform, surface_batch)
            da_s = safe_grad(surf["theta_a"], surface_batch["r_norm"])
            dc_s = safe_grad(surf["theta_c"], surface_batch["r_norm"])
            ta, tc = transform.surface_flux_targets(work["current_norm"], da_s)
            losses["surface_flux"] = mse(da_s, ta) + mse(dc_s, tc)
        if not losses:
            zero = next(model.parameters()).sum() * 0.0
            return zero, {"total": 0.0}
        rare_loss_scale = _scheduled_scalar(work, "rare_loss_scale", next(iter(losses.values())), default=1.0)
        weighted = []
        weighted_log: dict[str, float] = {}
        for name, value in losses.items():
            weight = float(getattr(self.weights, name, 1.0))
            if name in RARE_LOSS_NAMES:
                term = weight * rare_loss_scale * value
                weighted_log[f"weighted_{name}"] = float(term.detach().cpu())
            else:
                term = weight * value
            weighted.append(term)
        total = torch.stack(weighted).sum()
        log = {name: float(value.detach().cpu()) for name, value in losses.items()}
        log.update(weighted_log)
        log["rare_loss_scale"] = float(rare_loss_scale.detach().cpu())
        log["rare_sample_weight_scale"] = float(_scheduled_scalar(work, "rare_sample_weight_scale", total, default=1.0).detach().cpu())
        log["total"] = float(total.detach().cpu())
        return total, log


def make_optimizer(model: nn.Module, *, lr: float = 2e-3, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
