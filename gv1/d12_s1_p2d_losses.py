"""D12-S1G low-residual-anchor / normal-budget P2D-like correction losses.

D12-S1G is designed after the observed S1C/S1D failure pattern:

* S1C reached the required low_target improvement, but normal/global leaked by
  roughly 6-10 mV.
* S1D normal-regret guards reduced leakage pressure, but they also suppressed the
  P2D branch so much that low_target improvement fell to only ~2-8 mV.

S1G keeps the D9.5.1 base objective and the train-inside P2D branch, but adds two
mechanisms that directly target the pain point:

1. low residual anchor: on low_target/deep-low samples, the P2D correction is
   explicitly trained toward the no-P2D residual ``baseline_without_p2d - target``;
2. normal correction budget: outside the low segment, the positive downward
   correction is constrained by a small allowed mV budget rather than globally
   suppressing the branch.

This is still a diagnostic candidate, not a D9.6/D9.5.1 mainline replacement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .d12_s1_p2d_model import D12S1P2DLocalPINN
from .d12_s1_p2d_transform import D12S1P2DOutputTransform
from .losses import GV1LossComputer, LossWeights, _scheduled_scalar, make_optimizer


@dataclass
class D12S1LossWeights(LossWeights):
    p2d_lowtarget_focus: float = 0.22
    p2d_deep_coverage: float = 0.08
    p2d_normal_preservation: float = 0.30
    p2d_rest_preservation: float = 0.12
    p2d_high_preservation: float = 0.10
    p2d_correction_l2: float = 0.015
    # S1B additional normal-region leakage guards.  The first term penalizes
    # mean correction bias in normal voltage regions; the second penalizes
    # pointwise correction beyond a small allowed shift.
    p2d_normal_bias_preservation: float = 0.0
    p2d_normal_shift_guard: float = 0.0
    p2d_normal_allowed_shift_V: float = 0.006
    # S1G asymmetric guards: penalize only downward normal-region drift relative
    # to the no-P2D baseline. S1 leaked downward into normal_target_gt_3p20;
    # S1B over-suppressed all correction. These terms target the actual failure.
    p2d_normal_down_bias_guard: float = 0.0
    p2d_normal_down_shift_guard: float = 0.0
    p2d_normal_down_allowed_shift_V: float = 0.007
    # S1G regret guards: directly penalize normal-region degradation relative
    # to the no-P2D baseline.  This is the main S1G change: it allows P2D
    # correction where it reduces error, but blocks correction when it worsens
    # normal_target_gt_3p20 / non-low regions.
    p2d_normal_regret_guard: float = 0.0
    p2d_normal_regret_allowed_V: float = 0.002
    p2d_nonlow_regret_guard: float = 0.0
    p2d_nonlow_regret_allowed_V: float = 0.003
    # S1G residual-anchor terms: make the low-voltage correction learn the
    # actual low-segment residual instead of relying only on generic low MAE.
    p2d_low_residual_anchor: float = 0.0
    p2d_deep_residual_anchor: float = 0.0
    p2d_low_anchor_max_V: float = 0.55
    p2d_low_anchor_huber_delta_V: float = 0.045
    # S1G normal/non-low correction budgets: limit downward leakage directly
    # through the P2D correction amplitude outside the low segment.
    p2d_normal_correction_budget: float = 0.0
    p2d_normal_correction_allowed_V: float = 0.0045
    p2d_nonlow_correction_budget: float = 0.0
    p2d_nonlow_correction_allowed_V: float = 0.0060
    # S1G high-voltage preservation: S1E-soft was promotion-ready except high_ok.
    p2d_high_regret_guard: float = 0.0
    p2d_high_regret_allowed_V: float = 0.0015
    p2d_high_correction_budget: float = 0.0
    p2d_high_correction_allowed_V: float = 0.0015
    p2d_high_overshoot_guard: float = 0.0
    p2d_high_overshoot_threshold_V: float = 4.35
    p2d_low_target_threshold_V: float = 3.00
    p2d_deep_low_threshold_V: float = 2.75
    p2d_normal_margin_V: float = 0.20
    p2d_preservation_huber_delta_V: float = 0.05
    p2d_coverage_temperature_V: float = 0.035

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "D12S1LossWeights":
        if not data:
            return cls()
        valid = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    m = mask.to(device=x.device, dtype=x.dtype)
    denom = torch.clamp(torch.sum(m), min=float(eps))
    return torch.sum(m * x) / denom


def _pseudo_huber(err: torch.Tensor, delta: float) -> torch.Tensor:
    d = max(float(delta), 1e-6)
    z = err / d
    return d * d * (torch.sqrt(1.0 + z * z) - 1.0)


class D12S1LossComputer(GV1LossComputer):
    """D9.5.1 loss plus train-inside localized P2D/preservation terms."""

    def __init__(self, weights: D12S1LossWeights | Mapping[str, Any] | None = None) -> None:
        self.weights = weights if isinstance(weights, D12S1LossWeights) else D12S1LossWeights.from_mapping(weights)
        # GV1LossComputer uses self.weights dynamically, so no separate base field is needed.

    def to_dict(self) -> dict[str, Any]:
        return self.weights.to_dict()

    @staticmethod
    def _forward_raw(model: D12S1P2DLocalPINN, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return model(
            batch["t_norm"],
            batch["r_norm"],
            batch["current_norm"],
            batch["temperature_norm"],
            batch["condition"],
        )

    def __call__(
        self,
        model: D12S1P2DLocalPINN,
        transform: D12S1P2DOutputTransform,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total, log = super().__call__(model, transform, batch)
        w = self.weights
        if not any(
            float(x) > 0
            for x in [
                w.p2d_lowtarget_focus,
                w.p2d_deep_coverage,
                w.p2d_normal_preservation,
                w.p2d_rest_preservation,
                w.p2d_high_preservation,
                w.p2d_correction_l2,
                w.p2d_normal_bias_preservation,
                w.p2d_normal_shift_guard,
                w.p2d_normal_down_bias_guard,
                w.p2d_normal_down_shift_guard,
                w.p2d_normal_regret_guard,
                w.p2d_nonlow_regret_guard,
                w.p2d_low_residual_anchor,
                w.p2d_deep_residual_anchor,
                w.p2d_normal_correction_budget,
                w.p2d_nonlow_correction_budget,
                w.p2d_high_regret_guard,
                w.p2d_high_correction_budget,
                w.p2d_high_overshoot_guard,
            ]
        ):
            return total, log

        work = dict(batch)
        work["t_norm"] = batch["t_norm"].detach().clone().requires_grad_(True)
        work["r_norm"] = batch["r_norm"].detach().clone().requires_grad_(True)
        raw = self._forward_raw(model, work)
        pred = transform(
            raw,
            r_norm=work["r_norm"],
            current_A=work["current_A"],
            current_norm=work["current_norm"],
            cbar_a=work["cbar_a"],
            cbar_c=work["cbar_c"],
            temperature_norm=work.get("temperature_norm"),
            condition=work.get("condition"),
        )
        base = transform.without_p2d(
            raw,
            r_norm=work["r_norm"],
            current_A=work["current_A"],
            current_norm=work["current_norm"],
            cbar_a=work["cbar_a"],
            cbar_c=work["cbar_c"],
            temperature_norm=work.get("temperature_norm"),
            condition=work.get("condition"),
        )
        pv = pred["phis_c"]
        bv = base["phis_c"].detach()
        y = work["voltage_exp"].to(device=pv.device, dtype=pv.dtype)
        cur = work["current_A"].to(device=pv.device, dtype=pv.dtype)
        corr = pred.get("voltage_p2d_transport_deficit", torch.zeros_like(pv))

        low_thr = float(w.p2d_low_target_threshold_V)
        deep_thr = float(w.p2d_deep_low_threshold_V)
        margin = max(float(w.p2d_normal_margin_V), 0.0)
        low_mask = (y <= low_thr).to(dtype=pv.dtype)
        deep_mask = (y <= deep_thr).to(dtype=pv.dtype)
        normal_mask = (y > low_thr + margin).to(dtype=pv.dtype)
        nonlow_mask = (y > low_thr).to(dtype=pv.dtype)
        rest_mask = (torch.abs(cur) <= 1.0e-10).to(dtype=pv.dtype)
        high_mask = (y >= float(w.high_coverage_threshold_V)).to(dtype=pv.dtype)

        rare_scale = _scheduled_scalar(work, "rare_loss_scale", pv, default=1.0)
        custom_terms: dict[str, torch.Tensor] = {}
        if float(w.p2d_lowtarget_focus) > 0:
            # Penalize low-target over-prediction much more than under-prediction.
            low_over = torch.relu(pv - y).pow(2)
            low_under = 0.12 * torch.relu(y - pv).pow(2)
            custom_terms["p2d_lowtarget_focus"] = _masked_mean(low_over + low_under, low_mask)
        if float(getattr(w, "p2d_low_residual_anchor", 0.0)) > 0:
            # Target-aware but train-only: the ideal deficit equals the positive
            # no-P2D residual.  At inference, only the learned target-free branch
            # is used. This directly counters S1D's under-active correction.
            max_anchor = max(float(getattr(w, "p2d_low_anchor_max_V", 0.55)), 0.0)
            delta = max(float(getattr(w, "p2d_low_anchor_huber_delta_V", 0.045)), 1e-4)
            ideal = torch.clamp(bv - y, min=0.0, max=max_anchor)
            anchor_err = _pseudo_huber(corr - ideal, delta)
            custom_terms["p2d_low_residual_anchor"] = _masked_mean(anchor_err, low_mask)
        if float(getattr(w, "p2d_deep_residual_anchor", 0.0)) > 0:
            max_anchor = max(float(getattr(w, "p2d_low_anchor_max_V", 0.55)), 0.0)
            delta = max(float(getattr(w, "p2d_low_anchor_huber_delta_V", 0.045)), 1e-4)
            ideal = torch.clamp(bv - y, min=0.0, max=max_anchor)
            anchor_err = _pseudo_huber(corr - ideal, delta)
            custom_terms["p2d_deep_residual_anchor"] = _masked_mean(anchor_err, deep_mask)
        if float(w.p2d_deep_coverage) > 0:
            tau = max(float(w.p2d_coverage_temperature_V), 1e-4)
            pred_deep_frac = torch.mean(torch.sigmoid((deep_thr - pv) / tau))
            true_deep_frac = torch.mean(deep_mask)
            # Also penalize deep over-prediction directly when deep samples exist.
            deep_over = _masked_mean(torch.relu(pv - y).pow(2), deep_mask)
            custom_terms["p2d_deep_coverage"] = (pred_deep_frac - true_deep_frac).pow(2) + 0.5 * deep_over
        normal_delta = pv - bv
        if float(w.p2d_normal_preservation) > 0:
            diff = _pseudo_huber(normal_delta, float(w.p2d_preservation_huber_delta_V))
            custom_terms["p2d_normal_preservation"] = _masked_mean(diff, normal_mask)
        if float(getattr(w, "p2d_normal_bias_preservation", 0.0)) > 0:
            # S1B: S1 failed because the P2D branch introduced a broad negative
            # bias in normal_target_gt_3p20.  Penalize the mean normal correction
            # explicitly, not only its per-point Huber magnitude.
            normal_bias = _masked_mean(normal_delta, normal_mask)
            custom_terms["p2d_normal_bias_preservation"] = normal_bias.pow(2)
        if float(getattr(w, "p2d_normal_shift_guard", 0.0)) > 0:
            allowed = max(float(getattr(w, "p2d_normal_allowed_shift_V", 0.006)), 0.0)
            shift_excess = torch.relu(torch.abs(normal_delta) - allowed).pow(2)
            custom_terms["p2d_normal_shift_guard"] = _masked_mean(shift_excess, normal_mask)
        if float(getattr(w, "p2d_normal_down_bias_guard", 0.0)) > 0:
            # S1G: only broad downward leakage is penalized. Upward correction is
            # not the observed failure mode, and low_target samples are excluded
            # by normal_mask so this does not turn off the low-voltage branch.
            allowed_down = max(float(getattr(w, "p2d_normal_down_allowed_shift_V", 0.007)), 0.0)
            normal_bias = _masked_mean(normal_delta, normal_mask)
            custom_terms["p2d_normal_down_bias_guard"] = torch.relu(-normal_bias - allowed_down).pow(2)
        if float(getattr(w, "p2d_normal_down_shift_guard", 0.0)) > 0:
            allowed_down = max(float(getattr(w, "p2d_normal_down_allowed_shift_V", 0.007)), 0.0)
            down_excess = torch.relu((-normal_delta) - allowed_down).pow(2)
            custom_terms["p2d_normal_down_shift_guard"] = _masked_mean(down_excess, normal_mask)
        if float(getattr(w, "p2d_normal_correction_budget", 0.0)) > 0:
            # Directly constrain positive downward P2D leakage in normal voltage
            # regions.  This is intentionally correction-based, not prediction-
            # side suppression, so low-target anchoring remains possible.
            allowed = max(float(getattr(w, "p2d_normal_correction_allowed_V", 0.0045)), 0.0)
            positive_corr_excess = torch.relu(corr - allowed).pow(2)
            custom_terms["p2d_normal_correction_budget"] = _masked_mean(positive_corr_excess, normal_mask)
        if float(getattr(w, "p2d_nonlow_correction_budget", 0.0)) > 0:
            allowed = max(float(getattr(w, "p2d_nonlow_correction_allowed_V", 0.0060)), 0.0)
            positive_corr_excess = torch.relu(corr - allowed).pow(2)
            custom_terms["p2d_nonlow_correction_budget"] = _masked_mean(positive_corr_excess, nonlow_mask)
        if float(getattr(w, "p2d_normal_regret_guard", 0.0)) > 0:
            allowed = max(float(getattr(w, "p2d_normal_regret_allowed_V", 0.002)), 0.0)
            # Penalize only actual degradation relative to the no-P2D baseline.
            # This directly targets the S1/S1C failure: normal MAE went up by
            # 5.8-10.1 mV while low_target improved.
            pred_abs = torch.abs(pv - y)
            base_abs = torch.abs(bv - y)
            regret = torch.relu(pred_abs - base_abs - allowed).pow(2)
            custom_terms["p2d_normal_regret_guard"] = _masked_mean(regret, normal_mask)
        if float(getattr(w, "p2d_nonlow_regret_guard", 0.0)) > 0:
            allowed = max(float(getattr(w, "p2d_nonlow_regret_allowed_V", 0.003)), 0.0)
            pred_abs = torch.abs(pv - y)
            base_abs = torch.abs(bv - y)
            regret = torch.relu(pred_abs - base_abs - allowed).pow(2)
            custom_terms["p2d_nonlow_regret_guard"] = _masked_mean(regret, nonlow_mask)
        if float(w.p2d_rest_preservation) > 0:
            diff = _pseudo_huber(pv - bv, float(w.p2d_preservation_huber_delta_V))
            custom_terms["p2d_rest_preservation"] = _masked_mean(diff, rest_mask)
        if float(w.p2d_high_preservation) > 0:
            diff = _pseudo_huber(pv - bv, float(w.p2d_preservation_huber_delta_V))
            custom_terms["p2d_high_preservation"] = _masked_mean(diff, high_mask)
        if float(getattr(w, "p2d_high_regret_guard", 0.0)) > 0:
            allowed = max(float(getattr(w, "p2d_high_regret_allowed_V", 0.0015)), 0.0)
            pred_abs = torch.abs(pv - y)
            base_abs = torch.abs(bv - y)
            regret = torch.relu(pred_abs - base_abs - allowed).pow(2)
            custom_terms["p2d_high_regret_guard"] = _masked_mean(regret, high_mask)
        if float(getattr(w, "p2d_high_correction_budget", 0.0)) > 0:
            allowed = max(float(getattr(w, "p2d_high_correction_allowed_V", 0.0015)), 0.0)
            high_excess = torch.relu(torch.abs(corr) - allowed).pow(2)
            custom_terms["p2d_high_correction_budget"] = _masked_mean(high_excess, high_mask)
        if float(getattr(w, "p2d_high_overshoot_guard", 0.0)) > 0:
            threshold = float(getattr(w, "p2d_high_overshoot_threshold_V", 4.35))
            overshoot = torch.relu(pv - threshold).pow(2)
            # All-sample mean catches pred>4.35 even when target is not in the
            # high_target_ge_4p10 mask, matching the scorecard failure mode.
            custom_terms["p2d_high_overshoot_guard"] = torch.mean(overshoot)
        if float(w.p2d_correction_l2) > 0:
            custom_terms["p2d_correction_l2"] = torch.mean(corr.pow(2))

        weighted = []
        for name, value in custom_terms.items():
            factor = float(getattr(w, name))
            # D12-S1G low/deep terms are warm-scheduled; preservation/guards remain active.
            if name in {"p2d_lowtarget_focus", "p2d_deep_coverage", "p2d_low_residual_anchor", "p2d_deep_residual_anchor"}:
                term = factor * rare_scale * value
            else:
                term = factor * value
            weighted.append(term)
            log[name] = float(value.detach().cpu())
            log[f"weighted_{name}"] = float(term.detach().cpu())
        if weighted:
            total = total + torch.stack(weighted).sum()
            log["total"] = float(total.detach().cpu())
        return total, log


__all__ = [
    "D12S1LossWeights",
    "D12S1LossComputer",
    "make_optimizer",
]
