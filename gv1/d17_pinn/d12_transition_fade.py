# -*- coding: utf-8 -*-
"""D12-S1K-style low/transition fade voltage helpers for D17.

This module deliberately contains only observed-signal formulae.  It never reads
or uses cs/theta/phie/phis soft-label arrays.  The goal is to migrate the D12-S1K
idea into D17 as a transparent gate/fade/preservation mechanism rather than as a
free pointwise voltage copier.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Mapping, Any

import torch


@dataclass(frozen=True)
class D12TransitionFadeConfig:
    low_v: float = 2.75
    normal_v: float = 3.05
    low_width_v: float = 0.055
    transition_width_v: float = 0.080
    discharge_softness_frac: float = 0.030
    transition_gain: float = 0.70
    non_low_preservation_floor: float = 0.02

    def to_jsonable(self) -> Dict[str, float]:
        return asdict(self)


def cfg_from_mapping(cfg: Mapping[str, Any] | None = None) -> D12TransitionFadeConfig:
    cfg = cfg or {}
    return D12TransitionFadeConfig(
        low_v=float(cfg.get("low_v", 2.75)),
        normal_v=float(cfg.get("normal_v", 3.05)),
        low_width_v=float(cfg.get("low_width_v", 0.055)),
        transition_width_v=float(cfg.get("transition_width_v", 0.080)),
        discharge_softness_frac=float(cfg.get("discharge_softness_frac", 0.030)),
        transition_gain=float(cfg.get("transition_gain", 0.70)),
        non_low_preservation_floor=float(cfg.get("non_low_preservation_floor", 0.02)),
    )


def current_discharge_gate(current_A: torch.Tensor, softness_frac: float = 0.030) -> torch.Tensor:
    """Smoothly identify discharge / loaded regions from observed current only."""
    max_i = torch.clamp(torch.max(torch.abs(current_A)), min=1.0e-6)
    softness = max(float(softness_frac), 1.0e-4)
    # Assumes XJTU replay convention: negative current is discharge.  This is
    # consistent with earlier D9-D12 measured-current replay work.
    return torch.sigmoid((-current_A / max_i - 0.025) / softness)


def d12_transition_fade_gates(
    voltage_exp: torch.Tensor,
    current_A: torch.Tensor,
    cfg: D12TransitionFadeConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Return low/transition/fade/preservation gates.

    D12-S1K succeeded because correction was localized to low voltage and
    smoothly faded back to baseline outside the transition region.  These gates
    reproduce that logic using observed V(t) and I(t) only.
    """
    cfg = cfg or D12TransitionFadeConfig()
    discharge = current_discharge_gate(current_A, cfg.discharge_softness_frac)
    low_core = torch.sigmoid((float(cfg.low_v) - voltage_exp) / float(cfg.low_width_v))
    below_normal = torch.sigmoid((float(cfg.normal_v) - voltage_exp) / float(cfg.transition_width_v))
    above_low = torch.sigmoid((voltage_exp - float(cfg.low_v)) / float(cfg.transition_width_v))
    transition = above_low * below_normal
    fade = discharge * torch.clamp(low_core + float(cfg.transition_gain) * transition, 0.0, 1.0)
    # Keep a tiny floor only inside loaded data to avoid zero gradient at the
    # exact boundary.  It is reported by preservation loss/budget audits.
    fade = torch.clamp(fade + float(cfg.non_low_preservation_floor) * discharge * (1.0 - below_normal), 0.0, 1.0)
    preserve = 1.0 - torch.clamp(fade, 0.0, 1.0)
    return {
        "discharge_gate": torch.clamp(discharge, 0.0, 1.0),
        "low_core_gate": torch.clamp(low_core * discharge, 0.0, 1.0),
        "transition_gate": torch.clamp(transition * discharge, 0.0, 1.0),
        "fade_gate": torch.clamp(fade, 0.0, 1.0),
        "preserve_gate": torch.clamp(preserve, 0.0, 1.0),
    }


def d12_transition_fade_basis(
    t_norm: torch.Tensor,
    q_norm: torch.Tensor,
    i_norm: torch.Tensor,
    voltage_exp: torch.Tensor,
    current_A: torch.Tensor,
    cfg: D12TransitionFadeConfig | None = None,
) -> torch.Tensor:
    """Low-dimensional correction basis that respects D12 fade-to-baseline.

    A profile can learn a handful of coefficients on these basis vectors.  This
    is materially different from assigning V_pred=V_exp pointwise: the residual
    is bounded, smooth-ish, low dimensional, and gated.
    """
    cfg = cfg or D12TransitionFadeConfig()
    gates = d12_transition_fade_gates(voltage_exp, current_A, cfg)
    t = torch.clamp(t_norm, 0.0, 1.0)
    q = torch.clamp(q_norm, -1.5, 1.5)
    i = torch.clamp(i_norm, -1.5, 1.5)
    low = gates["low_core_gate"]
    trans = gates["transition_gate"]
    fade = gates["fade_gate"]
    discharge = gates["discharge_gate"]
    # A compact set: offset/slope in low, offset/slope in transition, current- and
    # capacity-shaped terms, plus a very small loaded high-voltage drift basis.
    cols = [
        low,
        low * (2.0 * t - 1.0),
        low * q,
        low * i,
        trans,
        trans * (2.0 * t - 1.0),
        trans * q,
        trans * i,
        fade * (2.0 * q * q - 1.0),
        0.20 * discharge * (1.0 - fade) * (2.0 * t - 1.0),
    ]
    basis = torch.stack(cols, dim=-1)
    return basis / torch.sqrt(torch.tensor(float(len(cols)), device=basis.device, dtype=basis.dtype))


def gate_audit_numbers(
    voltage_exp: torch.Tensor,
    current_A: torch.Tensor,
    residual: torch.Tensor,
    cfg: D12TransitionFadeConfig | None = None,
) -> Dict[str, float]:
    cfg = cfg or D12TransitionFadeConfig()
    gates = d12_transition_fade_gates(voltage_exp, current_A, cfg)
    fade = gates["fade_gate"]
    preserve = gates["preserve_gate"]
    def safe_mean(x: torch.Tensor) -> float:
        return float(torch.mean(x.detach()).cpu()) if x.numel() else 0.0
    return {
        "d12_fade_gate_mean": safe_mean(fade),
        "d12_fade_gate_max": float(torch.max(fade.detach()).cpu()) if fade.numel() else 0.0,
        "d12_low_gate_mean": safe_mean(gates["low_core_gate"]),
        "d12_transition_gate_mean": safe_mean(gates["transition_gate"]),
        "d12_preservation_residual_abs_mean_V": safe_mean(torch.abs(preserve * residual)),
        "d12_preservation_residual_abs_max_V": float(torch.max(torch.abs(preserve * residual)).detach().cpu()) if residual.numel() else 0.0,
    }
