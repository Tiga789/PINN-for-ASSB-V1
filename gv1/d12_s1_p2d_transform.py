"""D12-S1G train-inside P2D-like localized output transform.

This transform extends the D9.6/D9.5.1 GV1 output map with the same trainable
``raw_p2d_deficit`` channel used in S1D, but S1G intentionally keeps the
prediction-side gates open enough for low-voltage residual anchoring to work.
Normal-region protection is handled mainly in the loss via correction-budget and
regret terms, not by broad suppression in the transform.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .output_transform import GV1OutputTransform, OutputTransformConfig


@dataclass
class D12S1OutputTransformConfig(OutputTransformConfig):
    # D12-S1G P2D-like train-inside localized correction.  Disabled when scale=0.
    enable_p2d_transport_deficit: bool = True
    p2d_transport_scale_V: float = 0.20
    p2d_transport_gate_center_V: float = 3.14
    p2d_transport_gate_width_V: float = 0.20
    p2d_transport_pred_center_V: float = 3.62
    p2d_transport_pred_width_V: float = 0.24
    p2d_discharge_gate_center: float = 0.02
    p2d_discharge_gate_width: float = 0.08
    p2d_current_event_gain: float = 0.35
    p2d_temperature_event_gain: float = 0.10
    p2d_protocol_gain: float = 0.18
    p2d_protocol_c_rate_center: float = 2.30
    p2d_protocol_c_rate_width: float = 0.45
    p2d_max_correction_V: float = 0.60
    # S1G optional leakage controls.  They use only prediction-side quantities, so they
    # are valid at inference time.  Values <=0 disable the extra gates.
    p2d_low_gate_power: float = 1.0
    p2d_pred_low_gate_power: float = 1.0
    p2d_normal_suppression_center_V: float = 0.0
    p2d_normal_suppression_width_V: float = 0.18
    p2d_normal_suppression_power: float = 1.0
    # S1G local high limiter is primarily loss-side, not transform-side.
    # S1F showed that broad prediction-side high suppression can pull the whole
    # curve downward.  Defaults below keep this gate disabled; only set center>0
    # for diagnostic ablations.
    p2d_high_suppression_center_V: float = 0.0
    p2d_high_suppression_width_V: float = 0.08
    p2d_high_suppression_power: float = 1.0
    # Negative correction means an upward voltage push because phis_c = base - correction.
    # S1E overshoot came from this path, so S1G defaults to no upward branch.
    p2d_allow_upward_correction_V: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "D12S1OutputTransformConfig":
        if not data:
            return cls()
        valid = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def _raw_or_zeros(raw: Mapping[str, torch.Tensor], key: str, ref: torch.Tensor) -> torch.Tensor:
    value = raw.get(key)
    if value is None:
        return torch.zeros_like(ref)
    return value


class D12S1P2DOutputTransform(GV1OutputTransform):
    """D9.x output map plus localized train-inside P2D-like deficit branch."""

    def __init__(self, config: D12S1OutputTransformConfig | Mapping[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, D12S1OutputTransformConfig) else D12S1OutputTransformConfig.from_mapping(config)
        super().__init__(self.config)

    def to_dict(self) -> dict[str, Any]:
        return self.config.to_dict()

    def _protocol_gate(self, condition: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if condition is None or condition.ndim != 2 or condition.shape[1] < 2:
            return torch.ones_like(ref)
        # The existing D9 condition vector uses approx_c_rate / 5 as slot 1.
        approx_c_rate = condition[:, 1:2].to(device=ref.device, dtype=ref.dtype) * 5.0
        width = max(float(cfg.p2d_protocol_c_rate_width), 1e-3)
        high_rate_gate = torch.sigmoid((approx_c_rate - float(cfg.p2d_protocol_c_rate_center)) / width)
        return 1.0 + float(cfg.p2d_protocol_gain) * high_rate_gate

    def __call__(
        self,
        raw: Mapping[str, torch.Tensor],
        *,
        r_norm: torch.Tensor,
        current_A: torch.Tensor,
        current_norm: torch.Tensor,
        cbar_a: torch.Tensor,
        cbar_c: torch.Tensor,
        temperature_norm: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        disable_p2d: bool = False,
    ) -> dict[str, torch.Tensor]:
        base = super().__call__(
            raw,
            r_norm=r_norm,
            current_A=current_A,
            current_norm=current_norm,
            cbar_a=cbar_a,
            cbar_c=cbar_c,
            temperature_norm=temperature_norm,
            condition=condition,
        )
        cfg = self.config
        ref = base["phis_c"]
        correction = torch.zeros_like(ref)
        if bool(cfg.enable_p2d_transport_deficit) and not bool(disable_p2d) and float(cfg.p2d_transport_scale_V) > 0:
            cur_norm = current_norm.reshape(-1, 1).to(device=ref.device, dtype=ref.dtype)
            temp_norm = torch.zeros_like(cur_norm) if temperature_norm is None else temperature_norm.reshape(-1, 1).to(device=ref.device, dtype=ref.dtype)
            raw_def = _raw_or_zeros(raw, "raw_p2d_deficit", ref)

            # Local gates.  No target voltage is used at inference time.
            low_gate = torch.sigmoid(
                (float(cfg.p2d_transport_gate_center_V) - base["voltage_ocv_baseline"])
                / max(float(cfg.p2d_transport_gate_width_V), 1e-3)
            )
            pred_low_gate = torch.sigmoid(
                (float(cfg.p2d_transport_pred_center_V) - base["phis_c"])
                / max(float(cfg.p2d_transport_pred_width_V), 1e-3)
            )
            # S1G: optional sharpening of existing low gates and optionally suppress correction
            # where the baseline prediction is already normal/high.  This is
            # deliberately target-free and complements the training loss.
            low_power = max(float(getattr(cfg, "p2d_low_gate_power", 1.0)), 0.05)
            pred_power = max(float(getattr(cfg, "p2d_pred_low_gate_power", 1.0)), 0.05)
            low_gate = torch.clamp(low_gate, 0.0, 1.0).pow(low_power)
            pred_low_gate = torch.clamp(pred_low_gate, 0.0, 1.0).pow(pred_power)
            normal_suppression_center = float(getattr(cfg, "p2d_normal_suppression_center_V", 0.0))
            if normal_suppression_center > 0.0:
                normal_gate = torch.sigmoid(
                    (normal_suppression_center - base["phis_c"])
                    / max(float(getattr(cfg, "p2d_normal_suppression_width_V", 0.18)), 1e-3)
                )
                normal_power = max(float(getattr(cfg, "p2d_normal_suppression_power", 1.0)), 0.05)
                pred_low_gate = pred_low_gate * torch.clamp(normal_gate, 0.0, 1.0).pow(normal_power)

            high_suppression_center = float(getattr(cfg, "p2d_high_suppression_center_V", 0.0))
            if high_suppression_center > 0.0:
                high_gate = torch.sigmoid(
                    (high_suppression_center - base["phis_c"])
                    / max(float(getattr(cfg, "p2d_high_suppression_width_V", 0.08)), 1e-3)
                )
                high_power = max(float(getattr(cfg, "p2d_high_suppression_power", 1.4)), 0.05)
                pred_low_gate = pred_low_gate * torch.clamp(high_gate, 0.0, 1.0).pow(high_power)
            discharge_gate = torch.sigmoid(
                ((-cur_norm) - float(cfg.p2d_discharge_gate_center)) / max(float(cfg.p2d_discharge_gate_width), 1e-3)
            )
            current_event = torch.clamp(torch.abs(cur_norm), 0.0, 1.5)
            event_gain = 1.0 + float(cfg.p2d_current_event_gain) * current_event
            temp_gain = 1.0 + float(cfg.p2d_temperature_event_gain) * torch.relu(torch.abs(temp_norm) - 0.25)
            protocol_gain = self._protocol_gate(condition, ref)

            # Downward deficit with zero initial value and non-zero derivative.
            # softplus(0)-log(2)=0, but d/draw=0.5, so the new channel can
            # learn immediately.  Small negative values are allowed during
            # optimization and are controlled by preservation/correction-L2.
            deficit_amp = torch.nn.functional.softplus(raw_def) - 0.6931471805599453
            correction = (
                float(cfg.p2d_transport_scale_V)
                * low_gate
                * pred_low_gate
                * discharge_gate
                * event_gain
                * temp_gain
                * protocol_gain
                * deficit_amp
            )
            max_corr = max(float(cfg.p2d_max_correction_V), 0.0)
            if max_corr > 0:
                upward_allow = max(float(getattr(cfg, "p2d_allow_upward_correction_V", 0.0)), 0.0)
                correction = torch.clamp(correction, -upward_allow, max_corr)
            base["phis_c"] = base["phis_c"] - correction
            base["voltage_exp_pred"] = base["phis_c"]

        base["voltage_p2d_transport_deficit"] = correction
        base["voltage_p2d_deficit_raw"] = _raw_or_zeros(raw, "raw_p2d_deficit", ref)
        return base

    def without_p2d(
        self,
        raw: Mapping[str, torch.Tensor],
        *,
        r_norm: torch.Tensor,
        current_A: torch.Tensor,
        current_norm: torch.Tensor,
        cbar_a: torch.Tensor,
        cbar_c: torch.Tensor,
        temperature_norm: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self(
            raw,
            r_norm=r_norm,
            current_A=current_A,
            current_norm=current_norm,
            cbar_a=cbar_a,
            cbar_c=cbar_c,
            temperature_norm=temperature_norm,
            condition=condition,
            disable_p2d=True,
        )
