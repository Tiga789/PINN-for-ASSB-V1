# -*- coding: utf-8 -*-
"""SOH predictor for ASSB ModelFin_111 strict30 experiment.

This module is intentionally independent from the ModelFin_107A four-state
state engine. ASSB-111 protects the frozen 107A outputs for cs_a, cs_c, phie and
phis_c, while this file only defines the cycle-level SOH head.

Two variants are supported:

``legacy_accumulative``
    The original cumulative non-negative damage model. It is kept for A/B
    comparison and for loading old smoke models.

``saturating_v2``
    A train-only supervised, physically constrained SOH recurrence with a
    learnable/regularized floor and a remaining-degradable-capacity gate. This
    prevents the strict30 model from extrapolating early-cycle slope until it
    numerically clamps at 0.4.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

PathLike = Union[str, Path]


@dataclass
class Assb111SOHHeadConfig:
    n_features: int
    hidden_dim: int = 32
    hidden_layers: int = 2
    dropout: float = 0.05
    activation: str = "silu"

    # Variant control. Missing value in old config files is interpreted as
    # legacy_accumulative by from_dict(). New training scripts pass saturating_v2.
    model_variant: str = "legacy_accumulative"

    # Legacy accumulative model parameters.
    rate_scale: float = 1.0e-3
    residual_bound: float = 0.015
    lam_max: float = 0.60
    window_loss_max: float = 0.45
    r_ohm_delta_max: float = 250.0
    r_ohm_base: float = 105.0
    soh_min: float = 0.40
    soh_max: float = 1.05

    # Saturating_v2 parameters.
    floor_min: float = 0.65
    floor_max: float = 0.85
    soh_floor_prior: float = 0.72
    soh0_min: float = 0.94
    soh0_max: float = 1.03
    damage_rate_scale: float = 5.0e-4
    gate_gamma: float = 1.0
    soh_numeric_min: float = 0.60
    tail_slope_guard: float = 0.0020

    # Loss weights.
    huber_delta: float = 0.02
    w_soh: float = 1.0
    w_q: float = 0.0
    w_smooth: float = 0.05
    w_rate: float = 0.01
    w_monotonic: float = 0.20
    w_residual: float = 0.10
    w_bounds: float = 0.20
    w_floor_prior: float = 0.02
    w_tail_guard: float = 0.05
    dtype: str = "float64"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Assb111SOHHeadConfig":
        dd = dict(d)
        # Old smoke models did not write model_variant. Keep them loadable as
        # legacy models instead of silently switching architecture semantics.
        if "model_variant" not in dd:
            dd["model_variant"] = "legacy_accumulative"
        return cls(**{k: v for k, v in dd.items() if k in cls.__dataclass_fields__})


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError(f"PyTorch is required for assb111_soh_model: {_TORCH_IMPORT_ERROR}")


def _activation(name: str):
    low = str(name).strip().lower()
    if low in {"silu", "swish"}:
        return nn.SiLU()
    if low == "relu":
        return nn.ReLU()
    if low == "tanh":
        return nn.Tanh()
    if low == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _logit(p: float) -> float:
    pp = min(max(float(p), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(pp / (1.0 - pp))


@dataclass
class Assb111SOHOutput:
    SOH_pred: Any
    SOH_struct: Any
    residual: Any
    f_LAM_c: Any
    theta_window_scale_c: Any
    R_ohm_eff: Any
    lam_rate: Any
    window_rate: Any
    r_rate: Any
    lam_damage: Any
    window_damage: Any
    r_ohm_growth: Any
    # New saturating_v2 diagnostics. Legacy forward fills them with compatible
    # values where possible.
    SOH_base: Any = None
    SOH_pred_unclipped: Any = None
    soh_floor: Any = None
    soh0: Any = None
    damage_rate_base: Any = None
    damage_rate_gated: Any = None
    remaining_degradable: Any = None
    active_clamp_mask: Any = None


if torch is not None:
    class Assb111SOHHead(nn.Module):
        """Small interpretable strict30 SOH predictor."""

        def __init__(self, cfg: Assb111SOHHeadConfig):
            super().__init__()
            if int(cfg.n_features) <= 0:
                raise ValueError("n_features must be positive")
            self.cfg = cfg
            layers: List[nn.Module] = []
            in_dim = int(cfg.n_features)
            for _ in range(int(cfg.hidden_layers)):
                layers.append(nn.Linear(in_dim, int(cfg.hidden_dim)))
                layers.append(_activation(cfg.activation))
                if float(cfg.dropout) > 0:
                    layers.append(nn.Dropout(float(cfg.dropout)))
                in_dim = int(cfg.hidden_dim)
            self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
            self.rate_head = nn.Linear(in_dim, 3)
            self.residual_head = nn.Linear(in_dim, 1)
            # Extra trainable scalars used by saturating_v2. They exist in the
            # module for all variants; load() uses strict=False for old states.
            p_floor = (float(cfg.soh_floor_prior) - float(cfg.floor_min)) / max(float(cfg.floor_max) - float(cfg.floor_min), 1.0e-12)
            self.floor_raw = nn.Parameter(torch.tensor(_logit(p_floor), dtype=torch.float64))
            p_soh0 = (1.0 - float(cfg.soh0_min)) / max(float(cfg.soh0_max) - float(cfg.soh0_min), 1.0e-12)
            self.soh0_raw = nn.Parameter(torch.tensor(_logit(p_soh0), dtype=torch.float64))
            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            # Conservative starting point; train labels determine the amplitude.
            nn.init.constant_(self.rate_head.bias, -2.0)
            nn.init.zeros_(self.residual_head.bias)

        def forward(self, x, *, delta_cycle=None) -> Assb111SOHOutput:
            variant = str(self.cfg.model_variant).strip().lower()
            if variant in {"saturating", "saturating_v2", "sat_v2"}:
                return self._forward_saturating_v2(x, delta_cycle=delta_cycle)
            if variant in {"legacy", "legacy_accumulative", "accumulative", "stageb_like"}:
                return self._forward_legacy(x, delta_cycle=delta_cycle)
            raise ValueError(f"Unsupported Assb111SOHHeadConfig.model_variant={self.cfg.model_variant!r}")

        def _prepare_inputs(self, x, delta_cycle=None):
            x = torch.as_tensor(x, dtype=self._dtype(), device=self._device())
            if x.ndim != 2:
                raise ValueError(f"features must be 2D [cycle, feature], got shape={tuple(x.shape)}")
            n = x.shape[0]
            if delta_cycle is None:
                delta = torch.ones(n, dtype=x.dtype, device=x.device)
            else:
                delta = torch.as_tensor(delta_cycle, dtype=x.dtype, device=x.device).reshape(-1)
                if delta.numel() != n:
                    raise ValueError("delta_cycle length must match number of feature rows")
                delta = torch.clamp(delta, min=0.0)
            return x, delta

        def _encoded_rates(self, x):
            h = self.encoder(x)
            rates_raw = self.rate_head(h)
            residual = float(self.cfg.residual_bound) * torch.tanh(self.residual_head(h).reshape(-1))
            return h, rates_raw, residual

        def _forward_legacy(self, x, *, delta_cycle=None) -> Assb111SOHOutput:
            x, delta = self._prepare_inputs(x, delta_cycle)
            h, rates_raw, residual = self._encoded_rates(x)
            rates = F.softplus(rates_raw) * float(self.cfg.rate_scale)
            lam_rate = rates[:, 0]
            window_rate = rates[:, 1]
            r_rate = rates[:, 2]
            lam_damage = torch.cumsum(lam_rate * delta, dim=0).clamp(min=0.0, max=float(self.cfg.lam_max))
            window_damage = torch.cumsum(window_rate * delta, dim=0).clamp(min=0.0, max=float(self.cfg.window_loss_max))
            r_growth_unit = torch.cumsum(r_rate * delta, dim=0).clamp(min=0.0, max=1.0)
            r_ohm_growth = r_growth_unit * float(self.cfg.r_ohm_delta_max)
            f_lam = (1.0 - lam_damage).clamp(min=max(1.0e-6, 1.0 - float(self.cfg.lam_max)), max=float(self.cfg.soh_max))
            win_scale = (1.0 - window_damage).clamp(min=max(1.0e-6, 1.0 - float(self.cfg.window_loss_max)), max=float(self.cfg.soh_max))
            soh_struct = (f_lam * win_scale).clamp(min=float(self.cfg.soh_min), max=float(self.cfg.soh_max))
            unclipped = soh_struct + residual
            soh_pred = unclipped.clamp(min=float(self.cfg.soh_min), max=float(self.cfg.soh_max))
            active = (unclipped != soh_pred)
            return Assb111SOHOutput(
                SOH_pred=soh_pred,
                SOH_struct=soh_struct,
                residual=residual,
                f_LAM_c=f_lam,
                theta_window_scale_c=win_scale,
                R_ohm_eff=torch.as_tensor(float(self.cfg.r_ohm_base), dtype=x.dtype, device=x.device) + r_ohm_growth,
                lam_rate=lam_rate,
                window_rate=window_rate,
                r_rate=r_rate,
                lam_damage=lam_damage,
                window_damage=window_damage,
                r_ohm_growth=r_ohm_growth,
                SOH_base=soh_struct,
                SOH_pred_unclipped=unclipped,
                soh_floor=torch.as_tensor(float(self.cfg.soh_min), dtype=x.dtype, device=x.device),
                soh0=torch.as_tensor(1.0, dtype=x.dtype, device=x.device),
                damage_rate_base=lam_rate + window_rate,
                damage_rate_gated=lam_rate + window_rate,
                remaining_degradable=torch.ones_like(soh_struct),
                active_clamp_mask=active,
            )

        def _forward_saturating_v2(self, x, *, delta_cycle=None) -> Assb111SOHOutput:
            x, delta = self._prepare_inputs(x, delta_cycle)
            _h, rates_raw, residual = self._encoded_rates(x)
            rates = F.softplus(rates_raw)
            lam_rate = rates[:, 0] * float(self.cfg.damage_rate_scale)
            window_rate = rates[:, 1] * float(self.cfg.damage_rate_scale)
            r_rate = rates[:, 2] * float(self.cfg.damage_rate_scale)
            base_rate = lam_rate + window_rate

            floor = torch.as_tensor(float(self.cfg.floor_min), dtype=x.dtype, device=x.device) + torch.sigmoid(self.floor_raw.to(dtype=x.dtype, device=x.device)) * (
                float(self.cfg.floor_max) - float(self.cfg.floor_min)
            )
            soh0 = torch.as_tensor(float(self.cfg.soh0_min), dtype=x.dtype, device=x.device) + torch.sigmoid(self.soh0_raw.to(dtype=x.dtype, device=x.device)) * (
                float(self.cfg.soh0_max) - float(self.cfg.soh0_min)
            )
            denom = torch.clamp(soh0 - floor, min=torch.as_tensor(1.0e-8, dtype=x.dtype, device=x.device))

            base_vals: List[Any] = []
            rem_vals: List[Any] = []
            gated_vals: List[Any] = []
            prev = soh0
            gamma = float(self.cfg.gate_gamma)
            for i in range(x.shape[0]):
                remaining = torch.clamp((prev - floor) / denom, min=0.0, max=1.0)
                gated = base_rate[i] * torch.pow(remaining, gamma)
                prev = floor + (prev - floor) * torch.exp(-gated * delta[i])
                base_vals.append(prev)
                rem_vals.append(remaining)
                gated_vals.append(gated)
            soh_base = torch.stack(base_vals) if base_vals else torch.empty(0, dtype=x.dtype, device=x.device)
            remaining_degradable = torch.stack(rem_vals) if rem_vals else torch.empty(0, dtype=x.dtype, device=x.device)
            gated_rate = torch.stack(gated_vals) if gated_vals else torch.empty(0, dtype=x.dtype, device=x.device)

            # Interpretable mode proxies. The SOH output uses the floor-aware
            # recurrence; these factors are exported to retain a link to LAM/window.
            lam_integral = torch.cumsum(lam_rate * remaining_degradable * delta, dim=0)
            window_integral = torch.cumsum(window_rate * remaining_degradable * delta, dim=0)
            f_lam = torch.exp(-lam_integral).clamp(min=1.0e-6, max=float(self.cfg.soh_max))
            win_scale = torch.exp(-window_integral).clamp(min=1.0e-6, max=float(self.cfg.soh_max))
            lam_damage = (1.0 - f_lam).clamp(min=0.0)
            window_damage = (1.0 - win_scale).clamp(min=0.0)
            r_growth_unit = torch.cumsum(r_rate * delta, dim=0).clamp(min=0.0, max=1.0)
            r_ohm_growth = r_growth_unit * float(self.cfg.r_ohm_delta_max)
            unclipped = soh_base + residual
            numeric_min = max(float(self.cfg.soh_numeric_min), 1.0e-6)
            soh_pred = unclipped.clamp(min=numeric_min, max=float(self.cfg.soh_max))
            active = (unclipped != soh_pred)
            return Assb111SOHOutput(
                SOH_pred=soh_pred,
                SOH_struct=soh_base,
                residual=residual,
                f_LAM_c=f_lam,
                theta_window_scale_c=win_scale,
                R_ohm_eff=torch.as_tensor(float(self.cfg.r_ohm_base), dtype=x.dtype, device=x.device) + r_ohm_growth,
                lam_rate=lam_rate,
                window_rate=window_rate,
                r_rate=r_rate,
                lam_damage=lam_damage,
                window_damage=window_damage,
                r_ohm_growth=r_ohm_growth,
                SOH_base=soh_base,
                SOH_pred_unclipped=unclipped,
                soh_floor=floor,
                soh0=soh0,
                damage_rate_base=base_rate,
                damage_rate_gated=gated_rate,
                remaining_degradable=remaining_degradable,
                active_clamp_mask=active,
            )

        def _dtype(self):
            return torch.float64 if str(self.cfg.dtype).lower() in {"float64", "double", "torch.float64"} else torch.float32

        def _device(self):
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        def save(
            self,
            model_dir: PathLike,
            *,
            feature_columns: Optional[Sequence[str]] = None,
            scaler: Optional[Mapping[str, Any]] = None,
            split_manifest: Optional[Mapping[str, Any]] = None,
            extra: Optional[Mapping[str, Any]] = None,
        ) -> None:
            path = Path(model_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "soh_head.pt")
            payload: Dict[str, Any] = {
                "config": self.cfg.to_dict(),
                "feature_columns": list(feature_columns) if feature_columns is not None else None,
                "scaler": dict(scaler) if scaler is not None else None,
                "split_manifest": dict(split_manifest) if split_manifest is not None else None,
            }
            if extra:
                payload["extra"] = dict(extra)
            with (path / "soh_head_config.json").open("w", encoding="utf-8") as f:
                json.dump(_json_clean(payload), f, ensure_ascii=False, indent=2, sort_keys=True)

        @classmethod
        def load(cls, model_dir: PathLike, *, map_location: Optional[Union[str, "torch.device"]] = None) -> "Assb111SOHHead":
            path = Path(model_dir)
            cfg_path = path / "soh_head_config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(cfg_path)
            with cfg_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            cfg = Assb111SOHHeadConfig.from_dict(payload["config"])
            model = cls(cfg)
            state = torch.load(path / "soh_head.pt", map_location=map_location or "cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                print(f"[Assb111SOHHead.load] unexpected state keys ignored: {unexpected}")
            if missing and str(cfg.model_variant).lower() not in {"legacy", "legacy_accumulative", "accumulative"}:
                print(f"[Assb111SOHHead.load] missing state keys initialized from defaults: {missing}")
            model.eval()
            return model
else:  # pragma: no cover
    class Assb111SOHHead:  # type: ignore
        def __init__(self, *args, **kwargs):
            _require_torch()


def assb111_soh_loss(
    output: Assb111SOHOutput,
    soh_obs,
    *,
    train_mask,
    cfg: Assb111SOHHeadConfig,
    q_obs_ah=None,
    q_ref_ah: Optional[float] = None,
) -> Tuple[Any, Dict[str, float]]:
    """Strict supervised loss for train split only.

    ``train_mask`` must select only train cycles. Test labels may exist in the
    stored dataset, but they must never be selected by this mask.
    """
    _require_torch()
    pred = output.SOH_pred
    dtype = pred.dtype
    device = pred.device
    obs = torch.as_tensor(soh_obs, dtype=dtype, device=device).reshape(-1)
    mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device).reshape(-1)
    if obs.numel() != pred.numel() or mask.numel() != pred.numel():
        raise ValueError("soh_obs, train_mask and prediction length must match")
    valid = mask & torch.isfinite(obs)
    if not torch.any(valid):
        raise RuntimeError("No valid training SOH labels selected by train_mask")
    err = pred[valid] - obs[valid]
    loss_soh = F.huber_loss(pred[valid], obs[valid], delta=float(cfg.huber_delta), reduction="mean")

    if q_obs_ah is not None and float(cfg.w_q) != 0.0:
        q = torch.as_tensor(q_obs_ah, dtype=dtype, device=device).reshape(-1)
        if q_ref_ah is None:
            q_ref = torch.nanmedian(q[valid] / torch.clamp(obs[valid], min=1e-12)).clamp(min=1e-12)
        else:
            q_ref = torch.as_tensor(float(q_ref_ah), dtype=dtype, device=device)
        q_pred = pred * q_ref
        loss_q = F.huber_loss((q_pred[valid] - q[valid]) / q_ref, torch.zeros_like(err), delta=float(cfg.huber_delta), reduction="mean")
    else:
        loss_q = torch.zeros((), dtype=dtype, device=device)

    base_for_smooth = torch.as_tensor(output.SOH_base if output.SOH_base is not None else pred, dtype=dtype, device=device).reshape(-1)
    if pred.numel() >= 3:
        second = base_for_smooth[2:] - 2.0 * base_for_smooth[1:-1] + base_for_smooth[:-2]
        loss_smooth = second.pow(2).mean()
    else:
        loss_smooth = torch.zeros((), dtype=dtype, device=device)

    rate_terms = []
    for r in (output.lam_rate, output.window_rate, output.r_rate, output.damage_rate_gated):
        if r is None:
            continue
        rr = torch.as_tensor(r, dtype=dtype, device=device).reshape(-1)
        if rr.numel() >= 2:
            rate_terms.append((rr[1:] - rr[:-1]).pow(2).mean())
    loss_rate = sum(rate_terms) / max(len(rate_terms), 1) if rate_terms else torch.zeros((), dtype=dtype, device=device)

    if pred.numel() >= 2:
        upward = torch.relu(pred[1:] - pred[:-1] - 1.0e-5)
        loss_mono = upward.pow(2).mean()
    else:
        loss_mono = torch.zeros((), dtype=dtype, device=device)

    residual = torch.as_tensor(output.residual, dtype=dtype, device=device).reshape(-1)
    loss_res = residual.pow(2).mean()
    unclipped = torch.as_tensor(output.SOH_pred_unclipped if output.SOH_pred_unclipped is not None else pred, dtype=dtype, device=device).reshape(-1)
    lower = float(cfg.soh_numeric_min) if str(cfg.model_variant).lower() in {"saturating", "saturating_v2", "sat_v2"} else float(cfg.soh_min)
    bounds = torch.relu(torch.as_tensor(lower, dtype=dtype, device=device) - unclipped).pow(2).mean() + torch.relu(unclipped - torch.as_tensor(float(cfg.soh_max), dtype=dtype, device=device)).pow(2).mean()

    floor = output.soh_floor
    if floor is not None and str(cfg.model_variant).lower() in {"saturating", "saturating_v2", "sat_v2"}:
        floor_t = torch.as_tensor(floor, dtype=dtype, device=device)
        loss_floor_prior = (floor_t - torch.as_tensor(float(cfg.soh_floor_prior), dtype=dtype, device=device)).pow(2)
    else:
        loss_floor_prior = torch.zeros((), dtype=dtype, device=device)

    if str(cfg.model_variant).lower() in {"saturating", "saturating_v2", "sat_v2"} and pred.numel() >= 2:
        slope = base_for_smooth[1:] - base_for_smooth[:-1]
        too_negative = torch.relu(-slope - float(cfg.tail_slope_guard))
        active = torch.as_tensor(output.active_clamp_mask, dtype=dtype, device=device).reshape(-1) if output.active_clamp_mask is not None else torch.zeros_like(pred)
        clamp_penalty = active.pow(2).mean()
        below_penalty = torch.relu(torch.as_tensor(float(cfg.soh_numeric_min), dtype=dtype, device=device) - unclipped).pow(2).mean()
        loss_tail_guard = too_negative.pow(2).mean() + clamp_penalty + below_penalty
    else:
        loss_tail_guard = torch.zeros((), dtype=dtype, device=device)

    total = (
        float(cfg.w_soh) * loss_soh
        + float(cfg.w_q) * loss_q
        + float(cfg.w_smooth) * loss_smooth
        + float(cfg.w_rate) * loss_rate
        + float(cfg.w_monotonic) * loss_mono
        + float(cfg.w_residual) * loss_res
        + float(cfg.w_bounds) * bounds
        + float(cfg.w_floor_prior) * loss_floor_prior
        + float(cfg.w_tail_guard) * loss_tail_guard
    )
    active_mask = output.active_clamp_mask
    active_count = int(torch.as_tensor(active_mask, device=device).bool().sum().detach().cpu().item()) if active_mask is not None else 0
    logs = {
        "loss_total": float(total.detach().cpu()),
        "loss_soh": float(loss_soh.detach().cpu()),
        "loss_q": float(loss_q.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_rate": float(loss_rate.detach().cpu()),
        "loss_monotonic": float(loss_mono.detach().cpu()),
        "loss_residual": float(loss_res.detach().cpu()),
        "loss_bounds": float(bounds.detach().cpu()),
        "loss_floor_prior": float(loss_floor_prior.detach().cpu()),
        "loss_tail_guard": float(loss_tail_guard.detach().cpu()),
        "active_clamp_count": active_count,
        "n_train_valid": int(valid.detach().cpu().sum().item()),
        "train_mae": float(torch.mean(torch.abs(err)).detach().cpu()),
    }
    if floor is not None:
        logs["soh_floor"] = float(torch.as_tensor(floor, dtype=dtype, device=device).detach().cpu())
    if output.soh0 is not None:
        logs["soh0"] = float(torch.as_tensor(output.soh0, dtype=dtype, device=device).detach().cpu())
    return total, logs


def _safe_corr(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return float("nan")
    o = obs[mask]
    p = pred[mask]
    if np.nanstd(o) <= 1e-15 or np.nanstd(p) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(o, p)[0, 1])


def _safe_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return float("nan")
    o = obs[mask]
    p = pred[mask]
    ss_res = float(np.sum((p - o) ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    if ss_tot <= 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def soh_metrics(obs: Sequence[float], pred: Sequence[float]) -> Dict[str, float]:
    obs_arr = np.asarray(obs, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    out: Dict[str, float] = {"n": int(np.sum(mask))}
    if not mask.any():
        out.update(SOH_MAE=float("nan"), SOH_RMSE=float("nan"), SOH_BIAS=float("nan"), SOH_R2=float("nan"), SOH_corr=float("nan"), NMAE=float("nan"), NRMSE=float("nan"))
        return out
    err = pred_arr[mask] - obs_arr[mask]
    denom = float(np.nanmax(obs_arr[mask]) - np.nanmin(obs_arr[mask]))
    if denom <= 1e-30:
        denom = 1.0
    out.update(
        SOH_MAE=float(np.mean(np.abs(err))),
        SOH_RMSE=float(np.sqrt(np.mean(err ** 2))),
        SOH_BIAS=float(np.mean(err)),
        SOH_R2=_safe_r2(obs_arr, pred_arr),
        SOH_corr=_safe_corr(obs_arr, pred_arr),
        NMAE=float(np.mean(np.abs(err)) / denom),
        NRMSE=float(np.sqrt(np.mean(err ** 2)) / denom),
        SOH_obs_min=float(np.nanmin(obs_arr[mask])),
        SOH_obs_max=float(np.nanmax(obs_arr[mask])),
        SOH_pred_min=float(np.nanmin(pred_arr[mask])),
        SOH_pred_max=float(np.nanmax(pred_arr[mask])),
    )
    return out


def metrics_by_split(frame, *, obs_col: str = "SOH_obs", pred_col: str = "SOH_pred", split_col: str = "split") -> Dict[str, Dict[str, float]]:
    data = frame.copy()
    result: Dict[str, Dict[str, float]] = {"all": soh_metrics(data[obs_col], data[pred_col])}
    if split_col in data.columns:
        for split in sorted(str(s) for s in data[split_col].dropna().unique()):
            sub = data[data[split_col].astype(str) == split]
            result[split] = soh_metrics(sub[obs_col], sub[pred_col])
    return result


def prediction_frame_from_output(base_frame, output: Assb111SOHOutput):
    import pandas as pd

    def cpu(x):
        if torch is not None and torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def col(x, n: int):
        arr = cpu(x)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=float)
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=float)
        if arr.size != n:
            raise ValueError(f"prediction output column has length {arr.size}, expected {n}")
        return arr

    out = base_frame.copy()
    n = len(out)
    out["SOH_pred"] = col(output.SOH_pred, n)
    out["SOH_struct"] = col(output.SOH_struct, n)
    out["SOH_base"] = col(output.SOH_base if output.SOH_base is not None else output.SOH_struct, n)
    out["SOH_pred_unclipped"] = col(output.SOH_pred_unclipped if output.SOH_pred_unclipped is not None else output.SOH_pred, n)
    out["soh_floor"] = col(output.soh_floor if output.soh_floor is not None else np.nan, n)
    out["soh0"] = col(output.soh0 if output.soh0 is not None else np.nan, n)
    out["soh_residual"] = col(output.residual, n)
    out["f_LAM_c"] = col(output.f_LAM_c, n)
    out["theta_window_scale_c"] = col(output.theta_window_scale_c, n)
    out["R_ohm_eff"] = col(output.R_ohm_eff, n)
    out["lam_rate"] = col(output.lam_rate, n)
    out["window_rate"] = col(output.window_rate, n)
    out["r_rate"] = col(output.r_rate, n)
    out["damage_rate_base"] = col(output.damage_rate_base if output.damage_rate_base is not None else np.nan, n)
    out["damage_rate_gated"] = col(output.damage_rate_gated if output.damage_rate_gated is not None else np.nan, n)
    out["remaining_degradable"] = col(output.remaining_degradable if output.remaining_degradable is not None else np.nan, n)
    out["active_clamp_mask"] = col(output.active_clamp_mask if output.active_clamp_mask is not None else np.zeros(n), n).astype(bool)
    out["lam_damage"] = col(output.lam_damage, n)
    out["window_damage"] = col(output.window_damage, n)
    out["r_ohm_growth"] = col(output.r_ohm_growth, n)
    return out


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if torch is not None and torch.is_tensor(x):
        return _json_clean(x.detach().cpu().numpy())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def save_json(obj: Mapping[str, Any], path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "Assb111SOHHeadConfig",
    "Assb111SOHOutput",
    "Assb111SOHHead",
    "assb111_soh_loss",
    "soh_metrics",
    "metrics_by_split",
    "prediction_frame_from_output",
    "save_json",
]
