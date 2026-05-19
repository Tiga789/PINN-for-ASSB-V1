# -*- coding: utf-8 -*-
"""SOH heads for ASSB ModelFin_111/112 strict30 experiments.

This replaces the previous ASSB-111 SOH head with a backward-compatible module
that keeps ``legacy_accumulative`` and ``saturating_v2`` while adding three D7
variants:

``robust_saturating``
    Saturating_v2 recurrence with feature dropout, Huber loss support, and
    stronger diagnostics for multi-seed sweeps.
``latent_health_ode``
    A low-dimensional health-state recurrence.  It is still cycle-level and
    train/val selected, but makes the health state explicit for future coupling.
``ensemble_distilled``
    Student architecture compatible with teacher-distillation workflows.  The
    trainer may add a teacher target; the model itself remains a single head.

The file is intentionally independent from ModelFin_107A state inference.  It is
safe to use for SOH-only strict30 training and for ModelFin_112A L1 packaging.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np
import pandas as pd

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
    model_variant: str = "saturating_v2"

    # Legacy accumulative model parameters.
    rate_scale: float = 1.0e-3
    residual_bound: float = 0.015
    lam_max: float = 0.60
    window_loss_max: float = 0.45
    r_ohm_delta_max: float = 250.0
    r_ohm_base: float = 105.0
    soh_min: float = 0.40
    soh_max: float = 1.05

    # Saturating_v2 / robust parameters.
    floor_min: float = 0.65
    floor_max: float = 0.85
    soh_floor_prior: float = 0.72
    soh0_min: float = 0.94
    soh0_max: float = 1.03
    damage_rate_scale: float = 5.0e-4
    gate_gamma: float = 1.0
    soh_numeric_min: float = 0.60
    tail_slope_guard: float = 0.0020

    # Robust/latent additions.
    feature_dropout: float = 0.05
    rate_tv_target: float = 0.0
    latent_dim: int = 2
    latent_scale: float = 0.03
    distill_weight: float = 0.0

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
    w_rate_tv: float = 0.02
    w_distill: float = 0.0
    dtype: str = "float64"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Assb111SOHHeadConfig":
        dd = dict(d)
        if "model_variant" not in dd:
            dd["model_variant"] = "legacy_accumulative"
        # Backward compatibility with old CLI/config spellings.
        if "soh_floor_total" in dd and "soh_floor_prior" not in dd:
            dd["soh_floor_prior"] = dd.pop("soh_floor_total")
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


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if torch is not None and isinstance(x, torch.Tensor):
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
    SOH_base: Any = None
    SOH_pred_unclipped: Any = None
    soh_floor: Any = None
    soh0: Any = None
    damage_rate_base: Any = None
    damage_rate_gated: Any = None
    remaining_degradable: Any = None
    active_clamp_mask: Any = None
    latent_health: Any = None


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
            self.latent_head = nn.Linear(in_dim, max(1, int(cfg.latent_dim)))
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
            nn.init.constant_(self.rate_head.bias, -2.0)
            nn.init.zeros_(self.residual_head.bias)
            nn.init.zeros_(self.latent_head.bias)

        def _dtype(self):
            return torch.float64 if str(self.cfg.dtype).lower() in {"float64", "double", "torch.float64"} else torch.float32

        def _device(self):
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

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
            return x, torch.clamp(delta, min=0.0)

        def _encoded_rates(self, x, *, feature_dropout: float = 0.0):
            if self.training and float(feature_dropout) > 0:
                x = F.dropout(x, p=float(feature_dropout), training=True)
            h = self.encoder(x)
            rates_raw = self.rate_head(h)
            residual = float(self.cfg.residual_bound) * torch.tanh(self.residual_head(h).reshape(-1))
            latent = torch.tanh(self.latent_head(h)) * float(self.cfg.latent_scale)
            return h, rates_raw, residual, latent

        def forward(self, x, *, delta_cycle=None) -> Assb111SOHOutput:
            variant = str(self.cfg.model_variant).strip().lower()
            if variant in {"legacy", "legacy_accumulative", "accumulative", "stageb_like"}:
                return self._forward_legacy(x, delta_cycle=delta_cycle)
            if variant in {"saturating", "saturating_v2", "sat_v2"}:
                return self._forward_saturating(x, delta_cycle=delta_cycle, robust=False, latent=False)
            if variant in {"robust", "robust_saturating", "robust_saturating_v2"}:
                return self._forward_saturating(x, delta_cycle=delta_cycle, robust=True, latent=False)
            if variant in {"latent", "latent_health", "latent_health_ode"}:
                return self._forward_saturating(x, delta_cycle=delta_cycle, robust=True, latent=True)
            if variant in {"distill", "ensemble_distilled", "student"}:
                return self._forward_saturating(x, delta_cycle=delta_cycle, robust=True, latent=False)
            raise ValueError(f"Unsupported Assb111SOHHeadConfig.model_variant={self.cfg.model_variant!r}")

        def _forward_legacy(self, x, *, delta_cycle=None) -> Assb111SOHOutput:
            x, delta = self._prepare_inputs(x, delta_cycle)
            _h, rates_raw, residual, latent = self._encoded_rates(x)
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
                latent_health=latent,
            )

        def _floor_soh0(self, x):
            floor = torch.as_tensor(float(self.cfg.floor_min), dtype=x.dtype, device=x.device) + torch.sigmoid(self.floor_raw.to(dtype=x.dtype, device=x.device)) * (
                float(self.cfg.floor_max) - float(self.cfg.floor_min)
            )
            soh0 = torch.as_tensor(float(self.cfg.soh0_min), dtype=x.dtype, device=x.device) + torch.sigmoid(self.soh0_raw.to(dtype=x.dtype, device=x.device)) * (
                float(self.cfg.soh0_max) - float(self.cfg.soh0_min)
            )
            return floor, soh0

        def _forward_saturating(self, x, *, delta_cycle=None, robust: bool = False, latent: bool = False) -> Assb111SOHOutput:
            x, delta = self._prepare_inputs(x, delta_cycle)
            fd = float(self.cfg.feature_dropout) if robust else 0.0
            _h, rates_raw, residual, latent_z = self._encoded_rates(x, feature_dropout=fd)
            rates = F.softplus(rates_raw)
            lam_rate = rates[:, 0] * float(self.cfg.damage_rate_scale)
            window_rate = rates[:, 1] * float(self.cfg.damage_rate_scale)
            r_rate = rates[:, 2] * float(self.cfg.damage_rate_scale)
            base_rate = lam_rate + window_rate
            if latent:
                # Latent state acts as a bounded, feature-conditioned rate adapter.
                latent_adapter = 1.0 + torch.tanh(latent_z[:, 0])
                base_rate = torch.clamp(base_rate * latent_adapter, min=0.0)
            floor, soh0 = self._floor_soh0(x)
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
                latent_health=latent_z,
            )

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
            state_path = path / "soh_head.pt"
            if not state_path.exists():
                # packaged unified models may store selected_model.pt only
                state_path = path / "selected_model.pt"
            state = torch.load(state_path, map_location=map_location or "cpu")
            if isinstance(state, Mapping) and "soh_head_state_dict" in state:
                state = state["soh_head_state_dict"]
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


def _masked_tensor(values, mask):
    return values[mask]


def assb111_soh_loss(
    output: Assb111SOHOutput,
    soh_obs,
    *,
    train_mask,
    cfg: Assb111SOHHeadConfig,
    q_obs_ah=None,
    q_ref_ah: Optional[float] = None,
    teacher_soh=None,
) -> Tuple[Any, Dict[str, float]]:
    """Strict supervised loss for train split only.

    ``train_mask`` must select only train cycles.  Test labels may exist in the
    dataframe, but they must never be true in this mask.
    """
    _require_torch()
    soh_obs = torch.as_tensor(soh_obs, dtype=output.SOH_pred.dtype, device=output.SOH_pred.device)
    mask = torch.as_tensor(train_mask, dtype=torch.bool, device=output.SOH_pred.device).reshape(-1)
    if mask.numel() != output.SOH_pred.numel():
        raise ValueError("train_mask length must match SOH prediction length")
    if int(mask.sum().item()) == 0:
        raise RuntimeError("No train rows selected for SOH loss")
    pred = output.SOH_pred[mask]
    target = soh_obs[mask]
    finite = torch.isfinite(pred) & torch.isfinite(target)
    if int(finite.sum().item()) == 0:
        raise RuntimeError("No finite SOH labels in train mask")
    pred = pred[finite]
    target = target[finite]
    delta = float(cfg.huber_delta)
    loss_soh = F.huber_loss(pred, target, delta=delta)
    losses = {"loss_soh": loss_soh * float(cfg.w_soh)}

    # Smooth/rate penalties on visible train sequence only.
    train_base = output.SOH_base[mask][finite]
    if train_base.numel() > 2:
        d1 = train_base[1:] - train_base[:-1]
        losses["loss_smooth"] = float(cfg.w_smooth) * torch.mean((d1[1:] - d1[:-1]) ** 2) if d1.numel() > 1 else torch.zeros_like(loss_soh)
        losses["loss_monotonic"] = float(cfg.w_monotonic) * torch.mean(torch.relu(d1) ** 2)
        losses["loss_tail_guard"] = float(cfg.w_tail_guard) * torch.mean(torch.relu(-d1 - float(cfg.tail_slope_guard)) ** 2)
    else:
        z = torch.zeros_like(loss_soh)
        losses["loss_smooth"] = z
        losses["loss_monotonic"] = z
        losses["loss_tail_guard"] = z

    rate = output.damage_rate_gated if output.damage_rate_gated is not None else output.damage_rate_base
    if rate is not None:
        r = rate[mask]
        losses["loss_rate"] = float(cfg.w_rate) * torch.mean(r ** 2)
        if r.numel() > 1:
            losses["loss_rate_tv"] = float(cfg.w_rate_tv) * torch.mean((r[1:] - r[:-1]) ** 2)
        else:
            losses["loss_rate_tv"] = torch.zeros_like(loss_soh)
    else:
        losses["loss_rate"] = torch.zeros_like(loss_soh)
        losses["loss_rate_tv"] = torch.zeros_like(loss_soh)

    losses["loss_residual"] = float(cfg.w_residual) * torch.mean(output.residual[mask] ** 2)
    if output.soh_floor is not None:
        floor = output.soh_floor if torch.is_tensor(output.soh_floor) else torch.as_tensor(output.soh_floor, dtype=pred.dtype, device=pred.device)
        losses["loss_floor_prior"] = float(cfg.w_floor_prior) * (floor - float(cfg.soh_floor_prior)) ** 2
    else:
        losses["loss_floor_prior"] = torch.zeros_like(loss_soh)

    if teacher_soh is not None and float(cfg.w_distill) > 0:
        teacher = torch.as_tensor(teacher_soh, dtype=output.SOH_pred.dtype, device=output.SOH_pred.device)[mask][finite]
        losses["loss_distill"] = float(cfg.w_distill) * F.huber_loss(pred, teacher, delta=delta)
    else:
        losses["loss_distill"] = torch.zeros_like(loss_soh)

    total = sum(losses.values())
    logs = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    logs["loss_total"] = float(total.detach().cpu().item())
    if output.active_clamp_mask is not None:
        logs["active_clamp_count"] = int(torch.as_tensor(output.active_clamp_mask)[mask].sum().detach().cpu().item())
    else:
        logs["active_clamp_count"] = 0
    return total, logs


def soh_metrics(obs: Sequence[float], pred: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(obs, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if int(np.sum(m)) == 0:
        return {"n": 0, "SOH_MAE": math.nan, "SOH_RMSE": math.nan, "SOH_R2": math.nan, "SOH_corr": math.nan, "SOH_BIAS": math.nan}
    yy = y[m]
    pp = p[m]
    err = pp - yy
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    denom = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - float(np.sum(err * err)) / denom if denom > 1e-12 else math.nan
    corr = float(np.corrcoef(yy, pp)[0, 1]) if len(yy) > 1 and np.std(yy) > 1e-12 and np.std(pp) > 1e-12 else math.nan
    rng = float(np.nanmax(yy) - np.nanmin(yy)) if len(yy) else math.nan
    return {
        "n": int(len(yy)),
        "SOH_MAE": mae,
        "SOH_RMSE": rmse,
        "SOH_NMAE": mae / rng if rng and np.isfinite(rng) and rng > 1e-12 else math.nan,
        "SOH_NRMSE": rmse / rng if rng and np.isfinite(rng) and rng > 1e-12 else math.nan,
        "SOH_R2": r2,
        "SOH_corr": corr,
        "SOH_BIAS": float(np.mean(err)),
    }


def metrics_by_split(frame: pd.DataFrame, *, obs_col: str = "SOH_obs", pred_col: str = "SOH_pred") -> Dict[str, Dict[str, float]]:
    if "split" not in frame.columns:
        return {"all": soh_metrics(frame[obs_col].to_numpy(), frame[pred_col].to_numpy())}
    out: Dict[str, Dict[str, float]] = {}
    for split, g in frame.groupby(frame["split"].astype(str).str.lower(), sort=True):
        out[str(split)] = soh_metrics(g[obs_col].to_numpy(), g[pred_col].to_numpy())
    out["all"] = soh_metrics(frame[obs_col].to_numpy(), frame[pred_col].to_numpy())
    return out


def prediction_frame_from_output(frame: pd.DataFrame, output: Assb111SOHOutput) -> pd.DataFrame:
    out = frame.copy()
    def _np(x):
        if torch is not None and torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    out["SOH_pred"] = _np(output.SOH_pred).reshape(-1)
    out["SOH_struct"] = _np(output.SOH_struct).reshape(-1)
    out["SOH_base"] = _np(output.SOH_base if output.SOH_base is not None else output.SOH_struct).reshape(-1)
    out["SOH_residual"] = _np(output.residual).reshape(-1)
    if output.soh_floor is not None:
        val = _np(output.soh_floor)
        out["soh_floor"] = float(val.reshape(-1)[0]) if val.size else np.nan
    if output.damage_rate_gated is not None:
        out["damage_rate_gated"] = _np(output.damage_rate_gated).reshape(-1)
    if output.remaining_degradable is not None:
        out["remaining_degradable"] = _np(output.remaining_degradable).reshape(-1)
    if output.active_clamp_mask is not None:
        out["active_clamp_mask"] = _np(output.active_clamp_mask).reshape(-1).astype(bool)
    return out


__all__ = [
    "Assb111SOHHeadConfig",
    "Assb111SOHHead",
    "Assb111SOHOutput",
    "assb111_soh_loss",
    "soh_metrics",
    "metrics_by_split",
    "prediction_frame_from_output",
    "save_json",
]
