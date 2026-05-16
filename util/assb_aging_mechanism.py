# -*- coding: utf-8 -*-
"""Low-dimensional monotone aging mechanism for ASSB ModelFin_110.

This module implements Stage-B mechanism learning from ASSB-aging-fix1.  It is
not a free SOH head: it produces physically named slow variables and derives
SOH from them.

Main variables
--------------
``f_LAM_c``
    Effective positive-electrode active material fraction.  Starts at one and
    is constrained to be non-increasing.
``theta_window_scale_c``
    Positive-electrode usable stoichiometry-window scale.  Starts at one and is
    constrained to be non-increasing.
``R_ohm_eff``
    Effective ohmic/contact resistance.  Starts from ``r_ohm_base`` and is
    constrained to be non-decreasing.  Stage B does not use it to drive SOH
    unless ``use_apparent_capacity`` is explicitly enabled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import json

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    Tensor = object  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore

from util.assb_aging_fix1_config import AgingFix1Config, load_aging_config, save_aging_config

PathLike = Union[str, Path]


@dataclass
class AgingProfiles:
    """Container for cycle-level aging profiles.

    All tensor fields have shape ``(n_cycles,)`` unless otherwise noted.
    """

    cycle_id: Tensor
    f_LAM_c: Tensor
    theta_window_scale_c: Tensor
    R_ohm_eff: Tensor
    SOH_struct: Tensor
    Q_pred_Ah: Tensor
    lam_damage: Tensor
    window_damage: Tensor
    r_ohm_growth: Tensor
    raw_rates: Tensor
    lam_rate: Tensor
    window_rate: Tensor
    r_rate: Tensor

    def as_dict(self) -> Dict[str, Tensor]:
        return {
            "cycle_id": self.cycle_id,
            "f_LAM_c": self.f_LAM_c,
            "theta_window_scale_c": self.theta_window_scale_c,
            "R_ohm_eff": self.R_ohm_eff,
            "SOH_struct": self.SOH_struct,
            "Q_pred_Ah": self.Q_pred_Ah,
            "lam_damage": self.lam_damage,
            "window_damage": self.window_damage,
            "r_ohm_growth": self.r_ohm_growth,
            "raw_rates": self.raw_rates,
            "lam_rate": self.lam_rate,
            "window_rate": self.window_rate,
            "r_rate": self.r_rate,
        }


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for util.assb_aging_mechanism")


def _dtype_from_config(cfg: AgingFix1Config):
    _require_torch()
    return torch.float64 if str(cfg.dtype).lower() in {"float64", "double", "torch.float64"} else torch.float32


def _make_mlp(in_dim: int, hidden_dim: int, hidden_layers: int, out_dim: int) -> "nn.Sequential":
    _require_torch()
    layers = []
    prev = in_dim
    for _ in range(int(hidden_layers)):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.Tanh())
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class AgingMechanismHead(nn.Module):
    """Monotone cycle-level aging mechanism.

    The network predicts non-negative per-cycle damage rates from normalized
    cycle features.  Cumulative damage is then normalized to zero at the first
    cycle and bounded by trainable amplitudes.  This makes the output physically
    interpretable and prevents the 109-style almost-flat curve caused by small
    fixed initialization.
    """

    def __init__(self, cfg: AgingFix1Config):
        _require_torch()
        super().__init__()
        self.cfg = cfg
        self.feature_dim = int(cfg.feature_dim)
        self.rate_mlp = _make_mlp(self.feature_dim, int(cfg.hidden_dim), int(cfg.hidden_layers), 3)
        self.amp_lam_logit = nn.Parameter(torch.tensor(float(cfg.init_amplitude_logit_lam)))
        self.amp_window_logit = nn.Parameter(torch.tensor(float(cfg.init_amplitude_logit_window)))
        self.amp_rohm_logit = nn.Parameter(torch.tensor(float(cfg.init_amplitude_logit_rohm)))
        self.rate_bias = nn.Parameter(torch.tensor(float(cfg.init_rate_bias)))
        self._reset_parameters()
        self.to(dtype=_dtype_from_config(cfg))

    def _reset_parameters(self) -> None:
        # Small final-layer weights make the initial profile smooth, while the
        # amplitude logits set a non-flat physically plausible starting point.
        for module in self.modules():
            if hasattr(nn, "Linear") and isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        last = None
        for module in self.rate_mlp.modules():
            if isinstance(module, nn.Linear):
                last = module
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    @staticmethod
    def _sort_by_cycle(features: Tensor, cycle_id: Optional[Tensor]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        if cycle_id is None:
            order = torch.arange(features.shape[0], device=features.device)
            return features, order, None
        cycle_id = cycle_id.to(device=features.device)
        order = torch.argsort(cycle_id)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=order.device)
        return features[order], order, inv

    def _bounded_cumulative_damage(self, positive_rate: Tensor, max_loss: float, amplitude_logit: Tensor) -> Tensor:
        # Start at exactly zero damage for the first provided cycle.
        if positive_rate.numel() == 0:
            return positive_rate
        cum = torch.cumsum(positive_rate, dim=0)
        cum = cum - cum[0]
        denom = torch.clamp(cum[-1].detach(), min=torch.as_tensor(self.cfg.eps, dtype=cum.dtype, device=cum.device))
        damage_shape = cum / denom
        amplitude = torch.as_tensor(float(max_loss), dtype=cum.dtype, device=cum.device) * torch.sigmoid(amplitude_logit.to(dtype=cum.dtype, device=cum.device))
        return amplitude * damage_shape

    def forward(self, cycle_features: Tensor, cycle_id: Optional[Tensor] = None, q_ref_ah: Optional[float] = None) -> AgingProfiles:
        _require_torch()
        if not torch.is_tensor(cycle_features):
            cycle_features = torch.as_tensor(cycle_features, dtype=_dtype_from_config(self.cfg))
        if cycle_features.ndim != 2:
            raise ValueError(f"cycle_features must be 2D, got shape {tuple(cycle_features.shape)}")
        if cycle_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: model expects {self.feature_dim}, got {cycle_features.shape[1]}. "
                "Regenerate cycle_table or adjust AGING_FEATURE_DIM."
            )
        features_sorted, order, inv = self._sort_by_cycle(cycle_features, cycle_id)
        raw_sorted = self.rate_mlp(features_sorted) + self.rate_bias.to(dtype=features_sorted.dtype, device=features_sorted.device)
        positive_rates = F.softplus(raw_sorted) + torch.as_tensor(float(self.cfg.min_rate), dtype=features_sorted.dtype, device=features_sorted.device)
        lam_rate = positive_rates[:, 0]
        window_rate = positive_rates[:, 1]
        r_rate = positive_rates[:, 2]

        lam_damage = self._bounded_cumulative_damage(lam_rate, self.cfg.lam_max, self.amp_lam_logit)
        window_damage = self._bounded_cumulative_damage(window_rate, self.cfg.window_loss_max, self.amp_window_logit)
        r_growth = self._bounded_cumulative_damage(r_rate, self.cfg.r_ohm_delta_max, self.amp_rohm_logit)

        f_lam = torch.clamp(1.0 - lam_damage, min=1.0e-6, max=1.0)
        window_scale = torch.clamp(1.0 - window_damage, min=1.0e-6, max=1.0)
        r_ohm = torch.as_tensor(float(self.cfg.r_ohm_base), dtype=features_sorted.dtype, device=features_sorted.device) + r_growth
        soh_struct = torch.clamp(f_lam * window_scale, min=1.0e-6, max=1.05)
        if self.cfg.use_apparent_capacity and self.cfg.apparent_gamma_r != 0.0:
            # Optional only.  Kept weak and explicit to avoid R_ohm/gauge double counting.
            r_norm = r_growth / torch.clamp(torch.as_tensor(float(self.cfg.r_ohm_delta_max), dtype=features_sorted.dtype, device=features_sorted.device), min=1.0e-12)
            soh_struct = torch.clamp(soh_struct - float(self.cfg.apparent_gamma_r) * r_norm, min=1.0e-6, max=1.05)
        q_ref = torch.as_tensor(1.0 if q_ref_ah is None else float(q_ref_ah), dtype=features_sorted.dtype, device=features_sorted.device)
        q_pred = q_ref * soh_struct

        if inv is not None:
            # Return outputs in the original order supplied by the caller.
            raw = raw_sorted[inv]
            lam_rate_out = lam_rate[inv]
            win_rate_out = window_rate[inv]
            r_rate_out = r_rate[inv]
            lam_damage = lam_damage[inv]
            window_damage = window_damage[inv]
            r_growth = r_growth[inv]
            f_lam = f_lam[inv]
            window_scale = window_scale[inv]
            r_ohm = r_ohm[inv]
            soh_struct = soh_struct[inv]
            q_pred = q_pred[inv]
            cycle_out = cycle_id.to(device=features_sorted.device)[order][inv]
        else:
            raw = raw_sorted
            lam_rate_out = lam_rate
            win_rate_out = window_rate
            r_rate_out = r_rate
            cycle_out = torch.arange(cycle_features.shape[0], dtype=torch.long, device=cycle_features.device) if cycle_id is None else cycle_id

        return AgingProfiles(
            cycle_id=cycle_out,
            f_LAM_c=f_lam,
            theta_window_scale_c=window_scale,
            R_ohm_eff=r_ohm,
            SOH_struct=soh_struct,
            Q_pred_Ah=q_pred,
            lam_damage=lam_damage,
            window_damage=window_damage,
            r_ohm_growth=r_growth,
            raw_rates=raw,
            lam_rate=lam_rate_out,
            window_rate=win_rate_out,
            r_rate=r_rate_out,
        )

    def save(self, output_dir: PathLike, *, extra: Optional[Dict[str, Any]] = None) -> None:
        _require_torch()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"state_dict": self.state_dict(), "config": self.cfg.to_dict()}
        if extra:
            payload["extra"] = extra
        torch.save(payload, output_dir / "aging_mechanism.pt")
        torch.save(payload, output_dir / "aging_state.pt")
        save_aging_config(self.cfg, output_dir / "aging_config.json")

    @classmethod
    def load(cls, model_dir_or_file: PathLike, *, map_location: Union[str, "torch.device"] = "cpu") -> "AgingMechanismHead":
        _require_torch()
        path = Path(model_dir_or_file)
        if path.is_dir():
            state_path = path / "aging_mechanism.pt"
            if not state_path.exists():
                state_path = path / "aging_state.pt"
            if not state_path.exists():
                raise FileNotFoundError(f"No aging_mechanism.pt or aging_state.pt found in {path}")
            cfg_path = path / "aging_config.json"
            cfg = load_aging_config(cfg_path) if cfg_path.exists() else None
        else:
            state_path = path
            cfg = None
        payload = torch.load(state_path, map_location=map_location)
        if isinstance(payload, dict) and "config" in payload:
            cfg = AgingFix1Config.from_dict(payload["config"])
        if cfg is None:
            raise RuntimeError(f"Cannot recover AgingFix1Config from {state_path}")
        model = cls(cfg)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(state, strict=True)
        model.to(map_location)
        model.eval()
        return model


def profiles_to_numpy(profiles: AgingProfiles) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in profiles.as_dict().items():
        if torch is not None and torch.is_tensor(value):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value)
    return out


def save_profiles_csv(profiles: AgingProfiles, path: PathLike, *, extra_columns: Optional[Dict[str, Sequence[Any]]] = None) -> None:
    import pandas as pd

    data = profiles_to_numpy(profiles)
    frame = pd.DataFrame({
        "cycle_id": data["cycle_id"].astype(int),
        "f_LAM_c": data["f_LAM_c"],
        "theta_window_scale_c": data["theta_window_scale_c"],
        "R_ohm_eff": data["R_ohm_eff"],
        "SOH_pred": data["SOH_struct"],
        "Q_pred_Ah": data["Q_pred_Ah"],
        "lam_damage": data["lam_damage"],
        "window_damage": data["window_damage"],
        "r_ohm_growth": data["r_ohm_growth"],
        "lam_rate": data["lam_rate"],
        "window_rate": data["window_rate"],
        "r_rate": data["r_rate"],
    })
    if extra_columns:
        for key, value in extra_columns.items():
            frame[key] = list(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


__all__ = [
    "AgingProfiles",
    "AgingMechanismHead",
    "profiles_to_numpy",
    "save_profiles_csv",
]
