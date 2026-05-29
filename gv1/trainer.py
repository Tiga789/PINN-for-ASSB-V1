"""Trainer for the D9.5.1 GV1 trend-preserving rare-regime PINN smoke.

D9.5.1 keeps the one-profile 40/200/500 ks diagnostics and the auditable preset
layer, but changes D9.5 into a trend-first warmup schedule.  Ordinary voltage,
range and correlation losses act from epoch 1; low/high-tail, ultra-quantile,
coverage and event penalties are ramped in gradually.  It still does not modify
the old ASSB ``main.py`` / ``util/*`` stack.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .losses import GV1LossComputer, LossWeights, make_optimizer
from .model import ConditionedEffectiveSPMPINN, ModelConfig, count_trainable_parameters
from .output_transform import (
    GV1OutputTransform,
    OutputTransformConfig,
    compute_cbar_baselines_numpy,
)


@dataclass
class TrainerConfig:
    solution_npz: str
    output_dir: str
    epochs: int = 300
    batch_size: int = 2048
    lr: float = 2e-3
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    max_time_points: int = 4096
    time_window_s: float | None = None
    start_time_s: float | None = None
    prediction_time_points: int = 2048
    prediction_radial_points: int = 64
    log_every: int = 25
    save_prediction: bool = True
    nominal_capacity_Ah: float = 2.0
    current_scale_A: float | None = None
    event_sampling_mix: float = 0.55
    sample_weight_exponent: float = 1.0
    low_voltage_threshold_V: float = 2.75
    high_voltage_threshold_V: float = 4.10
    low_voltage_quantile: float = 0.08
    high_voltage_quantile: float = 0.92
    high_current_quantile: float = 0.90
    transition_current_delta_quantile: float = 0.90
    temperature_extreme_quantile: float = 0.90
    rare_loss_warmup_start_frac: float = 0.30
    rare_loss_warmup_full_frac: float = 0.85
    rare_loss_warmup_power: float = 1.25
    rare_loss_start_scale: float = 0.05
    rare_loss_final_scale: float = 1.0
    rare_sample_start_scale: float = 0.30
    rare_sample_final_scale: float = 0.80
    model: dict[str, Any] = field(default_factory=dict)
    transform: dict[str, Any] = field(default_factory=dict)
    losses: dict[str, Any] = field(default_factory=dict)
    profile_adaptive_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str) -> torch.device:
    key = (device or "auto").lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(key)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _finite_1d(x: Any, n: int, default: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full(n, float(default), dtype=np.float64)
    arr = np.asarray(x).reshape(-1)
    if len(arr) != n:
        return np.full(n, float(default), dtype=np.float64)
    arr = arr.astype(np.float64, copy=False)
    arr = np.where(np.isfinite(arr), arr, float(default))
    return arr


def _uniform_subsample_indices(n: int, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(np.int64))


class GV1ReplayDataset:
    """In-memory short-window view of one replay profile NPZ."""

    def __init__(self, arrays: Mapping[str, np.ndarray], cfg: TrainerConfig) -> None:
        if "t_global_s" not in arrays or "I_profile" not in arrays:
            raise ValueError("solution_npz must contain t_global_s and I_profile")
        t_raw = np.asarray(arrays["t_global_s"], dtype=np.float64).reshape(-1)
        i_raw = np.asarray(arrays["I_profile"], dtype=np.float64).reshape(-1)
        if len(t_raw) != len(i_raw):
            raise ValueError(f"t_global_s and I_profile length mismatch: {len(t_raw)} vs {len(i_raw)}")
        n0 = len(t_raw)
        v_raw = _finite_1d(arrays.get("voltage_exp"), n0, default=np.nan)
        temp_raw = _finite_1d(arrays.get("temperature_C"), n0, default=25.0)

        mask = np.isfinite(t_raw) & np.isfinite(i_raw) & np.isfinite(v_raw)
        if mask.sum() < 2:
            raise ValueError("Need at least two finite rows with t/I/voltage for D9 smoke training")
        t = t_raw[mask]
        i = i_raw[mask]
        v = v_raw[mask]
        temp = temp_raw[mask]

        order = np.argsort(t, kind="mergesort")
        t, i, v, temp = t[order], i[order], v[order], temp[order]
        if cfg.start_time_s is not None:
            keep = t >= float(cfg.start_time_s)
            t, i, v, temp = t[keep], i[keep], v[keep], temp[keep]
        if cfg.time_window_s is not None and len(t):
            t0 = float(t[0])
            keep = t <= t0 + float(cfg.time_window_s)
            t, i, v, temp = t[keep], i[keep], v[keep], temp[keep]
        if len(t) < 2:
            raise ValueError("Selected time window contains fewer than two points")

        idx = _uniform_subsample_indices(len(t), int(cfg.max_time_points))
        self.t_s = t[idx].astype(np.float64)
        self.current_A = i[idx].astype(np.float32)
        self.voltage_exp = v[idx].astype(np.float32)
        self.temperature_C = temp[idx].astype(np.float32)
        self.n_time = int(len(self.t_s))

        self.t0_s = float(np.nanmin(self.t_s))
        self.t1_s = float(np.nanmax(self.t_s))
        self.time_scale_s = max(self.t1_s - self.t0_s, 1.0)
        self.t_norm = ((self.t_s - self.t0_s) / self.time_scale_s).astype(np.float32)

        max_abs_i = float(np.nanmax(np.abs(self.current_A))) if len(self.current_A) else 1.0
        self.current_scale_A = float(cfg.current_scale_A) if cfg.current_scale_A else max(max_abs_i, 1e-6)
        self.current_norm = (self.current_A / self.current_scale_A).astype(np.float32)
        self.temperature_norm = ((self.temperature_C - 25.0) / 25.0).astype(np.float32)

        # Build transform config early so voltage-range estimation can be driven
        # by CLI/config.  D9.1 defaults to profile_minmax because the v1
        # percentile estimator compressed the measured 2.5--4.2 V envelope.
        tcfg_data = dict(cfg.transform)
        tcfg_data.update(
            nominal_capacity_Ah=float(cfg.nominal_capacity_Ah),
            current_scale_A=float(self.current_scale_A),
        )
        tmp_transform_config = OutputTransformConfig.from_mapping(tcfg_data)
        v_finite = self.voltage_exp[np.isfinite(self.voltage_exp)]
        if len(v_finite):
            strategy = str(tmp_transform_config.voltage_range_strategy).lower().strip()
            if strategy == "fixed":
                voltage_min = float(tmp_transform_config.voltage_min_V)
                voltage_max = float(tmp_transform_config.voltage_max_V)
            elif strategy == "percentile":
                voltage_min = float(np.nanpercentile(v_finite, float(tmp_transform_config.voltage_low_percentile)))
                voltage_max = float(np.nanpercentile(v_finite, float(tmp_transform_config.voltage_high_percentile)))
            else:
                voltage_min = float(np.nanmin(v_finite))
                voltage_max = float(np.nanmax(v_finite))
            voltage_min -= float(tmp_transform_config.voltage_margin_V)
            voltage_max += float(tmp_transform_config.voltage_margin_V)
        else:
            voltage_min, voltage_max = 2.5, 4.2

        # D9.2 keeps only a wide physical guard.  Unlike D9.1, this does not
        # become a hard clamp for phis_c; it only determines affine scaling and
        # the soft guardrail loss.
        voltage_min = float(np.clip(voltage_min, float(tmp_transform_config.voltage_floor_V), 4.0))
        voltage_max = float(np.clip(voltage_max, 3.0, float(tmp_transform_config.voltage_ceil_V)))
        if voltage_max <= voltage_min + 0.2:
            voltage_min, voltage_max = 2.5, 4.2
        voltage_center = 0.5 * (voltage_min + voltage_max)
        voltage_span = max(voltage_max - voltage_min, 0.5)
        voltage_std = float(np.nanstd(v_finite)) if len(v_finite) else 0.35
        voltage_guard_low = min(float(tmp_transform_config.voltage_guard_low_V), voltage_min - 0.10)
        voltage_guard_high = max(float(tmp_transform_config.voltage_guard_high_V), voltage_max + 0.10)

        tcfg_data.update(
            voltage_min_V=voltage_min,
            voltage_max_V=voltage_max,
            voltage_center_V=voltage_center,
            voltage_span_V=voltage_span,
            voltage_std_V=voltage_std,
            voltage_guard_low_V=voltage_guard_low,
            voltage_guard_high_V=voltage_guard_high,
        )
        self.transform_config = OutputTransformConfig.from_mapping(tcfg_data)
        cbar = compute_cbar_baselines_numpy(self.t_s, self.current_A, self.transform_config)
        self.q_net_Ah = cbar["q_net_Ah_replay"]
        self.cbar_a = cbar["cbar_a_norm_replay"]
        self.cbar_c = cbar["cbar_c_norm_replay"]

        self.condition = self._build_condition_vector(voltage_min, voltage_max).astype(np.float32)
        self._build_event_weights(cfg)

    def _build_condition_vector(self, voltage_min: float, voltage_max: float) -> np.ndarray:
        cap = float(self.transform_config.nominal_capacity_Ah)
        mean_temp = float(np.nanmean(self.temperature_C)) if len(self.temperature_C) else 25.0
        std_temp = float(np.nanstd(self.temperature_C)) if len(self.temperature_C) else 0.0
        has_charge = 1.0 if np.nanmax(self.current_A) > 0 else 0.0
        has_discharge = 1.0 if np.nanmin(self.current_A) < 0 else 0.0
        approx_c_rate = self.current_scale_A / max(cap, 1e-6)
        return np.asarray(
            [
                cap / 5.0,
                approx_c_rate / 5.0,
                float(voltage_min) / 5.0,
                float(voltage_max) / 5.0,
                mean_temp / 60.0,
                std_temp / 30.0,
                has_charge,
                has_discharge,
            ],
            dtype=np.float32,
        )


    def _build_event_weights(self, cfg: TrainerConfig) -> None:
        """Create regime weights/probabilities for event-aware sampling.

        The markers are intentionally multi-source: fixed low/high-voltage tails,
        dynamic voltage quantiles, high-current regions, current transitions and
        temperature extremes.  This avoids hard-coding D9.3 as only a low-voltage
        fix while still giving rare low-voltage samples enough visibility.
        """
        v = self.voltage_exp.astype(np.float64)
        I = self.current_A.astype(np.float64)
        T = self.temperature_C.astype(np.float64)
        finite_v = v[np.isfinite(v)]
        if len(finite_v) == 0:
            q_low, q_high = 2.75, 4.10
        else:
            q_low = float(np.nanquantile(finite_v, float(cfg.low_voltage_quantile)))
            q_high = float(np.nanquantile(finite_v, float(cfg.high_voltage_quantile)))
        low_thr = max(float(cfg.low_voltage_threshold_V), q_low)
        high_thr = min(float(cfg.high_voltage_threshold_V), q_high)
        self.low_voltage_marker = ((v <= low_thr) | (v <= float(cfg.low_voltage_threshold_V))).astype(np.float32)
        self.high_voltage_marker = ((v >= high_thr) | (v >= float(cfg.high_voltage_threshold_V))).astype(np.float32)

        abs_i = np.abs(I)
        i_thr = float(np.nanquantile(abs_i[np.isfinite(abs_i)], float(cfg.high_current_quantile))) if np.isfinite(abs_i).any() else 0.0
        self.high_current_marker = (abs_i >= max(i_thr, 1e-12)).astype(np.float32)

        dI = np.zeros_like(I, dtype=np.float64)
        if len(I) > 1:
            dI[1:] = np.abs(np.diff(I))
        dI_thr = float(np.nanquantile(dI[np.isfinite(dI)], float(cfg.transition_current_delta_quantile))) if np.isfinite(dI).any() else 0.0
        self.current_transition_marker = (dI >= max(dI_thr, 1e-12)).astype(np.float32)

        t_dev = np.abs(T - np.nanmedian(T)) if np.isfinite(T).any() else np.zeros_like(T)
        t_thr = float(np.nanquantile(t_dev[np.isfinite(t_dev)], float(cfg.temperature_extreme_quantile))) if np.isfinite(t_dev).any() else 0.0
        self.temperature_event_marker = (t_dev >= max(t_thr, 1e-12)).astype(np.float32)

        event = np.clip(
            self.high_current_marker + self.current_transition_marker + self.temperature_event_marker,
            0.0,
            1.0,
        ).astype(np.float32)
        self.event_marker = event
        # D9.5.1 uses softer event weights than D9.5.  Rare regimes are still
        # visible to the sampler, but the trainer/loss warmup decides when they
        # are allowed to influence optimization strongly.
        w = (
            1.0
            + 3.2 * self.low_voltage_marker
            + 1.2 * self.high_voltage_marker
            + 0.9 * self.high_current_marker
            + 0.9 * self.current_transition_marker
            + 0.5 * self.temperature_event_marker
        ).astype(np.float64)
        expo = max(float(cfg.sample_weight_exponent), 0.0)
        if expo != 1.0:
            w = np.power(w, expo)
        self.sample_weight = w.astype(np.float32)
        n = max(int(self.n_time), 1)
        if np.isfinite(w).all() and float(w.sum()) > 0:
            weighted = w / float(w.sum())
        else:
            weighted = np.full(n, 1.0 / n, dtype=np.float64)
        uniform = np.full(n, 1.0 / n, dtype=np.float64)
        mix = float(np.clip(float(cfg.event_sampling_mix), 0.0, 1.0))
        prob = (1.0 - mix) * uniform + mix * weighted
        prob = prob / float(prob.sum())
        self.sample_probability = prob.astype(np.float64)
        self.event_sampling_summary = {
            "event_sampling_mix": mix,
            "sample_weight_exponent": expo,
            "low_voltage_threshold_effective_V": float(low_thr),
            "high_voltage_threshold_effective_V": float(high_thr),
            "high_current_threshold_A": float(i_thr),
            "transition_current_delta_threshold_A": float(dI_thr),
            "temperature_extreme_threshold_C": float(t_thr),
            "low_voltage_marker_frac": float(np.mean(self.low_voltage_marker)),
            "high_voltage_marker_frac": float(np.mean(self.high_voltage_marker)),
            "high_current_marker_frac": float(np.mean(self.high_current_marker)),
            "current_transition_marker_frac": float(np.mean(self.current_transition_marker)),
            "temperature_event_marker_frac": float(np.mean(self.temperature_event_marker)),
            "sample_weight_minmax": [float(np.nanmin(self.sample_weight)), float(np.nanmax(self.sample_weight))],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "n_time": self.n_time,
            "t0_s": self.t0_s,
            "t1_s": self.t1_s,
            "time_scale_s": self.time_scale_s,
            "current_min_A": float(np.nanmin(self.current_A)),
            "current_max_A": float(np.nanmax(self.current_A)),
            "current_scale_A": self.current_scale_A,
            "temperature_min_C": float(np.nanmin(self.temperature_C)),
            "temperature_max_C": float(np.nanmax(self.temperature_C)),
            "voltage_min_V": float(np.nanmin(self.voltage_exp)),
            "voltage_max_V": float(np.nanmax(self.voltage_exp)),
            "condition_vector": self.condition.tolist(),
            "transform_config": self.transform_config.to_dict(),
            "event_sampling": self.event_sampling_summary,
        }

    def sample_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.choice(self.n_time, size=int(batch_size), replace=True, p=self.sample_probability)
        r = np.random.rand(int(batch_size)).astype(np.float32)
        cond = np.repeat(self.condition.reshape(1, -1), int(batch_size), axis=0)
        def tens(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x, device=device, dtype=torch.float32).reshape(-1, 1)
        return {
            "t_norm": tens(self.t_norm[idx]),
            "r_norm": tens(r),
            "current_A": tens(self.current_A[idx]),
            "current_norm": tens(self.current_norm[idx]),
            "temperature_norm": tens(self.temperature_norm[idx]),
            "cbar_a": tens(self.cbar_a[idx]),
            "cbar_c": tens(self.cbar_c[idx]),
            "voltage_exp": tens(self.voltage_exp[idx]),
            "sample_weight": tens(self.sample_weight[idx]),
            "low_voltage_marker": tens(self.low_voltage_marker[idx]),
            "high_voltage_marker": tens(self.high_voltage_marker[idx]),
            "event_marker": tens(self.event_marker[idx]),
            "condition": torch.as_tensor(cond, device=device, dtype=torch.float32),
        }

    def prediction_time_indices(self, max_points: int) -> np.ndarray:
        return _uniform_subsample_indices(self.n_time, int(max_points))


def _json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class GV1Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = resolve_device(config.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        arrays = _load_npz(config.solution_npz)
        self.dataset = GV1ReplayDataset(arrays, config)

        model_cfg = ModelConfig.from_mapping(config.model)
        if model_cfg.condition_dim != len(self.dataset.condition):
            model_cfg.condition_dim = int(len(self.dataset.condition))
        self.model = ConditionedEffectiveSPMPINN(model_cfg).to(self.device)
        self.transform = GV1OutputTransform(self.dataset.transform_config)
        self.loss_computer = GV1LossComputer(LossWeights.from_mapping(config.losses))
        self.optimizer = make_optimizer(self.model, lr=config.lr, weight_decay=config.weight_decay)
        self.history: list[dict[str, Any]] = []

    def _checkpoint_dict(self, epoch: int, loss: float) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "loss": float(loss),
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.config.to_dict(),
            "transform_config": self.transform.to_dict(),
            "loss_weights": self.loss_computer.to_dict(),
            "trainer_config": self.config.to_dict(),
            "dataset_summary": self.dataset.summary(),
        }

    def save_checkpoint(self, name: str, epoch: int, loss: float) -> None:
        torch.save(self._checkpoint_dict(epoch, loss), self.output_dir / name)

    def _warmup_value(self, epoch: int, *, start_scale: float, final_scale: float) -> float:
        """Return scheduled rare-regime scale for the current epoch."""
        epochs = max(int(self.config.epochs), 1)
        start_epoch = max(1, int(round(float(self.config.rare_loss_warmup_start_frac) * epochs)))
        full_epoch = max(start_epoch + 1, int(round(float(self.config.rare_loss_warmup_full_frac) * epochs)))
        if epoch <= start_epoch:
            z = 0.0
        elif epoch >= full_epoch:
            z = 1.0
        else:
            z = (float(epoch) - float(start_epoch)) / max(float(full_epoch - start_epoch), 1.0)
        z = float(np.clip(z, 0.0, 1.0)) ** max(float(self.config.rare_loss_warmup_power), 0.1)
        return float(start_scale) + (float(final_scale) - float(start_scale)) * z

    def train(self) -> dict[str, Any]:
        best_loss = math.inf
        best_epoch = -1
        for epoch in range(1, int(self.config.epochs) + 1):
            self.model.train()
            batch = self.dataset.sample_batch(self.config.batch_size, self.device)
            rare_loss_scale = self._warmup_value(
                epoch,
                start_scale=float(self.config.rare_loss_start_scale),
                final_scale=float(self.config.rare_loss_final_scale),
            )
            rare_sample_scale = self._warmup_value(
                epoch,
                start_scale=float(self.config.rare_sample_start_scale),
                final_scale=float(self.config.rare_sample_final_scale),
            )
            batch["rare_loss_scale"] = torch.as_tensor(rare_loss_scale, device=self.device, dtype=torch.float32)
            batch["rare_sample_weight_scale"] = torch.as_tensor(rare_sample_scale, device=self.device, dtype=torch.float32)
            self.optimizer.zero_grad(set_to_none=True)
            loss, log = self.loss_computer(self.model, self.transform, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            row = {"epoch": epoch, **log, "lr": float(self.optimizer.param_groups[0]["lr"])}
            self.history.append(row)
            total = float(log["total"])
            if np.isfinite(total) and total < best_loss:
                best_loss = total
                best_epoch = epoch
                self.save_checkpoint("best.pt", epoch, best_loss)
            if epoch == 1 or epoch % max(1, int(self.config.log_every)) == 0 or epoch == int(self.config.epochs):
                print(json.dumps({"epoch": epoch, "total": total, "best_loss": best_loss}, ensure_ascii=False))

        final_loss = float(self.history[-1]["total"]) if self.history else math.nan
        self.save_checkpoint("final.pt", int(self.config.epochs), final_loss)
        _write_history_csv(self.output_dir / "training_history.csv", self.history)
        summary = {
            "ok": True,
            "stage": "GV1 conditioned effective-SPM PINN smoke",
            "status": "trained_smoke",
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "final_loss": final_loss,
            "device": str(self.device),
            "trainable_parameters": count_trainable_parameters(self.model),
            "solution_npz": self.config.solution_npz,
            "output_dir": str(self.output_dir),
            "dataset": self.dataset.summary(),
            "model_config": self.model.config.to_dict(),
            "loss_weights": self.loss_computer.to_dict(),
            "profile_adaptive_diagnostics": dict(self.config.profile_adaptive_diagnostics),
            "note": "D9.5.1 trend-first warmup rare-regime trainer: keep D9.3 event/low-tail channels, keep correlation active from epoch 1, and warm up explicit rare-tail/event/coverage losses. Use 40ks -> 200ks -> 500ks diagnostics before 24-profile training.",
        }
        _json_dump(self.output_dir / "training_summary.json", summary)
        _json_dump(self.output_dir / "config.json", {"trainer": self.config.to_dict(), **summary})
        if self.config.save_prediction:
            # D9.2 saves prediction from best.pt rather than the last epoch.
            # The D9.1 logs often had a lower best_loss than final_loss; using
            # the final weights made profile metrics less stable.
            best_path = self.output_dir / "best.pt"
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=self.device)
                state = ckpt.get("model_state_dict", ckpt)
                self.model.load_state_dict(state)
                summary["prediction_checkpoint"] = "best.pt"
            else:
                summary["prediction_checkpoint"] = "final.pt"
            pred_path = self.save_prediction_npz()
            summary["prediction_npz"] = str(pred_path)
            _json_dump(self.output_dir / "training_summary.json", summary)
        return summary

    @torch.no_grad()
    def save_prediction_npz(self) -> Path:
        self.model.eval()
        idx = self.dataset.prediction_time_indices(self.config.prediction_time_points)
        t_norm = torch.as_tensor(self.dataset.t_norm[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        current_A = torch.as_tensor(self.dataset.current_A[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        current_norm = torch.as_tensor(self.dataset.current_norm[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        temp_norm = torch.as_tensor(self.dataset.temperature_norm[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        cbar_a = torch.as_tensor(self.dataset.cbar_a[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        cbar_c = torch.as_tensor(self.dataset.cbar_c[idx], device=self.device, dtype=torch.float32).reshape(-1, 1)
        condition = torch.as_tensor(
            np.repeat(self.dataset.condition.reshape(1, -1), len(idx), axis=0),
            device=self.device,
            dtype=torch.float32,
        )
        r_surface = torch.ones_like(t_norm)
        raw = self.model(t_norm, r_surface, current_norm, temp_norm, condition)
        out = self.transform(
            raw,
            r_norm=r_surface,
            current_A=current_A,
            current_norm=current_norm,
            cbar_a=cbar_a,
            cbar_c=cbar_c,
            temperature_norm=temp_norm,
            condition=condition,
        )

        radial_points = max(2, int(self.config.prediction_radial_points))
        r_grid = np.linspace(0.0, 1.0, radial_points, dtype=np.float32)
        nt = len(idx)
        # Build a compact time-radial prediction grid.
        tg = np.repeat(self.dataset.t_norm[idx], radial_points).astype(np.float32)
        rg = np.tile(r_grid, nt).astype(np.float32)
        ig = np.repeat(self.dataset.current_norm[idx], radial_points).astype(np.float32)
        iAg = np.repeat(self.dataset.current_A[idx], radial_points).astype(np.float32)
        tempg = np.repeat(self.dataset.temperature_norm[idx], radial_points).astype(np.float32)
        cb_ag = np.repeat(self.dataset.cbar_a[idx], radial_points).astype(np.float32)
        cb_cg = np.repeat(self.dataset.cbar_c[idx], radial_points).astype(np.float32)
        condg = np.repeat(self.dataset.condition.reshape(1, -1), nt * radial_points, axis=0).astype(np.float32)

        # Chunk inference to avoid GPU/CPU memory spikes.
        cs_a_chunks: list[np.ndarray] = []
        cs_c_chunks: list[np.ndarray] = []
        chunk = 65536
        for start in range(0, len(tg), chunk):
            sl = slice(start, min(start + chunk, len(tg)))
            bt = torch.as_tensor(tg[sl], device=self.device).reshape(-1, 1)
            br = torch.as_tensor(rg[sl], device=self.device).reshape(-1, 1)
            bi = torch.as_tensor(ig[sl], device=self.device).reshape(-1, 1)
            biA = torch.as_tensor(iAg[sl], device=self.device).reshape(-1, 1)
            btemp = torch.as_tensor(tempg[sl], device=self.device).reshape(-1, 1)
            bca = torch.as_tensor(cb_ag[sl], device=self.device).reshape(-1, 1)
            bcc = torch.as_tensor(cb_cg[sl], device=self.device).reshape(-1, 1)
            bc = torch.as_tensor(condg[sl], device=self.device)
            raw_grid = self.model(bt, br, bi, btemp, bc)
            out_grid = self.transform(
                raw_grid,
                r_norm=br,
                current_A=biA,
                current_norm=bi,
                cbar_a=bca,
                cbar_c=bcc,
                temperature_norm=btemp,
                condition=bc,
            )
            cs_a_chunks.append(out_grid["cs_a"].detach().cpu().numpy().reshape(-1))
            cs_c_chunks.append(out_grid["cs_c"].detach().cpu().numpy().reshape(-1))

        cs_a = np.concatenate(cs_a_chunks).reshape(nt, radial_points).astype(np.float32)
        cs_c = np.concatenate(cs_c_chunks).reshape(nt, radial_points).astype(np.float32)

        arrays = {
            "t_global_s": self.dataset.t_s[idx].astype(np.float64),
            "t_norm": self.dataset.t_norm[idx].astype(np.float32),
            "I_profile": self.dataset.current_A[idx].astype(np.float32),
            "temperature_C": self.dataset.temperature_C[idx].astype(np.float32),
            "voltage_exp": self.dataset.voltage_exp[idx].astype(np.float32),
            "voltage_exp_pred": out["voltage_exp_pred"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "phis_c_pred": out["phis_c"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "phie_pred": out["phie"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_ocv_baseline": out.get("voltage_ocv_baseline", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_direct_head": out.get("voltage_direct_head", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_ohmic_baseline": out.get("voltage_ohmic_baseline", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_softsign_correction": out.get("voltage_softsign_correction", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_low_tail_correction": out.get("voltage_low_tail_correction", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_event_correction": out.get("voltage_event_correction", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_temperature_correction": out.get("voltage_temperature_correction", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_low_gate": out.get("voltage_low_gate", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_current_event_gate": out.get("voltage_current_event_gate", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_profile_event_gate": out.get("voltage_profile_event_gate", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_base_branch": out.get("voltage_base_branch", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "voltage_event_branch_delta": out.get("voltage_event_branch_delta", out["phis_c"] * 0.0).detach().cpu().numpy().reshape(-1).astype(np.float32),
            "cbar_a_norm_replay": self.dataset.cbar_a[idx].astype(np.float32),
            "cbar_c_norm_replay": self.dataset.cbar_c[idx].astype(np.float32),
            "cbar_a_norm_replay_pred": out["cbar_a_norm_replay"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "cbar_c_norm_replay_pred": out["cbar_c_norm_replay"].detach().cpu().numpy().reshape(-1).astype(np.float32),
            "r_norm": r_grid,
            "cs_a_pred": cs_a,
            "cs_c_pred": cs_c,
        }
        pred_path = self.output_dir / "prediction.npz"
        np.savez_compressed(pred_path, **arrays)
        return pred_path


def run_training(config: TrainerConfig | Mapping[str, Any]) -> dict[str, Any]:
    cfg = config if isinstance(config, TrainerConfig) else TrainerConfig(**dict(config))
    trainer = GV1Trainer(cfg)
    return trainer.train()
