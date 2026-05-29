"""Profile-adaptive presets for GV1 D9.5.1.

D9.5.1 is a **trend-first warmup** repair of D9.5.  The D9.5 40 ks
metrics showed that rare-regime objectives were too aggressive: low-voltage
coverage improved slightly, but B3/B4 trend quality and MAE weakened.  D9.5.1
therefore keeps the D9.3 event-aware voltage transform for all profiles, uses
ordinary voltage/range/correlation losses from epoch 1, and warms up explicit
rare-tail/coverage/ultra-quantile/event losses later in training.

The preset decision is auditable and uses only protocol/current/temperature
metadata from the replay profile path and arrays.  It does not inspect any
validation/test metrics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass
class ProfileDiagnostics:
    solution_npz: str
    requested_mode: str
    selected_mode: str
    protocol_hint: str
    n_time: int
    max_abs_current_A: float
    current_min_A: float
    current_max_A: float
    temperature_min_C: float | None
    temperature_max_C: float | None
    voltage_min_V: float | None
    voltage_max_V: float | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_1d(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def inspect_profile(solution_npz: str | Path, requested_mode: str = "auto") -> ProfileDiagnostics:
    p = Path(solution_npz)
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as d:
        arrays = {k: d[k] for k in d.files}
    I = _safe_1d(arrays, "I_profile")
    T = _safe_1d(arrays, "temperature_C")
    V = _safe_1d(arrays, "voltage_exp")
    n = int(len(np.asarray(arrays.get("t_global_s", I)).reshape(-1)))
    path_text = str(p).replace("\\", "/").lower()
    protocol = "unknown"
    if "_r2.5_" in path_text or "r2.5" in path_text:
        protocol = "R2.5"
    elif "_r3_" in path_text or "r3" in path_text:
        protocol = "R3"
    elif "_2c_" in path_text or "2c" in path_text:
        protocol = "2C"

    max_abs_i = float(np.nanmax(np.abs(I))) if len(I) else 0.0
    i_min = float(np.nanmin(I)) if len(I) else 0.0
    i_max = float(np.nanmax(I)) if len(I) else 0.0
    t_min = float(np.nanmin(T)) if len(T) else None
    t_max = float(np.nanmax(T)) if len(T) else None
    v_min = float(np.nanmin(V)) if len(V) else None
    v_max = float(np.nanmax(V)) if len(V) else None

    req = (requested_mode or "auto").strip().lower()
    if req in {"auto", "trend", "trend_tail", "trend_tail_hybrid", "d951", "hybrid", "warmup", "trend_warmup"}:
        high_temp = bool(t_max is not None and t_max >= 40.0)
        high_current = bool(max_abs_i >= 6.0)
        protocol_event = protocol in {"R2.5", "R3"}
        if protocol_event or high_current or high_temp:
            selected = "event_highrate_trend_warmup"
            reason = (
                "auto: event/high-rate profile; use D9.3 event transform with D9.5.1 warm-scheduled rare losses "
                f"(protocol={protocol}, max_abs_I={max_abs_i:.3g} A, temp_max={t_max})"
            )
        else:
            selected = "lowrate_trend_warmup"
            reason = (
                "auto: low-rate 2C profile; use D9.3 trend branch with D9.5.1 warm-scheduled rare losses "
                f"(protocol={protocol}, max_abs_I={max_abs_i:.3g} A, temp_max={t_max})"
            )
    elif req in {"lowrate", "2c", "b1", "b1_safe", "lowrate_trend_tail", "lowrate_trend_warmup", "smooth", "smooth_2c", "d92", "lowrate_smooth"}:
        selected = "lowrate_trend_warmup"
        reason = f"requested {requested_mode}; D9.5.1 uses D9.3 branch + warm-scheduled rare losses"
    elif req in {"event", "event_highrate", "highrate", "r25", "r3", "event_highrate_trend_tail", "event_highrate_trend_warmup", "d93"}:
        selected = "event_highrate_trend_warmup"
        reason = f"requested {requested_mode}; D9.5.1 uses D9.3 event branch + warm-scheduled rare losses"
    else:
        raise ValueError(
            "profile_adaptive_mode must be auto, d951/trend_warmup, lowrate_trend_warmup, "
            f"event_highrate_trend_warmup, or aliases; got {requested_mode!r}"
        )

    return ProfileDiagnostics(
        solution_npz=str(p),
        requested_mode=requested_mode,
        selected_mode=selected,
        protocol_hint=protocol,
        n_time=n,
        max_abs_current_A=max_abs_i,
        current_min_A=i_min,
        current_max_A=i_max,
        temperature_min_C=t_min,
        temperature_max_C=t_max,
        voltage_min_V=v_min,
        voltage_max_V=v_max,
        reason=reason,
    )


def _base_d93_transform() -> dict[str, Any]:
    """D9.3-style event-aware transform, without D9.4 smooth-branch routing."""
    return {
        "profile_adaptive_mode": "d951_d93_trend_branch",
        "profile_event_gate": 1.0,
        "profile_dynamic_event_gate": 0.20,
        "voltage_range_strategy": "profile_minmax",
        "voltage_margin_V": 0.03,
        "voltage_floor_V": 2.35,
        "voltage_ceil_V": 4.35,
        "voltage_guard_low_V": 2.30,
        "voltage_guard_high_V": 4.40,
        "enable_voltage_hard_clamp": False,
        "phis_c_head_mode": "linear",
        "phis_c_direct_scale": 0.52,
        "phis_c_correction_scale_V": 0.20,
        "low_voltage_gate_center_V": 3.08,
        "low_voltage_gate_width_V": 0.18,
        "phis_c_low_tail_scale_V": 0.82,
        "phis_c_event_scale_V": 0.22,
        "event_current_gain": 0.42,
        "temperature_polarization_scale_V": 0.030,
        "ocv_baseline_mix": 0.18,
        "direct_voltage_mix": 0.82,
        "ohmic_mix": 1.0,
    }


def _base_losses() -> dict[str, Any]:
    """D9.5.1 weaker rare-regime losses; trainer supplies warmup scales."""
    return {
        "voltage_data": 1.0,
        "voltage_tail": 0.30,
        "voltage_bias": 0.10,
        "voltage_range": 0.08,
        "voltage_quantile": 0.22,
        "voltage_asymmetry": 0.10,
        "voltage_event": 0.14,
        "voltage_corr": 0.45,
        "voltage_ultra_quantile": 0.10,
        "voltage_low_coverage": 0.06,
        "voltage_tail_balance": 0.04,
        "voltage_guardrail": 0.01,
        "tail_fraction": 0.22,
        "tail_weight_gain": 1.7,
        "low_tail_extra_gain": 2.4,
        "high_tail_extra_gain": 0.8,
        "event_weight_gain": 0.9,
        "huber_delta_V": 0.08,
        "low_coverage_threshold_V": 2.75,
        "high_coverage_threshold_V": 4.10,
        "coverage_temperature_V": 0.040,
        "center_symmetry": 0.05,
        "surface_flux": 0.05,
        "radial_smooth": 0.002,
        "potential_smooth": 0.0005,
    }


def _warmup_trainer_defaults(*, event: bool) -> dict[str, Any]:
    return {
        "event_sampling_mix": 0.42 if event else 0.34,
        "sample_weight_exponent": 0.80 if event else 0.65,
        "low_voltage_threshold_V": 2.75,
        "high_voltage_threshold_V": 4.10,
        "low_voltage_quantile": 0.08,
        "high_voltage_quantile": 0.92,
        "high_current_quantile": 0.90,
        "transition_current_delta_quantile": 0.90,
        "temperature_extreme_quantile": 0.90,
        "rare_loss_warmup_start_frac": 0.30,
        "rare_loss_warmup_full_frac": 0.85,
        "rare_loss_warmup_power": 1.25,
        "rare_loss_start_scale": 0.05,
        "rare_loss_final_scale": 0.95 if event else 0.90,
        "rare_sample_start_scale": 0.30,
        "rare_sample_final_scale": 0.85 if event else 0.75,
    }


def preset_for_mode(mode: str) -> dict[str, Any]:
    """Return D9.5.1 trainer/transform/loss defaults for a selected mode."""
    key = (mode or "lowrate_trend_warmup").strip().lower()
    if key in {"lowrate_trend_warmup", "lowrate_trend_tail", "lowrate", "2c", "b1", "b1_safe", "trend_tail_hybrid", "d951", "warmup", "trend_warmup"}:
        losses = _base_losses()
        losses.update(
            voltage_corr=0.50,
            voltage_tail=0.26,
            voltage_quantile=0.18,
            voltage_ultra_quantile=0.08,
            voltage_low_coverage=0.045,
            voltage_tail_balance=0.035,
            voltage_event=0.10,
            low_tail_extra_gain=2.0,
            event_weight_gain=0.65,
        )
        transform = _base_d93_transform()
        transform.update(
            profile_adaptive_mode="lowrate_trend_warmup_d951",
            profile_event_gate=1.0,
            profile_dynamic_event_gate=0.08,
            phis_c_low_tail_scale_V=0.78,
            phis_c_event_scale_V=0.18,
            event_current_gain=0.30,
        )
        return {"trainer": _warmup_trainer_defaults(event=False), "transform": transform, "losses": losses}
    if key in {"event_highrate_trend_warmup", "event_highrate_trend_tail", "event_highrate", "event", "highrate", "r25", "r3", "d93"}:
        losses = _base_losses()
        losses.update(
            voltage_corr=0.42,
            voltage_tail=0.32,
            voltage_quantile=0.22,
            voltage_ultra_quantile=0.10,
            voltage_low_coverage=0.055,
            voltage_tail_balance=0.04,
            voltage_event=0.20,
            event_weight_gain=1.05,
        )
        transform = _base_d93_transform()
        transform.update(
            profile_adaptive_mode="event_highrate_trend_warmup_d951",
            profile_event_gate=1.0,
            profile_dynamic_event_gate=0.28,
            phis_c_low_tail_scale_V=0.82,
            phis_c_event_scale_V=0.24,
            event_current_gain=0.48,
        )
        return {"trainer": _warmup_trainer_defaults(event=True), "transform": transform, "losses": losses}
    raise ValueError(f"Unsupported D9.5.1 preset mode: {mode!r}")


def resolve_profile_adaptive_preset(solution_npz: str | Path, requested_mode: str = "auto") -> tuple[ProfileDiagnostics, dict[str, Any]]:
    diagnostics = inspect_profile(solution_npz, requested_mode=requested_mode)
    return diagnostics, preset_for_mode(diagnostics.selected_mode)
