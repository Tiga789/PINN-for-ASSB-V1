from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .metrics import safe_corr
from .schema import normalize_step_labels


@dataclass(frozen=True)
class CycleSegment:
    cycle_id: int
    start: int
    stop: int
    position: str

    @property
    def size(self) -> int:
        return self.stop - self.start


def sanitize_cycle_id(cycle_id: np.ndarray | None, n_time: int) -> np.ndarray:
    if cycle_id is None:
        return np.zeros(n_time, dtype=np.int64)
    arr = np.asarray(cycle_id).reshape(-1)
    if arr.size != n_time:
        return np.zeros(n_time, dtype=np.int64)
    out = np.empty(n_time, dtype=np.int64)
    last = 0
    for i, value in enumerate(arr):
        try:
            number = int(float(value))
        except Exception:
            number = last
        out[i] = number
        last = number
    return out


def cycle_segments(cycle_id: np.ndarray | None, n_time: int) -> list[CycleSegment]:
    ids = sanitize_cycle_id(cycle_id, n_time)
    if n_time == 0:
        return []
    change = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [n_time]])
    unique_count = len(starts)
    segments: list[CycleSegment] = []
    for rank, (start, stop) in enumerate(zip(starts, stops)):
        frac = rank / max(1, unique_count - 1)
        position = "early" if frac < 1 / 3 else "middle" if frac < 2 / 3 else "late"
        segments.append(CycleSegment(int(ids[start]), int(start), int(stop), position))
    return segments


def cumulative_ah(time_s: np.ndarray, current_A: np.ndarray | None, absolute: bool = False) -> np.ndarray:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if current_A is None:
        return np.zeros(t.size, dtype=np.float64)
    i = np.asarray(current_A, dtype=np.float64).reshape(-1)
    n = min(t.size, i.size)
    t = t[:n]
    i = i[:n]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
    signal = np.abs(i) if absolute else i
    return np.cumsum(signal * dt) / 3600.0


def cycle_summary_rows(
    time_s: np.ndarray,
    cycle_id: np.ndarray | None,
    current_A: np.ndarray | None,
    voltage_V: np.ndarray | None,
    temperature_C: np.ndarray | None,
    step_type: np.ndarray | None,
) -> list[dict[str, Any]]:
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    n = t.size
    current = None if current_A is None else np.asarray(current_A, dtype=np.float64).reshape(-1)[:n]
    voltage = None if voltage_V is None else np.asarray(voltage_V, dtype=np.float64).reshape(-1)[:n]
    temp = None if temperature_C is None else np.asarray(temperature_C, dtype=np.float64).reshape(-1)[:n]
    labels = normalize_step_labels(step_type, current, n)
    cum_abs = cumulative_ah(t, current, absolute=True)
    segments = cycle_segments(cycle_id, n)
    rows: list[dict[str, Any]] = []
    for rank, seg in enumerate(segments):
        sl = slice(seg.start, seg.stop)
        tt = t[sl]
        ii = current[sl] if current is not None else None
        vv = voltage[sl] if voltage is not None else None
        temp_seg = temp[sl] if temp is not None else None
        lab = labels[sl]
        dt = np.diff(tt, prepend=tt[0]) if tt.size else np.zeros(0)
        dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
        q_signed = float(np.sum(ii * dt) / 3600.0) if ii is not None else 0.0
        q_abs = float(np.sum(np.abs(ii) * dt) / 3600.0) if ii is not None else 0.0
        q_charge = float(np.sum(np.maximum(ii, 0.0) * dt) / 3600.0) if ii is not None else 0.0
        q_discharge = float(np.sum(np.maximum(-ii, 0.0) * dt) / 3600.0) if ii is not None else 0.0
        duration = float(tt[-1] - tt[0]) if tt.size > 1 else 0.0

        def duration_for(name: str) -> float:
            return float(np.sum(dt[lab == name])) if dt.size else 0.0

        rows.append(
            {
                "cycle_id": seg.cycle_id,
                "cycle_rank": rank,
                "normalized_cycle_index": rank / max(1, len(segments) - 1),
                "cycle_position": seg.position,
                "start_index": seg.start,
                "stop_index": seg.stop,
                "n_points": seg.size,
                "start_time_s": float(tt[0]) if tt.size else float("nan"),
                "end_time_s": float(tt[-1]) if tt.size else float("nan"),
                "duration_s": duration,
                "charge_duration_s": duration_for("charge"),
                "rest_duration_s": duration_for("rest"),
                "discharge_duration_s": duration_for("discharge"),
                "q_signed_Ah": q_signed,
                "q_abs_Ah": q_abs,
                "q_charge_Ah": q_charge,
                "q_discharge_Ah": q_discharge,
                "cumulative_abs_Ah_end": float(cum_abs[seg.stop - 1]) if seg.stop > seg.start else 0.0,
                "efc_proxy_end": float(cum_abs[seg.stop - 1] / max(1e-12, 2.0 * max(q_charge, q_discharge, 1e-12))) if seg.stop > seg.start else 0.0,
                "current_mean_A": _finite_stat(ii, np.mean),
                "current_abs_mean_A": _finite_stat(None if ii is None else np.abs(ii), np.mean),
                "current_abs_max_A": _finite_stat(None if ii is None else np.abs(ii), np.max),
                "voltage_start_V": _first_finite(vv),
                "voltage_end_V": _last_finite(vv),
                "voltage_min_V": _finite_stat(vv, np.min),
                "voltage_max_V": _finite_stat(vv, np.max),
                "voltage_mean_V": _finite_stat(vv, np.mean),
                "temperature_mean_C": _finite_stat(temp_seg, np.mean),
                "temperature_max_C": _finite_stat(temp_seg, np.max),
            }
        )
    return rows


def assign_cycle_position(cycle_id: np.ndarray | None, n_time: int) -> np.ndarray:
    out = np.full(n_time, "unknown", dtype=object)
    for seg in cycle_segments(cycle_id, n_time):
        out[seg.start : seg.stop] = seg.position
    return out


def boundary_rows(
    cycle_id: np.ndarray | None,
    n_time: int,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    state: str,
    case_id: str,
) -> list[dict[str, Any]]:
    segments = cycle_segments(cycle_id, n_time)
    true_arr = np.asarray(true_values, dtype=np.float64)
    pred_arr = np.asarray(pred_values, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    true_range = float(np.nanmax(true_arr) - np.nanmin(true_arr)) if np.any(np.isfinite(true_arr)) else float("nan")
    for left, right in zip(segments[:-1], segments[1:]):
        # Dense S1 casepacks may contain separated early/middle/late windows.
        # Only adjacent source cycles represent a real cycle boundary.
        if right.cycle_id != left.cycle_id + 1:
            continue
        li = left.stop - 1
        ri = right.start
        true_jump = float(np.nanmean(true_arr[ri] - true_arr[li]))
        pred_jump = float(np.nanmean(pred_arr[ri] - pred_arr[li]))
        jump_error = pred_jump - true_jump
        rows.append(
            {
                "case_id": case_id,
                "state": state,
                "left_cycle_id": left.cycle_id,
                "right_cycle_id": right.cycle_id,
                "boundary_index": ri,
                "true_jump": true_jump,
                "pred_jump": pred_jump,
                "jump_error": jump_error,
                "normalized_abs_jump_error": abs(jump_error) / true_range if np.isfinite(true_range) and true_range > 1e-12 else float("nan"),
            }
        )
    return rows


def cycle_bias_rows(
    cycle_id: np.ndarray | None,
    n_time: int,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    state: str,
    case_id: str,
) -> list[dict[str, Any]]:
    true_arr = np.asarray(true_values, dtype=np.float64)
    pred_arr = np.asarray(pred_values, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for seg in cycle_segments(cycle_id, n_time):
        sl = slice(seg.start, seg.stop)
        err = pred_arr[sl] - true_arr[sl]
        rows.append(
            {
                "case_id": case_id,
                "state": state,
                "cycle_id": seg.cycle_id,
                "cycle_position": seg.position,
                "n_points": seg.size,
                "bias": float(np.nanmean(err)),
                "mae": float(np.nanmean(np.abs(err))),
                "rmse": float(np.sqrt(np.nanmean(err**2))),
            }
        )
    return rows


def cycle_bias_trend(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    data = list(rows)
    if len(data) < 2:
        return {"cycle_bias_corr": float("nan"), "cycle_bias_slope": float("nan")}
    x = np.asarray([float(r["cycle_id"]) for r in data], dtype=np.float64)
    y = np.asarray([float(r["bias"]) for r in data], dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 2 or np.std(x) < 1e-12:
        return {"cycle_bias_corr": float("nan"), "cycle_bias_slope": float("nan")}
    design = np.column_stack([x, np.ones_like(x)])
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    return {"cycle_bias_corr": safe_corr(x, y), "cycle_bias_slope": float(coeff[0])}


def _finite_stat(values: np.ndarray | None, func: Any) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(func(arr)) if arr.size else float("nan")


def _first_finite(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    good = arr[np.isfinite(arr)]
    return float(good[0]) if good.size else float("nan")


def _last_finite(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    good = arr[np.isfinite(arr)]
    return float(good[-1]) if good.size else float("nan")
