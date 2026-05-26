from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import numpy as np
import pandas as pd

from .capacity_integrator import cumulative_charge_discharge_Ah, cumulative_energy_Wh
from .step_classifier import add_step_columns


@dataclass(frozen=True)
class ReplayBuildOptions:
    """Options for constructing a measured-current replay profile."""

    time_zero: str = 'start'  # start | keep
    sort_by: tuple[str, ...] = ()  # preserve original row order by default
    current_threshold_A: float = 1e-9
    repair_nonmonotonic_time: bool = True
    sample_frequency_hz: float | None = 1.0
    infer_step_columns: bool = True
    require_voltage: bool = True
    metadata_columns: tuple[str, ...] = (
        'dataset_id', 'batch_id', 'battery_id', 'cell_id', 'cycle_id', 'step_id', 'step_type',
        'protocol_id', 'observed_control_mode', 'current_input_mode', 'source_file', 'source_format',
    )


@dataclass
class ReplayProfile:
    t_s: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray | None = None
    temperature_C: np.ndarray | None = None
    cycle_id: np.ndarray | None = None
    step_id: np.ndarray | None = None
    step_type: np.ndarray | None = None
    capacity_Ah: np.ndarray | None = None
    q_charge_Ah: np.ndarray | None = None
    q_discharge_Ah: np.ndarray | None = None
    q_net_Ah: np.ndarray | None = None
    throughput_Ah: np.ndarray | None = None
    e_charge_Wh: np.ndarray | None = None
    e_discharge_Wh: np.ndarray | None = None
    e_abs_Wh: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            't_global_s': np.asarray(self.t_s, dtype=float),
            'I_profile': np.asarray(self.current_A, dtype=float),
        }
        optional = {
            'voltage_exp': self.voltage_V,
            'temperature_C': self.temperature_C,
            'cycle_id': self.cycle_id,
            'step_id': self.step_id,
            'step_type': self.step_type,
            'capacity_Ah': self.capacity_Ah,
            'Q_charge_Ah': self.q_charge_Ah,
            'Q_discharge_Ah': self.q_discharge_Ah,
            'Q_net_Ah': self.q_net_Ah,
            'throughput_Ah': self.throughput_Ah,
            'E_charge_Wh': self.e_charge_Wh,
            'E_discharge_Wh': self.e_discharge_Wh,
            'E_abs_Wh': self.e_abs_Wh,
        }
        for key, value in optional.items():
            if value is not None:
                out[key] = np.asarray(value)
        return out

    def summary(self) -> dict[str, Any]:
        t = np.asarray(self.t_s, dtype=float)
        i = np.asarray(self.current_A, dtype=float)
        summary = {
            'n_points': int(len(t)),
            't_start_s': float(np.nanmin(t)) if len(t) else None,
            't_end_s': float(np.nanmax(t)) if len(t) else None,
            'current_min_A': float(np.nanmin(i)) if len(i) else None,
            'current_max_A': float(np.nanmax(i)) if len(i) else None,
            'has_charge': bool(np.nanmax(i) > 0) if len(i) else False,
            'has_discharge': bool(np.nanmin(i) < 0) if len(i) else False,
            'has_rest': bool(np.nanmin(np.abs(i)) <= 1e-9) if len(i) else False,
        }
        if self.cycle_id is not None:
            cyc = pd.to_numeric(pd.Series(self.cycle_id), errors='coerce').dropna()
            if not cyc.empty:
                summary.update({'cycle_min': int(cyc.min()), 'cycle_max': int(cyc.max()), 'cycle_count': int(cyc.nunique())})
        summary['metadata'] = self.metadata
        return summary



def _repair_time_if_needed(t: np.ndarray, options: ReplayBuildOptions, metadata: dict[str, Any]) -> np.ndarray:
    """Repair time if the raw time column is relative-to-step and resets.

    Many cycler exports contain a relative time column that restarts at each
    step.  In measured-current replay the chronological row order matters more
    than sorting by such a column.  If non-monotonic time is detected, this
    function rebuilds a global time axis from the row order and a robust median
    positive time step.  XJTU is nominally 1 Hz, so sample_frequency_hz=1.0 is
    the fallback.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 2 or not options.repair_nonmonotonic_time:
        return t
    dt = np.diff(t)
    has_bad = bool(np.any(~np.isfinite(dt)) or np.any(dt <= 0))
    if not has_bad:
        return t
    pos = dt[np.isfinite(dt) & (dt > 0)]
    if len(pos):
        step = float(np.nanmedian(pos))
    elif options.sample_frequency_hz and options.sample_frequency_hz > 0:
        step = 1.0 / float(options.sample_frequency_hz)
    else:
        step = 1.0
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    metadata['time_repaired'] = True
    metadata['time_repair_reason'] = 'non_monotonic_or_duplicate_time_detected'
    metadata['time_repair_step_s'] = step
    return np.arange(len(t), dtype=float) * step

def _series_or_none(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df:
        return None
    s = df[col]
    if s.isna().all():
        return None
    return s.to_numpy()


def build_replay_profile(df: pd.DataFrame, options: ReplayBuildOptions | None = None) -> ReplayProfile:
    """Build a single measured-current replay profile from a standard table."""
    options = options or ReplayBuildOptions()
    required = ['time_s', 'current_A'] + (['voltage_V'] if options.require_voltage else [])
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f'Missing required columns for replay profile: {missing}')
    work = df.copy()
    if options.infer_step_columns:
        work = add_step_columns(work)
    sort_cols = [c for c in options.sort_by if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols).reset_index(drop=True)
    t = pd.to_numeric(work['time_s'], errors='coerce').to_numpy(dtype=float)
    i = pd.to_numeric(work['current_A'], errors='coerce').to_numpy(dtype=float)
    v = pd.to_numeric(work['voltage_V'], errors='coerce').to_numpy(dtype=float) if 'voltage_V' in work else None
    mask = np.isfinite(t) & np.isfinite(i)
    if v is not None:
        mask &= np.isfinite(v)
    if mask.sum() < 2:
        raise ValueError('Replay profile needs at least two finite rows')
    work = work.loc[mask].reset_index(drop=True)
    t = pd.to_numeric(work['time_s'], errors='coerce').to_numpy(dtype=float)
    meta: dict[str, Any] = {}
    t = _repair_time_if_needed(t, options, meta)
    if options.time_zero == 'start' and len(t):
        t = t - float(np.nanmin(t))
    i = pd.to_numeric(work['current_A'], errors='coerce').to_numpy(dtype=float)
    v = pd.to_numeric(work['voltage_V'], errors='coerce').to_numpy(dtype=float) if 'voltage_V' in work else None
    q = cumulative_charge_discharge_Ah(t, i, current_threshold_A=options.current_threshold_A)
    e = cumulative_energy_Wh(t, i, v, current_threshold_A=options.current_threshold_A)
    for col in options.metadata_columns:
        if col in work:
            vals = work[col].dropna().astype(str).unique().tolist()
            if len(vals) == 1:
                meta[col] = vals[0]
            elif 1 < len(vals) <= 20:
                meta[col] = vals
            elif len(vals) > 20:
                meta[col] = {'unique_count': len(vals), 'first_values': vals[:20]}
    return ReplayProfile(
        t_s=t,
        current_A=i,
        voltage_V=v,
        temperature_C=pd.to_numeric(work['temperature_C'], errors='coerce').to_numpy(float) if 'temperature_C' in work else None,
        cycle_id=_series_or_none(work, 'cycle_id'),
        step_id=_series_or_none(work, 'step_id'),
        step_type=_series_or_none(work, 'step_type'),
        capacity_Ah=pd.to_numeric(work['capacity_Ah'], errors='coerce').to_numpy(float) if 'capacity_Ah' in work else None,
        q_charge_Ah=q['Q_charge_Ah'],
        q_discharge_Ah=q['Q_discharge_Ah'],
        q_net_Ah=q['Q_net_Ah'],
        throughput_Ah=q['throughput_Ah'],
        e_charge_Wh=e['E_charge_Wh'],
        e_discharge_Wh=e['E_discharge_Wh'],
        e_abs_Wh=e['E_abs_Wh'],
        metadata=meta,
    )


def profile_to_dataframe(profile: ReplayProfile) -> pd.DataFrame:
    data = profile.to_npz_dict()
    out = pd.DataFrame({k: v for k, v in data.items() if np.asarray(v).ndim == 1})
    return out


def save_replay_profile_npz(profile: ReplayProfile, output_npz: str | Path, *, summary_json: str | Path | None = None) -> None:
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    arrays = profile.to_npz_dict()
    # Convert object/string arrays safely for npz.
    clean = {}
    for k, v in arrays.items():
        arr = np.asarray(v)
        if arr.dtype == object:
            arr = arr.astype(str)
        clean[k] = arr
    np.savez_compressed(output_npz, **clean)
    if summary_json is not None:
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_json).write_text(json.dumps(profile.summary(), ensure_ascii=False, indent=2), encoding='utf-8')
