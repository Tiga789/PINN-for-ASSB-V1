from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepClassificationOptions:
    current_threshold_A: float = 1e-9
    cv_voltage_threshold_V: float = 4.18
    rest_label: str = 'rest'
    charge_label: str = 'charge'
    discharge_label: str = 'discharge'
    charge_cv_label: str = 'charge_cv_observed'


def classify_step_types(
    current_A: pd.Series | np.ndarray,
    voltage_V: pd.Series | np.ndarray | None = None,
    options: StepClassificationOptions | None = None,
) -> pd.Series:
    """Classify rows as charge/discharge/rest with optional CV annotation.

    This is intentionally generic: protocol labels can be refined by adapters.
    """
    options = options or StepClassificationOptions()
    cur = pd.to_numeric(pd.Series(current_A), errors='coerce')
    out = pd.Series('unknown', index=cur.index, dtype='string')
    thr = abs(float(options.current_threshold_A))
    out[cur > thr] = options.charge_label
    out[cur < -thr] = options.discharge_label
    out[cur.abs() <= thr] = options.rest_label
    if voltage_V is not None:
        v = pd.to_numeric(pd.Series(voltage_V, index=cur.index), errors='coerce')
        cv = (cur > thr) & (v >= float(options.cv_voltage_threshold_V))
        out[cv] = options.charge_cv_label
    return out


def assign_step_ids(step_type: pd.Series | np.ndarray, *, start: int = 1) -> pd.Series:
    """Assign a new step id whenever step_type changes."""
    s = pd.Series(step_type).astype('string').fillna('unknown')
    if s.empty:
        return pd.Series([], dtype='Int64')
    change = s.ne(s.shift(1)).fillna(True).astype(int)
    ids = change.cumsum() + int(start) - 1
    return ids.astype('Int64')


def add_step_columns(df: pd.DataFrame, options: StepClassificationOptions | None = None) -> pd.DataFrame:
    """Return a copy with step_type_auto and step_id_auto."""
    if 'current_A' not in df:
        raise ValueError('current_A is required for step classification')
    out = df.copy()
    voltage = out['voltage_V'] if 'voltage_V' in out else None
    step_type = classify_step_types(out['current_A'], voltage, options)
    out['step_type_auto'] = step_type
    out['step_id_auto'] = assign_step_ids(step_type)
    if 'step_type' not in out or out['step_type'].isna().all():
        out['step_type'] = step_type
    if 'step_id' not in out or out['step_id'].isna().all():
        out['step_id'] = out['step_id_auto']
    return out
