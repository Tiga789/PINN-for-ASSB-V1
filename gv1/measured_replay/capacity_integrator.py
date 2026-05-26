from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Robust trapezoidal integral compatible with old/new NumPy."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return float('nan')
    y = y[mask]
    x = x[mask]
    order = np.argsort(x)
    y = y[order]
    x = x[order]
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def cumulative_integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral with the same length as the input."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if len(y) < 2:
        return out
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    inc = avg * dx
    inc[~np.isfinite(inc)] = 0.0
    out[1:] = np.cumsum(inc)
    return out


def cumulative_charge_discharge_Ah(t_s: np.ndarray, current_A: np.ndarray, *, current_threshold_A: float = 1e-9) -> dict[str, np.ndarray]:
    """Return cumulative charge, discharge, net and absolute throughput in Ah.

    Convention: positive current is charge. Discharge capacity is the integral
    of -I over negative-current regions.
    """
    t_s = np.asarray(t_s, dtype=float)
    current_A = np.asarray(current_A, dtype=float)
    charge_current = np.where(current_A > abs(current_threshold_A), current_A, 0.0)
    discharge_current = np.where(current_A < -abs(current_threshold_A), -current_A, 0.0)
    abs_current = np.abs(np.where(np.isfinite(current_A), current_A, 0.0))
    return {
        'Q_charge_Ah': cumulative_integral(charge_current, t_s) / 3600.0,
        'Q_discharge_Ah': cumulative_integral(discharge_current, t_s) / 3600.0,
        'Q_net_Ah': cumulative_integral(current_A, t_s) / 3600.0,
        'throughput_Ah': cumulative_integral(abs_current, t_s) / 3600.0,
    }


def cumulative_energy_Wh(t_s: np.ndarray, current_A: np.ndarray, voltage_V: np.ndarray | None = None, *, current_threshold_A: float = 1e-9) -> dict[str, np.ndarray]:
    """Return cumulative charge/discharge energy in Wh.

    Energy is computed from measured V(t) and I(t). It is an audit/feature,
    not part of the effective SPM physics closure.
    """
    t_s = np.asarray(t_s, dtype=float)
    current_A = np.asarray(current_A, dtype=float)
    if voltage_V is None:
        zeros = np.zeros_like(t_s, dtype=float)
        return {'E_charge_Wh': zeros, 'E_discharge_Wh': zeros, 'E_abs_Wh': zeros}
    voltage_V = np.asarray(voltage_V, dtype=float)
    power_W = voltage_V * current_A
    charge_power = np.where(current_A > abs(current_threshold_A), np.maximum(power_W, 0.0), 0.0)
    discharge_power = np.where(current_A < -abs(current_threshold_A), np.maximum(-power_W, 0.0), 0.0)
    abs_power = np.abs(np.where(np.isfinite(power_W), power_W, 0.0))
    return {
        'E_charge_Wh': cumulative_integral(charge_power, t_s) / 3600.0,
        'E_discharge_Wh': cumulative_integral(discharge_power, t_s) / 3600.0,
        'E_abs_Wh': cumulative_integral(abs_power, t_s) / 3600.0,
    }


def _last_valid(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors='coerce').dropna()
    return float(vals.iloc[-1]) if len(vals) else float('nan')


def build_cycle_integrals(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ('dataset_id', 'batch_id', 'battery_id', 'cell_id', 'cycle_id'),
    current_threshold_A: float = 1e-9,
) -> pd.DataFrame:
    """Build cycle-level charge/discharge capacity and energy summary."""
    if 'time_s' not in df or 'current_A' not in df:
        raise ValueError('time_s and current_A are required')
    groups = [c for c in group_cols if c in df.columns]
    if not groups:
        groups = ['__all__']
        work = df.copy()
        work['__all__'] = 0
    else:
        work = df.copy()
    rows = []
    for keys, g in work.groupby(groups, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(groups, keys) if col != '__all__'}
        g = g.sort_values('time_s')
        t = pd.to_numeric(g['time_s'], errors='coerce').to_numpy(float)
        i = pd.to_numeric(g['current_A'], errors='coerce').to_numpy(float)
        v = pd.to_numeric(g['voltage_V'], errors='coerce').to_numpy(float) if 'voltage_V' in g else None
        q = cumulative_charge_discharge_Ah(t, i, current_threshold_A=current_threshold_A)
        e = cumulative_energy_Wh(t, i, v, current_threshold_A=current_threshold_A)
        row.update({name: float(arr[-1]) if len(arr) else float('nan') for name, arr in q.items()})
        row.update({name: float(arr[-1]) if len(arr) else float('nan') for name, arr in e.items()})
        row['n_rows'] = int(len(g))
        row['t_start_s'] = float(np.nanmin(t)) if len(t) else float('nan')
        row['t_end_s'] = float(np.nanmax(t)) if len(t) else float('nan')
        row['voltage_start_V'] = float(pd.to_numeric(g.get('voltage_V'), errors='coerce').dropna().iloc[0]) if 'voltage_V' in g and pd.to_numeric(g.get('voltage_V'), errors='coerce').dropna().size else float('nan')
        row['voltage_end_V'] = _last_valid(g['voltage_V']) if 'voltage_V' in g else float('nan')
        row['capacity_Ah_max'] = float(pd.to_numeric(g['capacity_Ah'], errors='coerce').max()) if 'capacity_Ah' in g else float('nan')
        rows.append(row)
    return pd.DataFrame(rows)
