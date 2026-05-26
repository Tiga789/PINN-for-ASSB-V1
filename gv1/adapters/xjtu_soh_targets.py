"""SOH target utilities for XJTU measured-current replay datasets.

These utilities operate on the GV1 standard table.  They intentionally avoid
using model predictions; SOH targets must come from measured discharge capacity
or capacity-test cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SOHTargetOptions:
    q_ref_strategy: str = "first_full_discharge"  # first_full_discharge | early_stable_mean
    early_stable_n: int = 5
    full_discharge_voltage_V: float = 2.5
    partial_discharge_voltage_V: float = 3.0
    full_discharge_voltage_tol_V: float = 0.03
    current_threshold_A: float = 1e-6
    min_discharge_Ah: float = 0.2


def _integrate_discharge_capacity_Ah(group: pd.DataFrame, current_threshold_A: float) -> float:
    if "time_s" not in group or "current_A" not in group:
        return float("nan")
    g = group.sort_values("time_s")
    t = pd.to_numeric(g["time_s"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(g["current_A"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(i)
    if mask.sum() < 2:
        return float("nan")
    t = t[mask]
    i = i[mask]
    neg = i < -abs(current_threshold_A)
    if neg.sum() < 2:
        return 0.0
    # Integrate only negative current portions.  If time is not exactly monotonic, sort already handled.
    discharge_i = np.where(neg, -i, 0.0)
    return float(np.trapz(discharge_i, t) / 3600.0)


def _last_discharge_voltage(group: pd.DataFrame, current_threshold_A: float) -> float:
    if "voltage_V" not in group or "current_A" not in group:
        return float("nan")
    g = group.sort_values("time_s") if "time_s" in group else group
    cur = pd.to_numeric(g["current_A"], errors="coerce")
    v = pd.to_numeric(g["voltage_V"], errors="coerce")
    dis = g.loc[cur < -abs(current_threshold_A)]
    if dis.empty:
        return float("nan")
    return float(pd.to_numeric(dis["voltage_V"], errors="coerce").dropna().iloc[-1])


def _capacity_column_discharge_Ah(group: pd.DataFrame, current_threshold_A: float) -> float:
    if "capacity_Ah" not in group:
        return float("nan")
    cap = pd.to_numeric(group["capacity_Ah"], errors="coerce")
    cur = pd.to_numeric(group.get("current_A"), errors="coerce") if "current_A" in group else None
    if cur is not None and cur.notna().any():
        sub = cap[cur < -abs(current_threshold_A)]
    else:
        sub = cap
    sub = sub.dropna()
    if sub.empty:
        return float("nan")
    # Some battery cyclers reset capacity per step; max in the discharge segment is robust.
    return float(sub.max())


def build_xjtu_cycle_capacity_table(df: pd.DataFrame, options: SOHTargetOptions | None = None) -> pd.DataFrame:
    """Return cycle-level capacity table from a GV1-standard XJTU table."""
    options = options or SOHTargetOptions()
    if "cycle_id" not in df:
        raise ValueError("cycle_id is required for cycle-level SOH targets")
    rows = []
    group_cols = [c for c in ["dataset_id", "batch_id", "battery_id", "cell_id", "protocol_id", "cycle_id"] if c in df.columns]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        q_int = _integrate_discharge_capacity_Ah(g, options.current_threshold_A)
        q_col = _capacity_column_discharge_Ah(g, options.current_threshold_A)
        q = q_col if np.isfinite(q_col) and q_col > 0 else q_int
        v_end = _last_discharge_voltage(g, options.current_threshold_A)
        is_full = bool(np.isfinite(v_end) and v_end <= options.full_discharge_voltage_V + options.full_discharge_voltage_tol_V)
        is_partial = bool(np.isfinite(v_end) and v_end > options.full_discharge_voltage_V + options.full_discharge_voltage_tol_V)
        row.update({
            "Q_discharge_Ah": q,
            "Q_discharge_integral_Ah": q_int,
            "Q_discharge_column_Ah": q_col,
            "discharge_end_voltage_V": v_end,
            "is_full_discharge": is_full,
            "is_partial_discharge": is_partial,
            "is_capacity_label_candidate": bool(is_full and np.isfinite(q) and q >= options.min_discharge_Ah),
            "n_rows": int(len(g)),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "cycle_id" in out:
        out = out.sort_values([c for c in ["batch_id", "battery_id", "cycle_id"] if c in out.columns]).reset_index(drop=True)
    return out


def _choose_q_ref(capacity_table: pd.DataFrame, options: SOHTargetOptions) -> float:
    candidates = capacity_table.loc[capacity_table["is_capacity_label_candidate"].astype(bool)].copy()
    if candidates.empty:
        raise ValueError("No full-discharge capacity-label candidates found")
    q = pd.to_numeric(candidates["Q_discharge_Ah"], errors="coerce").dropna()
    q = q[q > 0]
    if q.empty:
        raise ValueError("No positive discharge capacities found")
    if options.q_ref_strategy == "early_stable_mean":
        return float(q.iloc[: max(1, options.early_stable_n)].mean())
    return float(q.iloc[0])


def build_xjtu_soh_targets(capacity_table: pd.DataFrame, options: SOHTargetOptions | None = None) -> pd.DataFrame:
    """Add Q_ref_Ah and SOH to a cycle-capacity table.

    Reference capacity is computed per cell if a cell_id/battery_id exists.
    """
    options = options or SOHTargetOptions()
    if capacity_table.empty:
        return capacity_table.copy()
    group_cols = [c for c in ["dataset_id", "batch_id", "battery_id", "cell_id"] if c in capacity_table.columns]
    rows = []
    for _, g in capacity_table.groupby(group_cols, dropna=False) if group_cols else [(None, capacity_table)]:
        gg = g.copy()
        q_ref = _choose_q_ref(gg, options)
        gg["Q_ref_Ah"] = q_ref
        gg["SOH"] = pd.to_numeric(gg["Q_discharge_Ah"], errors="coerce") / q_ref
        gg.loc[~gg["is_capacity_label_candidate"].astype(bool), "SOH"] = np.nan
        rows.append(gg)
    return pd.concat(rows, ignore_index=True, sort=False)
