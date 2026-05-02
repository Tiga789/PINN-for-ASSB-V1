#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASSB cycle-capable SPM reference solver / soft-label generator.

Purpose
-------
This file is intended to be placed at:

    integration_spm/spm_int_assb_cycle.py

It generates ASSB soft labels that are consistent with the current
training-side effective SPM prior:

    util/spm_assb_train_discharge.py
    util/thermo_assb.py
    util/_losses.py
    util/_rescale.py

Main consistency points
-----------------------
1. Uses the same parameter dictionary from util.spm_assb_train_discharge.makeParams().
2. Uses the same current sign convention:
       +I = charge, -I = discharge.
3. Uses the same surface-flux closure:
       J_a(t) = -I(t) R_a / (3 eps_a F V_a)
       J_c(t) =  I(t) R_c / (3 eps_c F V_c)
4. Uses the same OCP / i0 / diffusivity callables from util.thermo_assb.
5. Uses the same output variables expected by PINNSTRIPES/PyTorch training:
       data_phie.npz, data_phis_c.npz, data_cs_a.npz, data_cs_c.npz

Important terminology fix
-------------------------
In this file the legacy suffixes are electrode-fixed variables, not
charge/discharge role labels:

       a = negative electrode = Li-In/In effective pseudo-particle
       c = positive electrode = NMC811 representative particle

During discharge the positive electrode is the cathode and the negative
electrode is the anode. During charge these reaction roles switch. The
material identity, geometry, OCP table, diffusivity and i0 callable of each
electrode do NOT switch. The reaction-role switch is represented by the
sign of I(t), hence by J_a/J_c, eta_a/eta_c and I*R_ohm. This avoids mixing
"positive/negative electrode" with "anode/cathode reaction role".

Default experimental current source
-----------------------------------
By default the script looks for the user's local ASSB CSV:

    C:/Users/Tiga_QJW/Desktop/ZHB_realDATA/record_extracted.csv

It skips activation cycles by default and uses cycle 5 unless a different
cycle is passed on the command line.

V3 update
---------
Compared with the previous v2 generator, this version adds a conservative
voltage-alignment step for the soft-label terminal potential. This patched
version also adds an optional --merge_cycles mode; the original single-cycle
mode is unchanged unless --merge_cycles is explicitly passed.

1. It still reads positive_ocp_curve.csv and negative_ocp_curve.csv directly.
2. It still keeps the ASSB effective SPM concentration dynamics unchanged.
3. It can automatically tune the cathode OCP usable window, one cathode OCP
   offset, and R_ohm_eff against the selected experimental cycle voltage.
4. The raw pre-alignment phis_c is saved as phis_c_raw in solution.npz.

This is intended to generate a v3 soft-label set for data-loss workflow tests.
The fitted voltage-alignment terms are written to soft_label_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("spm_int_assb_cycle.py requires pandas to read record_extracted.csv") from exc

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("spm_int_assb_cycle.py requires torch because ASSB thermo callables use torch tensors") from exc


# -----------------------------------------------------------------------------
# Path and import helpers
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_DIR.parent if THIS_DIR.name == "integration_spm" else THIS_DIR
UTIL_DIR = REPO_ROOT / "util"

for _path in (REPO_ROOT, UTIL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from util.spm_assb_train_discharge import makeParams  # type: ignore
except Exception:  # pragma: no cover
    try:
        from spm_assb_train_discharge import makeParams  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Cannot import makeParams from util/spm_assb_train_discharge.py. "
            "Place this file inside the project or run it from the project root."
        ) from exc


# -----------------------------------------------------------------------------
# Small numeric helpers
# -----------------------------------------------------------------------------

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _to_scalar(x: Any) -> np.float64:
    arr = _to_numpy(x).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot convert empty value to scalar.")
    return np.float64(arr[0])


def _torch_scalar(x: Any) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


# -----------------------------------------------------------------------------
# Experimental current profile reader
# -----------------------------------------------------------------------------

DEFAULT_RECORD_CSV = Path(r"C:/Users/Tiga_QJW/Desktop/ZHB_realDATA/record_extracted.csv")
DEFAULT_OCP_DIR = Path(r"C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/ocp_estimation_outputs")

CYCLE_COLUMNS = ("循环号", "cycle", "Cycle", "cycle_index", "Cycle_Index", "循环", "循环序号")
TIME_COLUMNS = ("总时间", "总时间(s)", "time_s", "Time(s)", "time", "Time", "时间", "测试时间")
CURRENT_COLUMNS = ("电流(A)", "电流", "current_A", "Current(A)", "current", "Current", "I_A", "I(A)")
VOLTAGE_COLUMNS = ("电压(V)", "电压", "voltage_V", "Voltage(V)", "voltage", "Voltage", "V")


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    cols = {str(c).strip(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    # Loose matching for files with spaces or BOM marks.
    norm = {str(c).strip().replace(" ", "").replace("\ufeff", ""): c for c in df.columns}
    for name in candidates:
        key = name.strip().replace(" ", "").replace("\ufeff", "")
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"Could not find any of columns {candidates}; available columns: {list(df.columns)}")
    return None


def _parse_time_seconds(values: Any) -> np.ndarray:
    s = pd.Series(values)
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().sum() >= max(1, int(0.8 * len(s))):
        out = numeric.to_numpy(dtype=np.float64)
    else:
        # Handles forms such as '00:01:23', '0 days 00:01:23', etc.
        td = pd.to_timedelta(s.astype(str), errors="coerce")
        if td.notna().sum() >= max(1, int(0.8 * len(s))):
            out = td.dt.total_seconds().to_numpy(dtype=np.float64)
        else:
            # Last resort: datetime-like values. Relative seconds are used.
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() == 0:
                raise ValueError("Could not parse time column as numeric, timedelta, or datetime.")
            t0 = dt.dropna().iloc[0]
            out = (dt - t0).dt.total_seconds().to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(out)):
        mask = np.isfinite(out)
        if mask.sum() < 2:
            raise ValueError("Too few finite time values after parsing.")
        # Fill occasional invalid entries by interpolation over row index.
        idx = np.arange(len(out), dtype=np.float64)
        out = np.interp(idx, idx[mask], out[mask])
    return out.astype(np.float64)


@dataclass
class CurrentProfile:
    time_s: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray | None
    cycle: int | None
    csv_path: Path | None

    @property
    def n_points(self) -> int:
        return int(self.time_s.size)

    @property
    def tmax(self) -> np.float64:
        return np.float64(self.time_s[-1]) if self.time_s.size else np.float64(0.0)

    @property
    def i_ref_abs(self) -> np.float64:
        if self.current_A.size == 0:
            return np.float64(0.0)
        return np.float64(np.nanmax(np.abs(self.current_A)))

    @property
    def charge_capacity_Ah(self) -> np.float64:
        pos = np.where(self.current_A > 0.0, self.current_A, 0.0)
        return np.float64(np.trapezoid(pos, self.time_s) / 3600.0)

    @property
    def discharge_capacity_Ah(self) -> np.float64:
        neg = np.where(self.current_A < 0.0, self.current_A, 0.0)
        return np.float64(-np.trapezoid(neg, self.time_s) / 3600.0)

    @property
    def capacity_Ah(self) -> np.float64:
        vals = [float(self.charge_capacity_Ah), float(self.discharge_capacity_Ah)]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        if vals:
            return np.float64(np.mean(vals))
        return np.float64(0.0)


def read_record_current_profile(
    record_csv: str | Path | None = None,
    cycle: int | None = 5,
    skip_activation_cycles: int = 4,
) -> CurrentProfile:
    """Read one stable ASSB cycle from record_extracted.csv.

    Cycle 1--4 are activation cycles by default, so cycle=5 is used unless
    specified otherwise. Time is shifted so that the selected cycle starts at 0 s.
    """
    path = Path(record_csv) if record_csv is not None else DEFAULT_RECORD_CSV
    if not path.exists():
        raise FileNotFoundError(f"ASSB record CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"ASSB record CSV is empty: {path}")

    cycle_col = _find_column(df, CYCLE_COLUMNS, required=False)
    if cycle_col is not None:
        cyc_values = pd.to_numeric(df[cycle_col], errors="coerce")
        available = sorted(int(v) for v in cyc_values.dropna().unique())
        if cycle is None:
            candidates = [c for c in available if c > int(skip_activation_cycles)]
            if not candidates:
                raise ValueError(f"No cycle > {skip_activation_cycles} found in {path}")
            cycle = candidates[0]
        selected = df.loc[cyc_values == int(cycle)].copy()
        if selected.empty:
            raise ValueError(f"Cycle {cycle} not found. Available cycles: {available[:20]}...")
    else:
        selected = df.copy()
        cycle = None

    time_col = _find_column(selected, TIME_COLUMNS, required=True)
    current_col = _find_column(selected, CURRENT_COLUMNS, required=True)
    voltage_col = _find_column(selected, VOLTAGE_COLUMNS, required=False)

    t_s = _parse_time_seconds(selected[time_col])
    current_A = pd.to_numeric(selected[current_col], errors="coerce").to_numpy(dtype=np.float64)
    voltage_V = None
    if voltage_col is not None:
        voltage_V = pd.to_numeric(selected[voltage_col], errors="coerce").to_numpy(dtype=np.float64)

    mask = np.isfinite(t_s) & np.isfinite(current_A)
    if voltage_V is not None:
        mask = mask & np.isfinite(voltage_V)
    t_s = t_s[mask]
    current_A = current_A[mask]
    if voltage_V is not None:
        voltage_V = voltage_V[mask]

    if t_s.size < 3:
        raise ValueError("Selected cycle has fewer than 3 usable points.")

    order = np.argsort(t_s)
    t_s = t_s[order]
    current_A = current_A[order]
    if voltage_V is not None:
        voltage_V = voltage_V[order]

    # Shift selected cycle to start at t=0, then merge duplicate timestamps.
    t_s = t_s - t_s[0]
    unique_t, inv = np.unique(t_s, return_inverse=True)
    if unique_t.size != t_s.size:
        current_u = np.zeros_like(unique_t, dtype=np.float64)
        voltage_u = np.zeros_like(unique_t, dtype=np.float64) if voltage_V is not None else None
        counts = np.zeros_like(unique_t, dtype=np.float64)
        for k, j in enumerate(inv):
            current_u[j] += current_A[k]
            if voltage_u is not None:
                voltage_u[j] += voltage_V[k]
            counts[j] += 1.0
        current_A = current_u / counts
        if voltage_u is not None:
            voltage_V = voltage_u / counts
        t_s = unique_t

    if not np.all(np.diff(t_s) > 0):
        raise ValueError("Selected time profile is not strictly increasing after cleanup.")

    return CurrentProfile(
        time_s=t_s.astype(np.float64),
        current_A=current_A.astype(np.float64),
        voltage_V=None if voltage_V is None else voltage_V.astype(np.float64),
        cycle=cycle,
        csv_path=path,
    )


def read_record_current_profile_merged(
    record_csv: str | Path | None = None,
    cycle_from: int | None = 5,
    cycle_to: int | None = None,
    skip_activation_cycles: int = 4,
) -> CurrentProfile:
    """Read one continuous time-series containing multiple stable ASSB cycles.

    This is an additive helper for the merged soft-label workflow. It does not
    change read_record_current_profile(), so the original single-cycle mode is
    preserved. The selected rows are filtered by cycle number, sorted by the
    original time column, shifted so that the first selected row starts at 0 s,
    and then exported as one continuous profile.
    """
    path = Path(record_csv) if record_csv is not None else DEFAULT_RECORD_CSV
    if not path.exists():
        raise FileNotFoundError(f"ASSB record CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"ASSB record CSV is empty: {path}")

    cycle_col = _find_column(df, CYCLE_COLUMNS, required=True)
    cyc_values = pd.to_numeric(df[cycle_col], errors="coerce")
    available = sorted(int(v) for v in cyc_values.dropna().unique())
    if cycle_from is None:
        cycle_from = int(skip_activation_cycles) + 1
    mask_cycles = np.isfinite(cyc_values.to_numpy(dtype=np.float64)) & (cyc_values.to_numpy(dtype=np.float64) >= int(cycle_from))
    if cycle_to is not None:
        mask_cycles = mask_cycles & (cyc_values.to_numpy(dtype=np.float64) <= int(cycle_to))
    selected = df.loc[mask_cycles].copy()
    if selected.empty:
        raise ValueError(
            f"No rows found for cycles >= {cycle_from}"
            + ("" if cycle_to is None else f" and <= {cycle_to}")
            + f". Available cycles: {available[:30]}..."
        )

    selected_cycle_values = pd.to_numeric(selected[cycle_col], errors="coerce").to_numpy(dtype=np.float64)
    selected_cycles = sorted(int(v) for v in pd.Series(selected_cycle_values).dropna().unique())

    time_col = _find_column(selected, TIME_COLUMNS, required=True)
    current_col = _find_column(selected, CURRENT_COLUMNS, required=True)
    voltage_col = _find_column(selected, VOLTAGE_COLUMNS, required=False)

    t_s = _parse_time_seconds(selected[time_col])
    current_A = pd.to_numeric(selected[current_col], errors="coerce").to_numpy(dtype=np.float64)
    voltage_V = None
    if voltage_col is not None:
        voltage_V = pd.to_numeric(selected[voltage_col], errors="coerce").to_numpy(dtype=np.float64)

    mask = np.isfinite(t_s) & np.isfinite(current_A) & np.isfinite(selected_cycle_values)
    if voltage_V is not None:
        mask = mask & np.isfinite(voltage_V)
    t_s = t_s[mask]
    current_A = current_A[mask]
    cycle_profile = selected_cycle_values[mask]
    if voltage_V is not None:
        voltage_V = voltage_V[mask]

    if t_s.size < 3:
        raise ValueError("Merged cycle profile has fewer than 3 usable points.")

    order = np.argsort(t_s, kind="mergesort")
    t_s = t_s[order]
    current_A = current_A[order]
    cycle_profile = cycle_profile[order]
    if voltage_V is not None:
        voltage_V = voltage_V[order]

    # Shift the merged stable-cycle block to start at t=0, then merge duplicate timestamps.
    t_s = t_s - t_s[0]
    unique_t, inv = np.unique(t_s, return_inverse=True)
    if unique_t.size != t_s.size:
        current_u = np.zeros_like(unique_t, dtype=np.float64)
        cycle_u = np.zeros_like(unique_t, dtype=np.float64)
        voltage_u = np.zeros_like(unique_t, dtype=np.float64) if voltage_V is not None else None
        counts = np.zeros_like(unique_t, dtype=np.float64)
        for k, j in enumerate(inv):
            current_u[j] += current_A[k]
            cycle_u[j] += cycle_profile[k]
            if voltage_u is not None:
                voltage_u[j] += voltage_V[k]
            counts[j] += 1.0
        current_A = current_u / np.maximum(counts, 1.0)
        cycle_profile = np.rint(cycle_u / np.maximum(counts, 1.0))
        if voltage_u is not None:
            voltage_V = voltage_u / np.maximum(counts, 1.0)
        t_s = unique_t

    if not np.all(np.diff(t_s) > 0):
        raise ValueError("Merged time profile is not strictly increasing after cleanup. Check whether the chosen time column is truly global time.")

    profile = CurrentProfile(
        time_s=t_s.astype(np.float64),
        current_A=current_A.astype(np.float64),
        voltage_V=None if voltage_V is None else voltage_V.astype(np.float64),
        cycle=None,
        csv_path=path,
    )
    # Additional metadata; single-cycle users are unaffected because these
    # attributes exist only for merged profiles.
    profile.merge_cycles = True
    profile.cycle_from = int(cycle_from)
    profile.cycle_to = None if cycle_to is None else int(cycle_to)
    profile.selected_cycles = selected_cycles
    profile.cycle_profile = cycle_profile.astype(np.int64)
    profile.profile_label = f"cycles {cycle_from}+" if cycle_to is None else f"cycles {cycle_from}-{cycle_to}"
    return profile



# -----------------------------------------------------------------------------
# OCP-table helpers for ASSB v2/v3 soft labels
# -----------------------------------------------------------------------------

def _read_ocp_table(path: Path, preferred_y: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"OCP table not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"OCP table is empty: {path}")

    x_col = None
    for cand in ("soc_0to1", "x", "theta", "soc", "SOC"):
        if cand in df.columns:
            x_col = cand
            break
    if x_col is None:
        x_col = df.columns[0]

    y_col = None
    for cand in preferred_y:
        if cand in df.columns:
            y_col = cand
            break
    if y_col is None:
        # choose the first numeric column that is not x_col
        for c in df.columns:
            if c == x_col:
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().sum() > 0:
                y_col = c
                break
    if y_col is None:
        raise ValueError(f"Could not identify voltage column in {path}; columns={list(df.columns)}")

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError(f"OCP table {path} has fewer than 2 finite points.")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    # Merge duplicate x values by averaging.
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size != x.size:
        yy = np.zeros_like(ux, dtype=np.float64)
        cnt = np.zeros_like(ux, dtype=np.float64)
        for k, j in enumerate(inv):
            yy[j] += y[k]
            cnt[j] += 1.0
        y = yy / np.maximum(cnt, 1.0)
        x = ux
    return x.astype(np.float64), y.astype(np.float64)


def _default_theta_c_top(params: dict, profile: CurrentProfile, theta_c_bottom: float) -> np.float64:
    """Estimate the cathode top-window stoichiometry from cycle capacity.

    This follows the v2 idea: use the selected cycle capacity and the cathode
    effective inventory F * eps_c * V_c * cscamax.
    """
    F = np.float64(params["F"])
    cscamax = np.float64(params["cscamax"])
    eps_c = np.float64(params["eps_s_c"])
    V_c = np.float64(params.get("V_c", params["A_c"] * params["L_c"]))
    cap_c_Ah = F * eps_c * V_c * cscamax / np.float64(3600.0)
    q_Ah = np.float64(params.get("C", profile.capacity_Ah))
    if not np.isfinite(q_Ah) or q_Ah <= 0:
        q_Ah = profile.capacity_Ah
    dtheta = q_Ah / max(float(cap_c_Ah), 1.0e-30)
    theta_top = np.float64(theta_c_bottom) - np.float64(dtheta)
    return np.float64(np.clip(theta_top, 0.20, np.float64(theta_c_bottom) - 0.05))


def install_assb_ocp_tables(
    params: dict,
    ocp_dir: str | Path | None,
    profile: CurrentProfile | None = None,
    theta_c_bottom: float | None = None,
    theta_c_top: float | None = None,
    csanmax_eff: float | None = None,
    up_offset_V: float = 0.0,
) -> dict:
    """Install differentiable torch-compatible OCP callables from CSV tables.

    positive_ocp_curve.csv is interpreted as the positive-electrode OCP table
    on the usable SOC/window coordinate. The solver maps cathode concentration
    to that coordinate by

        soc_c = (theta_c_bottom - theta_c) / (theta_c_bottom - theta_c_top)

    and clips to [0, 1]. negative_ocp_curve.csv is used for the Li-In OCP; for
    the current first-prior table it is essentially a flat 0.62464 V plateau.
    """
    if csanmax_eff is not None and np.isfinite(csanmax_eff) and csanmax_eff > 0:
        params["csanmax_original"] = np.float64(params.get("csanmax", csanmax_eff))
        theta_a0 = np.float64(params.get("theta_a0", params["cs_a0"] / params["csanmax"]))
        params["csanmax"] = np.float64(csanmax_eff)
        params["theta_a0"] = theta_a0
        params["cs_a0"] = np.float64(theta_a0 * params["csanmax"])

    odir = Path(ocp_dir) if ocp_dir is not None else DEFAULT_OCP_DIR
    pos_path = odir / "positive_ocp_curve.csv"
    neg_path = odir / "negative_ocp_curve.csv"
    pos_soc, pos_u = _read_ocp_table(pos_path, ("U_p_ocp_est_V", "Ueq", "U", "Voltage", "voltage"))
    neg_soc, neg_u = _read_ocp_table(neg_path, ("U_n_ocp_est_V", "Ueq", "U", "Voltage", "voltage"))

    if theta_c_bottom is None:
        theta_c_bottom = float(params.get("theta_c_bottom", params.get("theta_c0", 0.90)))
    if theta_c_top is None:
        if profile is not None:
            theta_c_top = float(_default_theta_c_top(params, profile, theta_c_bottom))
        else:
            theta_c_top = float(params.get("theta_c_top", 0.465))
    if theta_c_top >= theta_c_bottom:
        raise ValueError(f"theta_c_top must be < theta_c_bottom; got top={theta_c_top}, bottom={theta_c_bottom}")

    params["ocp_source_dir"] = str(odir)
    params["theta_c_bottom"] = np.float64(theta_c_bottom)
    params["theta_c_top"] = np.float64(theta_c_top)
    params["U_p_offset_V"] = np.float64(up_offset_V)
    params["_ocp_pos_soc"] = pos_soc
    params["_ocp_pos_u"] = pos_u
    params["_ocp_neg_soc"] = neg_soc
    params["_ocp_neg_u"] = neg_u
    params["Uocp_a0_table"] = np.float64(np.interp(0.5, neg_soc, neg_u))
    params["Uocp_c0_table"] = np.float64(np.interp(0.0, pos_soc, pos_u) + up_offset_V)

    def uocp_a_from_table(cs_a, csanmax):
        ref = cs_a if isinstance(cs_a, torch.Tensor) else None
        theta = _to_numpy(cs_a) / float(csanmax)
        soc = np.clip(theta, 0.0, 1.0)
        val = np.interp(soc, neg_soc, neg_u)
        if ref is not None:
            return torch.as_tensor(val, dtype=torch.float64, device=ref.device)
        return torch.as_tensor(val, dtype=torch.float64)

    def uocp_c_from_table(cs_c, cscamax):
        ref = cs_c if isinstance(cs_c, torch.Tensor) else None
        theta = _to_numpy(cs_c) / float(cscamax)
        soc = (float(theta_c_bottom) - theta) / max(float(theta_c_bottom) - float(theta_c_top), 1.0e-12)
        soc = np.clip(soc, 0.0, 1.0)
        val = np.interp(soc, pos_soc, pos_u) + float(up_offset_V)
        if ref is not None:
            return torch.as_tensor(val, dtype=torch.float64, device=ref.device)
        return torch.as_tensor(val, dtype=torch.float64)

    params["Uocp_a"] = uocp_a_from_table
    params["Uocp_c_raw"] = uocp_c_from_table
    params["Uocp_c"] = uocp_c_from_table

    return params

# -----------------------------------------------------------------------------
# Parameter synchronization with training side
# -----------------------------------------------------------------------------

def _first_nonzero_current(current_A: np.ndarray, threshold_A: float = 1.0e-8) -> float:
    idx = np.where(np.abs(current_A) > threshold_A)[0]
    if idx.size == 0:
        return 0.0
    return float(current_A[idx[0]])


# -----------------------------------------------------------------------------
# Charge/discharge role helpers
# -----------------------------------------------------------------------------

def current_mode_from_value(I_now: float, threshold_A: float = 1.0e-10) -> str:
    """Return charge/discharge/rest from the signed experimental current.

    Convention used throughout this project:
        +I = charge
        -I = discharge

    This function is deliberately separate from electrode material labels.
    The NMC811 positive electrode and Li-In/In negative electrode keep their
    own OCP/i0/D_s functions. Only their electrochemical reaction roles switch.
    """
    I_now = float(I_now)
    if I_now > threshold_A:
        return "charge"
    if I_now < -threshold_A:
        return "discharge"
    return "rest"


def electrode_role_map_from_current(I_now: float, threshold_A: float = 1.0e-10) -> dict[str, str]:
    """Map fixed electrodes to reaction roles for metadata/auditing only.

    The returned role labels must NOT be used to swap material parameters.
    They only document the electrochemical role implied by the current sign.
    """
    mode = current_mode_from_value(I_now, threshold_A=threshold_A)
    if mode == "charge":
        return {
            "mode": "charge",
            "positive_electrode": "anode_reaction_role",
            "negative_electrode": "cathode_reaction_role",
        }
    if mode == "discharge":
        return {
            "mode": "discharge",
            "positive_electrode": "cathode_reaction_role",
            "negative_electrode": "anode_reaction_role",
        }
    return {
        "mode": "rest",
        "positive_electrode": "near_equilibrium_no_net_role",
        "negative_electrode": "near_equilibrium_no_net_role",
    }


def current_role_code_arrays(current_A: np.ndarray, threshold_A: float = 1.0e-10) -> dict[str, np.ndarray]:
    """Return numeric role/sign arrays saved into solution.npz.

    Codes:
        current_mode_code: -1 discharge, 0 rest, +1 charge
        positive_electrode_role_code: -1 anode role, 0 rest, +1 cathode role
        negative_electrode_role_code: -1 anode role, 0 rest, +1 cathode role

    These arrays are provenance/debug metadata. They are not used to change
    the OCP, D_s, i0 or geometry of either fixed electrode.
    """
    current_A = np.asarray(current_A, dtype=np.float64)
    mode = np.zeros_like(current_A, dtype=np.int8)
    mode[current_A > threshold_A] = 1
    mode[current_A < -threshold_A] = -1

    # For discharge: positive=NMC cathode role, negative=Li-In anode role.
    # For charge:    positive=NMC anode role,   negative=Li-In cathode role.
    positive_role = np.zeros_like(mode, dtype=np.int8)
    negative_role = np.zeros_like(mode, dtype=np.int8)
    positive_role[mode < 0] = 1
    negative_role[mode < 0] = -1
    positive_role[mode > 0] = -1
    negative_role[mode > 0] = 1

    return {
        "current_mode_code": mode,
        "positive_electrode_role_code": positive_role,
        "negative_electrode_role_code": negative_role,
    }


def charge_discharge_role_summary(current_A: np.ndarray, threshold_A: float = 1.0e-10) -> dict[str, Any]:
    """Compact summary of charge/discharge role alignment for JSON metadata."""
    current_A = np.asarray(current_A, dtype=np.float64)
    codes = current_role_code_arrays(current_A, threshold_A=threshold_A)["current_mode_code"]
    n_charge = int(np.sum(codes > 0))
    n_discharge = int(np.sum(codes < 0))
    n_rest = int(np.sum(codes == 0))
    first_nonzero = _first_nonzero_current(current_A, threshold_A=threshold_A)
    return {
        "current_sign_convention": "+I charge, -I discharge",
        "fixed_electrode_a": "negative electrode, Li-In/In effective pseudo-particle",
        "fixed_electrode_c": "positive electrode, NMC811 representative particle",
        "role_rule_discharge": "positive electrode is cathode role; negative electrode is anode role",
        "role_rule_charge": "positive electrode is anode role; negative electrode is cathode role",
        "material_parameter_switching": "disabled: OCP/i0/D_s/geometry remain fixed to electrode identity",
        "n_charge_points": n_charge,
        "n_discharge_points": n_discharge,
        "n_rest_points": n_rest,
        "first_nonzero_current_A": float(first_nonzero),
        "first_nonzero_mode": current_mode_from_value(first_nonzero, threshold_A=threshold_A),
    }



def _apply_profile_to_params(params: dict, profile: CurrentProfile) -> dict:
    """Synchronize makeParams() output with the selected cycle profile."""
    params["time_profile"] = profile.time_s.copy()
    params["current_profile_A"] = profile.current_A.copy()
    params["current_profile"] = (params["time_profile"], params["current_profile_A"])
    params["I_profile"] = params["current_profile"]
    params["I_app_profile"] = params["current_profile"]
    params["I_discharge"] = np.float64(profile.current_A[0])
    params["I_ref_abs"] = profile.i_ref_abs
    params["tmin"] = np.float64(0.0)
    params["tmax"] = profile.tmax
    if profile.capacity_Ah > 0:
        params["C"] = profile.capacity_Ah

    # Match the training-side charge-first / discharge-first initial-state logic.
    first_i = _first_nonzero_current(profile.current_A)
    if first_i > 0:
        theta_a0 = np.float64(0.45)
        theta_c0 = np.float64(0.90)
        ic_note = "charge-first cycle initial state"
    elif first_i < 0:
        theta_a0 = np.float64(0.55)
        theta_c0 = np.float64(0.27)
        ic_note = "discharge-first cycle initial state"
    else:
        theta_a0 = np.float64(params.get("cs_a0", 0.55 * params["csanmax"]) / params["csanmax"])
        theta_c0 = np.float64(params.get("cs_c0", 0.27 * params["cscamax"]) / params["cscamax"])
        ic_note = "zero-current initial state from existing params"

    params["theta_a0"] = theta_a0
    params["theta_c0"] = theta_c0
    params["cs_a0"] = np.float64(theta_a0 * params["csanmax"])
    params["cs_c0"] = np.float64(theta_c0 * params["cscamax"])
    params["assb_ic_note"] = ic_note

    # Ensure explicit volumes are present, matching the training loss helper.
    params["V_a"] = np.float64(params.get("V_a", params["A_a"] * params["L_a"]))
    params["V_c"] = np.float64(params.get("V_c", params["A_c"] * params["L_c"]))

    # Effective solid-state ionic network ohmic term. Training-side files should
    # already define this. If absent, keep the first-prior value from the scheme.
    params["R_ohm_eff"] = np.float64(params.get("R_ohm_eff", 100.0))

    # Keep physical/code electrolyte scale visible for downstream checks.
    params["ce_eff_physical"] = np.float64(params.get("ce_eff_physical", 1.0))
    params["ce_ref_code"] = np.float64(params.get("ce_ref_code", 1.2))

    return params


def load_cycle_params(
    record_csv: str | Path | None = None,
    cycle: int | None = 5,
    skip_activation_cycles: int = 4,
    ocp_dir: str | Path | None = None,
    theta_c_bottom: float | None = None,
    theta_c_top: float | None = None,
    csanmax_eff: float | None = 6.0,
    r_ohm_eff: float | None = None,
    up_offset_V: float = 0.0,
) -> tuple[dict, CurrentProfile]:
    params = makeParams()
    profile = read_record_current_profile(record_csv, cycle=cycle, skip_activation_cycles=skip_activation_cycles)
    params = _apply_profile_to_params(params, profile)
    if r_ohm_eff is not None and np.isfinite(r_ohm_eff):
        params["R_ohm_eff"] = np.float64(r_ohm_eff)
    params = install_assb_ocp_tables(
        params=params,
        ocp_dir=ocp_dir,
        profile=profile,
        theta_c_bottom=theta_c_bottom,
        theta_c_top=theta_c_top,
        csanmax_eff=csanmax_eff,
        up_offset_V=up_offset_V,
    )
    return params, profile


def load_merged_cycle_params(
    record_csv: str | Path | None = None,
    cycle_from: int | None = 5,
    cycle_to: int | None = None,
    skip_activation_cycles: int = 4,
    ocp_dir: str | Path | None = None,
    theta_c_bottom: float | None = None,
    theta_c_top: float | None = None,
    csanmax_eff: float | None = 6.0,
    r_ohm_eff: float | None = None,
    up_offset_V: float = 0.0,
) -> tuple[dict, CurrentProfile]:
    """Load params for a continuous cycles>=cycle_from soft-label run.

    This mirrors load_cycle_params() and only swaps the profile reader. The
    training-side parameter/OCP installation logic is otherwise unchanged.
    """
    params = makeParams()
    profile = read_record_current_profile_merged(
        record_csv=record_csv,
        cycle_from=cycle_from,
        cycle_to=cycle_to,
        skip_activation_cycles=skip_activation_cycles,
    )
    params = _apply_profile_to_params(params, profile)
    if r_ohm_eff is not None and np.isfinite(r_ohm_eff):
        params["R_ohm_eff"] = np.float64(r_ohm_eff)
    params = install_assb_ocp_tables(
        params=params,
        ocp_dir=ocp_dir,
        profile=profile,
        theta_c_bottom=theta_c_bottom,
        theta_c_top=theta_c_top,
        csanmax_eff=csanmax_eff,
        up_offset_V=up_offset_V,
    )
    return params, profile


# -----------------------------------------------------------------------------
# SPM spatial and temporal domains
# -----------------------------------------------------------------------------

def get_r_domain(n_r: int, params: dict) -> dict:
    if int(n_r) < 3:
        raise ValueError("n_r must be at least 3.")
    r_a = np.linspace(0.0, float(params["Rs_a"]), int(n_r), dtype=np.float64)
    r_c = np.linspace(0.0, float(params["Rs_c"]), int(n_r), dtype=np.float64)
    return {
        "n_r": int(n_r),
        "r_a": r_a,
        "r_c": r_c,
        "dR_a": np.float64(r_a[1] - r_a[0]),
        "dR_c": np.float64(r_c[1] - r_c[0]),
    }


def get_t_domain(profile: CurrentProfile) -> dict:
    t = profile.time_s.astype(np.float64, copy=True)
    return {
        "t": t,
        "n_t": int(t.size),
        "dt_median": np.float64(np.median(np.diff(t))) if t.size > 1 else np.float64(0.0),
    }


def make_sim_config(t_dom: dict, r_dom: dict) -> dict:
    return {**t_dom, **r_dom}


# -----------------------------------------------------------------------------
# Core physics closures
# -----------------------------------------------------------------------------

def surface_flux_from_current(I_now: float, params: dict) -> tuple[np.float64, np.float64]:
    """Return (J_a, J_c) in kmol m^-2 s^-1 from applied current in A.

    Important:
        a is the fixed negative electrode and c is the fixed positive electrode.
        The signs below already encode charge/discharge role switching:

            +I charge    -> J_a < 0, J_c > 0
            -I discharge -> J_a > 0, J_c < 0

        Do not swap OCP/i0/D_s functions between electrodes here.
    """
    I_now = np.float64(I_now)
    F = np.float64(params["F"])
    V_a = np.float64(params.get("V_a", params["A_a"] * params["L_a"]))
    V_c = np.float64(params.get("V_c", params["A_c"] * params["L_c"]))
    j_a = -I_now * np.float64(params["Rs_a"]) / (
        np.float64(3.0) * np.float64(params["eps_s_a"]) * F * V_a
    )
    j_c = I_now * np.float64(params["Rs_c"]) / (
        np.float64(3.0) * np.float64(params["eps_s_c"]) * F * V_c
    )
    return np.float64(j_a), np.float64(j_c)


def _uocp_c_raw(params: dict) -> Callable:
    # Training-side thermo_assb.py should expose Uocp_c_raw when R_ohm is wrapped
    # into Uocp_c. If it does not, fall back to Uocp_c and still add dynamic R_ohm.
    return params.get("Uocp_c_raw", params["Uocp_c"])


def _safe_i0(value: Any, floor: float = 1.0e-30) -> np.float64:
    return np.float64(max(float(abs(_to_scalar(value))), floor))


def compute_potentials(
    cse_a: float,
    cse_c: float,
    I_now: float,
    j_a: float,
    j_c: float,
    params: dict,
    deg_i0_a: float = 1.0,
    linearized_bv: bool = True,
    include_ohmic: bool = True,
) -> tuple[np.float64, np.float64, np.float64, np.float64, np.float64, np.float64]:
    """Compute phie and phis_c for the current surface concentrations.

    Returns: phie, phis_c, U_a, U_c_raw, eta_a, eta_c.
    """
    ce = np.float64(params.get("ce0", 1.2))
    T = np.float64(params["T"])
    R = np.float64(params["R"])
    F = np.float64(params["F"])

    # Electrode-fixed OCPs:
    #   U_a = Li-In/In negative-electrode OCP
    #   U_c = NMC811 positive-electrode OCP
    # They are not swapped during charge. The charge/discharge role switch is
    # carried by J_a/J_c, eta_a/eta_c and I_now * R_ohm_eff.
    U_a = _to_scalar(params["Uocp_a"](_torch_scalar(cse_a), params["csanmax"]))
    U_c = _to_scalar(_uocp_c_raw(params)(_torch_scalar(cse_c), params["cscamax"]))

    i0_a = _safe_i0(
        params["i0_a"](
            _torch_scalar(cse_a),
            _torch_scalar(ce),
            T,
            params["alpha_a"],
            params["csanmax"],
            R,
            _torch_scalar(deg_i0_a),
        )
    )
    i0_c = _safe_i0(
        params["i0_c"](
            _torch_scalar(cse_c),
            _torch_scalar(ce),
            T,
            params["alpha_c"],
            params["cscamax"],
            R,
        )
    )

    if linearized_bv:
        eta_a = np.float64(j_a) * R * T / i0_a
        eta_c = np.float64(j_c) * R * T / i0_c
    else:
        eta_a = (np.float64(2.0) * R * T / F) * np.arcsinh(np.float64(j_a) * F / (np.float64(2.0) * i0_a))
        eta_c = (np.float64(2.0) * R * T / F) * np.arcsinh(np.float64(j_c) * F / (np.float64(2.0) * i0_c))

    # Negative-electrode solid potential is the reference: phi_s,a = 0.
    phie = np.float64(-U_a - eta_a)

    # Positive terminal voltage / cathode solid potential closure.
    # Dynamic ohmic shift follows the training-side current sign convention:
    # discharge I<0 lowers the terminal voltage; charge I>0 raises it.
    ohmic = np.float64(I_now) * np.float64(params.get("R_ohm_eff", 0.0)) if include_ohmic else np.float64(0.0)
    phis_c = np.float64(phie + U_c + eta_c + ohmic)

    return np.float64(phie), np.float64(phis_c), U_a, U_c, np.float64(eta_a), np.float64(eta_c)


def _ds_a_array(params: dict, n_r: int) -> np.ndarray:
    val = _to_numpy(params["D_s_a"](params["T"], params["R"])).reshape(-1)
    if val.size == 1:
        return np.full(n_r, float(val[0]), dtype=np.float64)
    if val.size != n_r:
        return np.full(n_r, float(val[0]), dtype=np.float64)
    return val.astype(np.float64)


def _ds_c_array(params: dict, cs_c: np.ndarray, deg_ds_c: float) -> np.ndarray:
    deg_arr = np.full_like(cs_c, float(deg_ds_c), dtype=np.float64)
    val = _to_numpy(params["D_s_c"](cs_c, params["T"], params["R"], params["cscamax"], deg_arr)).reshape(-1)
    if val.size == 1:
        return np.full_like(cs_c, float(val[0]), dtype=np.float64)
    return val.astype(np.float64)


def _grad_ds_c_numeric(params: dict, cs_c: np.ndarray, deg_ds_c: float, step: float = 1.0e-4) -> np.ndarray:
    h = np.float64(max(step, 1.0e-9))
    cmax = np.float64(params["cscamax"])
    ds_p = _ds_c_array(params, np.clip(cs_c + h, 0.0, cmax), deg_ds_c)
    ds_m = _ds_c_array(params, np.clip(cs_c - h, 0.0, cmax), deg_ds_c)
    return (ds_p - ds_m) / (np.float64(2.0) * h)


# -----------------------------------------------------------------------------
# Implicit radial diffusion update
# -----------------------------------------------------------------------------

def _tridiag(ds: np.ndarray, dt: float, dr: float) -> np.ndarray:
    ds = np.asarray(ds, dtype=np.float64)
    a = np.float64(1.0) + np.float64(2.0) * ds * dt / (dr**2)
    b = -ds * dt / (dr**2)
    mat = np.diag(a, 0)
    if ds.size > 1:
        mat += np.diag(b[1:], -1) + np.diag(b[:-1], 1)
    mat[0, :] = 0.0
    mat[-1, :] = 0.0
    mat[0, 0] = -1.0 / dr
    mat[0, 1] = 1.0 / dr
    mat[-1, -1] = 1.0 / dr
    mat[-1, -2] = -1.0 / dr
    return mat


def _rhs(dt: float, r: np.ndarray, ddr_cs: np.ndarray, ds: np.ndarray, dds_dcs: np.ndarray, cs: np.ndarray, bound_grad: float) -> np.ndarray:
    r_safe = np.clip(r, a_min=1.0e-12, a_max=None)
    out = dt * (np.float64(2.0) / r_safe) * ddr_cs * ds
    out += dt * (ddr_cs**2) * dds_dcs
    out += cs
    out[0] = 0.0
    out[-1] = bound_grad
    return out.astype(np.float64)


def _implicit_update_particle(
    cs_prev: np.ndarray,
    r: np.ndarray,
    dr: float,
    dt: float,
    ds: np.ndarray,
    dds_dcs: np.ndarray,
    j_surface: float,
    cs_max: float,
) -> np.ndarray:
    # Boundary condition: D_s * dc/dr |R = -J.
    ddr = np.gradient(cs_prev, r, axis=0, edge_order=2)
    ddr[0] = 0.0
    bound_grad = -np.float64(j_surface) / max(float(ds[-1]), 1.0e-30)
    ddr[-1] = bound_grad
    A = _tridiag(ds, dt, dr)
    b = _rhs(dt, r, ddr, ds, dds_dcs, cs_prev, bound_grad=bound_grad)
    out = np.linalg.solve(A, b)
    return np.clip(out, 0.0, float(cs_max)).astype(np.float64)


# -----------------------------------------------------------------------------
# Solver execution
# -----------------------------------------------------------------------------

def init_solution(config: dict, params: dict, profile: CurrentProfile, deg_i0_a: float, linearized_bv: bool, include_ohmic: bool) -> dict:
    n_t = int(config["n_t"])
    n_r = int(config["n_r"])
    sol = {
        "ce": np.float64(params.get("ce0", 1.2)),
        "phis_a": np.float64(0.0),
        "phie": np.zeros(n_t, dtype=np.float64),
        "phis_c": np.zeros(n_t, dtype=np.float64),
        "cs_a": np.zeros((n_t, n_r), dtype=np.float64),
        "cs_c": np.zeros((n_t, n_r), dtype=np.float64),
        "j_a": np.zeros(n_t, dtype=np.float64),
        "j_c": np.zeros(n_t, dtype=np.float64),
        "I_profile": profile.current_A.astype(np.float64, copy=True),
        "eta_a": np.zeros(n_t, dtype=np.float64),
        "eta_c": np.zeros(n_t, dtype=np.float64),
        "Uocp_a": np.zeros(n_t, dtype=np.float64),
        "Uocp_c": np.zeros(n_t, dtype=np.float64),
    }
    sol["cs_a"][0, :] = np.float64(params["cs_a0"])
    sol["cs_c"][0, :] = np.float64(params["cs_c0"])
    j_a0, j_c0 = surface_flux_from_current(profile.current_A[0], params)
    sol["j_a"][0] = j_a0
    sol["j_c"][0] = j_c0
    phie, phis_c, Ua, Uc, eta_a, eta_c = compute_potentials(
        sol["cs_a"][0, -1],
        sol["cs_c"][0, -1],
        profile.current_A[0],
        j_a0,
        j_c0,
        params,
        deg_i0_a=deg_i0_a,
        linearized_bv=linearized_bv,
        include_ohmic=include_ohmic,
    )
    sol["phie"][0] = phie
    sol["phis_c"][0] = phis_c
    sol["Uocp_a"][0] = Ua
    sol["Uocp_c"][0] = Uc
    sol["eta_a"][0] = eta_a
    sol["eta_c"][0] = eta_c
    return sol


def integrate_cycle(
    params: dict,
    profile: CurrentProfile,
    n_r: int = 65,
    deg_i0_a: float = 1.0,
    deg_ds_c: float = 1.0,
    linearized_bv: bool = True,
    include_ohmic: bool = True,
    verbose: bool = True,
) -> tuple[dict, dict]:
    r_dom = get_r_domain(n_r, params)
    t_dom = get_t_domain(profile)
    config = make_sim_config(t_dom, r_dom)
    sol = init_solution(config, params, profile, deg_i0_a, linearized_bv, include_ohmic)

    t = config["t"]
    r_a = config["r_a"]
    r_c = config["r_c"]
    dR_a = config["dR_a"]
    dR_c = config["dR_c"]
    n_t = config["n_t"]

    start = time.time()
    if verbose:
        print(
            f"INFO: ASSB cycle integration | profile={getattr(profile, 'profile_label', profile.cycle)} | n_t={n_t} | n_r={n_r} | "
            f"tmax={float(profile.tmax):.6g}s | I_min={profile.current_A.min():.6g}A | I_max={profile.current_A.max():.6g}A"
        )

    for k in range(1, n_t):
        dt = np.float64(t[k] - t[k - 1])
        if dt <= 0:
            raise ValueError(f"Non-positive dt at step {k}: {dt}")
        I_now = np.float64(profile.current_A[k])
        j_a, j_c = surface_flux_from_current(I_now, params)
        sol["j_a"][k] = j_a
        sol["j_c"][k] = j_c

        # Anode / Li-In effective pseudo-particle.
        ds_a = _ds_a_array(params, n_r)
        dds_a = np.zeros(n_r, dtype=np.float64)
        sol["cs_a"][k, :] = _implicit_update_particle(
            cs_prev=sol["cs_a"][k - 1, :],
            r=r_a,
            dr=dR_a,
            dt=dt,
            ds=ds_a,
            dds_dcs=dds_a,
            j_surface=j_a,
            cs_max=params["csanmax"],
        )

        # Cathode / NMC811 representative particle.
        ds_c = _ds_c_array(params, sol["cs_c"][k - 1, :], deg_ds_c)
        dds_c = _grad_ds_c_numeric(params, sol["cs_c"][k - 1, :], deg_ds_c)
        sol["cs_c"][k, :] = _implicit_update_particle(
            cs_prev=sol["cs_c"][k - 1, :],
            r=r_c,
            dr=dR_c,
            dt=dt,
            ds=ds_c,
            dds_dcs=dds_c,
            j_surface=j_c,
            cs_max=params["cscamax"],
        )

        phie, phis_c, Ua, Uc, eta_a, eta_c = compute_potentials(
            sol["cs_a"][k, -1],
            sol["cs_c"][k, -1],
            I_now,
            j_a,
            j_c,
            params,
            deg_i0_a=deg_i0_a,
            linearized_bv=linearized_bv,
            include_ohmic=include_ohmic,
        )
        sol["phie"][k] = phie
        sol["phis_c"][k] = phis_c
        sol["Uocp_a"][k] = Ua
        sol["Uocp_c"][k] = Uc
        sol["eta_a"][k] = eta_a
        sol["eta_c"][k] = eta_c

        if verbose and (k == 1 or k == n_t - 1 or k % max(1, n_t // 10) == 0):
            print(f"  step {k:5d}/{n_t-1:5d} | t={t[k]:10.3f}s | I={I_now:+.6e} A | V={phis_c:+.6f}")

    if verbose:
        print(f"INFO: integration completed in {time.time() - start:.2f} s")
    return config, sol



# -----------------------------------------------------------------------------
# V3 voltage-alignment / soft-label calibration helpers
# -----------------------------------------------------------------------------

def _evaluate_positive_ocp_window(params: dict, theta_c: np.ndarray, theta_c_bottom: float, theta_c_top: float, up_offset_V: float = 0.0) -> np.ndarray:
    pos_soc = np.asarray(params["_ocp_pos_soc"], dtype=np.float64)
    pos_u = np.asarray(params["_ocp_pos_u"], dtype=np.float64)
    soc = (float(theta_c_bottom) - np.asarray(theta_c, dtype=np.float64)) / max(float(theta_c_bottom) - float(theta_c_top), 1.0e-12)
    soc = np.clip(soc, 0.0, 1.0)
    return np.interp(soc, pos_soc, pos_u) + float(up_offset_V)


def recompute_potentials_for_solution(
    params: dict,
    profile: CurrentProfile,
    sol: dict,
    deg_i0_a: float = 1.0,
    linearized_bv: bool = True,
    include_ohmic: bool = True,
) -> None:
    n_t = int(sol["phis_c"].shape[0])
    for k in range(n_t):
        phie, phis_c, Ua, Uc, eta_a, eta_c = compute_potentials(
            sol["cs_a"][k, -1],
            sol["cs_c"][k, -1],
            profile.current_A[k],
            sol["j_a"][k],
            sol["j_c"][k],
            params,
            deg_i0_a=deg_i0_a,
            linearized_bv=linearized_bv,
            include_ohmic=include_ohmic,
        )
        sol["phie"][k] = phie
        sol["phis_c"][k] = phis_c
        sol["Uocp_a"][k] = Ua
        sol["Uocp_c"][k] = Uc
        sol["eta_a"][k] = eta_a
        sol["eta_c"][k] = eta_c


def fit_v3_voltage_alignment(
    params: dict,
    profile: CurrentProfile,
    sol: dict,
    theta_bottom_min: float = 0.86,
    theta_bottom_max: float = 0.96,
    theta_top_min: float = 0.25,
    theta_top_max: float = 0.55,
    theta_grid_n: int = 41,
    up_offset_min: float = -0.35,
    up_offset_max: float = 0.05,
    r_ohm_min: float = 0.0,
    r_ohm_max: float = 500.0,
) -> dict:
    """Fit v3 cathode OCP window + OCP offset + R_ohm_eff to the measured voltage.

    The concentration trajectories are not changed. Only terminal potential
    closure terms are calibrated:

        V = U_p(window, theta_c) + U_p_offset - U_n - eta_a + eta_c + I R_ohm_eff

    The purpose is to produce a voltage-consistent soft-label set for data-loss
    testing while recording all fitted terms explicitly.
    """
    if profile.voltage_V is None:
        return {"enabled": False, "reason": "no experimental voltage column"}
    if len(profile.voltage_V) != len(sol["phis_c"]):
        return {"enabled": False, "reason": "voltage length does not match solution length"}

    E = np.asarray(profile.voltage_V, dtype=np.float64)
    I = np.asarray(profile.current_A, dtype=np.float64)
    theta_c = np.asarray(sol["cs_c"][:, -1], dtype=np.float64) / float(params["cscamax"])
    # base term excluding positive-electrode OCP and ohmic contribution
    base = -np.asarray(sol["Uocp_a"], dtype=np.float64) - np.asarray(sol["eta_a"], dtype=np.float64) + np.asarray(sol["eta_c"], dtype=np.float64)

    theta_grid_n = max(int(theta_grid_n), 3)
    bottoms = np.linspace(theta_bottom_min, theta_bottom_max, theta_grid_n)
    tops = np.linspace(theta_top_min, theta_top_max, theta_grid_n)

    best: dict[str, Any] | None = None
    X = np.column_stack([np.ones_like(I), I])
    for bottom in bottoms:
        for top in tops:
            if top >= bottom - 0.05:
                continue
            Uc0 = _evaluate_positive_ocp_window(params, theta_c, bottom, top, up_offset_V=0.0)
            rhs = E - (Uc0 + base)
            try:
                up_offset, r_ohm = np.linalg.lstsq(X, rhs, rcond=None)[0]
            except Exception:
                continue
            if not (up_offset_min <= up_offset <= up_offset_max):
                continue
            if not (r_ohm_min <= r_ohm <= r_ohm_max):
                continue
            Vfit = Uc0 + base + up_offset + I * r_ohm
            err = Vfit - E
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            bias = float(np.mean(err))
            max_abs = float(np.max(np.abs(err)))
            corr = float(np.corrcoef(Vfit, E)[0, 1]) if len(Vfit) > 2 else float("nan")
            rec = {
                "enabled": True,
                "theta_c_bottom": float(bottom),
                "theta_c_top": float(top),
                "U_p_offset_V": float(up_offset),
                "R_ohm_eff": float(r_ohm),
                "V_mae_model_exp": mae,
                "V_rmse_model_exp": rmse,
                "V_bias_model_exp": bias,
                "V_max_abs_model_exp": max_abs,
                "V_corr_model_exp": corr,
            }
            if best is None or mae < best["V_mae_model_exp"]:
                best = rec

    if best is None:
        return {
            "enabled": False,
            "reason": "no candidate inside bounds",
            "bounds": {
                "theta_bottom": [theta_bottom_min, theta_bottom_max],
                "theta_top": [theta_top_min, theta_top_max],
                "U_p_offset_V": [up_offset_min, up_offset_max],
                "R_ohm_eff": [r_ohm_min, r_ohm_max],
            },
        }

    sol["phis_c_raw"] = np.asarray(sol["phis_c"], dtype=np.float64).copy()
    params["theta_c_bottom"] = np.float64(best["theta_c_bottom"])
    params["theta_c_top"] = np.float64(best["theta_c_top"])
    params["U_p_offset_V"] = np.float64(best["U_p_offset_V"])
    params["R_ohm_eff"] = np.float64(best["R_ohm_eff"])
    params = install_assb_ocp_tables(
        params=params,
        ocp_dir=params.get("ocp_source_dir", DEFAULT_OCP_DIR),
        profile=profile,
        theta_c_bottom=best["theta_c_bottom"],
        theta_c_top=best["theta_c_top"],
        csanmax_eff=float(params["csanmax"]),
        up_offset_V=best["U_p_offset_V"],
    )
    recompute_potentials_for_solution(
        params=params,
        profile=profile,
        sol=sol,
        deg_i0_a=float(params.get("deg_i0_a_current", 1.0)),
        linearized_bv=True,
        include_ohmic=True,
    )
    sol["voltage_alignment_V"] = np.asarray(sol["phis_c"], dtype=np.float64) - np.asarray(sol["phis_c_raw"], dtype=np.float64)
    return best

# -----------------------------------------------------------------------------
# Dataset export compatible with util/init_pinn.py
# -----------------------------------------------------------------------------

def _params_matrix(n: int, deg_i0_a: float, deg_ds_c: float) -> np.ndarray:
    out = np.zeros((int(n), 2), dtype=np.float64)
    out[:, 0] = np.float64(deg_i0_a)
    out[:, 1] = np.float64(deg_ds_c)
    return out


def _save_state_dataset(path: Path, x_train: np.ndarray, y_train: np.ndarray, deg_i0_a: float, deg_ds_c: float) -> None:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    x_params_train = _params_matrix(y_train.shape[0], deg_i0_a, deg_ds_c)
    np.savez(
        path,
        x_train=x_train,
        y_train=y_train,
        x_params_train=x_params_train,
    )



def save_solution_and_datasets(
    output_dir: str | Path,
    config: dict,
    sol: dict,
    params: dict,
    profile: CurrentProfile,
    deg_i0_a: float = 1.0,
    deg_ds_c: float = 1.0,
    fit_report: dict | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = np.asarray(config["t"], dtype=np.float64)
    r_a = np.asarray(config["r_a"], dtype=np.float64)
    r_c = np.asarray(config["r_c"], dtype=np.float64)
    n_t = int(config["n_t"])
    n_r = int(config["n_r"])

    phis_c_raw = np.asarray(sol.get("phis_c_raw", sol["phis_c"]), dtype=np.float64)
    voltage_alignment = np.asarray(sol.get("voltage_alignment_V", sol["phis_c"] - phis_c_raw), dtype=np.float64)
    role_codes = current_role_code_arrays(sol["I_profile"])

    solution_path = out / "solution.npz"
    np.savez(
        solution_path,
        t=t,
        r_a=r_a,
        r_c=r_c,
        cs_a=sol["cs_a"],
        cs_c=sol["cs_c"],
        phie=sol["phie"],
        phis_c=sol["phis_c"],
        phis_c_raw=phis_c_raw,
        voltage_alignment_V=voltage_alignment,
        phis_a=np.zeros_like(sol["phie"]),
        ce=np.full_like(sol["phie"], np.float64(params.get("ce0", 1.2))),
        I_profile=sol["I_profile"],
        current_mode_code=role_codes["current_mode_code"],
        positive_electrode_role_code=role_codes["positive_electrode_role_code"],
        negative_electrode_role_code=role_codes["negative_electrode_role_code"],
        j_a=sol["j_a"],
        j_c=sol["j_c"],
        eta_a=sol["eta_a"],
        eta_c=sol["eta_c"],
        Uocp_a=sol["Uocp_a"],
        Uocp_c=sol["Uocp_c"],
        voltage_exp=np.array([]) if profile.voltage_V is None else profile.voltage_V,
        cycle=np.array([-1 if profile.cycle is None else profile.cycle], dtype=np.int64),
        cycle_profile=(
            np.array([], dtype=np.int64)
            if not hasattr(profile, "cycle_profile")
            else np.asarray(getattr(profile, "cycle_profile"), dtype=np.int64)
        ),
    )

    # Potential datasets: x = [t]
    x_t = t.reshape(-1, 1)
    _save_state_dataset(out / "data_phie.npz", x_t, sol["phie"], deg_i0_a, deg_ds_c)
    _save_state_dataset(out / "data_phis_c.npz", x_t, sol["phis_c"], deg_i0_a, deg_ds_c)

    # Concentration datasets: x = [t, r]
    tt_a, rr_a = np.meshgrid(t, r_a, indexing="ij")
    tt_c, rr_c = np.meshgrid(t, r_c, indexing="ij")
    x_cs_a = np.column_stack([tt_a.reshape(-1), rr_a.reshape(-1)])
    x_cs_c = np.column_stack([tt_c.reshape(-1), rr_c.reshape(-1)])
    _save_state_dataset(out / "data_cs_a.npz", x_cs_a, sol["cs_a"].reshape(-1), deg_i0_a, deg_ds_c)
    _save_state_dataset(out / "data_cs_c.npz", x_cs_c, sol["cs_c"].reshape(-1), deg_i0_a, deg_ds_c)

    fit_report = {} if fit_report is None else fit_report
    metadata = {
        "source": "spm_int_assb_cycle.py",
        "version": "v3_voltage_aligned_electrode_fixed_sign_v4",
        "record_csv": None if profile.csv_path is None else str(profile.csv_path),
        "cycle": profile.cycle,
        "merge_cycles": bool(getattr(profile, "merge_cycles", False)),
        "cycle_from": getattr(profile, "cycle_from", None),
        "cycle_to": getattr(profile, "cycle_to", None),
        "selected_cycles": getattr(profile, "selected_cycles", None),
        "n_t": n_t,
        "n_r": n_r,
        "tmax_s": float(t[-1]),
        "I_min_A": float(np.min(sol["I_profile"])),
        "I_max_A": float(np.max(sol["I_profile"])),
        "charge_discharge_role_summary": charge_discharge_role_summary(sol["I_profile"]),
        "solution_role_code_description": {
            "current_mode_code": "-1 discharge, 0 rest, +1 charge",
            "positive_electrode_role_code": "-1 anode role, 0 rest, +1 cathode role",
            "negative_electrode_role_code": "-1 anode role, 0 rest, +1 cathode role",
        },
        "capacity_charge_Ah": float(profile.charge_capacity_Ah),
        "capacity_discharge_Ah": float(profile.discharge_capacity_Ah),
        "capacity_used_Ah": float(params.get("C", 0.0)),
        "theta_a0": float(params.get("theta_a0", params["cs_a0"] / params["csanmax"])),
        "theta_c0": float(params.get("theta_c0", params["cs_c0"] / params["cscamax"])),
        "theta_c_bottom": float(params.get("theta_c_bottom", np.nan)),
        "theta_c_top": float(params.get("theta_c_top", np.nan)),
        "U_p_offset_V": float(params.get("U_p_offset_V", 0.0)),
        "T_K": float(params["T"]),
        "csanmax": float(params["csanmax"]),
        "cscamax": float(params["cscamax"]),
        "Rs_a_m": float(params["Rs_a"]),
        "Rs_c_m": float(params["Rs_c"]),
        "eps_s_a": float(params["eps_s_a"]),
        "eps_s_c": float(params["eps_s_c"]),
        "V_a_m3": float(params["V_a"]),
        "V_c_m3": float(params["V_c"]),
        "R_ohm_eff": float(params.get("R_ohm_eff", 0.0)),
        "ocp_source_dir": str(params.get("ocp_source_dir", "")),
        "ocp_positive_min_V": float(np.min(params.get("_ocp_pos_u", [np.nan]))),
        "ocp_positive_max_V": float(np.max(params.get("_ocp_pos_u", [np.nan]))),
        "ocp_negative_min_V": float(np.min(params.get("_ocp_neg_u", [np.nan]))),
        "ocp_negative_max_V": float(np.max(params.get("_ocp_neg_u", [np.nan]))),
        "deg_i0_a": float(deg_i0_a),
        "deg_ds_c": float(deg_ds_c),
        "fit_report": fit_report,
    }
    metadata_path = out / "soft_label_summary.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "solution": solution_path,
        "data_phie": out / "data_phie.npz",
        "data_phis_c": out / "data_phis_c.npz",
        "data_cs_a": out / "data_cs_a.npz",
        "data_cs_c": out / "data_cs_c.npz",
        "summary": metadata_path,
    }


# -----------------------------------------------------------------------------
# Validation / CLI
# -----------------------------------------------------------------------------


def sanity_check_solution(params: dict, profile: CurrentProfile, sol: dict) -> dict[str, float]:
    cs_a = np.asarray(sol["cs_a"], dtype=np.float64)
    cs_c = np.asarray(sol["cs_c"], dtype=np.float64)
    phis_c = np.asarray(sol["phis_c"], dtype=np.float64)
    out = {
        "cs_a_min": float(np.min(cs_a)),
        "cs_a_max": float(np.max(cs_a)),
        "cs_c_min": float(np.min(cs_c)),
        "cs_c_max": float(np.max(cs_c)),
        "theta_a_min": float(np.min(cs_a) / params["csanmax"]),
        "theta_a_max": float(np.max(cs_a) / params["csanmax"]),
        "theta_c_min": float(np.min(cs_c) / params["cscamax"]),
        "theta_c_max": float(np.max(cs_c) / params["cscamax"]),
        "Uocp_a_min": float(np.min(sol["Uocp_a"])),
        "Uocp_a_max": float(np.max(sol["Uocp_a"])),
        "Uocp_c_min": float(np.min(sol["Uocp_c"])),
        "Uocp_c_max": float(np.max(sol["Uocp_c"])),
        "V_model_min": float(np.min(phis_c)),
        "V_model_max": float(np.max(phis_c)),
    }
    if "phis_c_raw" in sol:
        raw = np.asarray(sol["phis_c_raw"], dtype=np.float64)
        out["V_raw_min"] = float(np.min(raw))
        out["V_raw_max"] = float(np.max(raw))
    if profile.voltage_V is not None and len(profile.voltage_V) == len(phis_c):
        err = phis_c - profile.voltage_V
        out["V_exp_min"] = float(np.min(profile.voltage_V))
        out["V_exp_max"] = float(np.max(profile.voltage_V))
        out["V_mae_model_exp"] = float(np.mean(np.abs(err)))
        out["V_rmse_model_exp"] = float(np.sqrt(np.mean(err**2)))
        out["V_bias_model_exp"] = float(np.mean(err))
        out["V_max_abs_model_exp"] = float(np.max(np.abs(err)))
        out["V_corr_model_exp"] = float(np.corrcoef(phis_c, profile.voltage_V)[0, 1]) if len(phis_c) > 2 else float("nan")
    return out



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate ASSB SPM v3/v4 soft labels with electrode-fixed charge/discharge sign provenance.")
    p.add_argument("--record_csv", "--record-csv", dest="record_csv", type=str, default=str(DEFAULT_RECORD_CSV), help="Path to record_extracted.csv")
    p.add_argument("--ocp_dir", "--ocp-dir", dest="ocp_dir", type=str, default=str(DEFAULT_OCP_DIR), help="Path to ocp_estimation_outputs folder")
    p.add_argument("--cycle", type=int, default=5, help="Cycle number to generate soft labels for; default skips activation cycles and uses 5")
    p.add_argument("--merge_cycles", "--merge-cycles", dest="merge_cycles", action="store_true", help="Generate one continuous soft-label sequence for cycles >= --cycle_from instead of a single cycle")
    p.add_argument("--cycle_from", "--cycle-from", dest="cycle_from", type=int, default=5, help="First cycle included when --merge_cycles is used")
    p.add_argument("--cycle_to", "--cycle-to", dest="cycle_to", type=int, default=None, help="Optional last cycle included when --merge_cycles is used")
    p.add_argument("--skip_activation_cycles", "--skip-activation-cycles", dest="skip_activation_cycles", type=int, default=4, help="Activation cycles to skip when --cycle is omitted")
    p.add_argument("--output_dir", "--out-dir", dest="output_dir", type=str, default=str(REPO_ROOT / "Data" / "assb_soft_labels_cycle5_v3"), help="Output directory for solution.npz and data_*.npz")
    p.add_argument("--n_r", "--n-r", dest="n_r", type=int, default=64, help="Number of radial grid points per particle")

    # Effective SPM prior / v2 defaults.
    p.add_argument("--csanmax_eff", "--csanmax-eff", dest="csanmax_eff", type=float, default=6.0, help="Li-In effective csanmax used only by the soft-label generator")
    p.add_argument("--theta_c_bottom", "--theta-c-bottom", dest="theta_c_bottom", type=float, default=0.90, help="Initial cathode usable-window bottom stoichiometry")
    p.add_argument("--theta_c_top", "--theta-c-top", dest="theta_c_top", type=float, default=None, help="Initial cathode usable-window top stoichiometry; default inferred from capacity")
    p.add_argument("--r_ohm_eff", "--r-ohm-eff", dest="r_ohm_eff", type=float, default=100.0, help="Initial R_ohm_eff before v3 fit")
    p.add_argument("--up_offset_V", "--up-offset-V", dest="up_offset_V", type=float, default=0.0, help="Initial cathode OCP offset before v3 fit")

    # Data-generation parameter dimensions.
    p.add_argument("--deg_i0_a", type=float, default=1.0, help="Negative-electrode effective i0 scaling factor")
    p.add_argument("--deg_ds_c", type=float, default=1.0, help="Cathode diffusivity degradation factor")
    p.add_argument("--nonlinear_bv", action="store_true", help="Use nonlinear Butler-Volmer inversion instead of linearized BV")
    p.add_argument("--no_ohmic", action="store_true", help="Disable dynamic I(t)*R_ohm_eff terminal-voltage shift")

    # V3 fit controls.
    p.add_argument("--no_fit_v3", "--no-fit-v3", dest="no_fit_v3", action="store_true", help="Disable v3 voltage alignment and keep v2-like closure")
    p.add_argument("--fit_theta_bottom_min", type=float, default=0.86)
    p.add_argument("--fit_theta_bottom_max", type=float, default=0.96)
    p.add_argument("--fit_theta_top_min", type=float, default=0.25)
    p.add_argument("--fit_theta_top_max", type=float, default=0.55)
    p.add_argument("--fit_theta_grid_n", type=int, default=41)
    p.add_argument("--fit_up_offset_min", type=float, default=-0.35)
    p.add_argument("--fit_up_offset_max", type=float, default=0.05)
    p.add_argument("--fit_r_ohm_min", type=float, default=0.0)
    p.add_argument("--fit_r_ohm_max", type=float, default=500.0)

    p.add_argument("--quiet", action="store_true", help="Suppress step progress printing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    default_single_output = str(REPO_ROOT / "Data" / "assb_soft_labels_cycle5_v3")
    if args.merge_cycles and args.output_dir == default_single_output:
        args.output_dir = str(REPO_ROOT / "Data" / f"assb_soft_labels_cycles{args.cycle_from}plus_v3")

    if args.merge_cycles:
        params, profile = load_merged_cycle_params(
            record_csv=args.record_csv,
            cycle_from=args.cycle_from,
            cycle_to=args.cycle_to,
            skip_activation_cycles=args.skip_activation_cycles,
            ocp_dir=args.ocp_dir,
            theta_c_bottom=args.theta_c_bottom,
            theta_c_top=args.theta_c_top,
            csanmax_eff=args.csanmax_eff,
            r_ohm_eff=args.r_ohm_eff,
            up_offset_V=args.up_offset_V,
        )
    else:
        params, profile = load_cycle_params(
            record_csv=args.record_csv,
            cycle=args.cycle,
            skip_activation_cycles=args.skip_activation_cycles,
            ocp_dir=args.ocp_dir,
            theta_c_bottom=args.theta_c_bottom,
            theta_c_top=args.theta_c_top,
            csanmax_eff=args.csanmax_eff,
            r_ohm_eff=args.r_ohm_eff,
            up_offset_V=args.up_offset_V,
        )
    params["deg_i0_a_current"] = np.float64(args.deg_i0_a)

    config, sol = integrate_cycle(
        params=params,
        profile=profile,
        n_r=args.n_r,
        deg_i0_a=args.deg_i0_a,
        deg_ds_c=args.deg_ds_c,
        linearized_bv=not args.nonlinear_bv,
        include_ohmic=not args.no_ohmic,
        verbose=not args.quiet,
    )

    fit_report: dict[str, Any] = {"enabled": False, "reason": "disabled"}
    if not args.no_fit_v3 and profile.voltage_V is not None and not args.nonlinear_bv and not args.no_ohmic:
        fit_report = fit_v3_voltage_alignment(
            params=params,
            profile=profile,
            sol=sol,
            theta_bottom_min=args.fit_theta_bottom_min,
            theta_bottom_max=args.fit_theta_bottom_max,
            theta_top_min=args.fit_theta_top_min,
            theta_top_max=args.fit_theta_top_max,
            theta_grid_n=args.fit_theta_grid_n,
            up_offset_min=args.fit_up_offset_min,
            up_offset_max=args.fit_up_offset_max,
            r_ohm_min=args.fit_r_ohm_min,
            r_ohm_max=args.fit_r_ohm_max,
        )
        print("INFO: v3 voltage-alignment fit:")
        for k, v in fit_report.items():
            print(f"  {k:24s}: {v}")

    paths = save_solution_and_datasets(
        output_dir=args.output_dir,
        config=config,
        sol=sol,
        params=params,
        profile=profile,
        deg_i0_a=args.deg_i0_a,
        deg_ds_c=args.deg_ds_c,
        fit_report=fit_report,
    )
    checks = sanity_check_solution(params, profile, sol)
    print("INFO: soft-label files written:")
    for name, path in paths.items():
        print(f"  {name:12s}: {path}")
    print("INFO: sanity checks:")
    for k, v in checks.items():
        print(f"  {k:22s}: {v:.8g}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
