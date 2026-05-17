# -*- coding: utf-8 -*-
"""Cycle-level feature extraction for ASSB ModelFin_111.

This module extracts non-leaking SOH-prediction features from existing 107A
state outputs and the continuous soft-label solution. It never creates SOH
labels; capacity/SOH labels are merged later by scripts after leakage checks.

The extractor is defensive because historical ASSB evaluators saved arrays with
slightly different names. It prefers paired 107A prediction arrays such as
``cs_a_pred`` / ``cs_c_pred`` / ``phie_pred`` / ``phis_c_pred``. If those are
not available, it falls back to the continuous solution fields. The goal is to
produce one row per cycle with physical summaries, not to retrain or alter
ModelFin_107A.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np
import pandas as pd

try:
    from util.assb111_split import load_manifest, split_for_cycles
except Exception:  # pragma: no cover
    from assb111_split import load_manifest, split_for_cycles  # type: ignore

PathLike = Union[str, Path]

DEFAULT_CS_A_MAX = 6.0
DEFAULT_CS_C_MAX = 51.8
REST_I_EPS = 1.0e-12


# ---------------------------------------------------------------------------
# Generic array helpers
# ---------------------------------------------------------------------------

def _first_key(keys: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    key_set = set(keys)
    for c in candidates:
        if c in key_set:
            return c
    lower = {str(k).lower(): k for k in key_set}
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def _npz_to_dict(path: Optional[PathLike]) -> Dict[str, np.ndarray]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _as_1d(x: Any, *, name: str = "array", dtype=None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = np.ravel(arr)
    if arr.size == 0:
        raise RuntimeError(f"{name} is empty")
    return arr


def _as_state_2d(x: Any, *, name: str = "state") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        # A flattened concentration is ambiguous. Keep as (N,1) so cycle-level
        # summaries still work, but radial features will be zero.
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim > 2:
        return arr.reshape(arr.shape[0], -1)
    raise RuntimeError(f"{name} has invalid ndim={arr.ndim}")


def _safe_nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(np.nanmean(x[finite]))


def _safe_nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return 0.0
    return float(np.nanstd(x[finite]))


def _safe_range(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return float("nan")
    return float(np.nanmax(x[finite]) - np.nanmin(x[finite]))


def _trapz(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2 or y.size < 2:
        return 0.0
    order = np.argsort(t)
    tt = t[order]
    yy = y[order]
    return float(np.sum((tt[1:] - tt[:-1]) * (yy[1:] + yy[:-1]) * 0.5))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    xx = x[mask] - float(np.nanmean(x[mask]))
    yy = y[mask] - float(np.nanmean(y[mask]))
    den = float(np.sum(xx * xx))
    if den <= 1e-30:
        return 0.0
    return float(np.sum(xx * yy) / den)


def _normalize_series(vals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return out
    lo = float(np.nanmin(arr[mask]))
    hi = float(np.nanmax(arr[mask]))
    if hi - lo <= 1e-30:
        return out
    out[mask] = (arr[mask] - lo) / (hi - lo)
    return out


# ---------------------------------------------------------------------------
# State array discovery
# ---------------------------------------------------------------------------

def _get_time_cycle_current(solution: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_key = _first_key(solution.keys(), ["t_global_s", "time_s", "t_s", "t", "time"])
    c_key = _first_key(solution.keys(), ["cycle_id", "cycle", "cycle_index"])
    i_key = _first_key(solution.keys(), ["I_profile", "current_A", "I_A", "I", "current"])
    if t_key is None or c_key is None or i_key is None:
        raise KeyError(f"solution npz must contain time, cycle_id and current arrays. Available={list(solution.keys())}")
    t = _as_1d(solution[t_key], name=t_key, dtype=float)
    cycle_raw = _as_1d(solution[c_key], name=c_key, dtype=int)
    I = _as_1d(solution[i_key], name=i_key, dtype=float)
    if cycle_raw.size == 1 and t.size > 1:
        cycle = np.full(t.size, int(cycle_raw[0]), dtype=int)
    elif cycle_raw.size == t.size:
        cycle = cycle_raw.astype(int)
    else:
        raise RuntimeError(f"cycle array length mismatch: {cycle_raw.size} vs time {t.size}")
    if I.size != t.size:
        raise RuntimeError(f"current array length mismatch: {I.size} vs time {t.size}")
    order = np.argsort(t)
    return t[order], cycle[order], I[order]


def _get_prediction_array(
    state_npz: Mapping[str, np.ndarray],
    solution_npz: Mapping[str, np.ndarray],
    base: str,
) -> Optional[np.ndarray]:
    candidates = [f"{base}_pred", f"pred_{base}", f"{base}_prediction", f"y_pred_{base}", base]
    key = _first_key(state_npz.keys(), candidates)
    if key is not None:
        return np.asarray(state_npz[key])
    key = _first_key(solution_npz.keys(), [base])
    if key is not None:
        return np.asarray(solution_npz[key])
    return None


def _get_cycle_for_state(
    state_npz: Mapping[str, np.ndarray],
    solution_cycle: np.ndarray,
    state_rows: int,
    *,
    concentration: bool,
) -> Optional[np.ndarray]:
    candidates = ["cycle_id_cs", "cycle_id_state", "cycle_id_sampled", "cycle_id"] if concentration else [
        "cycle_id_potential", "cycle_id_phi", "cycle_id_state", "cycle_id"
    ]
    key = _first_key(state_npz.keys(), candidates)
    if key is not None:
        arr = _as_1d(state_npz[key], name=key, dtype=int)
        if arr.size == state_rows:
            return arr.astype(int)
    if solution_cycle.size == state_rows:
        return solution_cycle.astype(int)
    return None


def _align_potential_to_solution(
    values: Optional[np.ndarray],
    state_npz: Mapping[str, np.ndarray],
    solution_cycle: np.ndarray,
    solution_n: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if values is None:
        return None, None
    vals = _as_1d(values, name="potential", dtype=float)
    if vals.size == solution_n:
        return vals, solution_cycle
    cyc = _get_cycle_for_state(state_npz, solution_cycle, vals.size, concentration=False)
    return vals, cyc


def _align_concentration(
    values: Optional[np.ndarray],
    state_npz: Mapping[str, np.ndarray],
    solution_cycle: np.ndarray,
    solution_n: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if values is None:
        return None, None
    arr = _as_state_2d(values, name="concentration")
    if arr.shape[0] == solution_n:
        return arr, solution_cycle
    cyc = _get_cycle_for_state(state_npz, solution_cycle, arr.shape[0], concentration=True)
    return arr, cyc


# ---------------------------------------------------------------------------
# Feature calculations
# ---------------------------------------------------------------------------

def _mean_radial(arr: np.ndarray) -> np.ndarray:
    arr = _as_state_2d(arr)
    return np.nanmean(arr, axis=1)


def _surface_radial(arr: np.ndarray) -> np.ndarray:
    arr = _as_state_2d(arr)
    return arr[:, -1]


def _radial_energy(arr: np.ndarray) -> np.ndarray:
    arr = _as_state_2d(arr)
    cbar = np.nanmean(arr, axis=1, keepdims=True)
    return np.nanmean((arr - cbar) ** 2, axis=1)


def _surface_minus_mean(arr: np.ndarray) -> np.ndarray:
    arr = _as_state_2d(arr)
    return arr[:, -1] - np.nanmean(arr, axis=1)


def _state_cycle_summary(
    rows: Dict[str, float],
    *,
    prefix: str,
    arr: Optional[np.ndarray],
    cyc: Optional[np.ndarray],
    cid: int,
    cmax: float,
) -> None:
    if arr is None or cyc is None:
        # Keep the output schema stable even when a historical 107A eval NPZ
        # lacks cycle_id_cs / cycle_id_sampled for concentration arrays.
        # The strict schema expects theta_* state-window columns and cs_* radial
        # columns.  Do not accidentally create theta_*_surface_minus_mean_abs.
        for name in ["mean_start", "mean_end", "mean_window", "surface_start", "surface_end"]:
            rows[f"theta_{prefix}_{name}"] = np.nan
        for name in ["radial_energy_mean", "surface_minus_mean_abs"]:
            rows[f"cs_{prefix}_{name}"] = np.nan
        return
    m = np.asarray(cyc, dtype=int) == int(cid)
    if not np.any(m):
        return
    sub = _as_state_2d(arr[m])
    mean = _mean_radial(sub)
    surf = _surface_radial(sub)
    theta_mean = mean / float(cmax)
    theta_surf = surf / float(cmax)
    rows[f"theta_{prefix}_mean_start"] = float(theta_mean[0])
    rows[f"theta_{prefix}_mean_end"] = float(theta_mean[-1])
    rows[f"theta_{prefix}_mean_window"] = _safe_range(theta_mean)
    rows[f"theta_{prefix}_surface_start"] = float(theta_surf[0])
    rows[f"theta_{prefix}_surface_end"] = float(theta_surf[-1])
    rows[f"cs_{prefix}_radial_energy_mean"] = _safe_nanmean(_radial_energy(sub))
    rows[f"cs_{prefix}_surface_minus_mean_abs"] = _safe_nanmean(np.abs(_surface_minus_mean(sub)))


def _potential_cycle_summary(
    rows: Dict[str, float],
    *,
    t: np.ndarray,
    cycle: np.ndarray,
    I: np.ndarray,
    phie: Optional[np.ndarray],
    phie_cycle: Optional[np.ndarray],
    phis_c: Optional[np.ndarray],
    phis_c_cycle: Optional[np.ndarray],
    cid: int,
) -> None:
    # Potential arrays are normally aligned to solution; if not, compute summaries
    # on their own cycle arrays and skip current-normalized/rest features.
    if phie is not None and phie_cycle is not None:
        mp = np.asarray(phie_cycle, dtype=int) == int(cid)
        phie_sub = np.asarray(phie, dtype=float)[mp]
        rows["phie_mean"] = _safe_nanmean(phie_sub)
        rows["phie_std"] = _safe_nanstd(phie_sub)
    else:
        phie_sub = np.asarray([], dtype=float)
        rows["phie_mean"] = np.nan
        rows["phie_std"] = np.nan
    if phis_c is not None and phis_c_cycle is not None:
        ms = np.asarray(phis_c_cycle, dtype=int) == int(cid)
        phis_sub = np.asarray(phis_c, dtype=float)[ms]
        rows["phis_c_mean"] = _safe_nanmean(phis_sub)
        rows["phis_c_std"] = _safe_nanstd(phis_sub)
    else:
        phis_sub = np.asarray([], dtype=float)
        rows["phis_c_mean"] = np.nan
        rows["phis_c_std"] = np.nan
    # Differential/polarization only if arrays have matching cycle-local length.
    if phie_sub.size and phis_sub.size and phie_sub.size == phis_sub.size:
        pol = phis_sub - phie_sub
        rows["polarization_mean"] = _safe_nanmean(pol)
        rows["polarization_abs_mean"] = _safe_nanmean(np.abs(pol))
        rows["polarization_std"] = _safe_nanstd(pol)
    else:
        rows["polarization_mean"] = np.nan
        rows["polarization_abs_mean"] = np.nan
        rows["polarization_std"] = np.nan
    # Current-normalized and rest slopes require solution alignment.
    msol = cycle == int(cid)
    if np.any(msol) and phie is not None and phis_c is not None and len(phie) == len(t) and len(phis_c) == len(t):
        tt = t[msol]
        ii = I[msol]
        pe = np.asarray(phie, dtype=float)[msol]
        ps = np.asarray(phis_c, dtype=float)[msol]
        pol = ps - pe
        active = np.abs(ii) > 1.0e-8
        if np.any(active):
            rows["current_norm_polarization_abs_mean"] = _safe_nanmean(np.abs(pol[active]) / np.maximum(np.abs(ii[active]), 1.0e-12))
        else:
            rows["current_norm_polarization_abs_mean"] = 0.0
        rest = np.abs(ii) <= REST_I_EPS
        if np.sum(rest) >= 2:
            rows["rest_phis_relax_slope"] = _linear_slope(tt[rest], ps[rest])
            rows["rest_phie_relax_slope"] = _linear_slope(tt[rest], pe[rest])
        else:
            rows["rest_phis_relax_slope"] = 0.0
            rows["rest_phie_relax_slope"] = 0.0
    else:
        rows.setdefault("current_norm_polarization_abs_mean", np.nan)
        rows.setdefault("rest_phis_relax_slope", 0.0)
        rows.setdefault("rest_phie_relax_slope", 0.0)


def extract_assb111_cycle_features(
    *,
    solution_npz: PathLike,
    state_eval_npz: Optional[PathLike] = None,
    cycle_table_csv: Optional[PathLike] = None,
    split_manifest_json: Optional[PathLike] = None,
    cycle_from: int = 5,
    cycle_to: int = 521,
    cs_a_max: float = DEFAULT_CS_A_MAX,
    cs_c_max: float = DEFAULT_CS_C_MAX,
) -> pd.DataFrame:
    """Extract one non-label feature row per cycle.

    Parameters
    ----------
    solution_npz:
        Continuous v2 massclosed candidate solution; must contain time, cycle_id,
        and current. It may also contain state fields.
    state_eval_npz:
        Optional 107A paired evaluation npz. Prediction fields are preferred.
    cycle_table_csv:
        Optional historical cycle table. Used only for non-label operating
        columns; target columns are not copied.
    split_manifest_json:
        Optional strict30 manifest. When present, a ``split`` column is assigned
        from the manifest.
    """
    solution = _npz_to_dict(solution_npz)
    state_npz = _npz_to_dict(state_eval_npz)
    t, cycle, I = _get_time_cycle_current(solution)

    # Prefer 107A prediction arrays, but fall back to the continuous solution
    # fields if the selected 107A NPZ contains sampled concentration arrays
    # without cycle_id_cs/cycle_id_sampled provenance.  This keeps smoke runs
    # robust while preserving the strict no-SOH-label rule.
    cs_a_raw_state = _get_prediction_array(state_npz, {}, "cs_a")
    cs_c_raw_state = _get_prediction_array(state_npz, {}, "cs_c")
    phie_raw_state = _get_prediction_array(state_npz, {}, "phie")
    phis_raw_state = _get_prediction_array(state_npz, {}, "phis_c")

    cs_a_raw_solution = _get_prediction_array({}, solution, "cs_a")
    cs_c_raw_solution = _get_prediction_array({}, solution, "cs_c")
    phie_raw_solution = _get_prediction_array({}, solution, "phie")
    phis_raw_solution = _get_prediction_array({}, solution, "phis_c")

    cs_a, cs_a_cycle = _align_concentration(cs_a_raw_state, state_npz, cycle, len(t))
    if (cs_a is None or cs_a_cycle is None) and cs_a_raw_solution is not None:
        cs_a, cs_a_cycle = _align_concentration(cs_a_raw_solution, {}, cycle, len(t))

    cs_c, cs_c_cycle = _align_concentration(cs_c_raw_state, state_npz, cycle, len(t))
    if (cs_c is None or cs_c_cycle is None) and cs_c_raw_solution is not None:
        cs_c, cs_c_cycle = _align_concentration(cs_c_raw_solution, {}, cycle, len(t))

    phie, phie_cycle = _align_potential_to_solution(phie_raw_state, state_npz, cycle, len(t))
    if (phie is None or phie_cycle is None) and phie_raw_solution is not None:
        phie, phie_cycle = _align_potential_to_solution(phie_raw_solution, {}, cycle, len(t))

    phis_c, phis_cycle = _align_potential_to_solution(phis_raw_state, state_npz, cycle, len(t))
    if (phis_c is None or phis_cycle is None) and phis_raw_solution is not None:
        phis_c, phis_cycle = _align_potential_to_solution(phis_raw_solution, {}, cycle, len(t))

    cycle_ids = [int(c) for c in sorted(np.unique(cycle)) if int(cycle_from) <= int(c) <= int(cycle_to)]
    rows: List[Dict[str, float]] = []
    throughput_running = 0.0
    q_net_running = 0.0

    for cid in cycle_ids:
        m = cycle == cid
        if not np.any(m):
            continue
        tt = t[m]
        ii = I[m]
        charge_C = _trapz(tt, np.clip(ii, 0.0, None))
        discharge_C = _trapz(tt, np.clip(-ii, 0.0, None))
        through_C = _trapz(tt, np.abs(ii))
        net_C = _trapz(tt, ii)
        row: Dict[str, float] = {
            "cycle_id": float(cid),
            "n_points": float(np.sum(m)),
            "t_start_s": float(np.nanmin(tt)),
            "t_end_s": float(np.nanmax(tt)),
            "duration_s": float(np.nanmax(tt) - np.nanmin(tt)),
            "q_charge_cycle_C": charge_C,
            "q_discharge_cycle_C": discharge_C,
            "q_net_cycle_C": net_C,
            "throughput_cycle_C": through_C,
            "throughput_before_C": throughput_running,
            "throughput_end_C": throughput_running + through_C,
            "q_net_before_C": q_net_running,
            "q_net_end_C": q_net_running + net_C,
            "I_abs_mean_A": _safe_nanmean(np.abs(ii)),
            "I_abs_max_A": float(np.nanmax(np.abs(ii))) if ii.size else 0.0,
            "charge_current_level_A": _safe_nanmean(ii[ii > 1e-12]) if np.any(ii > 1e-12) else 0.0,
            "discharge_current_level_A": _safe_nanmean(np.abs(ii[ii < -1e-12])) if np.any(ii < -1e-12) else 0.0,
            "rest_fraction": float(np.mean(np.abs(ii) <= REST_I_EPS)) if ii.size else 0.0,
        }
        _state_cycle_summary(row, prefix="a", arr=cs_a, cyc=cs_a_cycle, cid=cid, cmax=cs_a_max)
        _state_cycle_summary(row, prefix="c", arr=cs_c, cyc=cs_c_cycle, cid=cid, cmax=cs_c_max)
        _potential_cycle_summary(
            row,
            t=t,
            cycle=cycle,
            I=I,
            phie=phie,
            phie_cycle=phie_cycle,
            phis_c=phis_c,
            phis_c_cycle=phis_cycle,
            cid=cid,
        )
        rows.append(row)
        throughput_running += through_C
        q_net_running += net_C

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No cycle features extracted")

    # Optional merge of non-target columns from the existing cycle table.
    if cycle_table_csv:
        ctab_path = Path(cycle_table_csv)
        if ctab_path.exists():
            ctab = pd.read_csv(ctab_path)
            ctab_cols = [
                c for c in ctab.columns
                if c in {"cycle_id", "complete_cycle", "complete_cycle_from_solution"}
                or (c.startswith("voltage_") and c not in {"voltage_exp"})
            ]
            if "cycle_id" in ctab_cols:
                frame = frame.merge(ctab[ctab_cols].drop_duplicates("cycle_id"), on="cycle_id", how="left")

    # Normalized non-leaking operating/history features. q_discharge_norm is
    # generated only as a diagnostic column and excluded by the strict schema.
    frame["cycle_norm"] = _normalize_series(frame["cycle_id"])
    frame["throughput_before_norm"] = _normalize_series(frame["throughput_before_C"])
    frame["throughput_end_norm"] = _normalize_series(frame["throughput_end_C"])
    frame["I_abs_mean_norm"] = _normalize_series(frame["I_abs_mean_A"])
    frame["I_abs_max_norm"] = _normalize_series(frame["I_abs_max_A"])
    frame["charge_current_level_norm"] = _normalize_series(frame["charge_current_level_A"])
    frame["discharge_current_level_norm"] = _normalize_series(frame["discharge_current_level_A"])
    frame["q_charge_norm"] = _normalize_series(frame["q_charge_cycle_C"])
    frame["q_discharge_norm"] = _normalize_series(frame["q_discharge_cycle_C"])
    frame["duration_norm"] = _normalize_series(frame["duration_s"])

    if split_manifest_json:
        manifest = load_manifest(split_manifest_json)
        frame["split"] = split_for_cycles(frame["cycle_id"].astype(int).to_numpy(), manifest)
    return frame.sort_values("cycle_id").reset_index(drop=True)


def feature_summary(frame: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = [c for c in frame.columns if c != "cycle_id" and pd.api.types.is_numeric_dtype(frame[c])]
    nan_counts = {c: int(frame[c].isna().sum()) for c in numeric_cols}
    return {
        "n_cycles": int(len(frame)),
        "cycle_min": int(frame["cycle_id"].min()) if len(frame) else None,
        "cycle_max": int(frame["cycle_id"].max()) if len(frame) else None,
        "columns": list(frame.columns),
        "numeric_columns": numeric_cols,
        "nan_counts": nan_counts,
        "split_counts": {str(k): int(v) for k, v in frame["split"].value_counts().to_dict().items()} if "split" in frame.columns else {},
    }


def write_feature_outputs(frame: pd.DataFrame, output_csv: PathLike, output_json: Optional[PathLike] = None) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")
    if output_json is not None:
        s = feature_summary(frame)
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(_json_clean(s), f, ensure_ascii=False, indent=2, sort_keys=True)


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


__all__ = [
    "extract_assb111_cycle_features",
    "feature_summary",
    "write_feature_outputs",
]
