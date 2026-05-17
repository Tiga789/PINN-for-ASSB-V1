# -*- coding: utf-8 -*-
"""Feature schema and scaler rules for ASSB ModelFin_111.

The strict ModelFin_111 SOH task is deliberately leakage-sensitive. This file
keeps one source of truth for which feature names are allowed in each mode and
which names are forbidden because they encode the target capacity/SOH.

Feature modes
-------------
p0_history
    Conservative baseline: cycle/order/current/rest history only.
p1_107a_strict
    Main ModelFin_111 mode: non-leaking 107A physical-summary features plus
    history features. This is the only mode intended for the strict-30 claim.
p2_upperbound
    Diagnostic-only mode. It may allow capacity-equivalent columns only when
    ``allow_upper_bound=True``. Results from this mode must not be reported as
    strict prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    provenance: str
    description: str = ""
    strict_allowed: bool = True


# Forbidden tokens are intentionally broad. If a future feature column contains
# any of these substrings, it must be renamed with a clear diagnostic prefix or
# excluded from the strict model.
FORBIDDEN_COLUMN_TOKENS: Tuple[str, ...] = (
    "soh_obs", "soh_true", "soh_label", "soh_target", "target_soh",
    "q_obs", "q_true", "q_label", "q_target", "q_ref", "capacity",
    "放电容量", "容量", "q_discharge", "q_dis_", "discharge_capacity",
    "label", "target", "y_true", "ground_truth",
)

# Current-cycle complete discharge duration can become a capacity equivalent
# under fixed-current cycling, so the strict schema does not include it.
CAPACITY_EQUIVALENT_TOKENS: Tuple[str, ...] = (
    "q_discharge", "q_dis_", "q_obs", "q_target", "capacity",
    "discharge_capacity", "full_discharge_duration", "cutoff_time",
)

P0_HISTORY_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("cycle_norm", "history", "cycle_id", "normalized cycle index"),
    FeatureSpec("throughput_before_norm", "history", "I_profile", "cumulative absolute throughput before this cycle"),
    FeatureSpec("throughput_end_norm", "history", "I_profile", "cumulative absolute throughput through this cycle"),
    FeatureSpec("I_abs_mean_norm", "history", "I_profile", "mean absolute current in this cycle"),
    FeatureSpec("I_abs_max_norm", "history", "I_profile", "max absolute current in this cycle"),
    FeatureSpec("charge_current_level_norm", "history", "I_profile", "mean positive current level"),
    FeatureSpec("discharge_current_level_norm", "history", "I_profile", "mean negative-current magnitude"),
    FeatureSpec("rest_fraction", "history", "I_profile", "fraction of points with zero current"),
)

P1_107A_STATE_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("theta_a_mean_start", "107A_state", "107A_pred", "negative mean stoichiometry at cycle start"),
    FeatureSpec("theta_a_mean_end", "107A_state", "107A_pred", "negative mean stoichiometry at cycle end"),
    FeatureSpec("theta_a_mean_window", "107A_state", "107A_pred", "negative mean stoichiometry range"),
    FeatureSpec("theta_c_mean_start", "107A_state", "107A_pred", "positive mean stoichiometry at cycle start"),
    FeatureSpec("theta_c_mean_end", "107A_state", "107A_pred", "positive mean stoichiometry at cycle end"),
    FeatureSpec("theta_c_mean_window", "107A_state", "107A_pred", "positive mean stoichiometry range"),
    FeatureSpec("theta_a_surface_start", "107A_state", "107A_pred", "negative surface stoichiometry at cycle start"),
    FeatureSpec("theta_a_surface_end", "107A_state", "107A_pred", "negative surface stoichiometry at cycle end"),
    FeatureSpec("theta_c_surface_start", "107A_state", "107A_pred", "positive surface stoichiometry at cycle start"),
    FeatureSpec("theta_c_surface_end", "107A_state", "107A_pred", "positive surface stoichiometry at cycle end"),
    FeatureSpec("cs_a_radial_energy_mean", "107A_state", "107A_pred", "negative radial deviation energy"),
    FeatureSpec("cs_c_radial_energy_mean", "107A_state", "107A_pred", "positive radial deviation energy"),
    FeatureSpec("cs_a_surface_minus_mean_abs", "107A_state", "107A_pred", "negative surface minus mean absolute average"),
    FeatureSpec("cs_c_surface_minus_mean_abs", "107A_state", "107A_pred", "positive surface minus mean absolute average"),
    FeatureSpec("polarization_mean", "107A_potential", "107A_pred", "mean phis_c - phie"),
    FeatureSpec("polarization_abs_mean", "107A_potential", "107A_pred", "mean absolute phis_c - phie"),
    FeatureSpec("polarization_std", "107A_potential", "107A_pred", "std of phis_c - phie"),
    FeatureSpec("current_norm_polarization_abs_mean", "107A_potential", "107A_pred+I_profile", "|phis_c-phie| normalized by |I|"),
    FeatureSpec("phie_mean", "107A_potential", "107A_pred", "mean effective ionic potential"),
    FeatureSpec("phis_c_mean", "107A_potential", "107A_pred", "mean positive solid potential"),
    FeatureSpec("rest_phis_relax_slope", "107A_potential", "107A_pred+I_profile", "rest-segment phis_c slope proxy"),
    FeatureSpec("rest_phie_relax_slope", "107A_potential", "107A_pred+I_profile", "rest-segment phie slope proxy"),
)

# Diagnostic features are explicitly not allowed in strict mode unless the user
# opts into p2_upperbound and records that the result is not a prediction claim.
P2_DIAGNOSTIC_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("q_charge_norm", "diagnostic", "I_profile", "charge Ah/C feature; diagnostic only", strict_allowed=False),
    FeatureSpec("q_discharge_norm", "diagnostic", "I_profile", "discharge Ah/C feature; capacity-equivalent diagnostic only", strict_allowed=False),
    FeatureSpec("duration_norm", "diagnostic", "time_profile", "cycle duration; may encode capacity under CC; diagnostic only", strict_allowed=False),
)


def specs_for_mode(mode: str, *, allow_upper_bound: bool = False) -> Tuple[FeatureSpec, ...]:
    m = str(mode).strip().lower()
    if m in {"p0", "p0_history", "history"}:
        return P0_HISTORY_FEATURES
    if m in {"p1", "p1_107a_strict", "107a_strict", "strict"}:
        return P0_HISTORY_FEATURES + P1_107A_STATE_FEATURES
    if m in {"p2", "p2_upperbound", "upperbound", "diagnostic"}:
        if not allow_upper_bound:
            raise ValueError("p2_upperbound requires allow_upper_bound=True; do not use it for strict prediction")
        return P0_HISTORY_FEATURES + P1_107A_STATE_FEATURES + P2_DIAGNOSTIC_FEATURES
    raise ValueError(f"Unknown ASSB111 feature mode: {mode}")


def canonical_feature_names(mode: str = "p1_107a_strict", *, allow_upper_bound: bool = False) -> List[str]:
    return [s.name for s in specs_for_mode(mode, allow_upper_bound=allow_upper_bound)]


def is_forbidden_column(name: str, *, strict: bool = True) -> bool:
    low = str(name).strip().lower()
    tokens = FORBIDDEN_COLUMN_TOKENS if strict else CAPACITY_EQUIVALENT_TOKENS
    return any(tok in low for tok in tokens)


def forbidden_columns(columns: Iterable[str], *, strict: bool = True) -> List[str]:
    return [str(c) for c in columns if is_forbidden_column(str(c), strict=strict)]


def select_feature_columns(
    frame: pd.DataFrame,
    mode: str = "p1_107a_strict",
    *,
    allow_upper_bound: bool = False,
    allow_missing: bool = False,
) -> List[str]:
    """Return feature columns from ``frame`` that are valid for the mode.

    Missing columns raise by default. For early pipeline smoke tests,
    ``allow_missing=True`` can return the subset that exists, but strict training
    scripts should keep the default.
    """
    wanted = canonical_feature_names(mode, allow_upper_bound=allow_upper_bound)
    missing = [c for c in wanted if c not in frame.columns]
    if missing and not allow_missing:
        raise KeyError(f"ASSB111 feature frame missing columns for {mode}: {missing}")
    cols = [c for c in wanted if c in frame.columns]
    bad = forbidden_columns(cols, strict=True)
    if bad and not allow_upper_bound:
        raise ValueError(f"Forbidden target/capacity-equivalent feature columns in strict mode: {bad}")
    return cols


def schema_dict(mode: str = "p1_107a_strict", *, allow_upper_bound: bool = False) -> Dict[str, Any]:
    specs = specs_for_mode(mode, allow_upper_bound=allow_upper_bound)
    return {
        "feature_mode": mode,
        "allow_upper_bound": bool(allow_upper_bound),
        "n_features": len(specs),
        "feature_columns": [s.name for s in specs],
        "features": [asdict(s) for s in specs],
        "forbidden_column_tokens": list(FORBIDDEN_COLUMN_TOKENS),
        "capacity_equivalent_tokens": list(CAPACITY_EQUIVALENT_TOKENS),
        "strict_note": "P1 strict features must not include observed SOH/capacity labels or current-cycle discharge-capacity equivalents.",
    }


def write_schema_json(path: PathLike, mode: str = "p1_107a_strict", *, allow_upper_bound: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(schema_dict(mode, allow_upper_bound=allow_upper_bound)), f, ensure_ascii=False, indent=2, sort_keys=True)


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


def fit_standard_scaler(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    fit_mask: Optional[Sequence[bool]] = None,
    fit_cycles: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Fit a simple standard scaler with explicit fit provenance."""
    if fit_mask is None:
        if fit_cycles is not None:
            if "cycle_id" not in frame.columns:
                raise KeyError("fit_cycles was provided but frame has no cycle_id column")
            mask = frame["cycle_id"].astype(int).isin([int(c) for c in fit_cycles]).to_numpy()
        else:
            mask = np.ones(len(frame), dtype=bool)
    else:
        mask = np.asarray(fit_mask, dtype=bool)
    if mask.size != len(frame):
        raise ValueError("fit_mask length does not match frame")
    if not np.any(mask):
        raise RuntimeError("No rows selected for scaler fit")
    x = frame.loc[mask, list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(med, inds[1])
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0
    fit_cycle_values: List[int] = []
    if "cycle_id" in frame.columns:
        fit_cycle_values = [int(c) for c in frame.loc[mask, "cycle_id"].astype(int).tolist()]
    return {
        "type": "standard",
        "feature_columns": list(feature_columns),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "median_impute": [float(v) if math.isfinite(float(v)) else 0.0 for v in med],
        "fit_cycles": fit_cycle_values,
        "fit_n_rows": int(np.sum(mask)),
    }


def transform_with_scaler(frame: pd.DataFrame, scaler: Mapping[str, Any]) -> np.ndarray:
    cols = list(scaler["feature_columns"])
    x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.asarray(scaler.get("median_impute", np.zeros(len(cols))), dtype=float)
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(med, inds[1])
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0
    return (x - mean) / std


def write_scaler_json(scaler: Mapping[str, Any], path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(scaler)), f, ensure_ascii=False, indent=2, sort_keys=True)


def load_scaler_json(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


__all__ = [
    "FeatureSpec",
    "FORBIDDEN_COLUMN_TOKENS",
    "CAPACITY_EQUIVALENT_TOKENS",
    "P0_HISTORY_FEATURES",
    "P1_107A_STATE_FEATURES",
    "P2_DIAGNOSTIC_FEATURES",
    "specs_for_mode",
    "canonical_feature_names",
    "is_forbidden_column",
    "forbidden_columns",
    "select_feature_columns",
    "schema_dict",
    "write_schema_json",
    "fit_standard_scaler",
    "transform_with_scaler",
    "write_scaler_json",
    "load_scaler_json",
]
