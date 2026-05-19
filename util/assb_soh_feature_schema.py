# -*- coding: utf-8 -*-
"""Feature schema for ASSB strict30 SOH robustness work (D7 / ModelFin_112).

This module is intentionally independent from the previous ASSB-111 schema.  It
keeps the same strict rule: SOH/capacity labels and capacity-equivalent fields
must never be model inputs for strict prediction.  It adds G0-G4 feature groups
for feature audit and robust SOH head training:

G0  cycle / throughput / current history baseline
G1  frozen ModelFin_107A state-summary features
G2  raw voltage-health features extracted from record/solution time series
G3  current-switching, polarization, and rest-relaxation features
G4  combined strict feature set = G0 + G1 + G2 + G3

The helper functions here are small and dependency-light so that audit scripts
can run before touching the main PINN training code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math
import re

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


# Broad token list by design.  If a safe feature accidentally matches one of
# these names, rename it with a non-label name or use allow_upper_bound=True and
# mark the experiment as diagnostic only.
FORBIDDEN_COLUMN_TOKENS: Tuple[str, ...] = (
    "soh_obs",
    "soh_true",
    "soh_label",
    "soh_target",
    "target_soh",
    "q_obs",
    "q_true",
    "q_label",
    "q_target",
    "q_ref",
    "capacity",
    "cap_ah",
    "cap_mAh".lower(),
    "放电容量",
    "充电容量",
    "容量",
    "q_discharge",
    "q_dis_",
    "discharge_capacity",
    "label",
    "target",
    "y_true",
    "ground_truth",
)

CAPACITY_EQUIVALENT_TOKENS: Tuple[str, ...] = (
    "q_discharge",
    "q_dis_",
    "q_obs",
    "q_target",
    "capacity",
    "full_discharge_duration",
    "cutoff_time",
    "discharge_duration_full",
)

G0_HISTORY: Tuple[FeatureSpec, ...] = (
    FeatureSpec("cycle_norm", "G0_history", "cycle_id", "normalized cycle id"),
    FeatureSpec("throughput_before_norm", "G0_history", "I_profile", "cumulative throughput before cycle"),
    FeatureSpec("throughput_end_norm", "G0_history", "I_profile", "cumulative throughput through cycle"),
    FeatureSpec("I_abs_mean_norm", "G0_history", "I_profile", "mean absolute current"),
    FeatureSpec("I_abs_max_norm", "G0_history", "I_profile", "max absolute current"),
    FeatureSpec("charge_current_level_norm", "G0_history", "I_profile", "mean positive current level"),
    FeatureSpec("discharge_current_level_norm", "G0_history", "I_profile", "mean negative-current magnitude"),
    FeatureSpec("rest_fraction", "G0_history", "I_profile", "fraction of rest points"),
)

G1_107A_STATE: Tuple[FeatureSpec, ...] = (
    FeatureSpec("theta_a_mean_start", "G1_107A_state", "107A_pred", "negative mean stoichiometry at cycle start"),
    FeatureSpec("theta_a_mean_end", "G1_107A_state", "107A_pred", "negative mean stoichiometry at cycle end"),
    FeatureSpec("theta_a_mean_window", "G1_107A_state", "107A_pred", "negative mean stoichiometry range"),
    FeatureSpec("theta_c_mean_start", "G1_107A_state", "107A_pred", "positive mean stoichiometry at cycle start"),
    FeatureSpec("theta_c_mean_end", "G1_107A_state", "107A_pred", "positive mean stoichiometry at cycle end"),
    FeatureSpec("theta_c_mean_window", "G1_107A_state", "107A_pred", "positive mean stoichiometry range"),
    FeatureSpec("theta_a_surface_start", "G1_107A_state", "107A_pred", "negative surface stoichiometry at cycle start"),
    FeatureSpec("theta_a_surface_end", "G1_107A_state", "107A_pred", "negative surface stoichiometry at cycle end"),
    FeatureSpec("theta_c_surface_start", "G1_107A_state", "107A_pred", "positive surface stoichiometry at cycle start"),
    FeatureSpec("theta_c_surface_end", "G1_107A_state", "107A_pred", "positive surface stoichiometry at cycle end"),
    FeatureSpec("cs_a_radial_energy_mean", "G1_107A_state", "107A_pred", "negative radial deviation energy"),
    FeatureSpec("cs_c_radial_energy_mean", "G1_107A_state", "107A_pred", "positive radial deviation energy"),
    FeatureSpec("cs_a_surface_minus_mean_abs", "G1_107A_state", "107A_pred", "negative surface-minus-mean abs"),
    FeatureSpec("cs_c_surface_minus_mean_abs", "G1_107A_state", "107A_pred", "positive surface-minus-mean abs"),
    FeatureSpec("polarization_mean", "G1_107A_potential", "107A_pred", "mean phis_c - phie"),
    FeatureSpec("polarization_abs_mean", "G1_107A_potential", "107A_pred", "mean abs phis_c - phie"),
    FeatureSpec("polarization_std", "G1_107A_potential", "107A_pred", "std phis_c - phie"),
    FeatureSpec("current_norm_polarization_abs_mean", "G1_107A_potential", "107A_pred+I", "current-normalized abs polarization"),
    FeatureSpec("phie_mean", "G1_107A_potential", "107A_pred", "mean effective ionic potential"),
    FeatureSpec("phis_c_mean", "G1_107A_potential", "107A_pred", "mean positive solid potential"),
    FeatureSpec("rest_phis_relax_slope", "G1_107A_potential", "107A_pred+rest", "rest phis_c relaxation slope"),
    FeatureSpec("rest_phie_relax_slope", "G1_107A_potential", "107A_pred+rest", "rest phie relaxation slope"),
)

# The voltage features are online current-cycle signatures.  They do not use
# observed SOH/capacity labels.  Experiments using them should be described as
# cycle-level online SOH estimation, not future-cycle forecasting.
G2_VOLTAGE: Tuple[FeatureSpec, ...] = (
    FeatureSpec("voltage_start", "G2_voltage_health", "record_voltage", "first voltage of cycle"),
    FeatureSpec("voltage_end", "G2_voltage_health", "record_voltage", "last voltage of cycle"),
    FeatureSpec("voltage_mean", "G2_voltage_health", "record_voltage", "cycle voltage mean"),
    FeatureSpec("voltage_std", "G2_voltage_health", "record_voltage", "cycle voltage std"),
    FeatureSpec("voltage_min", "G2_voltage_health", "record_voltage", "cycle voltage min"),
    FeatureSpec("voltage_max", "G2_voltage_health", "record_voltage", "cycle voltage max"),
    FeatureSpec("charge_voltage_mean", "G2_voltage_health", "record_voltage", "mean voltage in charge steps"),
    FeatureSpec("discharge_voltage_mean", "G2_voltage_health", "record_voltage", "mean voltage in discharge steps"),
    FeatureSpec("charge_voltage_slope", "G2_voltage_health", "record_voltage+time", "linear voltage/time slope in charge"),
    FeatureSpec("discharge_voltage_slope", "G2_voltage_health", "record_voltage+time", "linear voltage/time slope in discharge"),
    FeatureSpec("v_at_tfrac_010", "G2_voltage_health", "record_voltage+time", "voltage at 10% normalized cycle time"),
    FeatureSpec("v_at_tfrac_025", "G2_voltage_health", "record_voltage+time", "voltage at 25% normalized cycle time"),
    FeatureSpec("v_at_tfrac_050", "G2_voltage_health", "record_voltage+time", "voltage at 50% normalized cycle time"),
    FeatureSpec("v_at_tfrac_075", "G2_voltage_health", "record_voltage+time", "voltage at 75% normalized cycle time"),
    FeatureSpec("v_at_tfrac_090", "G2_voltage_health", "record_voltage+time", "voltage at 90% normalized cycle time"),
)

G3_SWITCH_POLARIZATION: Tuple[FeatureSpec, ...] = (
    FeatureSpec("step_voltage_jump_abs_mean", "G3_switch_polarization", "record_I+V", "mean abs voltage jump at current transitions"),
    FeatureSpec("step_voltage_jump_signed_mean", "G3_switch_polarization", "record_I+V", "mean signed voltage jump at current transitions"),
    FeatureSpec("r_step_proxy_abs_mean", "G3_switch_polarization", "record_I+V", "mean |dV/dI| at transitions"),
    FeatureSpec("rest_voltage_recovery_mean", "G3_switch_polarization", "record_rest", "mean rest voltage recovery"),
    FeatureSpec("rest_voltage_recovery_abs_mean", "G3_switch_polarization", "record_rest", "mean abs rest voltage recovery"),
    FeatureSpec("charge_discharge_voltage_gap", "G3_switch_polarization", "record_voltage", "charge mean voltage minus discharge mean voltage"),
    FeatureSpec("voltage_efficiency_proxy", "G3_switch_polarization", "record_voltage", "discharge mean / charge mean"),
)

# Diagnostic / upper-bound only.  Do not enable in strict claims.
G_DIAGNOSTIC: Tuple[FeatureSpec, ...] = (
    FeatureSpec("duration_norm", "G_diag", "record_time", "cycle duration; can be capacity-equivalent under CC", strict_allowed=False),
    FeatureSpec("q_charge_norm", "G_diag", "record_capacity", "charge Ah feature; diagnostic only", strict_allowed=False),
    FeatureSpec("q_discharge_norm", "G_diag", "record_capacity", "discharge Ah feature; diagnostic only", strict_allowed=False),
)


def specs_for_group(group: str, *, allow_upper_bound: bool = False) -> Tuple[FeatureSpec, ...]:
    g = str(group).strip().lower()
    if g in {"g0", "g0_history", "history", "p0_history"}:
        return G0_HISTORY
    if g in {"g1", "g1_107a", "g1_107a_state", "p1_107a"}:
        return G1_107A_STATE
    if g in {"g2", "g2_voltage", "g2_voltage_health", "voltage"}:
        return G2_VOLTAGE
    if g in {"g3", "g3_switch", "g3_polarization", "switch", "polarization"}:
        return G3_SWITCH_POLARIZATION
    if g in {"g4", "g4_all", "g4_all_strict", "all_strict", "strict"}:
        return G0_HISTORY + G1_107A_STATE + G2_VOLTAGE + G3_SWITCH_POLARIZATION
    if g in {"p1_107a_strict", "p1", "assb111_p1"}:
        return G0_HISTORY + G1_107A_STATE
    if g in {"diag", "diagnostic", "upperbound", "p2_upperbound"}:
        if not allow_upper_bound:
            raise ValueError("diagnostic/upperbound features require allow_upper_bound=True")
        return G0_HISTORY + G1_107A_STATE + G2_VOLTAGE + G3_SWITCH_POLARIZATION + G_DIAGNOSTIC
    raise ValueError(f"Unknown feature group/mode: {group}")


def canonical_feature_names(group: str = "g4_all_strict", *, allow_upper_bound: bool = False) -> List[str]:
    return [s.name for s in specs_for_group(group, allow_upper_bound=allow_upper_bound)]


def is_forbidden_column(name: str, *, strict: bool = True) -> bool:
    low = str(name).strip().lower()
    tokens = FORBIDDEN_COLUMN_TOKENS if strict else CAPACITY_EQUIVALENT_TOKENS
    return any(tok.lower() in low for tok in tokens)


def forbidden_columns(columns: Iterable[str], *, strict: bool = True) -> List[str]:
    return [str(c) for c in columns if is_forbidden_column(str(c), strict=strict)]


def select_feature_columns(
    frame: pd.DataFrame,
    group: str = "g4_all_strict",
    *,
    allow_upper_bound: bool = False,
    allow_missing: bool = False,
) -> List[str]:
    wanted = canonical_feature_names(group, allow_upper_bound=allow_upper_bound)
    missing = [c for c in wanted if c not in frame.columns]
    if missing and not allow_missing:
        raise KeyError(f"Feature frame missing columns for {group}: {missing}")
    cols = [c for c in wanted if c in frame.columns]
    bad = forbidden_columns(cols, strict=True)
    if bad and not allow_upper_bound:
        raise ValueError(f"Forbidden target/capacity-equivalent feature columns in strict mode: {bad}")
    return cols


def feature_group_for_column(column: str) -> str:
    for spec in G0_HISTORY + G1_107A_STATE + G2_VOLTAGE + G3_SWITCH_POLARIZATION + G_DIAGNOSTIC:
        if spec.name == column:
            return spec.group
    return "unknown"


def schema_dict(group: str = "g4_all_strict", *, allow_upper_bound: bool = False) -> Dict[str, Any]:
    specs = specs_for_group(group, allow_upper_bound=allow_upper_bound)
    return {
        "feature_group": group,
        "allow_upper_bound": bool(allow_upper_bound),
        "n_features": len(specs),
        "feature_columns": [s.name for s in specs],
        "features": [asdict(s) for s in specs],
        "forbidden_column_tokens": list(FORBIDDEN_COLUMN_TOKENS),
        "capacity_equivalent_tokens": list(CAPACITY_EQUIVALENT_TOKENS),
        "strict_note": (
            "G4 strict features are intended for online SOH estimation. They must not include "
            "observed SOH/capacity labels or full-discharge capacity equivalents."
        ),
    }


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


def write_schema_json(path: PathLike, group: str = "g4_all_strict", *, allow_upper_bound: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(schema_dict(group, allow_upper_bound=allow_upper_bound)), f, ensure_ascii=False, indent=2, sort_keys=True)


def fit_standard_scaler(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    fit_mask: Optional[Sequence[bool]] = None,
    fit_cycles: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if fit_mask is None:
        if fit_cycles is not None:
            if "cycle_id" not in frame.columns:
                raise KeyError("fit_cycles was provided but frame has no cycle_id column")
            fit_mask_arr = frame["cycle_id"].astype(int).isin([int(c) for c in fit_cycles]).to_numpy()
        elif "split" in frame.columns:
            fit_mask_arr = frame["split"].astype(str).str.lower().eq("train").to_numpy()
        else:
            fit_mask_arr = np.ones(len(frame), dtype=bool)
    else:
        fit_mask_arr = np.asarray(fit_mask, dtype=bool)
    if fit_mask_arr.size != len(frame):
        raise ValueError("fit_mask length does not match frame")
    if not np.any(fit_mask_arr):
        raise RuntimeError("No rows selected for scaler fit")
    x = frame.loc[fit_mask_arr, list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    med[~np.isfinite(med)] = 0.0
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(med, inds[1])
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[~np.isfinite(std) | (std < 1.0e-12)] = 1.0
    fit_cycle_values: List[int] = []
    if "cycle_id" in frame.columns:
        fit_cycle_values = [int(c) for c in frame.loc[fit_mask_arr, "cycle_id"].astype(int).tolist()]
    return {
        "type": "standard",
        "feature_columns": list(feature_columns),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "median_impute": [float(v) for v in med],
        "fit_cycles": fit_cycle_values,
        "fit_n_rows": int(np.sum(fit_mask_arr)),
    }


def transform_with_scaler(frame: pd.DataFrame, scaler: Mapping[str, Any]) -> np.ndarray:
    cols = list(scaler["feature_columns"])
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing scaler feature columns: {missing}")
    x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    med = np.asarray(scaler.get("median_impute", np.zeros(len(cols))), dtype=float)
    inds = np.where(~np.isfinite(x))
    if inds[0].size:
        x[inds] = np.take(med, inds[1])
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)
    std[~np.isfinite(std) | (std < 1.0e-12)] = 1.0
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


def audit_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    allow_upper_bound: bool = False,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    bad = forbidden_columns(feature_columns, strict=True)
    if bad and not allow_upper_bound:
        failures.append(f"Forbidden target/capacity-equivalent feature columns: {bad}")
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        failures.append(f"Missing feature columns: {missing}")
    nonfinite: Dict[str, int] = {}
    for c in feature_columns:
        if c in frame.columns:
            vals = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float)
            n_bad = int(np.sum(~np.isfinite(vals)))
            if n_bad:
                nonfinite[c] = n_bad
    if nonfinite:
        warnings.append(f"Non-finite feature values will be median-imputed: {nonfinite}")
    group_counts: Dict[str, int] = {}
    for c in feature_columns:
        g = feature_group_for_column(c)
        group_counts[g] = group_counts.get(g, 0) + 1
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "n_features": len(feature_columns),
        "group_counts": group_counts,
        "nonfinite_counts": nonfinite,
    }


__all__ = [
    "FeatureSpec",
    "FORBIDDEN_COLUMN_TOKENS",
    "CAPACITY_EQUIVALENT_TOKENS",
    "G0_HISTORY",
    "G1_107A_STATE",
    "G2_VOLTAGE",
    "G3_SWITCH_POLARIZATION",
    "G_DIAGNOSTIC",
    "specs_for_group",
    "canonical_feature_names",
    "select_feature_columns",
    "feature_group_for_column",
    "schema_dict",
    "write_schema_json",
    "fit_standard_scaler",
    "transform_with_scaler",
    "write_scaler_json",
    "load_scaler_json",
    "audit_feature_frame",
]
