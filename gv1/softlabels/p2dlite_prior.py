# -*- coding: utf-8 -*-
"""Utilities for loading, validating, hashing, and resolving the standalone P2Dlite prior file.

D14-P4A note
------------
The P2Dlite prior file is intentionally the single source of physical
information for XJTU soft-label generation and future model prediction/training.
Do not duplicate XJTU cell parameters inside generator, auditor, or trainer
scripts. If the user edits the prior file for another cell, all downstream
artifacts should change their resolved_spec_hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_prior(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"P2Dlite prior file not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    validate_prior(obj)
    obj["_prior_path"] = str(p)
    obj["_prior_hash"] = prior_hash(obj)
    return obj


def _clean_for_hash(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_for_hash(v) for k, v in sorted(obj.items()) if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_clean_for_hash(v) for v in obj]
    return obj


def prior_hash(prior: Dict[str, Any]) -> str:
    data = json.dumps(_clean_for_hash(prior), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def get_value(prior: Dict[str, Any], path: str, default: Optional[Any] = None) -> Any:
    cur: Any = prior
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            if default is not None:
                return default
            raise KeyError(f"Missing prior path: {path}")
        cur = cur[key]
    if isinstance(cur, dict) and "value" in cur:
        return cur["value"]
    return cur


def validate_prior(prior: Dict[str, Any]) -> None:
    required = [
        "schema_version",
        "cell.nominal_capacity_Ah.value",
        "cell.voltage_min_V.value",
        "cell.voltage_max_V.value",
        "chemistry.positive_electrode.name",
        "chemistry.negative_electrode.name",
        "geometry.positive.thickness_m.value",
        "geometry.negative.thickness_m.value",
        "geometry.positive.particle_radius_m.value",
        "geometry.negative.particle_radius_m.value",
        "solid_phase.positive.c_s_max_mol_m3.value",
        "solid_phase.negative.c_s_max_mol_m3.value",
        "solid_phase.positive.D_s_m2_s.value",
        "solid_phase.negative.D_s_m2_s.value",
        "softlabel.n_r_default",
    ]
    missing = []
    for path in required:
        try:
            get_value(prior, path)
        except Exception:
            missing.append(path)
    if missing:
        raise ValueError("Invalid P2Dlite prior. Missing required fields: " + ", ".join(missing))


def _default_soft_voltage_bounds(prior: Dict[str, Any]) -> Dict[str, Any]:
    vmin = float(get_value(prior, "cell.voltage_min_V.value"))
    vmax = float(get_value(prior, "cell.voltage_max_V.value"))
    return {
        "enabled": True,
        "upper_margin_V": 0.02,
        "lower_margin_V": 0.02,
        "upper_warn_V": vmax + 0.05,
        "upper_fail_V": vmax + 0.15,
        "lower_warn_V": vmin - 0.05,
        "lower_fail_V": vmin - 0.15,
        "apply_to": ["phis_c", "phis_c_soft"],
        "source": "default_from_nominal_voltage_limits",
    }


def build_resolved_spec(prior: Dict[str, Any], n_r_override: Optional[int] = None) -> Dict[str, Any]:
    F = float(get_value(prior, "constants.F_C_per_mol"))
    Q_Ah = float(get_value(prior, "cell.nominal_capacity_Ah.value"))
    Q_C = Q_Ah * 3600.0

    Lp = float(get_value(prior, "geometry.positive.thickness_m.value"))
    eps_sp = float(get_value(prior, "geometry.positive.epsilon_s.value"))
    cmax_p = float(get_value(prior, "solid_phase.positive.c_s_max_mol_m3.value"))
    theta_p_min = float(get_value(prior, "stoichiometry.positive.theta_min.value"))
    theta_p_max = float(get_value(prior, "stoichiometry.positive.theta_max.value"))
    dtheta_p = max(theta_p_max - theta_p_min, 1e-6)

    area = get_value(prior, "geometry.effective_area_m2.value", default=None)
    if area is None:
        area = Q_C / (F * eps_sp * Lp * cmax_p * dtheta_p)

    n_r = int(n_r_override or get_value(prior, "softlabel.n_r_default"))

    residual_cfg = dict(prior.get("voltage_closure", {}).get("residual_correction", {}))
    soft_bounds = dict(prior.get("voltage_closure", {}).get("soft_voltage_bounds", {}) or _default_soft_voltage_bounds(prior))

    resolved = {
        "schema_version": prior.get("schema_version"),
        "prior_name": prior.get("prior_name"),
        "prior_hash": prior.get("_prior_hash") or prior_hash(prior),
        "prior_path": prior.get("_prior_path", ""),
        "model_family": "P2Dlite",
        "n_r": n_r,
        "constants": {
            "F_C_per_mol": F,
            "R_J_per_mol_K": float(get_value(prior, "constants.R_J_per_mol_K")),
        },
        "cell": {
            "manufacturer": prior.get("cell", {}).get("manufacturer", ""),
            "model": prior.get("cell", {}).get("model", ""),
            "nominal_capacity_Ah": Q_Ah,
            "voltage_min_V": float(get_value(prior, "cell.voltage_min_V.value")),
            "voltage_max_V": float(get_value(prior, "cell.voltage_max_V.value")),
            "fallback_temperature_K": float(get_value(prior, "cell.fallback_temperature_K.value")),
        },
        "chemistry": prior.get("chemistry", {}),
        "geometry": {
            "effective_area_m2": float(area),
            "positive": {
                "L_m": Lp,
                "R_particle_m": float(get_value(prior, "geometry.positive.particle_radius_m.value")),
                "eps_s": eps_sp,
                "eps_e": float(get_value(prior, "geometry.positive.epsilon_e.value")),
            },
            "negative": {
                "L_m": float(get_value(prior, "geometry.negative.thickness_m.value")),
                "R_particle_m": float(get_value(prior, "geometry.negative.particle_radius_m.value")),
                "eps_s": float(get_value(prior, "geometry.negative.epsilon_s.value")),
                "eps_e": float(get_value(prior, "geometry.negative.epsilon_e.value")),
            },
        },
        "solid_phase": {
            "positive": {
                "cmax_mol_m3": cmax_p,
                "D_m2_s": float(get_value(prior, "solid_phase.positive.D_s_m2_s.value")),
                "theta_min": theta_p_min,
                "theta_max": theta_p_max,
            },
            "negative": {
                "cmax_mol_m3": float(get_value(prior, "solid_phase.negative.c_s_max_mol_m3.value")),
                "D_m2_s": float(get_value(prior, "solid_phase.negative.D_s_m2_s.value")),
                "theta_min": float(get_value(prior, "stoichiometry.negative.theta_min.value")),
                "theta_max": float(get_value(prior, "stoichiometry.negative.theta_max.value")),
            },
        },
        "kinetics": {
            "alpha": float(get_value(prior, "kinetics.alpha.value")),
            "i0_positive_A_m2": float(get_value(prior, "kinetics.positive.i0_A_m2.value")),
            "i0_negative_A_m2": float(get_value(prior, "kinetics.negative.i0_A_m2.value")),
        },
        "voltage_closure": {
            "R_ohm_Ohm": float(get_value(prior, "voltage_closure.R_ohm_Ohm.value")),
            "voltage_offset_V": float(get_value(prior, "voltage_closure.voltage_offset_V.value")),
            "residual_correction": residual_cfg,
            "soft_voltage_bounds": soft_bounds,
        },
        "interpretation": prior.get("interpretation", {}),
        "audit": prior.get("audit", {}),
    }
    return resolved
