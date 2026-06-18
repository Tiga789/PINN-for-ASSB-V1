# -*- coding: utf-8 -*-
"""
D17-P2 P2Dlite prior normalization.

This file converts a loose resolved_p2dlite_spec.json into numerical parameters
used by the differentiable D17 smoke model.  It deliberately rejects per-profile
state-answer keys such as theta0_oracle/cs_a/cs_c and provides conservative
smoke defaults only when the project is still using the P1 placeholder spec.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


FARADAY_C_PER_MOL = 96485.33212
R_GAS_J_PER_MOL_K = 8.314462618


SUSPICIOUS_STATE_KEYS = {
    "cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c",
    "theta0_oracle", "oracle_shift", "cs_a_soft", "cs_c_soft",
    "theta_a_soft", "theta_c_soft", "phie_soft", "phis_c_soft",
}


@dataclass
class ElectrodeParams:
    name: str
    particle_radius_m: float
    active_fraction: float
    electrode_volume_m3: float
    csmax_mol_m3: float
    Ds_m2_s: float
    i0_A_m2: float
    theta_min: float
    theta_max: float
    current_flux_sign: float


@dataclass
class D17P2DlitePrior:
    cell_format: str = "Lishen 18650"
    nominal_capacity_Ah: float = 2.0
    temperature_C: float = 25.0
    electrolyte_phi_ref_V: float = 0.0
    Rohm_Ohm: float = 0.035
    voltage_offset_V: float = 0.0
    qeff_scale_init: float = 1.0
    qeff_scale_min: float = 0.6
    qeff_scale_max: float = 1.2
    theta0_a_init: float = 0.72
    theta0_c_init: float = 0.48
    theta0_a_min: float = 0.05
    theta0_a_max: float = 0.95
    theta0_c_min: float = 0.05
    theta0_c_max: float = 0.95
    ocp_phase_shift_max: float = 0.08
    gauge_shift_max_V: float = 0.10
    residual_coeff_max_V: float = 0.05
    positive: ElectrodeParams = None  # type: ignore[assignment]
    negative: ElectrodeParams = None  # type: ignore[assignment]
    ocp_positive_theta: Optional[Sequence[float]] = None
    ocp_positive_U: Optional[Sequence[float]] = None
    ocp_negative_theta: Optional[Sequence[float]] = None
    ocp_negative_U: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        if self.positive is None:
            # Conservative NCM523-like smoke defaults. Formal runs should replace
            # them with the D15 resolved P2Dlite prior.
            self.positive = ElectrodeParams(
                name="cathode_NCM523_assumed",
                particle_radius_m=5.0e-6,
                active_fraction=0.58,
                electrode_volume_m3=1.20e-5,
                csmax_mol_m3=51554.0,
                Ds_m2_s=3.0e-14,
                i0_A_m2=0.60,
                theta_min=0.20,
                theta_max=0.93,
                current_flux_sign=+1.0,
            )
        if self.negative is None:
            self.negative = ElectrodeParams(
                name="anode_graphite_assumed",
                particle_radius_m=6.0e-6,
                active_fraction=0.62,
                electrode_volume_m3=1.10e-5,
                csmax_mol_m3=31370.0,
                Ds_m2_s=4.0e-14,
                i0_A_m2=0.75,
                theta_min=0.02,
                theta_max=0.92,
                current_flux_sign=-1.0,
            )
        if self.ocp_positive_theta is None or self.ocp_positive_U is None:
            th = np.linspace(0.0, 1.0, 101)
            # Smooth NCM-like decreasing U(theta) vs lithiation. This is only a
            # fallback smoke OCP; it should be replaced by resolved prior tables.
            U = 4.35 - 0.85 * th + 0.06 * np.tanh((0.45 - th) / 0.08) - 0.04 * np.tanh((th - 0.85) / 0.04)
            self.ocp_positive_theta = th.tolist()
            self.ocp_positive_U = U.tolist()
        if self.ocp_negative_theta is None or self.ocp_negative_U is None:
            th = np.linspace(0.0, 1.0, 101)
            # Smooth graphite-like low potential curve. Formal runs should use
            # the generator prior OCP table.
            U = 0.08 + 0.12 * np.exp(-6.0 * th) + 0.045 * np.tanh((0.18 - th) / 0.05) + 0.02 * th
            self.ocp_negative_theta = th.tolist()
            self.ocp_negative_U = U.tolist()


def _walk_keys(obj: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield p
            yield from _walk_keys(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            yield from _walk_keys(v, f"{prefix}[{i}]")


def _find_first(d: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    """Recursively find a value by candidate leaf names."""
    found = []
    def rec(obj: Any) -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                if k in names:
                    found.append(v)
                rec(v)
    rec(d)
    return found[0] if found else default


def _to_float(x: Any, default: float) -> float:
    try:
        if isinstance(x, (list, tuple)) and x:
            x = x[0]
        return float(x)
    except Exception:
        return float(default)


def _get_nested(d: Mapping[str, Any], path_options: Sequence[Sequence[str]], default: Any = None) -> Any:
    for path in path_options:
        cur: Any = d
        ok = True
        for p in path:
            if isinstance(cur, Mapping) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _electrode_from_spec(spec: Mapping[str, Any], side: str, default: ElectrodeParams) -> ElectrodeParams:
    # Accept multiple naming conventions used across D14/D15 drafts.
    side_aliases = {
        "positive": ["positive", "cathode", "c", "pos", "p"],
        "negative": ["negative", "anode", "a", "neg", "n"],
    }[side]
    blocks = []
    for root_name in ["electrodes", "electrode", "geometry", "transport", "kinetics", "params"]:
        root = spec.get(root_name)
        if isinstance(root, Mapping):
            for alias in side_aliases:
                val = root.get(alias)
                if isinstance(val, Mapping):
                    blocks.append(val)
    for alias in side_aliases:
        val = spec.get(alias)
        if isinstance(val, Mapping):
            blocks.append(val)
    merged: Dict[str, Any] = {}
    for b in blocks:
        merged.update(b)

    return ElectrodeParams(
        name=str(merged.get("name", default.name)),
        particle_radius_m=_to_float(_get_nested(merged, [["particle_radius_m"], ["R_particle_m"], ["Rp_m"], ["R_s_m"], ["radius_m"]], default.particle_radius_m), default.particle_radius_m),
        active_fraction=_to_float(_get_nested(merged, [["active_fraction"], ["eps_s"], ["epsilon_s"], ["eps_s_j"]], default.active_fraction), default.active_fraction),
        electrode_volume_m3=_to_float(_get_nested(merged, [["electrode_volume_m3"], ["V_electrode_m3"], ["V_j_m3"], ["volume_m3"]], default.electrode_volume_m3), default.electrode_volume_m3),
        csmax_mol_m3=_to_float(_get_nested(merged, [["csmax_mol_m3"], ["c_s_max_mol_m3"], ["cs_max"], ["cmax_mol_m3"]], default.csmax_mol_m3), default.csmax_mol_m3),
        Ds_m2_s=_to_float(_get_nested(merged, [["Ds_m2_s"], ["D_s_m2_s"], ["solid_diffusivity_m2_s"], ["Ds"]], default.Ds_m2_s), default.Ds_m2_s),
        i0_A_m2=_to_float(_get_nested(merged, [["i0_A_m2"], ["exchange_current_density_A_m2"], ["i0"], ["i00_A_m2"]], default.i0_A_m2), default.i0_A_m2),
        theta_min=_to_float(_get_nested(merged, [["theta_min"], ["stoich_min"], ["x_min"]], default.theta_min), default.theta_min),
        theta_max=_to_float(_get_nested(merged, [["theta_max"], ["stoich_max"], ["x_max"]], default.theta_max), default.theta_max),
        current_flux_sign=default.current_flux_sign,
    )


def _extract_ocp(spec: Mapping[str, Any], side: str) -> tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
    ocp = spec.get("ocp")
    if not isinstance(ocp, Mapping):
        return None, None
    aliases = {
        "positive": ["positive", "cathode", "c", "pos", "p"],
        "negative": ["negative", "anode", "a", "neg", "n"],
    }[side]
    candidates = []
    for alias in aliases:
        val = ocp.get(alias)
        if isinstance(val, Mapping):
            candidates.append(val)
    for b in candidates:
        theta = b.get("theta") or b.get("x") or b.get("stoich") or b.get("theta_grid")
        U = b.get("U") or b.get("voltage") or b.get("ocp_V") or b.get("U_V")
        if theta is not None and U is not None:
            return theta, U
    return None, None


def load_p2dlite_prior(path: str | Path | None, allow_smoke_defaults: bool = True) -> D17P2DlitePrior:
    base = D17P2DlitePrior()
    if path is None:
        if allow_smoke_defaults:
            return base
        raise FileNotFoundError("resolved spec path is required")
    path = Path(path)
    if not path.exists():
        if allow_smoke_defaults:
            return base
        raise FileNotFoundError(f"resolved spec not found: {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise ValueError("resolved spec must be a JSON object")
    bad = [k for k in _walk_keys(spec) if k.split(".")[-1] in SUSPICIOUS_STATE_KEYS]
    if bad:
        raise ValueError(f"Resolved spec appears to contain state-answer keys forbidden in D17 mainline: {bad[:20]}")

    prior = D17P2DlitePrior()
    prior.cell_format = str(_get_nested(spec, [["cell", "format"], ["cell_format"]], prior.cell_format))
    prior.nominal_capacity_Ah = _to_float(_get_nested(spec, [["cell", "nominal_capacity_Ah"], ["capacity", "Q_nominal_Ah"], ["Q_nominal_Ah"]], prior.nominal_capacity_Ah), prior.nominal_capacity_Ah)
    prior.temperature_C = _to_float(_get_nested(spec, [["temperature", "default_C"], ["temperature_C"], ["cell", "temperature_C"]], prior.temperature_C), prior.temperature_C)
    prior.Rohm_Ohm = _to_float(_get_nested(spec, [["resistance", "Rohm_Ohm"], ["transport", "Rohm_Ohm"], ["Rohm_Ohm"], ["R_ohm_Ohm"]], prior.Rohm_Ohm), prior.Rohm_Ohm)
    prior.voltage_offset_V = _to_float(_get_nested(spec, [["voltage", "offset_V"], ["voltage", "voltage_offset_V"], ["voltage_offset_V"], ["b_V"], ["alignment", "voltage_offset_V"]], prior.voltage_offset_V), prior.voltage_offset_V)

    # P3.4: resolved-spec alignment must center the trainable profile latent
    # variables at the same nominal choices used by the generator / voltage-only
    # alignment pass.  P3.3 only read ranges, which forced raw=0 to broad-range
    # midpoints and made the forward core start with a systematic bias.
    prior.qeff_scale_init = _to_float(_get_nested(spec, [["capacity", "Q_eff_scale_init"], ["capacity", "qeff_scale_init"], ["Q_eff_scale_init"], ["qeff_scale_init"], ["alignment", "qeff_scale_init"]], prior.qeff_scale_init), prior.qeff_scale_init)
    qrange = _get_nested(spec, [["capacity", "Q_eff_scale_range"], ["capacity", "qeff_scale_range"], ["Q_eff_scale_range"], ["qeff_scale_range"]], None)
    if isinstance(qrange, (list, tuple)) and len(qrange) >= 2:
        prior.qeff_scale_min, prior.qeff_scale_max = float(qrange[0]), float(qrange[1])
    prior.qeff_scale_min = _to_float(_get_nested(spec, [["capacity", "Q_eff_scale_min"], ["qeff_scale_min"]], prior.qeff_scale_min), prior.qeff_scale_min)
    prior.qeff_scale_max = _to_float(_get_nested(spec, [["capacity", "Q_eff_scale_max"], ["qeff_scale_max"]], prior.qeff_scale_max), prior.qeff_scale_max)

    prior.theta0_a_init = _to_float(_get_nested(spec, [["initial_state", "theta_a0"], ["initial_state", "theta0_a"], ["theta0_a_init"], ["theta_a0_init"], ["alignment", "theta0_a_init"]], prior.theta0_a_init), prior.theta0_a_init)
    prior.theta0_c_init = _to_float(_get_nested(spec, [["initial_state", "theta_c0"], ["initial_state", "theta0_c"], ["theta0_c_init"], ["theta_c0_init"], ["alignment", "theta0_c_init"]], prior.theta0_c_init), prior.theta0_c_init)
    prior.theta0_a_min = _to_float(_get_nested(spec, [["initial_state", "theta_a0_min"], ["theta0_a_min"], ["theta_a0_min"]], prior.theta0_a_min), prior.theta0_a_min)
    prior.theta0_a_max = _to_float(_get_nested(spec, [["initial_state", "theta_a0_max"], ["theta0_a_max"], ["theta_a0_max"]], prior.theta0_a_max), prior.theta0_a_max)
    prior.theta0_c_min = _to_float(_get_nested(spec, [["initial_state", "theta_c0_min"], ["theta0_c_min"], ["theta_c0_min"]], prior.theta0_c_min), prior.theta0_c_min)
    prior.theta0_c_max = _to_float(_get_nested(spec, [["initial_state", "theta_c0_max"], ["theta0_c_max"], ["theta_c0_max"]], prior.theta0_c_max), prior.theta0_c_max)

    prior.ocp_phase_shift_max = _to_float(_get_nested(spec, [["ocp", "phase_shift_max"], ["ocp_phase_shift_max"], ["adapter", "ocp_phase_shift_max"]], prior.ocp_phase_shift_max), prior.ocp_phase_shift_max)
    prior.gauge_shift_max_V = _to_float(_get_nested(spec, [["gauge", "shift_max_V"], ["gauge_shift_max_V"], ["adapter", "gauge_shift_max_V"]], prior.gauge_shift_max_V), prior.gauge_shift_max_V)
    prior.residual_coeff_max_V = _to_float(_get_nested(spec, [["voltage", "residual_coeff_max_V"], ["residual_coeff_max_V"], ["adapter", "residual_coeff_max_V"]], prior.residual_coeff_max_V), prior.residual_coeff_max_V)
    prior.positive = _electrode_from_spec(spec, "positive", prior.positive)
    prior.negative = _electrode_from_spec(spec, "negative", prior.negative)
    thp, Up = _extract_ocp(spec, "positive")
    thn, Un = _extract_ocp(spec, "negative")
    if thp is not None and Up is not None:
        prior.ocp_positive_theta, prior.ocp_positive_U = list(map(float, thp)), list(map(float, Up))
    if thn is not None and Un is not None:
        prior.ocp_negative_theta, prior.ocp_negative_U = list(map(float, thn)), list(map(float, Un))
    return prior


def prior_to_jsonable(prior: D17P2DlitePrior) -> Dict[str, Any]:
    return asdict(prior)
