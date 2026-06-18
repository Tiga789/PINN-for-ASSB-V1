# -*- coding: utf-8 -*-
"""D17-P3.4 resolved-spec alignment utilities.

P3.4 is the final P3 gate before P4.  Its purpose is to stop training the
forward electrochemical core against a placeholder prior.  This module builds a
D17-compatible resolved P2Dlite-RG spec using only observed replay signals
I(t), V(t), T(t), manifest metadata and project/generator prior JSON files.  It
never reads cs/theta/phie/phis soft-label arrays.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .dataset import FORBIDDEN_STATE_FIELDS, load_observed_profile
from .p2dlite_prior import D17P2DlitePrior, FARADAY_C_PER_MOL, R_GAS_J_PER_MOL_K, load_p2dlite_prior, prior_to_jsonable
from .p3_trainer import D17ProfileDataset, normalize_protocol, select_balanced_records
from .trainer import crop_time_window

SUSPICIOUS_FILE_PARTS = ("placeholder", "p33", "p34", "smoke_summary", "prediction")


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _has_forbidden_state_keys(obj: Any) -> bool:
    bad = set(FORBIDDEN_STATE_FIELDS) | {"theta0_oracle", "oracle_shift", "cs_a_soft", "cs_c_soft", "theta_a_soft", "theta_c_soft", "phie_soft", "phis_c_soft"}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k) in bad:
                return True
            if _has_forbidden_state_keys(v):
                return True
    elif isinstance(obj, list):
        for v in obj[:50]:
            if _has_forbidden_state_keys(v):
                return True
    return False


def _candidate_score(path: Path, obj: Mapping[str, Any]) -> float:
    name = path.name.lower()
    full = str(path).lower()
    if any(part in full for part in SUSPICIOUS_FILE_PARTS):
        return -1e9
    score = 0.0
    if "p2dlite" in name: score += 10
    if "rg" in name: score += 4
    if "resolved" in name: score += 5
    if "prior" in name: score += 3
    if "spec" in name: score += 3
    text_keys = json.dumps(list(obj.keys()), ensure_ascii=False).lower()
    for kw in ["electrodes", "positive", "negative", "ocp", "capacity", "rohm", "theta"]:
        if kw in text_keys:
            score += 1.0
    # Prefer real spec files, not output summaries.
    if "summary" in name or "manifest" in name:
        score -= 8.0
    try:
        score += min(path.stat().st_size / 50000.0, 2.0)
    except Exception:
        pass
    return score


def discover_prior_candidates(search_roots: Sequence[str | Path], max_files_per_root: int = 4000) -> List[Dict[str, Any]]:
    """Find likely resolved P2Dlite/P2Dlite-RG prior JSON files.

    The function is intentionally conservative: it rejects files containing
    state-answer keys and ranks spec/prior/resolved JSON higher than summaries.
    """
    patterns = ["*p2dlite*spec*.json", "*p2dlite*prior*.json", "*resolved*spec*.json", "*resolved*prior*.json", "*p2dlite*rg*.json"]
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        count = 0
        for pat in patterns:
            for p in root_path.rglob(pat):
                if not p.is_file():
                    continue
                sp = str(p.resolve())
                if sp in seen:
                    continue
                seen.add(sp)
                count += 1
                if count > max_files_per_root:
                    break
                obj = _read_json(p)
                if not obj:
                    continue
                forbidden = _has_forbidden_state_keys(obj)
                score = _candidate_score(p, obj)
                if forbidden:
                    score -= 1e6
                out.append({"path": str(p), "score": score, "forbidden_state_keys": forbidden, "size_bytes": p.stat().st_size})
    out.sort(key=lambda d: float(d["score"]), reverse=True)
    return out


def _ocp_interp(theta: np.ndarray, theta_grid: Sequence[float], U_grid: Sequence[float]) -> np.ndarray:
    th = np.asarray(theta_grid, dtype=np.float64).reshape(-1)
    uu = np.asarray(U_grid, dtype=np.float64).reshape(-1)
    order = np.argsort(th)
    th, uu = th[order], uu[order]
    return np.interp(theta, th, uu, left=float(uu[0]), right=float(uu[-1]))


def _cumtrapz_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    dx = np.diff(x, prepend=x[0])
    dx = np.maximum(dx, 0.0)
    avg = 0.5 * (y + np.concatenate([[y[0]], y[:-1]]))
    return np.cumsum(avg * dx)


def _bv_inverse_eta(J: np.ndarray, i0_A_m2: float, T_C: np.ndarray) -> np.ndarray:
    T_K = T_C + 273.15
    arg = FARADAY_C_PER_MOL * J / max(2.0 * float(i0_A_m2), 1e-9)
    return (2.0 * R_GAS_J_PER_MOL_K * T_K / FARADAY_C_PER_MOL) * np.arcsinh(arg)


def _profile_arrays(profile: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(profile.get("t_global_s", profile.get("time_s")), dtype=np.float64).reshape(-1)
    I = np.asarray(profile.get("I_profile", profile.get("current_A")), dtype=np.float64).reshape(-1)
    V = np.asarray(profile.get("voltage_exp"), dtype=np.float64).reshape(-1)
    T = np.asarray(profile.get("temperature_C", np.full_like(t, 25.0)), dtype=np.float64).reshape(-1)
    n = min(len(t), len(I), len(V), len(T))
    if n < 8:
        raise ValueError("profile too short for P3.4 voltage-only prior alignment")
    return t[:n], I[:n], V[:n], T[:n]


def _current_rel_theta(t: np.ndarray, I: np.ndarray, elec: Any, qeff: float) -> np.ndarray:
    J = float(elec.current_flux_sign) * I * float(elec.particle_radius_m) / (3.0 * float(elec.active_fraction) * FARADAY_C_PER_MOL * float(elec.electrode_volume_m3) * max(float(qeff), 1e-6))
    return (-(3.0 / float(elec.particle_radius_m)) * _cumtrapz_np(J, t)) / float(elec.csmax_mol_m3)


def _predict_forward_voltage(prior: D17P2DlitePrior, t: np.ndarray, I: np.ndarray, T: np.ndarray, theta_a0: float, theta_c0: float, qeff: float, Rohm: float, bV: float) -> np.ndarray:
    pa, pc = prior.negative, prior.positive
    rel_a = _current_rel_theta(t, I, pa, qeff)
    rel_c = _current_rel_theta(t, I, pc, qeff)
    ta = np.clip(theta_a0 + rel_a, pa.theta_min, pa.theta_max)
    tc = np.clip(theta_c0 + rel_c, pc.theta_min, pc.theta_max)
    Ua = _ocp_interp(ta, prior.ocp_negative_theta, prior.ocp_negative_U)
    Uc = _ocp_interp(tc, prior.ocp_positive_theta, prior.ocp_positive_U)
    Ja = float(pa.current_flux_sign) * I * float(pa.particle_radius_m) / (3.0 * float(pa.active_fraction) * FARADAY_C_PER_MOL * float(pa.electrode_volume_m3) * max(float(qeff), 1e-6))
    Jc = float(pc.current_flux_sign) * I * float(pc.particle_radius_m) / (3.0 * float(pc.active_fraction) * FARADAY_C_PER_MOL * float(pc.electrode_volume_m3) * max(float(qeff), 1e-6))
    etaa = _bv_inverse_eta(Ja, pa.i0_A_m2, T)
    etac = _bv_inverse_eta(Jc, pc.i0_A_m2, T)
    return Uc - Ua + etac - etaa + I * float(Rohm) + float(bV)


def _fit_bv_rohm_offset(y: np.ndarray, I: np.ndarray, base_no_ir: np.ndarray, Rohm_prior: float) -> Tuple[float, float, float]:
    """Fit bV and Rohm with clipped least squares from observed voltage only."""
    rhs = y - base_no_ir
    X = np.stack([np.ones_like(I), I], axis=1)
    # Robust first pass: downweight extreme residuals.
    try:
        beta, *_ = np.linalg.lstsq(X, rhs, rcond=None)
        pred = X @ beta
        e = rhs - pred
        mad = np.median(np.abs(e - np.median(e))) + 1e-9
        keep = np.abs(e) <= 5.0 * mad
        if int(np.sum(keep)) >= max(20, int(0.3 * len(rhs))):
            beta, *_ = np.linalg.lstsq(X[keep], rhs[keep], rcond=None)
        b, R = float(beta[0]), float(beta[1])
    except Exception:
        b, R = float(np.median(rhs)), float(Rohm_prior)
    R = float(np.clip(R, 0.001, 0.18))
    b = float(np.clip(b, -0.28, 0.28))
    rmse = float(np.sqrt(np.mean((base_no_ir + b + I * R - y) ** 2)))
    return R, b, rmse


def voltage_only_global_alignment(prior: D17P2DlitePrior, profiles: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Coarse voltage-only fit of theta0/qeff/Rohm/bV.

    This is not a state-label fit.  It uses only observed terminal voltage and
    current to place the generator choices in the right gauge before PINN
    training, replacing the P3.3 placeholder midpoint initialization.
    """
    align = cfg.get("p34_spec_alignment", {}) if isinstance(cfg.get("p34_spec_alignment", {}), Mapping) else {}
    max_points = int(align.get("max_points_per_profile", 256))
    arrays: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for prof in profiles:
        t, I, V, T = _profile_arrays(prof)
        if len(t) > max_points:
            idx = np.linspace(0, len(t)-1, max_points).round().astype(int)
            t, I, V, T = t[idx], I[idx], V[idx], T[idx]
        arrays.append((t, I, V, T))
    if not arrays:
        raise RuntimeError("no observed profiles available for P3.4 alignment")

    ta_center = float(prior.theta0_a_init)
    tc_center = float(prior.theta0_c_init)
    q_center = float(prior.qeff_scale_init)
    theta_a_span = float(align.get("theta_a0_span", 0.18))
    theta_c_span = float(align.get("theta_c0_span", 0.14))
    q_span = float(align.get("qeff_span", 0.18))
    n_ta = int(align.get("theta_a0_grid", 9))
    n_tc = int(align.get("theta_c0_grid", 9))
    n_q = int(align.get("qeff_grid", 5))
    ta_grid = np.linspace(max(prior.theta0_a_min, prior.negative.theta_min + 0.03, ta_center - theta_a_span), min(prior.theta0_a_max, prior.negative.theta_max - 0.03, ta_center + theta_a_span), n_ta)
    tc_grid = np.linspace(max(prior.theta0_c_min, prior.positive.theta_min + 0.03, tc_center - theta_c_span), min(prior.theta0_c_max, prior.positive.theta_max - 0.03, tc_center + theta_c_span), n_tc)
    q_grid = np.linspace(max(prior.qeff_scale_min, q_center - q_span), min(prior.qeff_scale_max, q_center + q_span), n_q)

    best: Dict[str, Any] = {"rmse_V": float("inf")}
    V_all = np.concatenate([a[2] for a in arrays])
    I_all = np.concatenate([a[1] for a in arrays])
    for ta in ta_grid:
        for tc in tc_grid:
            for qeff in q_grid:
                base_parts = []
                ok = True
                for t, I, V, T in arrays:
                    try:
                        # no Rohm, no offset; fit them once globally after stacking
                        base0 = _predict_forward_voltage(prior, t, I, T, float(ta), float(tc), float(qeff), 0.0, 0.0)
                    except Exception:
                        ok = False; break
                    if not np.all(np.isfinite(base0)):
                        ok = False; break
                    base_parts.append(base0)
                if not ok:
                    continue
                base_all = np.concatenate(base_parts)
                R, b, rmse = _fit_bv_rohm_offset(V_all, I_all, base_all, prior.Rohm_Ohm)
                if rmse < float(best["rmse_V"]):
                    best = {"theta0_a_init": float(ta), "theta0_c_init": float(tc), "qeff_scale_init": float(qeff), "Rohm_Ohm": float(R), "voltage_offset_V": float(b), "rmse_V": float(rmse)}
    if not math.isfinite(float(best.get("rmse_V", float("inf")))):
        # Fallback: median constant offset at prior center.
        base_parts = [_predict_forward_voltage(prior, t, I, T, ta_center, tc_center, q_center, prior.Rohm_Ohm, 0.0) for t, I, V, T in arrays]
        base_all = np.concatenate(base_parts)
        b = float(np.clip(np.median(V_all - base_all), -0.28, 0.28))
        best = {"theta0_a_init": ta_center, "theta0_c_init": tc_center, "qeff_scale_init": q_center, "Rohm_Ohm": prior.Rohm_Ohm, "voltage_offset_V": b, "rmse_V": float(np.sqrt(np.mean((base_all + b - V_all) ** 2)))}
    best["profile_count"] = len(arrays)
    best["points_total"] = int(sum(len(a[0]) for a in arrays))
    best["method"] = "observed_voltage_only_coarse_theta_qeff_rohm_offset_fit_no_state_labels"
    return best


def _base_spec_from_prior(prior: D17P2DlitePrior) -> Dict[str, Any]:
    return {
        "schema": "D17_P3.4_resolved_p2dlite_rg_prior_v1",
        "cell_format": prior.cell_format,
        "cell": {"format": prior.cell_format, "nominal_capacity_Ah": prior.nominal_capacity_Ah, "temperature_C": prior.temperature_C},
        "capacity": {"Q_nominal_Ah": prior.nominal_capacity_Ah, "Q_eff_scale_init": prior.qeff_scale_init, "Q_eff_scale_range": [prior.qeff_scale_min, prior.qeff_scale_max]},
        "temperature_C": prior.temperature_C,
        "transport": {"Rohm_Ohm": prior.Rohm_Ohm},
        "voltage": {"offset_V": prior.voltage_offset_V, "residual_coeff_max_V": prior.residual_coeff_max_V},
        "initial_state": {
            "theta_a0": prior.theta0_a_init,
            "theta_c0": prior.theta0_c_init,
            "theta_a0_min": prior.theta0_a_min,
            "theta_a0_max": prior.theta0_a_max,
            "theta_c0_min": prior.theta0_c_min,
            "theta_c0_max": prior.theta0_c_max,
        },
        "adapter": {"ocp_phase_shift_max": prior.ocp_phase_shift_max, "gauge_shift_max_V": prior.gauge_shift_max_V, "residual_coeff_max_V": prior.residual_coeff_max_V},
        "electrodes": {
            "positive": asdict(prior.positive),
            "negative": asdict(prior.negative),
        },
        "ocp": {
            "positive": {"theta": list(map(float, prior.ocp_positive_theta)), "U": list(map(float, prior.ocp_positive_U))},
            "negative": {"theta": list(map(float, prior.ocp_negative_theta)), "U": list(map(float, prior.ocp_negative_U))},
            "phase_shift_max": prior.ocp_phase_shift_max,
        },
    }


def select_p34_profiles(split_manifest: str | Path, train_count: int, validation_count: int, time_window_s: float, max_time_points: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Mapping[str, Any]]]:
    train_ds = D17ProfileDataset(split_manifest=split_manifest, split="train", allow_softlabel_npz_profile_source=False)
    val_ds = D17ProfileDataset(split_manifest=split_manifest, split="validation", allow_softlabel_npz_profile_source=False)
    train_recs = select_balanced_records(train_ds, profile_count=train_count)
    val_recs = select_balanced_records(val_ds, profile_count=validation_count)
    profiles: List[Mapping[str, Any]] = []
    for rec in list(train_recs) + list(val_recs):
        prof = load_observed_profile(rec["replay_npz"])
        prof = crop_time_window(prof, time_window_s=time_window_s, max_time_points=max_time_points)
        profiles.append(prof)
    return train_recs, val_recs, profiles


def build_p34_resolved_spec(cfg: Mapping[str, Any], out_json: str | Path) -> Dict[str, Any]:
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), Mapping) else {}
    align_cfg = cfg.get("p34_spec_alignment", {}) if isinstance(cfg.get("p34_spec_alignment", {}), Mapping) else {}
    split_manifest = str(paths.get("split_manifest", ""))
    if not split_manifest:
        raise RuntimeError("P3.4 requires paths.split_manifest")
    search_roots: List[str] = []
    for k in ["resolved_spec", "base_resolved_spec", "softlabel_root", "replay_search_root", "output_root"]:
        v = paths.get(k)
        if v:
            pp = Path(str(v))
            search_roots.append(str(pp if pp.is_dir() else pp.parent))
    search_roots.append("configs")
    candidates = discover_prior_candidates(search_roots)
    selected_candidate: Optional[Dict[str, Any]] = None
    base_prior: D17P2DlitePrior
    for cand in candidates:
        if float(cand.get("score", -1e9)) < 0 or cand.get("forbidden_state_keys"):
            continue
        try:
            base_prior = load_p2dlite_prior(cand["path"], allow_smoke_defaults=False)
            selected_candidate = cand
            break
        except Exception:
            continue
    else:
        base_path = paths.get("base_resolved_spec") or paths.get("resolved_spec")
        base_prior = load_p2dlite_prior(base_path, allow_smoke_defaults=True)
        selected_candidate = None

    train_count = int(align_cfg.get("profile_count", cfg.get("train", {}).get("profile_count", 12)))
    val_count = int(align_cfg.get("validation_profile_count", cfg.get("validation", {}).get("profile_count", 6)))
    time_window_s = float(cfg.get("train", {}).get("time_window_s", 40000.0))
    max_time_points = int(align_cfg.get("max_time_points", cfg.get("train", {}).get("max_time_points", 512)))
    train_recs, val_recs, profiles = select_p34_profiles(split_manifest, train_count, val_count, time_window_s, max_time_points)
    fit = voltage_only_global_alignment(base_prior, profiles, cfg) if bool(align_cfg.get("voltage_only_fit", True)) else {}

    prior = deepcopy(base_prior)
    if fit:
        prior.theta0_a_init = float(fit["theta0_a_init"])
        prior.theta0_c_init = float(fit["theta0_c_init"])
        prior.qeff_scale_init = float(fit["qeff_scale_init"])
        prior.Rohm_Ohm = float(fit["Rohm_Ohm"])
        prior.voltage_offset_V = float(fit["voltage_offset_V"])
        # A fitted DC offset should be a core/gauge choice, not a residual budget.
        prior.residual_coeff_max_V = min(float(prior.residual_coeff_max_V), 0.06)
    spec = _base_spec_from_prior(prior)
    spec["alignment"] = {
        "protocol": "D17-P3.4_RESOLVED_SPEC_ALIGNMENT",
        "uses_observed_voltage": True,
        "uses_observed_current": True,
        "uses_state_softlabels": False,
        "forbidden_state_fields": list(FORBIDDEN_STATE_FIELDS),
        "selected_candidate_prior": selected_candidate,
        "voltage_only_fit": fit,
        "train_profiles": [{"canonical_cell_uid": r.get("canonical_cell_uid"), "protocol": normalize_protocol(r), "replay_npz": r.get("replay_npz")} for r in train_recs],
        "validation_profiles": [{"canonical_cell_uid": r.get("canonical_cell_uid"), "protocol": normalize_protocol(r), "replay_npz": r.get("replay_npz")} for r in val_recs],
        "notes": "This file is generated from observed replay voltage/current plus prior JSON candidates; it does not read cs/theta/phie/phis arrays.",
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(spec), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"resolved_spec": str(out_json), "selected_candidate_prior": selected_candidate, "voltage_only_fit": fit, "train_count": len(train_recs), "validation_count": len(val_recs)}


def ensure_p34_resolved_spec(cfg: Mapping[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    paths = cfg.setdefault("paths", {}) if isinstance(cfg, dict) else cfg.get("paths", {})
    align_cfg = cfg.get("p34_spec_alignment", {}) if isinstance(cfg.get("p34_spec_alignment", {}), Mapping) else {}
    auto = bool(align_cfg.get("enabled", True))
    target = align_cfg.get("out_resolved_spec") or Path(out_dir) / "D17_P34_RESOLVED_P2DLITE_RG_SPEC_ALIGNED.json"
    target = Path(str(target))
    resolved_spec = str(paths.get("resolved_spec", ""))
    looks_placeholder = "placeholder" in resolved_spec.lower() or not resolved_spec
    if auto or looks_placeholder or not Path(resolved_spec).exists():
        info = build_p34_resolved_spec(cfg, target)
        if isinstance(cfg, dict):
            cfg.setdefault("paths", {})["resolved_spec"] = info["resolved_spec"]
        return info
    return {"resolved_spec": resolved_spec, "selected_candidate_prior": None, "voltage_only_fit": None, "reused_existing_spec": True}
