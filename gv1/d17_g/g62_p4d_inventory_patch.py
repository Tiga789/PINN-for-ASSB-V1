from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .g1_data import (
    OBS_I_KEYS,
    OBS_TIME_KEYS,
    OBS_V_KEYS,
    _find_1d_any,
    _find_1d_exact,
    _load_npz_dict,
    _semantics_for,
    load_profile_pack,
    load_semantics_map,
    load_split_records,
)
from .g3_frozen_audit import (
    augment_profile_features,
    build_model_from_checkpoint,
    parse_vocab_from_feature_names,
    read_json,
    resolve_checkpoint_path,
    safe_float,
    torch_load_safe,
    write_csv,
)
from .g13_trainer import _device_from_arg
from .g6_full_cycle_audit import RunningStats, cycle_ranges, load_cycle_ids_for_profile, predict_denorm

P4D_BRANCH = "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH"
DEFAULT_P4D_CONFIG = "configs/d15_p4d_full_remaining14_config.json"
DEFAULT_PRIOR = "configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json"
TARGETS_DEFAULT = ["theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c"]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _as_float_array(x: Any, name: str = "array") -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S", "O"}:
        raise TypeError(f"{name} is not numeric")
    return arr.astype(np.float64).reshape(-1)


def _interp_to_target(src_y: np.ndarray, src_t: Optional[np.ndarray], target_t: np.ndarray, fill: float = 0.0) -> np.ndarray:
    src_y = np.asarray(src_y, dtype=np.float64).reshape(-1)
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    if src_y.size == target_t.size:
        return src_y.astype(np.float64)
    if src_y.size == 0:
        return np.full(target_t.size, float(fill), dtype=np.float64)
    if src_t is None or np.asarray(src_t).reshape(-1).size != src_y.size:
        x_old = np.linspace(0.0, 1.0, src_y.size, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, target_t.size, dtype=np.float64)
        return np.interp(x_new, x_old, src_y).astype(np.float64)
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    order = np.argsort(src_t)
    x = src_t[order]
    y = src_y[order]
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return np.full(target_t.size, float(fill), dtype=np.float64)
    ux, ui = np.unique(x, return_index=True)
    y = y[ui]
    if ux.size == 1:
        return np.full(target_t.size, float(y[0]), dtype=np.float64)
    return np.interp(target_t, ux, y, left=float(y[0]), right=float(y[-1])).astype(np.float64)


def _find_observed_1d(soft: Mapping[str, Any], replay: Mapping[str, Any], keys: Sequence[str], target_t: np.ndarray, fill: float = 0.0) -> Tuple[str, np.ndarray]:
    n = int(np.asarray(target_t).reshape(-1).size)
    # For P4D soft labels, observed I/V/t are copied into solution_softlabels.npz.
    # These are allowed inputs, not state answers.
    for src_name, src in [("soft", soft), ("replay", replay)]:
        k, exact = _find_1d_exact(src, keys, n)
        if exact is not None:
            return f"{src_name}:{k}:exact", np.asarray(exact, dtype=np.float64).reshape(-1)
    for src_name, src in [("soft", soft), ("replay", replay)]:
        k, arr = _find_1d_any(src, keys)
        if arr is None:
            continue
        tk, tt = _find_1d_exact(src, OBS_TIME_KEYS, int(np.asarray(arr).reshape(-1).size))
        if tt is None:
            _, tt = _find_1d_any(src, OBS_TIME_KEYS)
        return f"{src_name}:{k}:interp", _interp_to_target(arr, tt, target_t, fill=fill)
    return "filled", np.full(n, fill, dtype=np.float64)


def read_csv_dicts(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_p4d_config_and_prior(project_root: str | Path = ".", p4d_config: str | Path = "", prior_json: str | Path = "") -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    root = Path(project_root)
    cfg_path = Path(p4d_config) if p4d_config else root / DEFAULT_P4D_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing P4D config: {cfg_path}. Expected {DEFAULT_P4D_CONFIG} in project root.")
    cfg = read_json(cfg_path, default={}) or {}
    gen = dict(cfg.get("generation") or {})
    prior_path = Path(prior_json) if prior_json else Path(str(cfg.get("prior_json") or DEFAULT_PRIOR))
    if not prior_path.is_absolute():
        prior_path = root / prior_path
    if not prior_path.exists():
        raise FileNotFoundError(f"Missing P4D prior json: {prior_path}. Provide --prior_json explicitly if the config moved.")
    prior = read_json(prior_path, default={}) or {}
    meta = {"p4d_config": str(cfg_path), "prior_json": str(prior_path), "generation": gen}
    return gen, prior, meta


def _import_rg_solver():
    try:
        from gv1.p2dlite_rg.radial_solver import ElectrodeRGParams, generate_rg_profile, infer_surface_flux_from_cbar
        return ElectrodeRGParams, generate_rg_profile, infer_surface_flux_from_cbar
    except Exception as exc:
        raise ImportError(
            "Cannot import gv1.p2dlite_rg.radial_solver. D17-G6.2 requires the actual D15 generator radial solver "
            "to avoid re-inventing P4D semantics. Make sure the project contains gv1/p2dlite_rg/radial_solver.py."
        ) from exc


def electrode_params_from_prior(prior: Mapping[str, Any], electrode: str):
    ElectrodeRGParams, _, _ = _import_rg_solver()
    rg = dict(prior.get("radial_gradient") or {})
    if electrode == "a":
        spec = prior["electrodes"]["negative"]
        alpha_D = float(rg.get("alpha_D_negative", 1.0))
        alpha_J = float(rg.get("alpha_J_negative", 1.0))
        name = "negative_graphite_d17g62_p4d_override"
    else:
        spec = prior["electrodes"]["positive"]
        alpha_D = float(rg.get("alpha_D_positive", 1.0))
        alpha_J = float(rg.get("alpha_J_positive", 1.0))
        name = "positive_NCM523_d17g62_p4d_override"
    return ElectrodeRGParams(
        name=name,
        radius_m=float(spec["particle_radius_m"]),
        diffusivity_m2_s=float(spec["solid_diffusivity_m2_s"]),
        csmax_mol_m3=float(spec["csmax_mol_m3"]),
        alpha_D=alpha_D,
        alpha_J=alpha_J,
        gradient_clip_normalized=float(spec.get("gradient_clip_normalized", rg.get("gradient_clip_normalized", 0.12))),
        theta_min_clip=0.0,
        theta_max_clip=1.0,
    )


def cum_theta_from_current(t: np.ndarray, I: np.ndarray, theta0: float, window: float, capacity_Ah: float, sign: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    I = np.asarray(I, dtype=np.float64).reshape(-1)
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt), dt, 0.0)
    dt[dt < 0] = 0.0
    q_Ah = np.cumsum(I * dt) / 3600.0
    theta = float(theta0) + float(sign) * (q_Ah / max(float(capacity_Ah), 1e-12)) * float(window)
    return np.clip(theta, 0.0, 1.0)


def observed_arrays_on_target_grid(softlabel_npz: str | Path, replay_npz: str | Path, target_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    keys = list(set(OBS_TIME_KEYS + OBS_I_KEYS + OBS_V_KEYS + ["temperature_C", "T_C", "T", "temperature"]))
    soft = _load_npz_dict(softlabel_npz, keys)
    replay = _load_npz_dict(replay_npz, keys) if replay_npz and Path(replay_npz).exists() else {}
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    I_key, I = _find_observed_1d(soft, replay, OBS_I_KEYS, target_t, fill=0.0)
    V_key, V = _find_observed_1d(soft, replay, OBS_V_KEYS, target_t, fill=0.0)
    # Temperature is only used for feature consistency; P4D deterministic inventory does not use it.
    T_key, T = _find_observed_1d(soft, replay, ["temperature_C", "T_C", "T", "temperature"], target_t, fill=25.0)
    return I, V, T, {"I_source": I_key, "V_source": V_key, "T_source": T_key}


def p4d_deterministic_outputs(
    t: np.ndarray,
    I: np.ndarray,
    V: np.ndarray,
    gen: Mapping[str, Any],
    prior: Mapping[str, Any],
    nr: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    _, generate_rg_profile, infer_surface_flux_from_cbar = _import_rg_solver()
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    I = np.asarray(I, dtype=np.float64).reshape(-1)
    V = np.asarray(V, dtype=np.float64).reshape(-1)
    if not (t.size == I.size == V.size):
        raise ValueError(f"P4D observed arrays have inconsistent sizes: t={t.size}, I={I.size}, V={V.size}")
    cap_Ah = float(gen.get("capacity_scale_Ah", 2.0))
    theta_pos0 = float(gen.get("theta_positive_initial", 0.90))
    theta_neg0 = float(gen.get("theta_negative_initial", 0.08))
    phie_scale = float(gen.get("phie_ohmic_scale_V_per_A", -0.015))
    p_a = electrode_params_from_prior(prior, "a")
    p_c = electrode_params_from_prior(prior, "c")
    pos_spec = prior["electrodes"]["positive"]
    neg_spec = prior["electrodes"]["negative"]
    win_c = float(pos_spec.get("theta_max", 0.9149)) - float(pos_spec.get("theta_min", 0.2535))
    win_a = float(neg_spec.get("theta_max", 0.8544)) - float(neg_spec.get("theta_min", 0.0079))
    theta_c_mean = cum_theta_from_current(t, I, theta_pos0, win_c, cap_Ah, sign=-1.0)
    theta_a_mean = cum_theta_from_current(t, I, theta_neg0, win_a, cap_Ah, sign=+1.0)
    cbar_c = theta_c_mean * p_c.csmax_mol_m3
    cbar_a = theta_a_mean * p_a.csmax_mol_m3
    J_c = infer_surface_flux_from_cbar(t, cbar_c, p_c.R)
    J_a = infer_surface_flux_from_cbar(t, cbar_a, p_a.R)
    use_nr = int(nr or gen.get("n_r", 17))
    max_sub = float((prior.get("radial_gradient") or {}).get("implicit_step_subdivide_dt_s", 10.0))
    cs_a, diag_a = generate_rg_profile(t, cbar_a, J_a, np.full(use_nr, cbar_a[0], dtype=float), p_a, nr=use_nr, max_substep_s=max_sub)
    cs_c, diag_c = generate_rg_profile(t, cbar_c, J_c, np.full(use_nr, cbar_c[0], dtype=float), p_c, nr=use_nr, max_substep_s=max_sub)
    outs: Dict[str, np.ndarray] = {
        "theta_a": (cs_a / p_a.csmax_mol_m3).astype(np.float32),
        "theta_c": (cs_c / p_c.csmax_mol_m3).astype(np.float32),
        "cs_a": cs_a.astype(np.float32),
        "cs_c": cs_c.astype(np.float32),
        "phie": (phie_scale * I).astype(np.float32).reshape(-1, 1),
        "phis_c": V.astype(np.float32).reshape(-1, 1),
        "cbar_a": cbar_a.astype(np.float32),
        "cbar_c": cbar_c.astype(np.float32),
        "J_a_eff_rg": np.asarray(diag_a.get("J_used"), dtype=np.float32),
        "J_c_eff_rg": np.asarray(diag_c.get("J_used"), dtype=np.float32),
    }
    meta = {
        "capacity_scale_Ah": cap_Ah,
        "theta_positive_initial": theta_pos0,
        "theta_negative_initial": theta_neg0,
        "phie_ohmic_scale_V_per_A": phie_scale,
        "n_r": use_nr,
        "theta_window_positive": win_c,
        "theta_window_negative": win_a,
        "csmax_a": float(p_a.csmax_mol_m3),
        "csmax_c": float(p_c.csmax_mol_m3),
    }
    return outs, meta


def patch_p4d_prediction(
    profile: Any,
    pred: np.ndarray,
    target_slices: Mapping[str, Tuple[int, int]],
    gen: Mapping[str, Any],
    prior: Mapping[str, Any],
    apply_protocols: Sequence[str] | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if str(profile.branch) != P4D_BRANCH:
        return pred, {"applied": False, "reason": "not_p4d_branch"}
    if apply_protocols and str(profile.protocol) not in {str(x) for x in apply_protocols}:
        return pred, {"applied": False, "reason": f"protocol_{profile.protocol}_not_in_apply_list"}
    patched = np.asarray(pred, dtype=np.float32).copy()
    I, V, _T, obs_meta = observed_arrays_on_target_grid(profile.softlabel_npz, profile.replay_npz, profile.t_global_s)
    # Determine radial dimension from target slice, not from a magic constant.
    nr = None
    if "cs_a" in target_slices:
        a, b = target_slices["cs_a"]
        nr = int(b - a)
    outs, meta = p4d_deterministic_outputs(profile.t_global_s, I, V, gen, prior, nr=nr)
    replaced: List[str] = []
    for key in ["theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c"]:
        if key not in target_slices:
            continue
        a, b = target_slices[key]
        arr = np.asarray(outs[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != patched.shape[0] or arr.shape[1] != (b - a):
            raise ValueError(f"P4D override shape mismatch for {key}: override={arr.shape}, target_slice={(patched.shape[0], b-a)}")
        patched[:, a:b] = arr
        replaced.append(key)
    return patched, {"applied": True, "replaced_targets": replaced, "observed_sources": obs_meta, "p4d_meta": meta}


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return float("nan")
    yt = yt[mask]
    yp = yp[mask]
    sse = float(np.sum((yp - yt) ** 2))
    ybar = float(np.mean(yt))
    sst = float(np.sum((yt - ybar) ** 2))
    return 1.0 - sse / sst if sst > 1e-18 else float("nan")


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return {"n_points": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "nmae": float("nan"), "nrmse": float("nan"), "bias": float("nan")}
    yt = yt[mask]
    yp = yp[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    rng = float(np.max(yt) - np.min(yt)) if yt.size else float("nan")
    denom = rng if math.isfinite(rng) and rng > 1e-12 else float("nan")
    return {
        "n_points": int(yt.size),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(yt, yp)),
        "nmae": float(mae / denom) if math.isfinite(denom) else float("nan"),
        "nrmse": float(rmse / denom) if math.isfinite(denom) else float("nan"),
        "bias": float(np.mean(err)),
        "target_range": rng,
        "target_std": float(np.std(yt)),
        "target_min": float(np.min(yt)),
        "target_max": float(np.max(yt)),
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
    }


def select_records_by_contains(records: Sequence[Mapping[str, Any]], contains: Sequence[str] | None = None, splits: Sequence[str] = ("all",), include_flagged: bool = False, limit: int = 0) -> List[Dict[str, Any]]:
    contains = [str(x) for x in (contains or []) if str(x).strip()]
    split_set = {str(s) for s in splits}
    use_all = "all" in split_set
    out: List[Dict[str, Any]] = []
    for r in records:
        sp = str(r.get("split") or "")
        is_flagged = bool(r.get("is_flagged_probe")) or sp == "flagged_probe"
        if is_flagged and not include_flagged:
            continue
        if not (use_all or sp in split_set):
            continue
        uid = " ".join([str(r.get("canonical_cell_uid") or ""), str(r.get("cell_uid") or ""), str(r.get("softlabel_npz") or "")])
        if contains and not any(c in uid for c in contains):
            continue
        out.append(dict(r))
    if int(limit) > 0:
        out = out[: int(limit)]
    return out


def aggregate_rows(rows: Sequence[Mapping[str, Any]], prefix: str = "") -> Dict[str, Any]:
    vals = [safe_float(r.get("r2")) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    out: Dict[str, Any] = {
        f"{prefix}profile_target_count": len(vals),
        f"{prefix}mean_r2": float(np.mean(vals)) if vals else float("nan"),
        f"{prefix}min_r2": float(np.min(vals)) if vals else float("nan"),
        f"{prefix}max_r2": float(np.max(vals)) if vals else float("nan"),
    }
    for target in sorted({str(r.get("target")) for r in rows}):
        tv = [safe_float(r.get("r2")) for r in rows if str(r.get("target")) == target]
        tv = [v for v in tv if math.isfinite(v)]
        if tv:
            out[f"{prefix}{target}_r2_mean"] = float(np.mean(tv))
            out[f"{prefix}{target}_r2_min"] = float(np.min(tv))
    return out


def find_worst(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    finite = [dict(r) for r in rows if math.isfinite(safe_float(r.get("r2")))]
    if not finite:
        return None
    return min(finite, key=lambda r: safe_float(r.get("r2"), 1e99))


def run_p4d_equivalence_smoke(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    out_dir: str | Path,
    project_root: str | Path = ".",
    p4d_config: str | Path = "",
    prior_json: str | Path = "",
    profile_contains: Sequence[str] | None = None,
    max_time_points: int = 0,
    time_window_s: float = 0.0,
    r2_mean_threshold: float = 0.98,
    r2_min_threshold: float = 0.95,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen, prior, p4d_meta = load_p4d_config_and_prior(project_root, p4d_config, prior_json)
    records, manifest = load_split_records(split_manifest)
    sem_map = load_semantics_map(g0_profile_semantics_csv)
    selected = select_records_by_contains(records, contains=profile_contains, splits=("all",), include_flagged=False)
    if not selected:
        raise ValueError(f"No records matched profile_contains={profile_contains}")
    protocol_vocab = sorted({str(r.get("protocol") or "UNKNOWN") for r in records})
    branches = sorted({str(_semantics_for(r, sem_map).get("semantic_branch") or "UNKNOWN_OR_MIXED_BRANCH") for r in records})
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for r in selected:
        sem = _semantics_for(r, sem_map)
        branch = str(sem.get("semantic_branch") or "")
        if branch != P4D_BRANCH:
            continue
        try:
            prof = load_profile_pack(r, sem, protocol_vocab, branches, int(max_time_points), float(time_window_s))
            I, V, _T, obs_meta = observed_arrays_on_target_grid(prof.softlabel_npz, prof.replay_npz, prof.t_global_s)
            nr = int(prof.target_slices["cs_a"][1] - prof.target_slices["cs_a"][0])
            outs, meta = p4d_deterministic_outputs(prof.t_global_s, I, V, gen, prior, nr=nr)
            for target in TARGETS_DEFAULT:
                if target not in prof.target_slices:
                    continue
                a, b = prof.target_slices[target]
                yt = prof.targets[:, a:b]
                yp = outs[target]
                if yp.ndim == 1:
                    yp = yp.reshape(-1, 1)
                row = {
                    "split": prof.split,
                    "canonical_cell_uid": prof.canonical_cell_uid,
                    "cell_uid": prof.cell_uid,
                    "protocol": prof.protocol,
                    "semantic_branch": prof.branch,
                    "target": target,
                    **metric_row(yt, yp),
                }
                rows.append(row)
        except Exception as e:
            failures.append({"canonical_cell_uid": r.get("canonical_cell_uid") or r.get("cell_uid"), "error": repr(e)})
    write_csv(rows, out / "D17_G62_P4D_EQUIVALENCE_PROFILE_TARGET_METRICS.csv")
    write_csv(failures, out / "D17_G62_P4D_EQUIVALENCE_FAILURES.csv")
    agg = aggregate_rows(rows)
    blockers: List[str] = []
    if failures:
        blockers.append(f"{len(failures)} P4D profiles failed deterministic equivalence evaluation")
    if safe_float(agg.get("mean_r2")) < float(r2_mean_threshold) or safe_float(agg.get("min_r2")) < float(r2_min_threshold):
        blockers.append(f"P4D deterministic equivalence below gate: mean={agg.get('mean_r2')}, min={agg.get('min_r2')}")
    status = "PASS" if not failures else "REVIEW"
    promotion_status = "PASS" if status == "PASS" and not blockers else "REVIEW"
    summary = {
        "protocol": "D17-G6.2_P4D_GENERATOR_EQUIVALENCE_SMOKE",
        "created_at_utc": utc_now(),
        "status": status,
        "promotion_status": promotion_status,
        "g62_patch_ready": bool(promotion_status == "PASS"),
        "recommendation": "RUN_G62_PATCHED_AUDIT" if promotion_status == "PASS" else "REVIEW_P4D_FORMULA_OR_PRIOR_BEFORE_PATCHED_AUDIT",
        "blockers": blockers,
        "policy": {
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "state_softlabels_used_for_training": False,
            "state_softlabels_used_for_metrics_only": True,
            "deterministic_inputs": "t/I/V from replay or copied observed fields, plus D15-P4D config/prior",
        },
        "p4d_meta": p4d_meta,
        "dataset": {"manifest_hash_sha256": manifest.get("manifest_hash_sha256"), "record_counts": manifest.get("counts"), "selected_record_count": len(selected), "evaluated_profile_count": len({r.get("canonical_cell_uid") for r in rows})},
        "aggregate": agg,
        "worst_profile_target": find_worst(rows),
        "files": {
            "summary_json": str(out / "D17_G62_P4D_INVENTORY_EQUIVALENCE_SMOKE_SUMMARY.json"),
            "profile_target_metrics_csv": str(out / "D17_G62_P4D_EQUIVALENCE_PROFILE_TARGET_METRICS.csv"),
            "failures_csv": str(out / "D17_G62_P4D_EQUIVALENCE_FAILURES.csv"),
        },
    }
    write_json(summary, out / "D17_G62_P4D_INVENTORY_EQUIVALENCE_SMOKE_SUMMARY.json")
    return summary


def run_p4d_patched_audit(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    candidate_dir: str | Path,
    candidate_summary: str | Path,
    out_dir: str | Path,
    project_root: str | Path = ".",
    p4d_config: str | Path = "",
    prior_json: str | Path = "",
    checkpoint_path: str | Path = "",
    splits: Sequence[str] = ("all",),
    include_flagged_probe: bool = False,
    profile_limit: int = 0,
    profile_contains: Sequence[str] | None = None,
    max_time_points: int = 0,
    time_window_s: float = 0.0,
    predict_batch_size: int = 8192,
    device_arg: str = "auto",
    apply_protocols: Sequence[str] | None = None,
    r2_mean_threshold: float = 0.95,
    r2_min_threshold: float = 0.90,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gen, prior, p4d_meta = load_p4d_config_and_prior(project_root, p4d_config, prior_json)
    candidate = read_json(candidate_summary, default={}) or {}
    ckpt_path = resolve_checkpoint_path(str(checkpoint_path or ""), candidate, candidate_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint for G6.2 patched audit: {ckpt_path}")
    ckpt = torch_load_safe(ckpt_path, map_location="cpu")
    device = _device_from_arg(device_arg)
    model = build_model_from_checkpoint(ckpt, device)
    feature_names = list(ckpt.get("feature_names") or [])
    local_input_dim = int(ckpt.get("local_input_dim", 0))
    _, protocol_vocab, branch_vocab = parse_vocab_from_feature_names(feature_names, local_input_dim)
    if not protocol_vocab or not branch_vocab:
        raise ValueError("Could not parse protocol/branch vocab from checkpoint feature_names")
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(ckpt.get("target_slices") or {}).items()}
    records, manifest = load_split_records(split_manifest)
    sem_map = load_semantics_map(g0_profile_semantics_csv)
    selected = select_records_by_contains(records, contains=profile_contains, splits=splits, include_flagged=include_flagged_probe, limit=int(profile_limit))
    if not selected:
        raise ValueError("No records selected for D17-G6.2 patched audit")

    profile_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    patch_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    total_points = 0
    t0 = time.time()
    model.eval()
    for idx, rec in enumerate(selected):
        canonical = str(rec.get("canonical_cell_uid") or rec.get("cell_uid") or f"profile_{idx}")
        try:
            sem = _semantics_for(rec, sem_map)
            prof = load_profile_pack(rec, sem, protocol_vocab, branch_vocab, int(max_time_points), float(time_window_s))
            cycles, cycle_info = load_cycle_ids_for_profile(prof)
            X_aug, finfo, aug_names = augment_profile_features(prof)
            if X_aug.shape[1] != x_mean.size:
                raise ValueError(f"feature dim mismatch for {canonical}: X_aug={X_aug.shape[1]}, checkpoint={x_mean.size}")
            if list(aug_names) != feature_names:
                raise ValueError(f"feature name mismatch for {canonical}; refusing silent schema drift")
            total_points += int(prof.targets.shape[0])
            profile_pred_chunks: List[np.ndarray] = []
            for j0 in range(0, X_aug.shape[0], max(1, int(predict_batch_size))):
                j1 = min(X_aug.shape[0], j0 + max(1, int(predict_batch_size)))
                profile_pred_chunks.append(predict_denorm(model, X_aug[j0:j1], x_mean, x_std, y_mean, y_std, device))
            pred_raw = np.concatenate(profile_pred_chunks, axis=0) if profile_pred_chunks else np.zeros_like(prof.targets)
            pred, patch_meta = patch_p4d_prediction(prof, pred_raw, target_slices, gen, prior, apply_protocols=apply_protocols)
            patch_rows.append({
                "profile_index": idx,
                "split": prof.split,
                "canonical_cell_uid": prof.canonical_cell_uid,
                "cell_uid": prof.cell_uid,
                "protocol": prof.protocol,
                "semantic_branch": prof.branch,
                "n_time": int(prof.targets.shape[0]),
                "cycle_ranges": cycle_ranges(cycles),
                "patch_applied": bool(patch_meta.get("applied")),
                "patch_reason": patch_meta.get("reason", ""),
                "patch_replaced_targets": ";".join(patch_meta.get("replaced_targets") or []),
            })
            for target, (a, b) in target_slices.items():
                yt = prof.targets[:, a:b]
                yp = pred[:, a:b]
                row = {"split": prof.split, "canonical_cell_uid": prof.canonical_cell_uid, "cell_uid": prof.cell_uid, "protocol": prof.protocol, "semantic_branch": prof.branch, "target": target, **metric_row(yt, yp)}
                profile_rows.append(row)
                for cv in np.unique(cycles):
                    mask = cycles == cv
                    cycle_rows.append({"split": prof.split, "canonical_cell_uid": prof.canonical_cell_uid, "cell_uid": prof.cell_uid, "protocol": prof.protocol, "semantic_branch": prof.branch, "cycle_id": str(cv), "target": target, **metric_row(yt[mask], yp[mask])})
            del prof, X_aug, pred_raw, pred
        except Exception as e:
            failures.append({"profile_index": idx, "canonical_cell_uid": canonical, "cell_uid": rec.get("cell_uid"), "split": rec.get("split"), "protocol": rec.get("protocol"), "error": repr(e)})
    elapsed = max(time.time() - t0, 1e-9)
    write_csv(profile_rows, out / "D17_G62_PATCHED_PROFILE_TARGET_METRICS.csv")
    write_csv(cycle_rows, out / "D17_G62_PATCHED_CYCLE_TARGET_METRICS.csv")
    write_csv(patch_rows, out / "D17_G62_PATCH_APPLICATION_MANIFEST.csv")
    write_csv(failures, out / "D17_G62_PATCHED_AUDIT_FAILURES.csv")
    agg = aggregate_rows(profile_rows)
    blockers: List[str] = []
    if failures:
        blockers.append(f"{len(failures)} profiles failed G6.2 patched audit")
    if safe_float(agg.get("mean_r2")) < float(r2_mean_threshold) or safe_float(agg.get("min_r2")) < float(r2_min_threshold):
        blockers.append(f"G6.2 patched profile-target R2 below gate: mean={agg.get('mean_r2')}, min={agg.get('min_r2')}")
    status = "PASS" if not failures else "REVIEW"
    promotion_status = "PASS" if status == "PASS" and not blockers else "REVIEW"
    summary = {
        "protocol": "D17-G6.2_P4D_GEO_SEMANTIC_INVENTORY_PATCHED_AUDIT",
        "created_at_utc": utc_now(),
        "status": status,
        "promotion_status": promotion_status,
        "g6_streaming_ready": bool(promotion_status == "PASS"),
        "recommendation": "RUN_G6C_CYCLE_WISE_STREAMING_AUDIT" if promotion_status == "PASS" else "REVIEW_G62_PATCHED_AUDIT_BEFORE_FULL_STREAMING",
        "blockers": blockers,
        "purpose": "Evaluate the G6.1 full-cycle candidate after replacing P4D branch theta/cs/phie/phis predictions with a deterministic D15-P4D code-equivalent inventory layer. No new training is performed.",
        "policy": {
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "candidate_modified": False,
            "p4d_patch_uses_state_softlabels": False,
            "p4d_patch_inputs": "observed t/I/V + D15-P4D config/prior + p2dlite_rg radial_solver",
            "softlabels_report_only_for_metrics": True,
        },
        "candidate": {"candidate_summary": str(candidate_summary), "candidate_protocol": candidate.get("protocol"), "candidate_status": candidate.get("status"), "candidate_g6_ready": candidate.get("g6_ready"), "checkpoint": str(ckpt_path), "checkpoint_best_epoch": ckpt.get("best_epoch")},
        "p4d_meta": p4d_meta,
        "dataset": {"manifest_hash_sha256": manifest.get("manifest_hash_sha256"), "record_counts": manifest.get("counts"), "selected_record_count": len(selected), "evaluated_profile_count": len({r.get("canonical_cell_uid") for r in profile_rows}), "total_time_points_evaluated": int(total_points), "max_time_points": int(max_time_points), "time_window_s": float(time_window_s)},
        "runtime": {"elapsed_s": float(elapsed), "points_per_second": float(total_points / elapsed), "device": str(device), "predict_batch_size": int(predict_batch_size)},
        "aggregate": agg,
        "worst_profile_target": find_worst(profile_rows),
        "worst_cycle_target": find_worst(cycle_rows),
        "patch_application_counts": {"applied": int(sum(1 for r in patch_rows if r.get("patch_applied") in [True, "True", "true", 1])), "not_applied": int(sum(1 for r in patch_rows if not (r.get("patch_applied") in [True, "True", "true", 1])))} ,
        "files": {"summary_json": str(out / "D17_G62_P4D_GEO_PATCHED_AUDIT_SUMMARY.json"), "scorecard_json": str(out / "D17_G62_SCORECARD.json"), "profile_target_metrics_csv": str(out / "D17_G62_PATCHED_PROFILE_TARGET_METRICS.csv"), "cycle_target_metrics_csv": str(out / "D17_G62_PATCHED_CYCLE_TARGET_METRICS.csv"), "patch_manifest_csv": str(out / "D17_G62_PATCH_APPLICATION_MANIFEST.csv"), "failures_csv": str(out / "D17_G62_PATCHED_AUDIT_FAILURES.csv")},
    }
    write_json(summary, out / "D17_G62_P4D_GEO_PATCHED_AUDIT_SUMMARY.json")
    scorecard = {
        "protocol": summary["protocol"],
        "status": summary["status"],
        "promotion_status": summary["promotion_status"],
        "g6_streaming_ready": summary["g6_streaming_ready"],
        "blockers": blockers,
        "aggregate": agg,
        "worst_profile_target": summary["worst_profile_target"],
        "worst_cycle_target": summary["worst_cycle_target"],
        "patch_application_counts": summary["patch_application_counts"],
        "policy": summary["policy"],
    }
    write_json(scorecard, out / "D17_G62_SCORECARD.json")
    return summary
