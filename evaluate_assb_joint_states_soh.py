# -*- coding: utf-8 -*-
"""Joint five-state evaluator for ASSB StageB/107A hybrid and ASSB-111.

Default mode (`stageB_hybrid`) keeps the original D5 fix2 intent:
- four electrochemical states are read from a 107A evaluation NPZ;
- paired true/pred arrays inside that NPZ are authoritative;
- mismatched concentration arrays are never silently flattened/truncated;
- SOH is read from ModelFin_110_stageB evaluation CSV.

ASSB-111 mode delegates to scripts/evaluate_assb111_five_state.py, which is the
preferred strict30 evaluator for ModelFin_111.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STATE_KEYS: Tuple[str, ...] = ("cs_a", "cs_c", "phie", "phis_c")
PRED_ALIASES: Dict[str, List[str]] = {
    "cs_a": ["cs_a_pred", "cs_a_prediction", "pred_cs_a", "prediction_cs_a", "csa_pred", "cs_a_pred_corrected", "cs_a_corrected_pred", "cs_a_hat"],
    "cs_c": ["cs_c_pred", "cs_c_prediction", "pred_cs_c", "prediction_cs_c", "csc_pred", "cs_c_pred_corrected", "cs_c_corrected_pred", "cs_c_hat"],
    "phie": ["phie_pred", "phie_prediction", "pred_phie", "prediction_phie", "phi_e_pred", "phie_hat"],
    "phis_c": ["phis_c_pred", "phis_c_prediction", "pred_phis_c", "prediction_phis_c", "phi_s_c_pred", "phis_pred", "phis_c_hat"],
}
TRUE_ALIASES: Dict[str, List[str]] = {
    "cs_a": ["cs_a_true", "true_cs_a", "cs_a_ref", "ref_cs_a", "cs_a_reference", "reference_cs_a", "cs_a_label", "label_cs_a", "csa_true", "csa_ref"],
    "cs_c": ["cs_c_true", "true_cs_c", "cs_c_ref", "ref_cs_c", "cs_c_reference", "reference_cs_c", "cs_c_label", "label_cs_c", "csc_true", "csc_ref"],
    "phie": ["phie_true", "true_phie", "phie_ref", "ref_phie", "phie_reference", "reference_phie", "phie_label", "label_phie"],
    "phis_c": ["phis_c_true", "true_phis_c", "phis_c_ref", "ref_phis_c", "phis_c_reference", "reference_phis_c", "phis_c_label", "label_phis_c"],
}
GENERIC_ALIASES: Dict[str, List[str]] = {
    "cs_a": ["cs_a", "csa"],
    "cs_c": ["cs_c", "csc"],
    "phie": ["phie", "phi_e"],
    "phis_c": ["phis_c", "phi_s_c", "phis"],
}
CYCLE_ALIASES_BY_VAR: Dict[str, List[str]] = {
    "cs_a": ["cycle_id_cs", "cycle_id_csa", "cycle_id_concentration", "cycle_id_sampled", "cycle_id"],
    "cs_c": ["cycle_id_cs", "cycle_id_csc", "cycle_id_concentration", "cycle_id_sampled", "cycle_id"],
    "phie": ["cycle_id_potential", "cycle_id_phie", "cycle_id_phi", "cycle_id"],
    "phis_c": ["cycle_id_potential", "cycle_id_phis_c", "cycle_id_phi", "cycle_id"],
}
TIME_ALIASES_BY_VAR: Dict[str, List[str]] = {
    "cs_a": ["t_cs", "t_concentration", "t_sampled", "t_global_s", "time_s", "t"],
    "cs_c": ["t_cs", "t_concentration", "t_sampled", "t_global_s", "time_s", "t"],
    "phie": ["t_potential", "t_phi", "t_global_s", "time_s", "t"],
    "phis_c": ["t_potential", "t_phi", "t_global_s", "time_s", "t"],
}
PRED_NPZ_PREFERRED_NAMES: Tuple[str, ...] = (
    "eval_sampled_arrays_ModelFin107A_csA_corrected.npz",
    "eval_sampled_arrays_corrected.npz",
    "eval_sampled_arrays.npz",
    "states_corrected.npz",
    "predictions_corrected.npz",
    "eval_predictions_corrected.npz",
    "prediction_corrected.npz",
    "predictions.npz",
    "prediction.npz",
    "eval_predictions.npz",
    "state_prediction.npz",
    "pinn_predictions.npz",
    "results.npz",
)


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        val = float(x)
        return None if not math.isfinite(val) else val
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find_key(files: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    direct = {str(k).lower(): str(k) for k in files}
    for alias in aliases:
        hit = direct.get(str(alias).lower())
        if hit is not None:
            return hit
    relaxed = {_norm_name(k): str(k) for k in files}
    for alias in aliases:
        hit = relaxed.get(_norm_name(alias))
        if hit is not None:
            return hit
    return None


def _load_array(z: np.lib.npyio.NpzFile, aliases: Sequence[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    key = _find_key(z.files, aliases)
    if key is None:
        return None, None
    return key, np.asarray(z[key])


def _metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if obs.shape != pred.shape:
        raise ValueError(f"shape mismatch: obs {obs.shape} vs pred {pred.shape}")
    mask = np.isfinite(obs) & np.isfinite(pred)
    out: Dict[str, Any] = {
        "n": int(np.sum(mask)), "MAE": float("nan"), "RMSE": float("nan"),
        "BIAS": float("nan"), "MAX": float("nan"), "corr": float("nan"),
        "R2": float("nan"), "obs_min": float("nan"), "obs_max": float("nan"),
        "obs_range": float("nan"), "pred_min": float("nan"), "pred_max": float("nan"),
        "NMAE": float("nan"), "NRMSE": float("nan"),
    }
    if not mask.any():
        return out
    y = obs[mask]
    p = pred[mask]
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    out.update({
        "MAE": mae, "RMSE": rmse, "BIAS": float(np.mean(e)), "MAX": float(np.max(np.abs(e))),
        "obs_min": float(np.min(y)), "obs_max": float(np.max(y)),
        "pred_min": float(np.min(p)), "pred_max": float(np.max(p)),
    })
    obs_range = float(out["obs_max"] - out["obs_min"])
    out["obs_range"] = obs_range
    if obs_range > 1.0e-30:
        out["NMAE"] = float(mae / obs_range)
        out["NRMSE"] = float(rmse / obs_range)
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot > 1.0e-30:
        out["R2"] = float(1.0 - float(np.sum(e * e)) / ss_tot)
    if y.size >= 2 and float(np.std(y)) > 1.0e-30 and float(np.std(p)) > 1.0e-30:
        out["corr"] = float(np.corrcoef(y, p)[0, 1])
    return out


def _metrics_or_unavailable(reason: str, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": False, "reason": reason, "n": 0,
        "MAE": float("nan"), "RMSE": float("nan"), "BIAS": float("nan"), "MAX": float("nan"),
        "corr": float("nan"), "R2": float("nan"), "NMAE": float("nan"), "NRMSE": float("nan"),
    }
    out.update(extra)
    return out


def _sample_indices(n: int, max_points: int) -> Optional[np.ndarray]:
    if max_points is None or max_points <= 0 or n <= max_points:
        return None
    return np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))


def _maybe_sample_pair(obs: np.ndarray, pred: np.ndarray, max_time_points: int, max_radial_points: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {"sampled_time": False, "sampled_radial": False}
    if obs.shape != pred.shape:
        raise ValueError(f"shape mismatch before sampling: obs {obs.shape}, pred {pred.shape}")
    if obs.ndim >= 1:
        tidx = _sample_indices(obs.shape[0], max_time_points)
        if tidx is not None:
            obs = obs[tidx]
            pred = pred[tidx]
            info["sampled_time"] = True
            info["n_time_after_sampling"] = int(len(tidx))
    if obs.ndim >= 2:
        ridx = _sample_indices(obs.shape[1], max_radial_points)
        if ridx is not None:
            obs = obs[:, ridx]
            pred = pred[:, ridx]
            info["sampled_radial"] = True
            info["n_radial_after_sampling"] = int(len(ridx))
    return obs, pred, info


def _nearest_indices(source_t: np.ndarray, target_t: np.ndarray, tol: float) -> np.ndarray:
    source = np.asarray(source_t, dtype=np.float64).reshape(-1)
    target = np.asarray(target_t, dtype=np.float64).reshape(-1)
    if source.size == 0 or target.size == 0:
        raise ValueError("empty time axis for alignment")
    order = np.argsort(source)
    sorted_t = source[order]
    pos = np.searchsorted(sorted_t, target)
    pos0 = np.clip(pos - 1, 0, sorted_t.size - 1)
    pos1 = np.clip(pos, 0, sorted_t.size - 1)
    d0 = np.abs(sorted_t[pos0] - target)
    d1 = np.abs(sorted_t[pos1] - target)
    choose = np.where(d1 < d0, pos1, pos0)
    dist = np.minimum(d0, d1)
    if tol >= 0 and dist.size and float(np.nanmax(dist)) > tol:
        raise ValueError(f"nearest-time alignment max distance {float(np.nanmax(dist)):.6g} > tolerance {tol}")
    return order[choose]


def _cycle_array_from_npz(z: np.lib.npyio.NpzFile, var: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    return _load_array(z, CYCLE_ALIASES_BY_VAR[var])


def _time_array_from_npz(z: np.lib.npyio.NpzFile, var: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    return _load_array(z, TIME_ALIASES_BY_VAR[var])


def _score_npz(path: Path) -> Dict[str, Any]:
    try:
        with np.load(path, allow_pickle=True) as z:
            files = list(z.files)
            score = 0
            paired = 0
            details: Dict[str, Dict[str, Optional[str]]] = {}
            for var in STATE_KEYS:
                pred = _find_key(files, PRED_ALIASES[var])
                true = _find_key(files, TRUE_ALIASES[var])
                generic = _find_key(files, GENERIC_ALIASES[var])
                if pred:
                    score += 3
                if true:
                    score += 3
                if pred and true:
                    score += 5
                    paired += 1
                if generic:
                    score += 1
                details[var] = {"pred_key": pred, "true_key": true, "generic_key": generic}
            if path.name in PRED_NPZ_PREFERRED_NAMES:
                score += 3
            return {"path": str(path), "score": score, "n_paired_variables": paired, "details": details, "keys_sample": files[:50]}
    except Exception as exc:
        return {"path": str(path), "score": -1, "error": str(exc)}


def discover_state_npz(state_eval_dir: Optional[Path], explicit_npz: Optional[Path], output_dir: Path) -> Path:
    candidates: List[Path] = []
    if explicit_npz and explicit_npz.exists():
        candidates.append(explicit_npz)
    elif state_eval_dir and state_eval_dir.exists():
        for name in PRED_NPZ_PREFERRED_NAMES:
            p = state_eval_dir / name
            if p.exists():
                candidates.append(p)
        candidates += [p for p in sorted(state_eval_dir.rglob("*.npz")) if p not in candidates]
    scored = [_score_npz(p) for p in candidates]
    valid = [d for d in scored if int(d.get("score", -1)) >= 8]
    selected = max(valid, key=lambda d: (int(d.get("n_paired_variables", 0)), int(d.get("score", -1)))) if valid else None
    save_json({"state_eval_dir": str(state_eval_dir) if state_eval_dir else "", "explicit_npz": str(explicit_npz) if explicit_npz else "", "selected": selected, "candidates": scored}, output_dir / "state_npz_discovery.json")
    if not selected:
        raise FileNotFoundError(f"Cannot find a state prediction NPZ with paired state arrays in {state_eval_dir or explicit_npz}")
    return Path(selected["path"])


def _resolve_state_pair(var: str, pred_z: np.lib.npyio.NpzFile, ref_z: Optional[np.lib.npyio.NpzFile], align_tolerance_s: float) -> Dict[str, Any]:
    pred_key, pred_arr = _load_array(pred_z, PRED_ALIASES[var])
    true_key, true_arr = _load_array(pred_z, TRUE_ALIASES[var])
    cycle_key, cycle_arr = _cycle_array_from_npz(pred_z, var)
    time_key, time_arr = _time_array_from_npz(pred_z, var)

    if pred_arr is not None and true_arr is not None:
        if pred_arr.shape != true_arr.shape:
            return {"available": False, "reason": f"paired internal shape mismatch: {true_key}{true_arr.shape} vs {pred_key}{pred_arr.shape}", "alignment_mode": "paired_npz_internal_shape_mismatch"}
        return {"available": True, "obs": true_arr, "pred": pred_arr, "pred_key": pred_key, "ref_key": true_key, "cycle_key": cycle_key, "cycle_id": cycle_arr, "time_key": time_key, "time": time_arr, "alignment_mode": "paired_npz_internal", "source": "state_prediction_npz_internal_true_pred"}

    if pred_arr is None:
        generic_key, generic_arr = _load_array(pred_z, GENERIC_ALIASES[var])
        if generic_arr is not None:
            pred_key, pred_arr = generic_key, generic_arr
    if pred_arr is None:
        return {"available": False, "reason": f"missing prediction array for {var}", "alignment_mode": "unavailable"}
    if ref_z is None:
        return {"available": False, "reason": f"{var} has prediction array but no internal true array and no reference_npz", "alignment_mode": "missing_reference"}

    ref_key, ref_arr = _load_array(ref_z, TRUE_ALIASES[var] + GENERIC_ALIASES[var])
    if ref_arr is None:
        return {"available": False, "reason": f"reference_npz missing array for {var}", "alignment_mode": "reference_missing"}
    if ref_arr.shape == pred_arr.shape:
        ref_cycle_key, ref_cycle = _cycle_array_from_npz(ref_z, var)
        return {"available": True, "obs": ref_arr, "pred": pred_arr, "pred_key": pred_key, "ref_key": ref_key, "cycle_key": cycle_key or ref_cycle_key, "cycle_id": cycle_arr if cycle_arr is not None else ref_cycle, "time_key": time_key, "time": time_arr, "alignment_mode": "reference_npz_exact_shape", "source": "state_prediction_npz_vs_reference_npz_exact_shape"}

    ref_time_key, ref_time = _time_array_from_npz(ref_z, var)
    if time_arr is not None and ref_time is not None:
        try:
            idx = _nearest_indices(np.asarray(ref_time).reshape(-1), np.asarray(time_arr).reshape(-1), align_tolerance_s)
            if ref_arr.ndim >= 1 and ref_arr.shape[0] >= int(np.max(idx)) + 1:
                ref_aligned = ref_arr[idx]
                if ref_aligned.shape == pred_arr.shape:
                    return {"available": True, "obs": ref_aligned, "pred": pred_arr, "pred_key": pred_key, "ref_key": ref_key, "cycle_key": cycle_key, "cycle_id": cycle_arr, "time_key": time_key, "time": time_arr, "alignment_mode": "reference_npz_nearest_time", "time_alignment_reference_key": ref_time_key, "source": "state_prediction_npz_vs_reference_npz_time_aligned"}
        except Exception as exc:
            return {"available": False, "reason": f"shape mismatch and time alignment failed for {var}: {exc}", "alignment_mode": "reference_npz_time_alignment_failed", "pred_shape": tuple(pred_arr.shape), "ref_shape": tuple(ref_arr.shape)}
    return {"available": False, "reason": f"shape mismatch with no valid alignment for {var}: pred {pred_key}{pred_arr.shape}, ref {ref_key}{ref_arr.shape}. Refusing silent flatten/truncate.", "alignment_mode": "shape_mismatch_no_truncation", "pred_shape": tuple(pred_arr.shape), "ref_shape": tuple(ref_arr.shape)}


def _cycle_metric_rows(var: str, obs: np.ndarray, pred: np.ndarray, cycle_id: Optional[np.ndarray], split_map: Optional[Mapping[int, str]]) -> List[Dict[str, Any]]:
    if cycle_id is None:
        return []
    cyc = np.asarray(cycle_id).reshape(-1)
    if obs.ndim == 0 or obs.shape[0] != cyc.size:
        return []
    rows: List[Dict[str, Any]] = []
    cyc_int = cyc.astype(int)
    for c in np.unique(cyc_int):
        mask = cyc_int == int(c)
        if not np.any(mask):
            continue
        m = _metrics(obs[mask], pred[mask])
        m.update({"cycle_id": int(c), "variable": var})
        if split_map is not None:
            m["split"] = split_map.get(int(c), "")
        rows.append(m)
    return rows


def _load_split_map(cycle_table_csv: Optional[Path]) -> Optional[Dict[int, str]]:
    if cycle_table_csv is None or not cycle_table_csv.exists():
        return None
    try:
        frame = pd.read_csv(cycle_table_csv)
    except Exception:
        return None
    if not {"cycle_id", "split"}.issubset(frame.columns):
        return None
    return {int(r["cycle_id"]): str(r["split"]) for _, r in frame.iterrows() if pd.notna(r.get("cycle_id"))}


def evaluate_states(state_prediction_npz: Path, reference_npz: Optional[Path], cycle_table_csv: Optional[Path], output_dir: Path, max_time_points: int, max_radial_points: int, align_tolerance_s: float) -> Dict[str, Any]:
    split_map = _load_split_map(cycle_table_csv)
    states: Dict[str, Any] = {}
    global_rows: List[Dict[str, Any]] = []
    by_cycle_rows: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {"state_prediction_npz": str(state_prediction_npz), "reference_npz": str(reference_npz) if reference_npz else "", "variables": {}}

    with np.load(state_prediction_npz, allow_pickle=True) as pred_z:
        ref_z_ctx = np.load(reference_npz, allow_pickle=True) if reference_npz else None
        try:
            for var in STATE_KEYS:
                resolved = _resolve_state_pair(var, pred_z, ref_z_ctx, align_tolerance_s)
                if not resolved.get("available", False):
                    block = _metrics_or_unavailable(str(resolved.get("reason", "unavailable")), **{k: v for k, v in resolved.items() if k not in {"obs", "pred"}})
                    states[var] = block
                    global_rows.append({"variable": var, **block})
                    provenance["variables"][var] = block
                    continue
                obs = np.asarray(resolved["obs"])
                pred = np.asarray(resolved["pred"])
                obs_s, pred_s, sample_info = _maybe_sample_pair(obs, pred, max_time_points=max_time_points, max_radial_points=max_radial_points)
                m = _metrics(obs_s, pred_s)
                m.update({"available": True, "variable": var, "pred_key": resolved.get("pred_key"), "ref_key": resolved.get("ref_key"), "cycle_key": resolved.get("cycle_key"), "time_key": resolved.get("time_key"), "source": resolved.get("source"), "alignment_mode": resolved.get("alignment_mode"), "obs_shape": tuple(obs.shape), "pred_shape": tuple(pred.shape), **sample_info})
                states[var] = m
                global_rows.append({"variable": var, **m})
                provenance["variables"][var] = {k: m.get(k) for k in ["pred_key", "ref_key", "cycle_key", "time_key", "source", "alignment_mode", "obs_shape", "pred_shape", "sampled_time", "sampled_radial"]}
                by_cycle_rows.extend(_cycle_metric_rows(var, obs, pred, resolved.get("cycle_id"), split_map))
        finally:
            if ref_z_ctx is not None:
                ref_z_ctx.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(global_rows).to_csv(output_dir / "metrics_states_global.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(by_cycle_rows).to_csv(output_dir / "metrics_states_by_cycle.csv", index=False, encoding="utf-8-sig")
    save_json(states, output_dir / "metrics_states_global.json")
    save_json(provenance, output_dir / "state_array_alignment_provenance.json")
    return states


def _discover_stageb_csv(stageb_eval_dir: Optional[Path], explicit: Optional[Path]) -> Path:
    candidates: List[Path] = []
    if explicit and explicit.exists():
        candidates.append(explicit)
    if stageb_eval_dir and stageb_eval_dir.exists():
        candidates.extend([stageb_eval_dir / "mechanism_by_cycle.csv", stageb_eval_dir / "soh_stageB_by_cycle.csv"])
        candidates.extend(stageb_eval_dir.rglob("mechanism_by_cycle.csv"))
        candidates.extend(stageb_eval_dir.rglob("soh_stageB_by_cycle.csv"))
    seen: set[str] = set()
    for p in candidates:
        ps = str(p)
        if ps in seen or not p.exists():
            continue
        seen.add(ps)
        try:
            frame = pd.read_csv(p)
        except Exception:
            continue
        if {"cycle_id", "SOH_obs", "SOH_pred"}.issubset(frame.columns):
            return p
    raise FileNotFoundError("Cannot find StageB CSV with cycle_id/SOH_obs/SOH_pred")


def _complete_mask(frame: pd.DataFrame) -> pd.Series:
    for col in ("complete_cycle", "complete_cycle_from_solution", "is_complete"):
        if col in frame.columns:
            vals = frame[col]
            if vals.dtype == bool:
                return vals.fillna(False)
            return vals.astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return pd.Series(np.ones(len(frame), dtype=bool), index=frame.index)


def _soh_metrics_by_split(frame: pd.DataFrame, complete_only: bool) -> Dict[str, Any]:
    x = frame.copy()
    if complete_only:
        x = x[_complete_mask(x)].copy()
    out: Dict[str, Any] = {}
    if len(x) == 0:
        out["all"] = _metrics_or_unavailable("no rows after complete_only filter")
        return out
    out["all"] = _metrics(x["SOH_obs"].to_numpy(float), x["SOH_pred"].to_numpy(float))
    if "split" in x.columns:
        for split, g in x.groupby("split"):
            out[str(split)] = _metrics(g["SOH_obs"].to_numpy(float), g["SOH_pred"].to_numpy(float))
    return out


def evaluate_stageb_soh(stageb_eval_dir: Optional[Path], stageb_mechanism_csv: Optional[Path], output_dir: Path) -> Dict[str, Any]:
    csv_path = _discover_stageb_csv(stageb_eval_dir, stageb_mechanism_csv)
    frame = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "soh_stageB_by_cycle.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "available": True,
        "source_csv": str(csv_path),
        "capacity_by_split": _soh_metrics_by_split(frame, complete_only=False),
        "capacity_by_split_complete_only": _soh_metrics_by_split(frame, complete_only=True),
    }
    save_json(metrics, output_dir / "soh_stageB_metrics.json")
    return metrics


def _write_scorecard(states: Dict[str, Any], soh_metrics: Optional[Dict[str, Any]], output_dir: Path, *, soh_source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for var in STATE_KEYS:
        m = dict(states.get(var, {}))
        rows.append({
            "variable": var, "source": m.get("source", "state_prediction_npz"), "n": m.get("n", 0),
            "MAE": m.get("MAE"), "RMSE": m.get("RMSE"), "NMAE": m.get("NMAE"),
            "NRMSE": m.get("NRMSE"), "R2": m.get("R2"), "corr": m.get("corr"),
            "available": m.get("available", False),
        })
    if soh_metrics:
        # For StageB legacy, complete_only/all is the main reported SOH value.
        m = soh_metrics.get("capacity_by_split_complete_only", {}).get("all", {})
        rows.append({
            "variable": "SOH", "source": soh_source, "n": m.get("n", 0),
            "MAE": m.get("MAE"), "RMSE": m.get("RMSE"), "NMAE": m.get("NMAE"),
            "NRMSE": m.get("NRMSE"), "R2": m.get("R2"), "corr": m.get("corr"),
            "available": True,
        })
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(output_dir / "five_state_scorecard.csv", index=False, encoding="utf-8-sig")
    return scorecard


def run_stageb_hybrid(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_npz = discover_state_npz(Path(args.state_eval_dir) if args.state_eval_dir else None, Path(args.state_prediction_npz) if args.state_prediction_npz else None, output_dir)
    ref_npz = Path(args.reference_npz) if args.reference_npz else None
    states = evaluate_states(
        state_prediction_npz=state_npz,
        reference_npz=ref_npz if ref_npz and ref_npz.exists() else None,
        cycle_table_csv=Path(args.cycle_table_csv) if args.cycle_table_csv else None,
        output_dir=output_dir,
        max_time_points=int(args.max_time_points),
        max_radial_points=int(args.max_radial_points),
        align_tolerance_s=float(args.align_tolerance_s),
    )
    soh_metrics = evaluate_stageb_soh(Path(args.stageB_eval_dir) if args.stageB_eval_dir else None, Path(args.stageB_mechanism_csv) if args.stageB_mechanism_csv else None, output_dir)
    scorecard = _write_scorecard(states, soh_metrics, output_dir, soh_source="StageB_complete_only")
    save_json({"mode": "stageB_hybrid", "state_npz": str(state_npz), "reference_npz": str(ref_npz) if ref_npz else "", "stageB_eval_dir": str(args.stageB_eval_dir), "note": "Default mode reproduces D5 hybrid scorecard. StageB SOH may be full-cycle calibration depending on how StageB was trained."}, output_dir / "joint_evaluator_provenance.json")
    print(scorecard.to_string(index=False))
    return 0


def run_assb111_delegate(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    script = root / "scripts" / "evaluate_assb111_five_state.py"
    if not script.exists():
        raise FileNotFoundError(f"ASSB111 dedicated evaluator not found: {script}")
    cmd = [sys.executable, str(script),
           "--model111_dir", str(args.model111_dir),
           "--dataset_csv", str(args.dataset_csv),
           "--split_manifest_json", str(args.split_manifest_json),
           "--state_eval_dir", str(args.state_eval_dir),
           "--output_dir", str(args.output_dir),
           "--feature_mode", str(args.feature_mode),
           "--device", str(args.device)]
    if args.state_prediction_npz:
        cmd += ["--state_eval_npz", str(args.state_prediction_npz)]
    if args.scaler_json:
        cmd += ["--scaler_json", str(args.scaler_json)]
    if args.allow_cpu:
        cmd += ["--allow_cpu"]
    if args.soft_fail:
        cmd += ["--soft_fail"]
    print("[evaluate_assb_joint_states_soh] Delegating ASSB111 evaluation:")
    print(" ".join(cmd))
    return int(subprocess.run(cmd, check=False).returncode)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASSB joint five-state evaluator: StageB/107A hybrid or ASSB111 strict30")
    p.add_argument("--mode", choices=["stageB_hybrid", "hybrid_stageB", "stageB107A", "stageb", "hybrid", "assb111"], default="stageB_hybrid")

    # Legacy StageB/107A arguments.
    p.add_argument("--stageB_eval_dir", default=r".\EvalFin_110_stageB_aging")
    p.add_argument("--stageB_mechanism_csv", "--stageb_mechanism_csv", dest="stageB_mechanism_csv", default="")
    p.add_argument("--reference_npz", default=r"..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz")
    p.add_argument("--state_eval_dir", default=r".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only")
    p.add_argument("--state_prediction_npz", "--state_eval_npz", dest="state_prediction_npz", default="")
    p.add_argument("--cycle_table_csv", default=r".\Data\assb_aging_fix1\cycle_table.csv")
    p.add_argument("--output_dir", default=r".\EvalFin_110_joint_StageB_SOH_107A_states_fix2")
    p.add_argument("--max_time_points", type=int, default=20000)
    p.add_argument("--max_radial_points", type=int, default=64)
    p.add_argument("--align_tolerance_s", type=float, default=1.0e-6)

    # ASSB111 delegated arguments.
    p.add_argument("--model111_dir", "--model111", "--model_dir", dest="model111_dir", default=r".\ModelFin_111")
    p.add_argument("--dataset_csv", default=r".\Data\assb111\dataset_strict30.csv")
    p.add_argument("--split_manifest_json", default=r".\Data\assb111\split_manifest.json")
    p.add_argument("--scaler_json", default="")
    p.add_argument("--feature_mode", default="p1_107a_strict")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--soft_fail", action="store_true")
    p.add_argument("--allow_soh_only", action="store_true", help="accepted for compatibility with older fix2 commands")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "assb111":
        return run_assb111_delegate(args)
    return run_stageb_hybrid(args)


if __name__ == "__main__":
    raise SystemExit(main())
