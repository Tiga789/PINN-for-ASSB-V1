
# -*- coding: utf-8 -*-
r"""Corrected joint five-state evaluator for ASSB ModelFin110 Stage-B SOH + 107A states.

Fix2 purpose
------------
The previous joint evaluator could silently compare sampled concentration predictions
(cs_a/cs_c; e.g. 20,000 sampled times x 64 radial points) against the beginning of a
full-length reference solution.npz (e.g. 373,235 time points x 64 radial points).  That
produced near-zero correlations for cs_a/cs_c even when the 107A evaluation npz already
contained correctly paired fields such as cs_a_true/cs_a_pred and cs_c_true/cs_c_pred.

This version follows the old 107A evaluation-output convention:
  - Prefer paired arrays inside the state NPZ: <var>_true vs <var>_pred.
  - Use matching cycle arrays from the same NPZ: cycle_id_cs for cs_a/cs_c and
    cycle_id_potential for phie/phis_c.
  - Never silently truncate mismatched arrays.  Shapes must match, or an explicit
    time-index alignment must be possible.
  - Save key/shape/alignment provenance for every variable.

Typical command
---------------
python evaluate_assb_joint_states_soh.py ^
  --stageB_eval_dir .\EvalFin_110_stageB_aging ^
  --reference_npz ..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz ^
  --state_eval_dir .\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only ^
  --cycle_table_csv .\Data\assb_aging_fix1\cycle_table.csv ^
  --output_dir .\EvalFin_110_joint_StageB_SOH_107A_states
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

STATE_KEYS: Tuple[str, ...] = ("cs_a", "cs_c", "phie", "phis_c")
CONCENTRATION_KEYS = {"cs_a", "cs_c"}
POTENTIAL_KEYS = {"phie", "phis_c"}

# For paired NPZs, true/pred keys must be preferred over generic key names.
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
    if isinstance(x, dict):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
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


def _as_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _finite_pair(obs: np.ndarray, pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    obs = _as_float_array(obs).reshape(-1)
    pred = _as_float_array(pred).reshape(-1)
    if obs.shape != pred.shape:
        raise ValueError(f"Internal error: finite_pair received mismatched flat shapes {obs.shape} vs {pred.shape}")
    mask = np.isfinite(obs) & np.isfinite(pred)
    return obs[mask], pred[mask]


def _metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    y, p = _finite_pair(obs, pred)
    out: Dict[str, Any] = {
        "n": int(y.size), "MAE": float("nan"), "RMSE": float("nan"), "BIAS": float("nan"), "MAX": float("nan"),
        "corr": float("nan"), "R2": float("nan"), "obs_min": float("nan"), "obs_max": float("nan"),
        "obs_range": float("nan"), "pred_min": float("nan"), "pred_max": float("nan"),
        "NMAE": float("nan"), "NRMSE": float("nan"),
    }
    if y.size == 0:
        return out
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    out.update({
        "MAE": mae,
        "RMSE": rmse,
        "BIAS": float(np.mean(e)),
        "MAX": float(np.max(np.abs(e))),
        "obs_min": float(np.min(y)),
        "obs_max": float(np.max(y)),
        "pred_min": float(np.min(p)),
        "pred_max": float(np.max(p)),
    })
    obs_range = float(out["obs_max"] - out["obs_min"])
    out["obs_range"] = obs_range
    if obs_range > 1.0e-30:
        out["NMAE"] = float(mae / obs_range)
        out["NRMSE"] = float(rmse / obs_range)
    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot > 1.0e-30:
        out["R2"] = float(1.0 - ss_res / ss_tot)
    if y.size >= 2 and float(np.std(y)) > 1.0e-30 and float(np.std(p)) > 1.0e-30:
        out["corr"] = float(np.corrcoef(y, p)[0, 1])
    return out


def _metrics_or_unavailable(reason: str, **extra: Any) -> Dict[str, Any]:
    out = {
        "available": False,
        "reason": reason,
        "n": 0,
        "MAE": float("nan"), "RMSE": float("nan"), "BIAS": float("nan"), "MAX": float("nan"),
        "corr": float("nan"), "R2": float("nan"), "NMAE": float("nan"), "NRMSE": float("nan"),
    }
    out.update(extra)
    return out


def _select_rows(arr: np.ndarray, idx: Optional[np.ndarray]) -> np.ndarray:
    arr = np.asarray(arr)
    if idx is None:
        return arr
    if arr.ndim == 0:
        return arr
    return arr[idx]


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
    if tol >= 0 and np.nanmax(dist) > tol:
        raise ValueError(f"nearest-time alignment max distance {float(np.nanmax(dist)):.6g} > tolerance {tol}")
    return order[choose]


def _load_stageb_frame(stageb_eval_dir: Optional[Path], stageb_mechanism_csv: Optional[Path]) -> pd.DataFrame:
    candidates: List[Path] = []
    if stageb_mechanism_csv and stageb_mechanism_csv.exists():
        candidates.append(stageb_mechanism_csv)
    if stageb_eval_dir:
        candidates += [stageb_eval_dir / "mechanism_by_cycle.csv", stageb_eval_dir / "soh_stageB_by_cycle.csv"]
        if stageb_eval_dir.exists():
            candidates += list(stageb_eval_dir.rglob("mechanism_by_cycle.csv"))
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.exists():
            frame = pd.read_csv(p)
            if {"cycle_id", "SOH_obs", "SOH_pred"}.issubset(frame.columns):
                return frame
    raise FileNotFoundError("Cannot find Stage-B mechanism_by_cycle.csv with cycle_id/SOH_obs/SOH_pred")


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
    result: Dict[str, Any] = {}
    if len(x) == 0:
        result["all"] = _metrics_or_unavailable("no rows after complete_only filter")
        return result
    result["all"] = _metrics(x["SOH_obs"].to_numpy(float), x["SOH_pred"].to_numpy(float))
    if "split" in x.columns:
        for split, g in x.groupby("split"):
            if len(g):
                result[str(split)] = _metrics(g["SOH_obs"].to_numpy(float), g["SOH_pred"].to_numpy(float))
    return result


def _write_soh_outputs(frame: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "soh_stageB_by_cycle.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "available": True,
        "capacity_by_split": _soh_metrics_by_split(frame, complete_only=False),
        "capacity_by_split_complete_only": _soh_metrics_by_split(frame, complete_only=True),
    }
    save_json(metrics, output_dir / "soh_stageB_metrics.json")
    return metrics


def _pred_score_for_npz(path: Path) -> Dict[str, Any]:
    try:
        with np.load(path, allow_pickle=True) as z:
            files = list(z.files)
            score = 0
            paired = 0
            details = {}
            for v in STATE_KEYS:
                pred_key = _find_key(files, PRED_ALIASES[v])
                true_key = _find_key(files, TRUE_ALIASES[v])
                generic_key = _find_key(files, GENERIC_ALIASES[v])
                if pred_key is not None:
                    score += 3
                if true_key is not None:
                    score += 3
                if pred_key is not None and true_key is not None:
                    paired += 1
                    score += 5
                if generic_key is not None:
                    score += 1
                details[v] = {"pred_key": pred_key, "true_key": true_key, "generic_key": generic_key}
            if path.name in PRED_NPZ_PREFERRED_NAMES:
                score += 3
            return {"path": str(path), "score": score, "n_paired_variables": paired, "keys_sample": files[:50], "details": details}
    except Exception as exc:
        return {"path": str(path), "score": -1, "error": str(exc)}


def _discover_state_npz(state_eval_dir: Optional[Path], output_dir: Path) -> Optional[Path]:
    if not state_eval_dir or not state_eval_dir.exists():
        save_json({"state_eval_dir": str(state_eval_dir) if state_eval_dir else "", "candidates": [], "selected": None}, output_dir / "state_npz_discovery.json")
        return None
    candidates: List[Path] = []
    for name in PRED_NPZ_PREFERRED_NAMES:
        p = state_eval_dir / name
        if p.exists():
            candidates.append(p)
    candidates += [p for p in sorted(state_eval_dir.rglob("*.npz")) if p not in candidates]
    scored = [_pred_score_for_npz(p) for p in candidates]
    valid = [d for d in scored if d.get("score", -1) >= 8]
    selected = max(valid, key=lambda d: (int(d.get("n_paired_variables", 0)), float(d.get("score", -1)))) if valid else None
    save_json({"state_eval_dir": str(state_eval_dir), "candidates": scored, "selected": selected}, output_dir / "state_npz_discovery.json")
    return Path(selected["path"]) if selected else None


def _cycle_array_from_npz(z: np.lib.npyio.NpzFile, var: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    return _load_array(z, CYCLE_ALIASES_BY_VAR[var])


def _time_array_from_npz(z: np.lib.npyio.NpzFile, var: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    return _load_array(z, TIME_ALIASES_BY_VAR[var])


def _resolve_state_pair(
    var: str,
    *,
    pred_z: np.lib.npyio.NpzFile,
    ref_z: Optional[np.lib.npyio.NpzFile],
    align_tolerance_s: float,
) -> Dict[str, Any]:
    files = list(pred_z.files)
    pred_key, pred_arr = _load_array(pred_z, PRED_ALIASES[var])
    true_key, true_arr = _load_array(pred_z, TRUE_ALIASES[var])
    cycle_key, cycle_arr = _cycle_array_from_npz(pred_z, var)
    time_key, time_arr = _time_array_from_npz(pred_z, var)

    # P0 fix: paired arrays in the 107A evaluation NPZ are authoritative.
    if pred_arr is not None and true_arr is not None:
        if true_arr.shape != pred_arr.shape:
            return {"available": False, "reason": f"paired internal shape mismatch: {true_key}{true_arr.shape} vs {pred_key}{pred_arr.shape}", "pred_key": pred_key, "ref_key": true_key, "alignment_mode": "paired_npz_internal_shape_mismatch"}
        return {
            "available": True,
            "obs": true_arr,
            "pred": pred_arr,
            "pred_key": pred_key,
            "ref_key": true_key,
            "cycle_key": cycle_key,
            "cycle_id": cycle_arr,
            "time_key": time_key,
            "time": time_arr,
            "alignment_mode": "paired_npz_internal",
            "source": "state_prediction_npz_internal_true_pred",
        }

    # Fallback only when paired internal true/pred are unavailable.
    if pred_arr is None:
        generic_pred_key, generic_pred_arr = _load_array(pred_z, GENERIC_ALIASES[var])
        if generic_pred_arr is not None:
            pred_key, pred_arr = generic_pred_key, generic_pred_arr
    if pred_arr is None:
        return {"available": False, "reason": f"missing prediction array for {var}; tried {PRED_ALIASES[var]} and {GENERIC_ALIASES[var]}", "alignment_mode": "unavailable"}

    if ref_z is None:
        return {"available": False, "reason": f"{var} has prediction array but no internal *_true array and no reference_npz", "pred_key": pred_key, "alignment_mode": "missing_reference"}

    ref_key, ref_arr = _load_array(ref_z, TRUE_ALIASES[var] + GENERIC_ALIASES[var])
    if ref_arr is None:
        return {"available": False, "reason": f"reference_npz missing array for {var}", "pred_key": pred_key, "alignment_mode": "reference_missing"}

    if ref_arr.shape == pred_arr.shape:
        ref_cycle_key, ref_cycle = _cycle_array_from_npz(ref_z, var)
        return {
            "available": True,
            "obs": ref_arr,
            "pred": pred_arr,
            "pred_key": pred_key,
            "ref_key": ref_key,
            "cycle_key": cycle_key or ref_cycle_key,
            "cycle_id": cycle_arr if cycle_arr is not None else ref_cycle,
            "time_key": time_key,
            "time": time_arr,
            "alignment_mode": "reference_npz_exact_shape",
            "source": "state_prediction_npz_vs_reference_npz_exact_shape",
        }

    # Explicit time-index alignment for sampled prediction arrays vs full reference.
    ref_time_key, ref_time = _time_array_from_npz(ref_z, var)
    if time_arr is not None and ref_time is not None:
        try:
            idx = _nearest_indices(np.asarray(ref_time).reshape(-1), np.asarray(time_arr).reshape(-1), align_tolerance_s)
            if ref_arr.ndim >= 1 and ref_arr.shape[0] >= np.max(idx) + 1:
                ref_aligned = ref_arr[idx]
                if ref_aligned.shape == pred_arr.shape:
                    return {
                        "available": True,
                        "obs": ref_aligned,
                        "pred": pred_arr,
                        "pred_key": pred_key,
                        "ref_key": ref_key,
                        "cycle_key": cycle_key,
                        "cycle_id": cycle_arr,
                        "time_key": time_key,
                        "time": time_arr,
                        "alignment_mode": "reference_npz_nearest_time",
                        "time_alignment_reference_key": ref_time_key,
                        "source": "state_prediction_npz_vs_reference_npz_time_aligned",
                    }
        except Exception as exc:
            return {
                "available": False,
                "reason": f"shape mismatch and time alignment failed for {var}: {exc}; pred {pred_key}{pred_arr.shape}, ref {ref_key}{ref_arr.shape}",
                "pred_key": pred_key,
                "ref_key": ref_key,
                "pred_shape": tuple(pred_arr.shape),
                "ref_shape": tuple(ref_arr.shape),
                "alignment_mode": "reference_npz_time_alignment_failed",
            }

    return {
        "available": False,
        "reason": f"shape mismatch with no valid alignment for {var}: pred {pred_key}{pred_arr.shape}, ref {ref_key}{ref_arr.shape}. Refusing silent flatten/truncate.",
        "pred_key": pred_key,
        "ref_key": ref_key,
        "pred_shape": tuple(pred_arr.shape),
        "ref_shape": tuple(ref_arr.shape),
        "alignment_mode": "shape_mismatch_no_truncation",
    }


def _cycle_metric_rows(var: str, obs: np.ndarray, pred: np.ndarray, cycle_id: Optional[np.ndarray], split_map: Optional[Mapping[int, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if cycle_id is None:
        return rows
    cyc = np.asarray(cycle_id).reshape(-1)
    if obs.ndim == 0 or obs.shape[0] != cyc.size:
        return rows
    for c in np.unique(cyc.astype(int)):
        mask = cyc.astype(int) == int(c)
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


def evaluate_states(
    *,
    state_prediction_npz: Path,
    reference_npz: Optional[Path],
    cycle_table_csv: Optional[Path],
    output_dir: Path,
    max_time_points: int,
    max_radial_points: int,
    align_tolerance_s: float,
) -> Dict[str, Any]:
    split_map = _load_split_map(cycle_table_csv)
    states: Dict[str, Any] = {}
    global_rows: List[Dict[str, Any]] = []
    by_cycle_rows: List[Dict[str, Any]] = []
    provenance: Dict[str, Any] = {}

    with np.load(state_prediction_npz, allow_pickle=True) as pred_z:
        ref_z_ctx = np.load(reference_npz, allow_pickle=True) if reference_npz else None
        try:
            for var in STATE_KEYS:
                resolved = _resolve_state_pair(var, pred_z=pred_z, ref_z=ref_z_ctx, align_tolerance_s=align_tolerance_s)
                if not resolved.get("available", False):
                    block = _metrics_or_unavailable(str(resolved.get("reason", "unavailable")), **{k: v for k, v in resolved.items() if k not in {"obs", "pred"}})
                    states[var] = block
                    global_rows.append({"variable": var, **block})
                    provenance[var] = block
                    continue
                obs = np.asarray(resolved["obs"])
                pred = np.asarray(resolved["pred"])
                obs_shape_before = tuple(obs.shape)
                pred_shape_before = tuple(pred.shape)
                try:
                    obs_s, pred_s, sample_info = _maybe_sample_pair(obs, pred, max_time_points=max_time_points, max_radial_points=max_radial_points)
                    m = _metrics(obs_s, pred_s)
                    m.update({
                        "available": True,
                        "variable": var,
                        "pred_key": resolved.get("pred_key"),
                        "ref_key": resolved.get("ref_key"),
                        "cycle_key": resolved.get("cycle_key"),
                        "time_key": resolved.get("time_key"),
                        "source": resolved.get("source"),
                        "alignment_mode": resolved.get("alignment_mode"),
                        "obs_shape": obs_shape_before,
                        "pred_shape": pred_shape_before,
                        **sample_info,
                    })
                    states[var] = m
                    global_rows.append({"variable": var, **m})
                    provenance[var] = {k: m.get(k) for k in ["pred_key", "ref_key", "cycle_key", "time_key", "source", "alignment_mode", "obs_shape", "pred_shape", "sampled_time", "sampled_radial"]}
                    by_cycle_rows.extend(_cycle_metric_rows(var, obs, pred, resolved.get("cycle_id"), split_map))
                except Exception as exc:
                    block = _metrics_or_unavailable(str(exc), pred_key=resolved.get("pred_key"), ref_key=resolved.get("ref_key"), alignment_mode=resolved.get("alignment_mode"), obs_shape=obs_shape_before, pred_shape=pred_shape_before)
                    states[var] = block
                    global_rows.append({"variable": var, **block})
                    provenance[var] = block
        finally:
            if ref_z_ctx is not None:
                ref_z_ctx.close()

    # Save global CSV with stable columns.
    global_df = pd.DataFrame(global_rows)
    ordered = ["variable", "available", "n", "MAE", "RMSE", "BIAS", "MAX", "corr", "R2", "obs_min", "obs_max", "obs_range", "pred_min", "pred_max", "NMAE", "NRMSE", "pred_key", "ref_key", "source", "alignment_mode", "obs_shape", "pred_shape", "cycle_key", "time_key", "reason"]
    for col in ordered:
        if col not in global_df.columns:
            global_df[col] = np.nan
    global_df[ordered].to_csv(output_dir / "state_metrics_global.csv", index=False, encoding="utf-8-sig")

    if by_cycle_rows:
        by_cycle_df = pd.DataFrame(by_cycle_rows)
        by_cycle_df.to_csv(output_dir / "state_metrics_by_cycle.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["cycle_id", "variable", "n", "MAE", "RMSE", "NMAE", "NRMSE", "R2", "corr"]).to_csv(output_dir / "state_metrics_by_cycle.csv", index=False, encoding="utf-8-sig")

    result = {
        "available": True,
        "state_prediction_npz": str(state_prediction_npz),
        "reference_npz": str(reference_npz) if reference_npz else "",
        "states": states,
        "provenance": provenance,
        "note": "For 107A sampled concentration arrays, paired internal *_true/*_pred keys are preferred and no silent flatten/truncate is allowed.",
    }
    save_json(result, output_dir / "state_metrics_global.json")
    save_json(provenance, output_dir / "state_array_alignment_provenance.json")
    return result


def _append_soh_by_cycle_to_long(output_dir: Path, soh_frame: pd.DataFrame) -> None:
    state_csv = output_dir / "state_metrics_by_cycle.csv"
    if not state_csv.exists():
        return
    state_df = pd.read_csv(state_csv)
    rows = []
    for _, r in soh_frame.iterrows():
        y = np.asarray([float(r["SOH_obs"])])
        p = np.asarray([float(r["SOH_pred"])])
        m = _metrics(y, p)
        m.update({
            "cycle_id": int(r["cycle_id"]),
            "variable": "SOH",
            "split": str(r.get("split", "")),
            "complete_cycle": bool(r.get("complete_cycle", r.get("complete_cycle_from_solution", True))),
        })
        rows.append(m)
    combined = pd.concat([state_df, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(output_dir / "joint_metrics_by_cycle_long.csv", index=False, encoding="utf-8-sig")


def _make_scorecard(state_metrics: Dict[str, Any], soh_metrics: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    states = state_metrics.get("states", {}) if isinstance(state_metrics, dict) else {}
    for var in STATE_KEYS:
        m = states.get(var, {}) if isinstance(states, dict) else {}
        rows.append({
            "variable": var,
            "source": m.get("source", "state_prediction_npz"),
            "alignment_mode": m.get("alignment_mode", ""),
            "n": m.get("n", 0),
            "MAE": m.get("MAE", np.nan),
            "RMSE": m.get("RMSE", np.nan),
            "NMAE": m.get("NMAE", np.nan),
            "NRMSE": m.get("NRMSE", np.nan),
            "R2": m.get("R2", np.nan),
            "corr": m.get("corr", np.nan),
            "pred_key": m.get("pred_key", ""),
            "ref_key": m.get("ref_key", ""),
            "reason": m.get("reason", ""),
        })
    soh_all = soh_metrics.get("capacity_by_split_complete_only", {}).get("all", {}) if isinstance(soh_metrics, dict) else {}
    rows.append({
        "variable": "SOH",
        "source": "StageB_complete_only",
        "alignment_mode": "cycle_id_complete_only",
        "n": soh_all.get("n", 0),
        "MAE": soh_all.get("MAE", np.nan),
        "RMSE": soh_all.get("RMSE", np.nan),
        "NMAE": soh_all.get("NMAE", np.nan),
        "NRMSE": soh_all.get("NRMSE", np.nan),
        "R2": soh_all.get("R2", np.nan),
        "corr": soh_all.get("corr", np.nan),
        "pred_key": "SOH_pred",
        "ref_key": "SOH_obs",
        "reason": "",
    })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "five_state_scorecard.csv", index=False, encoding="utf-8-sig")
    save_json({"rows": df.to_dict(orient="records")}, output_dir / "five_state_scorecard.json")
    return df


def _plot_outputs(scorecard: pd.DataFrame, soh_frame: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    try:
        vals = pd.to_numeric(scorecard["NMAE"], errors="coerce").to_numpy()
        x = np.arange(len(scorecard))
        plt.figure(figsize=(8, 4.5))
        plt.bar(x, vals)
        plt.xticks(x, scorecard["variable"].astype(str).tolist())
        plt.ylabel("NMAE")
        plt.title("Five-state normalized MAE")
        plt.tight_layout()
        plt.savefig(output_dir / "five_state_nmae_scorecard.png", dpi=180)
        plt.close()
    except Exception:
        pass
    try:
        plt.figure(figsize=(8, 4.5))
        plt.plot(soh_frame["cycle_id"], soh_frame["SOH_obs"], label="SOH obs")
        plt.plot(soh_frame["cycle_id"], soh_frame["SOH_pred"], label="SOH StageB pred")
        if "complete_cycle" in soh_frame.columns:
            incomplete = soh_frame[~_complete_mask(soh_frame)]
            if len(incomplete):
                plt.scatter(incomplete["cycle_id"], incomplete["SOH_obs"], label="incomplete obs")
        plt.xlabel("cycle")
        plt.ylabel("SOH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "soh_stageB_pred_vs_obs.png", dpi=180)
        plt.close()
    except Exception:
        pass
    try:
        if {"f_LAM_c", "theta_window_scale_c", "R_ohm_eff"}.issubset(soh_frame.columns):
            fig, ax1 = plt.subplots(figsize=(8, 4.5))
            ax1.plot(soh_frame["cycle_id"], soh_frame["f_LAM_c"], label="f_LAM_c")
            ax1.plot(soh_frame["cycle_id"], soh_frame["theta_window_scale_c"], label="theta window scale")
            ax1.set_xlabel("cycle")
            ax1.set_ylabel("fraction / scale")
            ax2 = ax1.twinx()
            ax2.plot(soh_frame["cycle_id"], soh_frame["R_ohm_eff"], linestyle="--", label="R_ohm_eff")
            ax2.set_ylabel("R_ohm_eff")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / "stageB_mechanism_profiles.png", dpi=180)
            plt.close(fig)
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Corrected joint cs_a/cs_c/phie/phis_c + StageB SOH evaluator")
    parser.add_argument("--stageB_eval_dir", default="EvalFin_110_stageB_aging")
    parser.add_argument("--stageB_mechanism_csv", default="")
    parser.add_argument("--reference_npz", default="", help="Optional full reference solution.npz; paired *_true arrays in state npz are preferred.")
    parser.add_argument("--state_prediction_npz", default="", help="NPZ containing paired four-state arrays, preferably eval_sampled_arrays_ModelFin107A_csA_corrected.npz")
    parser.add_argument("--state_eval_dir", default="", help="Directory to auto-discover the 107A state prediction NPZ")
    parser.add_argument("--cycle_table_csv", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_time_points", type=int, default=0, help="Optional downsampling for global metrics; 0 means use all rows.")
    parser.add_argument("--max_radial_points", type=int, default=0, help="Optional radial downsampling for concentration metrics; 0 means use all radial points.")
    parser.add_argument("--align_tolerance_s", type=float, default=1.0, help="Tolerance for fallback nearest-time alignment if paired true/pred keys are absent.")
    parser.add_argument("--allow_soh_only", action="store_true", help="Write SOH-only outputs if state NPZ is not found.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stageb_dir = Path(args.stageB_eval_dir) if args.stageB_eval_dir else None
    stageb_csv = Path(args.stageB_mechanism_csv) if args.stageB_mechanism_csv else None
    soh_frame = _load_stageb_frame(stageb_dir, stageb_csv)
    soh_metrics = _write_soh_outputs(soh_frame, output_dir)

    state_pred = Path(args.state_prediction_npz) if args.state_prediction_npz else None
    if state_pred is None or not state_pred.exists():
        state_pred = _discover_state_npz(Path(args.state_eval_dir) if args.state_eval_dir else None, output_dir)
    else:
        save_json({"state_prediction_npz": str(state_pred), "manual": True, "score": _pred_score_for_npz(state_pred)}, output_dir / "state_npz_discovery.json")

    if state_pred is None or not state_pred.exists():
        state_metrics = {"available": False, "reason": "No valid state prediction NPZ found. Provide --state_prediction_npz or --state_eval_dir.", "states": {}}
        save_json(state_metrics, output_dir / "state_metrics_global.json")
        scorecard = _make_scorecard(state_metrics, soh_metrics, output_dir)
        _plot_outputs(scorecard, soh_frame, output_dir)
        save_json({"state_available": False, "soh_available": True, "state_prediction_npz": None}, output_dir / "joint_evaluation_summary.json")
        if args.allow_soh_only:
            print("[joint eval fix2] SOH-only outputs written; state NPZ unavailable.")
            return 0
        raise SystemExit("No valid state prediction NPZ found. See state_npz_discovery.json.")

    reference_npz = Path(args.reference_npz) if args.reference_npz else None
    if reference_npz is not None and not reference_npz.exists():
        raise FileNotFoundError(f"reference_npz not found: {reference_npz}")
    cycle_table_csv = Path(args.cycle_table_csv) if args.cycle_table_csv else None
    if cycle_table_csv is not None and not cycle_table_csv.exists():
        cycle_table_csv = None

    state_metrics = evaluate_states(
        state_prediction_npz=state_pred,
        reference_npz=reference_npz,
        cycle_table_csv=cycle_table_csv,
        output_dir=output_dir,
        max_time_points=args.max_time_points,
        max_radial_points=args.max_radial_points,
        align_tolerance_s=args.align_tolerance_s,
    )
    _append_soh_by_cycle_to_long(output_dir, soh_frame)
    scorecard = _make_scorecard(state_metrics, soh_metrics, output_dir)
    _plot_outputs(scorecard, soh_frame, output_dir)

    summary = {
        "state_available": bool(state_metrics.get("available", False)),
        "soh_available": True,
        "stageB_eval_dir": str(stageb_dir) if stageb_dir else "",
        "state_prediction_npz": str(state_pred),
        "reference_npz": str(reference_npz) if reference_npz else "",
        "cycle_table_csv": str(cycle_table_csv) if cycle_table_csv else "",
        "complete_only_soh_all": soh_metrics.get("capacity_by_split_complete_only", {}).get("all", {}),
        "scorecard_csv": str(output_dir / "five_state_scorecard.csv"),
        "state_metrics_global_csv": str(output_dir / "state_metrics_global.csv"),
        "state_metrics_by_cycle_csv": str(output_dir / "state_metrics_by_cycle.csv"),
        "fix2_key_change": "cs_a/cs_c/phie/phis_c now prefer paired *_true vs *_pred arrays inside the 107A state NPZ; mismatched arrays are not silently truncated.",
    }
    save_json(summary, output_dir / "joint_evaluation_summary.json")

    print("[joint eval fix2] wrote", output_dir)
    cols = ["variable", "source", "alignment_mode", "n", "MAE", "RMSE", "NMAE", "NRMSE", "R2", "corr", "pred_key", "ref_key"]
    for col in cols:
        if col not in scorecard.columns:
            scorecard[col] = ""
    print(scorecard[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
