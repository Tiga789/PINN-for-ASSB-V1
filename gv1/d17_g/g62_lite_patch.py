from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .g1_data import (
    OBS_TIME_KEYS, OBS_I_KEYS, OBS_V_KEYS,
    _load_npz_dict, _semantics_for,
    load_split_records, load_semantics_map, load_profile_pack,
)
from .g3_frozen_audit import (
    torch_load_safe, build_model_from_checkpoint, parse_vocab_from_feature_names,
    resolve_checkpoint_path, augment_profile_features,
)
from .g6_full_cycle_audit import predict_denorm
from .g13_trainer import _device_from_arg


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def json_dump(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _r2_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if yt.size == 0:
        return {"n_points": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "nmae": float("nan"), "nrmse": float("nan"), "bias": float("nan"), "target_range": float("nan"), "target_std": float("nan"), "pred_min": float("nan"), "pred_max": float("nan")}
    err = yp - yt
    sse = float(np.sum(err * err))
    mean = float(np.mean(yt))
    sst = float(np.sum((yt - mean) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 1e-30 else (1.0 if sse < 1e-30 else float("nan"))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    tr = float(np.max(yt) - np.min(yt)) if yt.size else float("nan")
    return {
        "n_points": int(yt.size),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nmae": float(mae / tr) if tr > 1e-30 else float("nan"),
        "nrmse": float(rmse / tr) if tr > 1e-30 else float("nan"),
        "bias": float(np.mean(err)),
        "target_range": tr,
        "target_std": float(np.std(yt)),
        "target_min": float(np.min(yt)),
        "target_max": float(np.max(yt)),
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
    }


def _first_numeric_1d(d: Mapping[str, Any], keys: Sequence[str]) -> Tuple[str, Optional[np.ndarray]]:
    for k in keys:
        if k not in d:
            continue
        try:
            arr = np.asarray(d[k])
            if arr.dtype.kind in {"U", "S", "O"}:
                continue
            arr = arr.astype(np.float64).reshape(-1)
            if arr.size:
                return k, arr
        except Exception:
            continue
    return "", None


def _align_to_target(src_y: Optional[np.ndarray], src_t: Optional[np.ndarray], target_t: np.ndarray, fill: float = 0.0) -> np.ndarray:
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    if src_y is None or np.asarray(src_y).size == 0:
        return np.full(target_t.size, float(fill), dtype=np.float64)
    src_y = np.asarray(src_y, dtype=np.float64).reshape(-1)
    if src_y.size == target_t.size:
        return src_y.astype(np.float64)
    if src_t is None or np.asarray(src_t).reshape(-1).size != src_y.size:
        x0 = np.linspace(0.0, 1.0, src_y.size)
        x1 = np.linspace(0.0, 1.0, target_t.size)
        return np.interp(x1, x0, src_y).astype(np.float64)
    src_t = np.asarray(src_t, dtype=np.float64).reshape(-1)
    good = np.isfinite(src_t) & np.isfinite(src_y)
    src_t = src_t[good]
    src_y = src_y[good]
    if src_t.size == 0:
        return np.full(target_t.size, float(fill), dtype=np.float64)
    order = np.argsort(src_t)
    src_t = src_t[order]
    src_y = src_y[order]
    ux, idx = np.unique(src_t, return_index=True)
    uy = src_y[idx]
    if ux.size == 1:
        return np.full(target_t.size, float(uy[0]), dtype=np.float64)
    return np.interp(target_t, ux, uy, left=float(uy[0]), right=float(uy[-1])).astype(np.float64)


def load_observed_current_voltage(profile: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    target_t = np.asarray(profile.t_global_s, dtype=np.float64).reshape(-1)
    replay = _load_npz_dict(profile.replay_npz, None) if getattr(profile, "replay_npz", "") and Path(profile.replay_npz).exists() else {}
    soft_obs = _load_npz_dict(profile.softlabel_npz, list(set(OBS_TIME_KEYS + OBS_I_KEYS + OBS_V_KEYS))) if getattr(profile, "softlabel_npz", "") and Path(profile.softlabel_npz).exists() else {}
    src = replay if replay else soft_obs
    tk, src_t = _first_numeric_1d(src, OBS_TIME_KEYS)
    ik, src_i = _first_numeric_1d(src, OBS_I_KEYS)
    vk, src_v = _first_numeric_1d(src, OBS_V_KEYS)
    I = _align_to_target(src_i, src_t, target_t, fill=0.0)
    V = _align_to_target(src_v, src_t, target_t, fill=0.0)
    return I, V, {"observed_source": "replay" if replay else "softlabel_obs_fallback", "time_key": tk, "I_key": ik, "V_key": vk}


def current_integral_theta(t: np.ndarray, I: np.ndarray, theta0: float, window: float, capacity_Ah: float, sign: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    I = np.asarray(I, dtype=np.float64).reshape(-1)
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)
    q_Ah = np.cumsum(I * dt) / 3600.0
    return np.clip(float(theta0) + float(sign) * q_Ah / max(float(capacity_Ah), 1e-12) * float(window), 0.0, 1.0)


def p4d_formula_theta(profile: Any, cfg: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    gen = dict(cfg.get("p4d_generation", {}) if isinstance(cfg.get("p4d_generation"), Mapping) else {})
    capacity_Ah = float(gen.get("capacity_scale_Ah", 2.0))
    theta_pos0 = float(gen.get("theta_positive_initial", 0.90))
    theta_neg0 = float(gen.get("theta_negative_initial", 0.08))
    pos_min = float(gen.get("theta_positive_min", 0.2535))
    pos_max = float(gen.get("theta_positive_max", 0.9149))
    neg_min = float(gen.get("theta_negative_min", 0.0079))
    neg_max = float(gen.get("theta_negative_max", 0.8544))
    csmax_c = float(gen.get("csmax_positive_mol_m3", 5.05e4))
    csmax_a = float(gen.get("csmax_negative_mol_m3", 3.15e4))
    phie_scale = float(gen.get("phie_ohmic_scale_V_per_A", -0.015))
    sign_c = float(gen.get("theta_c_current_sign", -1.0))
    sign_a = float(gen.get("theta_a_current_sign", +1.0))
    t = np.asarray(profile.t_global_s, dtype=np.float64).reshape(-1)
    I, V, obs_info = load_observed_current_voltage(profile)
    th_c = current_integral_theta(t, I, theta_pos0, pos_max - pos_min, capacity_Ah, sign=sign_c)
    th_a = current_integral_theta(t, I, theta_neg0, neg_max - neg_min, capacity_Ah, sign=sign_a)
    arr = {
        "theta_a_mean": th_a.astype(np.float32),
        "theta_c_mean": th_c.astype(np.float32),
        "cs_a_mean": (th_a * csmax_a).astype(np.float32),
        "cs_c_mean": (th_c * csmax_c).astype(np.float32),
        "phie": (phie_scale * I).astype(np.float32),
        "phis_c": V.astype(np.float32),
    }
    info = {"mode": "fast_current_integral_mean_only_no_radial_solver", "capacity_scale_Ah": capacity_Ah, "theta_positive_initial": theta_pos0, "theta_negative_initial": theta_neg0, "theta_positive_window": pos_max - pos_min, "theta_negative_window": neg_max - neg_min, "csmax_positive_mol_m3": csmax_c, "csmax_negative_mol_m3": csmax_a, "phie_ohmic_scale_V_per_A": phie_scale, **obs_info}
    return arr, info


def _select_records(records: Sequence[Mapping[str, Any]], terms: Sequence[str], splits: Sequence[str], include_flagged: bool, limit: int = 0) -> List[Dict[str, Any]]:
    split_set = {str(x).strip() for x in splits if str(x).strip()}
    use_all = "all" in split_set
    terms = [str(x).strip() for x in terms if str(x).strip()]
    out: List[Dict[str, Any]] = []
    for r in records:
        sp = str(r.get("split") or "")
        is_flagged = sp == "flagged_probe" or bool(r.get("is_flagged_probe"))
        if is_flagged and not include_flagged:
            continue
        if not (use_all or sp in split_set):
            continue
        hay = " ".join([str(r.get(k, "")) for k in ["canonical_cell_uid", "cell_uid", "protocol", "battery", "softlabel_npz", "replay_npz"]])
        if terms and not any(term in hay for term in terms):
            continue
        out.append(dict(r))
    if limit and int(limit) > 0:
        out = out[: int(limit)]
    return out


def _resolve_checkpoint(candidate_dir: str | Path, candidate_summary: Mapping[str, Any], checkpoint_path: str | Path = "") -> Path:
    if checkpoint_path and Path(checkpoint_path).exists():
        return Path(checkpoint_path)
    files = candidate_summary.get("files") if isinstance(candidate_summary.get("files"), Mapping) else {}
    for key in ["best_model_pt", "best_checkpoint", "model_pt", "last_model_pt"]:
        p = files.get(key) if files else None
        if p and Path(p).exists():
            return Path(p)
    for rel in ["model/best_model.pt", "model/best_model_and_latents.pt", "model/last_model.pt", "best_model.pt"]:
        p = Path(candidate_dir) / rel
        if p.exists():
            return p
    return resolve_checkpoint_path(str(checkpoint_path or ""), candidate_summary, candidate_dir)


def _patch_pred_array(pred: np.ndarray, profile: Any, target_slices: Mapping[str, Tuple[int, int]], cfg: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    patched = np.asarray(pred, dtype=np.float32).copy()
    formula, info = p4d_formula_theta(profile, cfg)
    gen = dict(cfg.get("p4d_generation", {}) if isinstance(cfg.get("p4d_generation"), Mapping) else {})
    csmax_c = float(gen.get("csmax_positive_mol_m3", 5.05e4))
    csmax_a = float(gen.get("csmax_negative_mol_m3", 3.15e4))
    # Shift NN-predicted radial cs field to deterministic P4D mean; keep learned radial shape.
    for cs_key, th_key, mean_key, csmax in [("cs_a", "theta_a", "cs_a_mean", csmax_a), ("cs_c", "theta_c", "cs_c_mean", csmax_c)]:
        if cs_key in target_slices:
            a, b = target_slices[cs_key]
            field = patched[:, a:b]
            desired = formula[mean_key].reshape(-1, 1).astype(np.float32)
            current_mean = np.mean(field, axis=1, keepdims=True)
            shifted = field + (desired - current_mean)
            shifted = np.clip(shifted, 0.0, float(csmax)).astype(np.float32)
            patched[:, a:b] = shifted
            if th_key in target_slices:
                ta, tb = target_slices[th_key]
                # Interpolate if target radial resolution differs.
                th = shifted / float(csmax)
                if th.shape[1] != tb - ta:
                    x_old = np.linspace(0.0, 1.0, th.shape[1])
                    x_new = np.linspace(0.0, 1.0, tb - ta)
                    th = np.stack([np.interp(x_new, x_old, row) for row in th], axis=0).astype(np.float32)
                patched[:, ta:tb] = th.astype(np.float32)
    if "phie" in target_slices and bool(cfg.get("override_phie", True)):
        a, b = target_slices["phie"]
        patched[:, a:b] = formula["phie"].reshape(-1, 1).astype(np.float32)
    if "phis_c" in target_slices and bool(cfg.get("override_phis_c", True)):
        a, b = target_slices["phis_c"]
        patched[:, a:b] = formula["phis_c"].reshape(-1, 1).astype(np.float32)
    info["patch_mode"] = "mean_shift_cs_theta_plus_optional_phie_phis_override_no_training_no_radial_solver"
    return patched, info


def run_formula_only(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    out_dir: str | Path,
    config: Mapping[str, Any],
    profile_contains: Sequence[str],
    splits: Sequence[str],
    include_flagged_probe: bool,
    profile_limit: int,
    max_time_points: int,
    time_window_s: float,
) -> Dict[str, Any]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    records, manifest = load_split_records(split_manifest)
    sem_map = load_semantics_map(g0_profile_semantics_csv)
    selected = _select_records(records, profile_contains, splits, include_flagged_probe, profile_limit)
    if not selected:
        raise ValueError("No profiles selected; check --profile_contains and --splits")
    protocols = sorted({str(r.get("protocol") or "UNKNOWN") for r in selected})
    branches = sorted({str(_semantics_for(r, sem_map).get("semantic_branch") or "UNKNOWN_OR_MIXED_BRANCH") for r in selected})
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, rec in enumerate(selected):
        canonical = str(rec.get("canonical_cell_uid") or rec.get("cell_uid") or i)
        try:
            prof = load_profile_pack(rec, _semantics_for(rec, sem_map), protocols, branches, max_time_points, time_window_s)
            formula, info = p4d_formula_theta(prof, config)
            infos.append({"profile_index": i, "canonical_cell_uid": prof.canonical_cell_uid, "protocol": prof.protocol, "semantic_branch": prof.branch, **info})
            for key, formula_key in [("theta_a", "theta_a_mean"), ("theta_c", "theta_c_mean"), ("cs_a", "cs_a_mean"), ("cs_c", "cs_c_mean"), ("phie", "phie"), ("phis_c", "phis_c")]:
                if key not in prof.target_slices:
                    continue
                a, b = prof.target_slices[key]
                yt = prof.targets[:, a:b]
                if key in {"theta_a", "theta_c", "cs_a", "cs_c"}:
                    yp = formula[formula_key].reshape(-1, 1)
                    yt_eval = np.mean(yt, axis=1, keepdims=True)
                    metric_note = "formula mean vs softlabel radial mean"
                else:
                    yp = formula[formula_key].reshape(-1, 1)
                    yt_eval = yt.reshape(-1, 1)
                    metric_note = "formula scalar vs softlabel scalar"
                rows.append({
                    "profile_index": i,
                    "split": prof.split,
                    "canonical_cell_uid": prof.canonical_cell_uid,
                    "protocol": prof.protocol,
                    "semantic_branch": prof.branch,
                    "target": key,
                    "metric_note": metric_note,
                    **_r2_mae_rmse(yt_eval, yp),
                })
        except Exception as e:
            failures.append({"profile_index": i, "canonical_cell_uid": canonical, "error": repr(e)})
    write_csv(rows, out / "D17_G62L_FORMULA_ONLY_TARGET_METRICS.csv")
    write_csv(infos, out / "D17_G62L_FORMULA_INFO.csv")
    write_csv(failures, out / "D17_G62L_LOAD_FAILURES.csv")
    r2s = [_safe_float(r.get("r2")) for r in rows]
    r2s = [x for x in r2s if math.isfinite(x)]
    mean_r2 = float(np.mean(r2s)) if r2s else float("nan")
    min_r2 = float(np.min(r2s)) if r2s else float("nan")
    theta_rows = [r for r in rows if str(r.get("target")) in {"theta_a", "theta_c", "cs_a", "cs_c"}]
    theta_r2s = [_safe_float(r.get("r2")) for r in theta_rows]
    theta_r2s = [x for x in theta_r2s if math.isfinite(x)]
    inv_mean = float(np.mean(theta_r2s)) if theta_r2s else float("nan")
    inv_min = float(np.min(theta_r2s)) if theta_r2s else float("nan")
    inv_gate = float(config.get("formula_inventory_min_r2_threshold", 0.90))
    blockers: List[str] = []
    if failures:
        blockers.append(f"{len(failures)} profiles failed to load")
    if inv_min < inv_gate:
        blockers.append(f"formula inventory min R2 below gate {inv_gate}: {inv_min:.6g}")
    summary = {
        "protocol": "D17-G6.2L_FAST_P4D_GEO_FORMULA_ONLY_INVENTORY_CHECK",
        "created_at_utc": utc_now(),
        "status": "PASS" if not failures else "REVIEW",
        "promotion_status": "PASS" if not blockers else "REVIEW",
        "g62_patch_formula_ready": not blockers,
        "recommendation": "RUN_G62L_MODEL_PATCH_SMOKE" if not blockers else "STOP_REVIEW_P4D_FORMULA_SEMANTICS",
        "blockers": blockers,
        "training_performed": False,
        "radial_solver_used": False,
        "model_loaded": False,
        "selected_profile_count": len(selected),
        "evaluated_profile_count": len({r.get("canonical_cell_uid") for r in rows}),
        "max_time_points": int(max_time_points),
        "time_window_s": float(time_window_s),
        "all_target_mean_r2": mean_r2,
        "all_target_min_r2": min_r2,
        "inventory_mean_r2": inv_mean,
        "inventory_min_r2": inv_min,
        "files": {"summary_json": str(out / "D17_G62L_FORMULA_ONLY_SUMMARY.json"), "metrics_csv": str(out / "D17_G62L_FORMULA_ONLY_TARGET_METRICS.csv")},
        "elapsed_s": float(time.time() - t0),
    }
    json_dump(summary, out / "D17_G62L_FORMULA_ONLY_SUMMARY.json")
    return summary


def run_model_patch_smoke(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    candidate_dir: str | Path,
    candidate_summary: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path,
    config: Mapping[str, Any],
    profile_contains: Sequence[str],
    splits: Sequence[str],
    include_flagged_probe: bool,
    profile_limit: int,
    max_time_points: int,
    time_window_s: float,
    predict_batch_size: int,
    device_arg: str,
    save_predictions: bool = False,
) -> Dict[str, Any]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cand = read_json(candidate_summary, default={}) or {}
    ckpt_path = _resolve_checkpoint(candidate_dir, cand, checkpoint)
    ckpt = torch_load_safe(ckpt_path, map_location="cpu")
    device = _device_from_arg(device_arg)
    model = build_model_from_checkpoint(ckpt, device)
    feature_names = list(ckpt.get("feature_names") or [])
    local_input_dim = int(ckpt.get("local_input_dim", 0))
    _, protocol_vocab, branch_vocab = parse_vocab_from_feature_names(feature_names, local_input_dim)
    x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
    x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    target_slices = {str(k): (int(v[0]), int(v[1])) for k, v in dict(ckpt.get("target_slices") or {}).items()}
    records, manifest = load_split_records(split_manifest)
    sem_map = load_semantics_map(g0_profile_semantics_csv)
    selected = _select_records(records, profile_contains, splits, include_flagged_probe, profile_limit)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    patch_infos: List[Dict[str, Any]] = []
    pred_manifest: List[Dict[str, Any]] = []
    t0 = time.time()
    for idx, rec in enumerate(selected):
        canonical = str(rec.get("canonical_cell_uid") or rec.get("cell_uid") or idx)
        try:
            prof = load_profile_pack(rec, _semantics_for(rec, sem_map), protocol_vocab, branch_vocab, max_time_points, time_window_s)
            X_aug, finfo, aug_names = augment_profile_features(prof)
            if X_aug.shape[1] != x_mean.size:
                raise ValueError(f"feature dim mismatch: X_aug={X_aug.shape[1]}, ckpt={x_mean.size}")
            if list(aug_names) != feature_names:
                raise ValueError("feature names differ from checkpoint; refusing silent schema drift")
            preds: List[np.ndarray] = []
            model.eval()
            for i0 in range(0, X_aug.shape[0], max(1, int(predict_batch_size))):
                i1 = min(X_aug.shape[0], i0 + max(1, int(predict_batch_size)))
                preds.append(predict_denorm(model, X_aug[i0:i1], x_mean, x_std, y_mean, y_std, device))
            pred = np.concatenate(preds, axis=0).astype(np.float32)
            pred_patch, pinfo = _patch_pred_array(pred, prof, target_slices, config)
            patch_infos.append({"profile_index": idx, "canonical_cell_uid": prof.canonical_cell_uid, "protocol": prof.protocol, "semantic_branch": prof.branch, **pinfo})
            for target, (a, b) in target_slices.items():
                yt = prof.targets[:, a:b]
                yb = pred[:, a:b]
                yp = pred_patch[:, a:b]
                rows.append({"profile_index": idx, "split": prof.split, "canonical_cell_uid": prof.canonical_cell_uid, "protocol": prof.protocol, "semantic_branch": prof.branch, "target": target, "stage": "before_patch", **_r2_mae_rmse(yt, yb)})
                rows.append({"profile_index": idx, "split": prof.split, "canonical_cell_uid": prof.canonical_cell_uid, "protocol": prof.protocol, "semantic_branch": prof.branch, "target": target, "stage": "after_patch", **_r2_mae_rmse(yt, yp)})
            if save_predictions:
                safe = prof.canonical_cell_uid.replace("\\", "_").replace("/", "_")
                p = out / "predictions" / f"D17_G62L_PATCHED_{idx:03d}_{safe}.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                arrays: Dict[str, Any] = {"t_global_s": prof.t_global_s.astype(np.float32), "canonical_cell_uid": np.array(prof.canonical_cell_uid), "protocol": np.array(prof.protocol), "semantic_branch": np.array(prof.branch)}
                for target, (a, b) in target_slices.items():
                    arrays[f"{target}_pred_before"] = pred[:, a:b].astype(np.float32)
                    arrays[f"{target}_pred"] = pred_patch[:, a:b].astype(np.float32)
                    arrays[f"{target}_true_report_only"] = prof.targets[:, a:b].astype(np.float32)
                np.savez_compressed(p, **arrays)
                pred_manifest.append({"profile_index": idx, "canonical_cell_uid": prof.canonical_cell_uid, "pred_npz": str(p), "n_time": int(prof.targets.shape[0])})
        except Exception as e:
            failures.append({"profile_index": idx, "canonical_cell_uid": canonical, "error": repr(e)})
    write_csv(rows, out / "D17_G62L_MODEL_PATCH_TARGET_METRICS.csv")
    write_csv(patch_infos, out / "D17_G62L_PATCH_INFO.csv")
    write_csv(failures, out / "D17_G62L_LOAD_FAILURES.csv")
    write_csv(pred_manifest, out / "D17_G62L_PREDICTION_MANIFEST.csv")
    after = [r for r in rows if r.get("stage") == "after_patch"]
    r2s = [_safe_float(r.get("r2")) for r in after]
    r2s = [x for x in r2s if math.isfinite(x)]
    mean_r2 = float(np.mean(r2s)) if r2s else float("nan")
    min_r2 = float(np.min(r2s)) if r2s else float("nan")
    gates = dict(config.get("model_patch_gates", {}) if isinstance(config.get("model_patch_gates"), Mapping) else {})
    mean_gate = float(gates.get("mean_r2", 0.95))
    min_gate = float(gates.get("min_r2", 0.90))
    blockers: List[str] = []
    if failures:
        blockers.append(f"{len(failures)} profiles failed to load/evaluate")
    if mean_r2 < mean_gate or min_r2 < min_gate:
        blockers.append(f"patched selected-profile R2 below gate: mean={mean_r2:.6g}, min={min_r2:.6g}")
    summary = {
        "protocol": "D17-G6.2L_FAST_P4D_GEO_MODEL_PATCH_SMOKE",
        "created_at_utc": utc_now(),
        "status": "PASS" if not failures else "REVIEW",
        "promotion_status": "PASS" if not blockers else "REVIEW",
        "g6c_streaming_smoke_ready": not blockers,
        "recommendation": "RUN_CYCLEWISE_STREAMING_SMOKE_SELECTED_PROFILES" if not blockers else "STOP_REVIEW_G62L_PATCH",
        "blockers": blockers,
        "training_performed": False,
        "radial_solver_used": False,
        "candidate_weights_modified": False,
        "model_loaded": True,
        "checkpoint": str(ckpt_path),
        "selected_profile_count": len(selected),
        "evaluated_profile_count": len({r.get("canonical_cell_uid") for r in after}),
        "max_time_points": int(max_time_points),
        "time_window_s": float(time_window_s),
        "after_patch_mean_r2": mean_r2,
        "after_patch_min_r2": min_r2,
        "files": {"summary_json": str(out / "D17_G62L_MODEL_PATCH_SMOKE_SUMMARY.json"), "metrics_csv": str(out / "D17_G62L_MODEL_PATCH_TARGET_METRICS.csv")},
        "elapsed_s": float(time.time() - t0),
    }
    json_dump(summary, out / "D17_G62L_MODEL_PATCH_SMOKE_SUMMARY.json")
    return summary
