#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D11-S9 trainable localized P2D-like transport-deficit correction.

This script does NOT modify GV1 mainline code and does NOT retrain the PINN.
It reads existing baseline prediction.npz files, fits a small deterministic
trainable correction head on a visible profile subset, and writes corrected
prediction.npz files for diagnostic scoring.

Correction form:
    V_corr = V_base - clip(max(0, X @ w), 0, max_deficit_V)
where X contains low-voltage, discharge/high-rate, SOC/time proxy and
transport-like gates computed from replay/prediction arrays.  The regression
learns a local voltage deficit target max(V_base - V_true, 0), with high weight
on low-target segments and preservation weight on normal segments.

This is a P2D-like transport/polarization *surrogate correction head* rather
than a full P2D solver.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return default


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def first_array(data: np.lib.npyio.NpzFile, names: Sequence[str]) -> Optional[np.ndarray]:
    keys = set(data.files)
    for name in names:
        if name in keys:
            return np.asarray(data[name])
    # fuzzy fallback: exact lower-case
    lower_map = {k.lower(): k for k in data.files}
    for name in names:
        if name.lower() in lower_map:
            return np.asarray(data[lower_map[name.lower()]])
    return None


def force_1d(x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        # If shape is (N,1) or (1,N), flatten.  For matrix outputs, use first column.
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            arr = arr.reshape(arr.shape[0], -1)[:, 0]
    arr = arr.astype(float, copy=False).reshape(-1)
    if n is not None:
        if arr.size < n:
            pad = np.full(n - arr.size, np.nan)
            arr = np.concatenate([arr, pad])
        elif arr.size > n:
            arr = arr[:n]
    return arr


def metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return {"n": 0, "MAE_V": float("nan"), "RMSE_V": float("nan"), "corr": float("nan"), "bias_V": float("nan")}
    yt = y_true[m]
    yp = y_pred[m]
    err = yp - yt
    if yt.size >= 2 and np.nanstd(yt) > 1e-12 and np.nanstd(yp) > 1e-12:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": int(yt.size),
        "MAE_V": float(np.mean(np.abs(err))),
        "RMSE_V": float(np.sqrt(np.mean(err ** 2))),
        "corr": corr,
        "bias_V": float(np.mean(err)),
        "pred_over_frac": float(np.mean(err > 0)),
        "pred_under_frac": float(np.mean(err < 0)),
        "target_min_V": float(np.nanmin(yt)),
        "target_max_V": float(np.nanmax(yt)),
        "pred_min_V": float(np.nanmin(yp)),
        "pred_max_V": float(np.nanmax(yp)),
    }


@dataclass
class PredictionRecord:
    profile: str
    protocol: str
    mode: str
    path: Path
    y_true: np.ndarray
    y_pred: np.ndarray
    t: np.ndarray
    I: np.ndarray
    T: np.ndarray
    ocv: np.ndarray
    low_gate_existing: np.ndarray
    component_arrays: Dict[str, np.ndarray]
    keys: List[str]


def infer_profile_from_path(path: Path) -> str:
    parts = list(path.parts)
    # Look for directory component containing Batch or battery.
    for part in reversed(parts):
        if re.search(r"Batch[-_]?\d|battery[-_]?\d", part, flags=re.I):
            if part.lower() not in {"prediction.npz"}:
                return part.replace(" ", "_")
    return path.parent.name


def infer_mode_from_path(path: Path) -> str:
    s = str(path).replace("\\", "/")
    known = [
        "baseline_copy", "baseline_d951", "p2dlike_transport_mild", "p2dlike_transport_medium",
        "p2dlike_transport_strong", "p2dlike_transport_discharge_only", "lowtarget_amplify_down_1p25",
        "lowtarget_amplify_down_1p50", "lowtarget_amplify_down_1p75_guarded",
    ]
    for k in known:
        if k in s:
            return k
    # fallback: parent directory
    return path.parent.name


def infer_protocol(profile: str, path: Path) -> str:
    text = f"{profile} {path}".lower()
    if "r2.5" in text or "r25" in text or "batch-3" in text or "batch_3" in text or "batch3" in text:
        return "R2.5"
    if "r3" in text or "batch-4" in text or "batch_4" in text or "batch4" in text:
        return "R3"
    if "2c" in text or "batch-1" in text or "batch_1" in text or "batch1" in text:
        return "2C"
    return "unknown"


def load_prediction(path: Path, forced_mode: str = "baseline") -> PredictionRecord:
    with np.load(path, allow_pickle=True) as data:
        y_true = first_array(data, [
            "voltage_exp", "voltage_true", "target_voltage", "voltage_target",
            "y_true", "target", "V_true", "V_exp",
        ])
        y_pred = first_array(data, [
            "voltage_exp_pred", "voltage_pred", "pred_voltage", "y_pred",
            "phis_c_pred", "V_pred", "pred",
        ])
        if y_true is None or y_pred is None:
            raise ValueError(f"Cannot find voltage true/pred arrays in {path}. Keys={data.files}")
        n = min(np.asarray(y_true).size, np.asarray(y_pred).size)
        y_true = force_1d(y_true, n)
        y_pred = force_1d(y_pred, n)

        t = first_array(data, ["t_global_s", "time_s", "t_s", "t", "time"])
        if t is None:
            t = np.arange(n, dtype=float)
        t = force_1d(t, n)

        I = first_array(data, ["I_profile", "current_A", "I_A", "current", "I"])
        if I is None:
            I = np.zeros(n, dtype=float)
        I = force_1d(I, n)

        T = first_array(data, ["temperature_C", "T_C", "temperature", "T"])
        if T is None:
            T = np.full(n, np.nan)
        T = force_1d(T, n)

        ocv = first_array(data, ["voltage_ocv_baseline", "ocv_baseline", "voltage_ocv", "V_ocv"])
        if ocv is None:
            ocv = y_pred.copy()
        ocv = force_1d(ocv, n)

        low_gate_existing = first_array(data, ["voltage_low_gate", "low_voltage_gate", "low_gate"])
        if low_gate_existing is None:
            low_gate_existing = np.full(n, np.nan)
        low_gate_existing = force_1d(low_gate_existing, n)

        comp: Dict[str, np.ndarray] = {}
        for k in data.files:
            if any(tok in k.lower() for tok in ["gate", "correction", "branch", "baseline", "head"]):
                try:
                    comp[k] = force_1d(np.asarray(data[k]), n)
                except Exception:
                    pass
        keys = list(data.files)

    profile = infer_profile_from_path(path)
    protocol = infer_protocol(profile, path)
    mode = forced_mode or infer_mode_from_path(path)
    return PredictionRecord(profile, protocol, mode, path, y_true, y_pred, t, I, T, ocv, low_gate_existing, comp, keys)


def find_baseline_predictions(root: Path, mode_filter: str = "baseline") -> List[Path]:
    if not root.exists():
        return []
    all_npz = sorted(root.rglob("prediction.npz"))
    selected: List[Path] = []
    for p in all_npz:
        s = str(p).replace("\\", "/").lower()
        if mode_filter.lower() in s:
            selected.append(p)
    if not selected:
        # If no mode filter match, take npz under root, but avoid candidate names when possible.
        for p in all_npz:
            s = str(p).lower()
            if any(x in s for x in ["p2dlike_transport", "lowtarget_amplify", "escape_mild", "escape_medium", "escape_strong"]):
                continue
            selected.append(p)
    # Deduplicate profiles by choosing first path per profile
    by_prof: Dict[str, Path] = {}
    for p in selected:
        prof = infer_profile_from_path(p)
        by_prof.setdefault(prof, p)
    return [by_prof[k] for k in sorted(by_prof)]


def alternating_split(records: List[PredictionRecord]) -> Dict[str, str]:
    # Split each protocol group alternately into train/eval.  If protocol unknown, use sorted profile alternation.
    groups: Dict[str, List[str]] = {}
    for r in records:
        groups.setdefault(r.protocol, []).append(r.profile)
    split: Dict[str, str] = {}
    for protocol, profiles in groups.items():
        uniq = sorted(set(profiles))
        for i, p in enumerate(uniq):
            split[p] = "train" if i % 2 == 0 else "eval"
    return split


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=float)
    lo, hi = np.nanpercentile(x[m], [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(x[m]), np.nanmax(x[m])
    if hi <= lo:
        return np.zeros_like(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def build_features(r: PredictionRecord, cfg: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = r.y_true.size
    tnorm = normalize01(r.t)
    absI = np.abs(r.I)
    finite_absI = absI[np.isfinite(absI)]
    scaleI = float(np.nanpercentile(finite_absI, 90)) if finite_absI.size else 1.0
    if not np.isfinite(scaleI) or scaleI <= 1e-12:
        scaleI = 1.0
    absI_norm = np.clip(absI / scaleI, 0.0, 3.0)
    # Datasets use charge positive / discharge negative in earlier project stages.
    discharge_gate = sigmoid((-r.I) / max(scaleI * 0.10, 1e-12))
    highrate_gate = np.clip(absI_norm, 0.0, 1.5) / 1.5
    ocv_ref = np.where(np.isfinite(r.ocv), r.ocv, r.y_pred)
    low_center = float(cfg.get("low_center_V", 3.05))
    low_width = max(float(cfg.get("low_width_V", 0.22)), 1e-3)
    low_gate = sigmoid((low_center - ocv_ref) / low_width)
    if np.isfinite(r.low_gate_existing).any():
        # Blend with existing model low gate if available, but never use target voltage.
        ex = np.nan_to_num(r.low_gate_existing, nan=0.0, posinf=1.0, neginf=0.0)
        low_gate = np.clip(0.5 * low_gate + 0.5 * ex, 0.0, 1.0)
    # Low SOC proxy: later in time and lower OCV/predicted voltage.
    pred_low_gate = sigmoid((float(cfg.get("pred_low_center_V", 3.35)) - r.y_pred) / max(float(cfg.get("pred_low_width_V", 0.25)), 1e-3))
    transport_gate = np.clip(low_gate * (0.25 + 0.75 * discharge_gate) * (0.35 + 0.65 * highrate_gate), 0.0, 1.0)
    # Feature matrix.  Use only measurable/predicted quantities, not y_true.
    X_cols = [
        transport_gate,
        transport_gate * low_gate,
        transport_gate * discharge_gate,
        transport_gate * highrate_gate,
        transport_gate * tnorm,
        transport_gate * pred_low_gate,
        transport_gate * absI_norm,
        low_gate * discharge_gate,
        pred_low_gate * discharge_gate,
        np.ones(n) * 0.0 + 0.0,  # placeholder for stable shape; replaced by bias column below
    ]
    X = np.vstack(X_cols[:-1] + [np.ones(n)]).T.astype(float)
    gates = {
        "low_gate_s9": low_gate,
        "discharge_gate_s9": discharge_gate,
        "highrate_gate_s9": highrate_gate,
        "pred_low_gate_s9": pred_low_gate,
        "transport_gate_s9": transport_gate,
        "time_norm_s9": tnorm,
        "absI_norm_s9": absI_norm,
    }
    return X, gates


def segment_masks(y_true: np.ndarray, y_pred: np.ndarray, I: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    n = y_true.size
    m_all = np.isfinite(y_true) & np.isfinite(y_pred)
    absI = np.abs(I)
    finite_absI = absI[np.isfinite(absI)]
    low_I_thr = max(float(np.nanpercentile(finite_absI, 10)) if finite_absI.size else 0.0, 1e-12)
    return {
        "all": m_all,
        "low_target": m_all & (y_true <= 3.0),
        "low_target_le_2p75": m_all & (y_true <= 2.75),
        "normal_preserve": m_all & (y_true > 3.1) & (y_true < 4.1),
        "rest_I_zero": m_all & (absI <= low_I_thr),
        "high_target_ge_4p10": m_all & (y_true >= 4.10),
        "charge_I_positive": m_all & (I > low_I_thr),
        "discharge_I_negative": m_all & (I < -low_I_thr),
    }


def build_fit_arrays(records: List[PredictionRecord], split: Dict[str, str], cfg: Dict[str, float], use_split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    feature_names = [
        "transport_gate", "transport_low_gate", "transport_discharge", "transport_highrate",
        "transport_time", "transport_pred_low", "transport_absI", "low_discharge", "predlow_discharge", "bias",
    ]
    for r in records:
        if split.get(r.profile, "train") != use_split:
            continue
        X, _ = build_features(r, cfg)
        masks = segment_masks(r.y_true, r.y_pred, r.I, r.t)
        err = r.y_pred - r.y_true
        target_deficit = np.clip(err, 0.0, float(cfg.get("target_deficit_clip_V", 1.2)))
        # Weighted fit: low-target gets strong positive target, normal/high/rest get preservation target.
        w = np.ones_like(target_deficit) * float(cfg.get("normal_weight", 2.0))
        w[masks["low_target"]] = float(cfg.get("low_weight", 30.0))
        w[masks["low_target_le_2p75"]] = float(cfg.get("deep_low_weight", 50.0))
        w[masks["high_target_ge_4p10"]] = float(cfg.get("high_weight", 5.0))
        w[masks["rest_I_zero"]] = np.maximum(w[masks["rest_I_zero"]], float(cfg.get("rest_weight", 4.0)))
        # Do not learn to subtract where the base already underpredicts heavily.
        under = err < -0.03
        target_deficit[under] = 0.0
        w[under] = np.maximum(w[under], float(cfg.get("underpreserve_weight", 6.0)))
        good = np.all(np.isfinite(X), axis=1) & np.isfinite(target_deficit) & np.isfinite(w)
        if np.any(good):
            Xs.append(X[good])
            ys.append(target_deficit[good])
            ws.append(w[good])
    if not Xs:
        raise RuntimeError("No training rows found for D11-S9 correction head.")
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), feature_names


def fit_weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, nonnegative_except_bias: bool = True) -> np.ndarray:
    # Weighted ridge closed form.  Then project non-bias coeffs to nonnegative to enforce deficit-like behavior.
    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    Xw = X * sw[:, None]
    yw = y * sw
    n_feat = X.shape[1]
    reg = np.eye(n_feat) * float(lam)
    reg[-1, -1] = float(lam) * 0.1  # weaker bias reg
    try:
        beta = np.linalg.solve(Xw.T @ Xw + reg, Xw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw.T @ Xw + reg, Xw.T @ yw, rcond=None)[0]
    if nonnegative_except_bias:
        beta[:-1] = np.maximum(beta[:-1], 0.0)
        beta[-1] = max(beta[-1], 0.0)
    return beta


def apply_correction(r: PredictionRecord, beta: np.ndarray, cfg: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    X, gates = build_features(r, cfg)
    raw = X @ beta
    deficit = np.maximum(raw, 0.0)
    deficit *= float(cfg.get("deficit_scale", 1.0))
    max_deficit = float(cfg.get("max_deficit_V", 0.9))
    deficit = np.clip(deficit, 0.0, max_deficit)
    # Preserve high-voltage and rest regions with soft reduction factors.
    yref = np.where(np.isfinite(r.ocv), r.ocv, r.y_pred)
    high_guard = 1.0 - sigmoid((yref - float(cfg.get("high_guard_center_V", 4.05))) / max(float(cfg.get("high_guard_width_V", 0.08)), 1e-3))
    if bool(cfg.get("enable_high_guard", True)):
        deficit = deficit * np.clip(0.15 + 0.85 * high_guard, 0.0, 1.0)
    if bool(cfg.get("enable_rest_guard", True)):
        absI = np.abs(r.I)
        finite_absI = absI[np.isfinite(absI)]
        scaleI = float(np.nanpercentile(finite_absI, 90)) if finite_absI.size else 1.0
        if scaleI <= 1e-12 or not np.isfinite(scaleI):
            scaleI = 1.0
        rest_factor = sigmoid((absI / scaleI - 0.05) / 0.03)
        deficit = deficit * np.clip(0.25 + 0.75 * rest_factor, 0.0, 1.0)
    v_corr = r.y_pred - deficit
    extras = dict(gates)
    extras["p2dlike_trainable_deficit_V"] = deficit
    extras["p2dlike_trainable_raw_deficit_V"] = np.maximum(raw, 0.0)
    extras["p2dlike_trainable_high_guard"] = high_guard
    return v_corr, extras


def save_prediction(out_path: Path, r: PredictionRecord, v_corr: np.ndarray, extras: Dict[str, np.ndarray], mode: str, split_tag: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "voltage_exp": r.y_true.astype(np.float32),
        "voltage_exp_pred": v_corr.astype(np.float32),
        "voltage_exp_base_pred": r.y_pred.astype(np.float32),
        "t_global_s": r.t.astype(np.float64),
        "I_profile": r.I.astype(np.float32),
        "temperature_C": r.T.astype(np.float32),
        "voltage_ocv_baseline": r.ocv.astype(np.float32),
        "profile": np.array(r.profile),
        "protocol": np.array(r.protocol),
        "mode": np.array(mode),
        "split": np.array(split_tag),
    }
    for k, v in extras.items():
        payload[k] = np.asarray(v).astype(np.float32)
    np.savez_compressed(out_path, **payload)


def config_grid() -> Dict[str, Dict[str, float]]:
    return {
        "baseline_copy": {"baseline": 1.0},
        "p2dtrain_local_mild": {
            "low_weight": 25.0, "deep_low_weight": 45.0, "normal_weight": 3.5, "high_weight": 8.0, "rest_weight": 5.0,
            "ridge_lambda": 3.0, "max_deficit_V": 0.55, "deficit_scale": 0.75,
            "low_center_V": 3.08, "low_width_V": 0.22, "enable_high_guard": 1.0, "enable_rest_guard": 1.0,
        },
        "p2dtrain_local_medium": {
            "low_weight": 40.0, "deep_low_weight": 75.0, "normal_weight": 4.0, "high_weight": 10.0, "rest_weight": 6.0,
            "ridge_lambda": 2.0, "max_deficit_V": 0.70, "deficit_scale": 0.95,
            "low_center_V": 3.10, "low_width_V": 0.24, "enable_high_guard": 1.0, "enable_rest_guard": 1.0,
        },
        "p2dtrain_local_guarded": {
            "low_weight": 60.0, "deep_low_weight": 110.0, "normal_weight": 6.0, "high_weight": 14.0, "rest_weight": 8.0,
            "ridge_lambda": 4.0, "max_deficit_V": 0.78, "deficit_scale": 1.05,
            "low_center_V": 3.12, "low_width_V": 0.24, "high_guard_center_V": 3.95, "high_guard_width_V": 0.10,
            "enable_high_guard": 1.0, "enable_rest_guard": 1.0,
        },
        "p2dtrain_deeplow_focus": {
            "low_weight": 35.0, "deep_low_weight": 160.0, "normal_weight": 5.5, "high_weight": 16.0, "rest_weight": 8.0,
            "ridge_lambda": 5.0, "max_deficit_V": 0.82, "deficit_scale": 1.10,
            "low_center_V": 3.15, "low_width_V": 0.20, "enable_high_guard": 1.0, "enable_rest_guard": 1.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="D11-S9 trainable localized P2D-like correction head from baseline predictions.")
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--baseline_prediction_root", default="", help="Directory containing baseline prediction.npz files. Auto-detected if empty.")
    ap.add_argument("--baseline_mode_filter", default="baseline", help="Substring used to select baseline prediction paths.")
    ap.add_argument("--out_root", default="", help="Output prediction root. Defaults under cache_root.")
    ap.add_argument("--max_profiles", type=int, default=6)
    args = ap.parse_args()

    cache = Path(args.cache_root)
    candidate_roots = []
    if args.baseline_prediction_root:
        candidate_roots.append(Path(args.baseline_prediction_root))
    candidate_roots.extend([
        cache / "xjtu_batch134_d11_s8_p2dlike_transport_correction",
        cache / "xjtu_batch134_d11_s7_lowvoltage_escape",
        cache / "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair",
        cache / "xjtu_batch134_d11_s5a_lowtarget_sign_gate_diagnosis",
        cache / "xjtu_batch134_d11_s4_lowtail_correction_smoke",
    ])
    baseline_paths: List[Path] = []
    used_root: Optional[Path] = None
    for root in candidate_roots:
        paths = find_baseline_predictions(root, args.baseline_mode_filter)
        if paths:
            baseline_paths = paths
            used_root = root
            break
    if not baseline_paths:
        raise RuntimeError("No baseline prediction.npz files found. Run prior D11-S5/S7/S8 baseline predictions first or pass --baseline_prediction_root.")
    # Keep 6 profiles; exclude battery-8 if present.
    filt = [p for p in baseline_paths if "battery-8" not in str(p).lower() and "battery_8" not in str(p).lower()]
    baseline_paths = filt[: args.max_profiles]
    records = [load_prediction(p, forced_mode="baseline_copy") for p in baseline_paths]
    split = alternating_split(records)

    out_root = Path(args.out_root) if args.out_root else cache / "xjtu_batch134_d11_s9_trainable_p2dlike_correction"
    if out_root.exists():
        # Do not delete previous output automatically; archive minimal marker.
        pass
    out_root.mkdir(parents=True, exist_ok=True)

    configs = config_grid()
    fit_rows: List[Dict[str, Any]] = []
    coeff_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    # Baseline copy
    for r in records:
        mode = "baseline_copy"
        split_tag = split.get(r.profile, "train")
        out_path = out_root / mode / r.profile / "prediction.npz"
        save_prediction(out_path, r, r.y_pred, {}, mode, split_tag)
        run_rows.append({
            "mode": mode, "profile": r.profile, "protocol": r.protocol, "split": split_tag,
            "source_prediction": str(r.path), "output_prediction": str(out_path), "status": "written",
        })

    # Train candidate heads on train split, apply to all profiles.
    for mode, cfg in configs.items():
        if mode == "baseline_copy":
            continue
        X, y, w, feature_names = build_fit_arrays(records, split, cfg, use_split="train")
        beta = fit_weighted_ridge(X, y, w, lam=float(cfg.get("ridge_lambda", 2.0)))
        # Fit diagnostics
        pred_train = np.clip(np.maximum(X @ beta, 0.0) * float(cfg.get("deficit_scale", 1.0)), 0.0, float(cfg.get("max_deficit_V", 0.9)))
        fit_mae = float(np.mean(np.abs(pred_train - y)))
        fit_rows.append({
            "mode": mode, "train_rows": int(X.shape[0]), "feature_count": int(X.shape[1]),
            "fit_deficit_mae_V": fit_mae, "target_deficit_mean_V": float(np.mean(y)),
            "pred_deficit_mean_V": float(np.mean(pred_train)), "ridge_lambda": cfg.get("ridge_lambda"),
            "max_deficit_V": cfg.get("max_deficit_V"), "deficit_scale": cfg.get("deficit_scale"),
        })
        for name, val in zip(feature_names, beta):
            coeff_rows.append({"mode": mode, "feature": name, "coef": float(val)})
        for r in records:
            split_tag = split.get(r.profile, "train")
            v_corr, extras = apply_correction(r, beta, cfg)
            out_path = out_root / mode / r.profile / "prediction.npz"
            save_prediction(out_path, r, v_corr, extras, mode, split_tag)
            base_m = metric(r.y_true, r.y_pred)
            corr_m = metric(r.y_true, v_corr)
            low_mask = segment_masks(r.y_true, r.y_pred, r.I, r.t)["low_target"]
            low_base = metric(r.y_true[low_mask], r.y_pred[low_mask]) if np.any(low_mask) else {}
            low_corr = metric(r.y_true[low_mask], v_corr[low_mask]) if np.any(low_mask) else {}
            run_rows.append({
                "mode": mode, "profile": r.profile, "protocol": r.protocol, "split": split_tag,
                "source_prediction": str(r.path), "output_prediction": str(out_path), "status": "written",
                "all_base_MAE_V": base_m.get("MAE_V"), "all_corr_MAE_V": corr_m.get("MAE_V"),
                "low_base_MAE_V": low_base.get("MAE_V"), "low_corr_MAE_V": low_corr.get("MAE_V"),
                "mean_deficit_V": float(np.nanmean(extras["p2dlike_trainable_deficit_V"])),
                "low_mean_deficit_V": float(np.nanmean(extras["p2dlike_trainable_deficit_V"][low_mask])) if np.any(low_mask) else float("nan"),
            })

    write_csv(out_root / "D11_S9_training_fit_summary.csv", fit_rows)
    write_csv(out_root / "D11_S9_feature_coefficients.csv", coeff_rows)
    write_csv(out_root / "D11_S9_prediction_manifest.csv", run_rows)
    json_dump(out_root / "D11_S9_summary.json", {
        "ok": True,
        "stage": "D11-S9 trainable localized P2D-like correction head generation",
        "baseline_root_used": str(used_root),
        "out_root": str(out_root),
        "profile_count": len(records),
        "profiles": [{"profile": r.profile, "protocol": r.protocol, "split": split.get(r.profile), "path": str(r.path)} for r in records],
        "modes": list(configs.keys()),
        "run_count_expected": len(records) * len(configs),
        "notes": [
            "No GV1 mainline files were modified.",
            "Correction is a deterministic trainable ridge head fitted on visible train profiles only.",
            "Use scorecard script to evaluate all/eval split and segment-level promotion rules.",
        ],
    })
    print(json.dumps({"ok": True, "out_root": str(out_root), "run_count_expected": len(records) * len(configs)}, indent=2))


if __name__ == "__main__":
    main()
