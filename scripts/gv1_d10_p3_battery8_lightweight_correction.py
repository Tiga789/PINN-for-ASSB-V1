#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D10-P3 lightweight post-hoc correction study for B1_2C battery-8.

Purpose
-------
D10-P0 classifies B1_2C battery-8 as a flagged late-2C discharge regime / outlier
case. D10-P1 validates that the D9.6/D9.5.1 mainline generalizes on the other
23 profiles. This script tests whether battery-8 can be handled by a *lightweight*
profile-level calibration without changing the D9.6 mainline or adding hard voltage
clamps.

It reads one D9.6 battery-8 prediction.npz and evaluates several simple candidates:
  - identity baseline
  - global bias
  - global affine:                 Vcorr = a * Vpred + b
  - discharge-only bias:            Vcorr = Vpred + b_dis for I < 0
  - discharge-only affine:          Vcorr = a_dis * Vpred + b_dis for I < 0
  - current-segment affine:         separate affine on charge / discharge / rest
  - discharge-time affine ridge:    discharge only, linear features [Vpred, t_norm]

No strong clamp is applied. The script reports full-fit and chronological holdout
metrics and produces a recommendation. A full-fit correction is a calibration
benchmark; it should not be confused with cross-cell generalization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting is optional
    plt = None

DEFAULT_CACHE_ROOT = Path(r"E:/XJTU battery dataset/_gv1_cache")
DEFAULT_PREDICTION_NPZ = DEFAULT_CACHE_ROOT / "xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96" / "prediction.npz"
DEFAULT_OUT_DIR = DEFAULT_CACHE_ROOT / "xjtu_batch134_d10_p3_battery8_lightweight_correction"


def _as1d(x: Any, n: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if n is not None:
        a = a[:n]
    return a


def _corr(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(y, dtype=bool)
    m = mask & np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 3:
        return math.nan
    yy = y[m].astype(float)
    pp = p[m].astype(float)
    if float(np.nanstd(yy)) <= 1e-12 or float(np.nanstd(pp)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(yy, pp)[0, 1])


def _r2(y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones_like(y, dtype=bool)
    m = mask & np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 3:
        return math.nan
    yy = y[m].astype(float)
    pp = p[m].astype(float)
    den = float(np.sum((yy - float(np.mean(yy))) ** 2))
    if den <= 1e-12:
        return math.nan
    return float(1.0 - np.sum((pp - yy) ** 2) / den)


def _metrics(prefix: str, y: np.ndarray, p: np.ndarray, I: np.ndarray, t: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    m = mask & np.isfinite(y) & np.isfinite(p)
    n = int(m.sum())
    row: dict[str, Any] = {"metric_scope": prefix, "n": n}
    if n == 0:
        return row
    err = p[m] - y[m]
    p_m = p[m]
    y_m = y[m]
    row.update({
        "mae_V": float(np.nanmean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.nanmean(err ** 2))),
        "bias_V": float(np.nanmean(err)),
        "corr": _corr(y, p, m),
        "r2": _r2(y, p, m),
        "target_min_V": float(np.nanmin(y_m)),
        "target_max_V": float(np.nanmax(y_m)),
        "target_range_V": float(np.nanmax(y_m) - np.nanmin(y_m)),
        "pred_min_V": float(np.nanmin(p_m)),
        "pred_max_V": float(np.nanmax(p_m)),
        "pred_range_V": float(np.nanmax(p_m) - np.nanmin(p_m)),
        "pred_upper_frac_ge_4p269": float(np.nanmean(p_m >= 4.269)),
        "pred_overshoot_frac_gt_4p35": float(np.nanmean(p_m > 4.35)),
        "pred_low_frac_le_2p75": float(np.nanmean(p_m <= 2.75)),
        "target_low_frac_le_2p75": float(np.nanmean(y_m <= 2.75)),
        "target_high_frac_ge_4p10": float(np.nanmean(y_m >= 4.10)),
    })
    return row


def _safe_float(v: Any, default: float = math.nan) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fit_bias(y: np.ndarray, p: np.ndarray, fit_mask: np.ndarray) -> float:
    m = fit_mask & np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 3:
        return 0.0
    return float(np.mean(y[m] - p[m]))


def _fit_affine(y: np.ndarray, p: np.ndarray, fit_mask: np.ndarray, *, ridge: float = 1e-8) -> tuple[float, float]:
    m = fit_mask & np.isfinite(y) & np.isfinite(p)
    if int(m.sum()) < 3 or float(np.nanstd(p[m])) <= 1e-12:
        return 1.0, _fit_bias(y, p, fit_mask)
    X = np.column_stack([p[m], np.ones(int(m.sum()))])
    yy = y[m]
    A = X.T @ X + float(ridge) * np.eye(2)
    b = X.T @ yy
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(X, yy, rcond=None)[0]
    a, c = float(coef[0]), float(coef[1])
    if not math.isfinite(a) or not math.isfinite(c):
        return 1.0, _fit_bias(y, p, fit_mask)
    return a, c


def _fit_linear_features(y: np.ndarray, features: np.ndarray, fit_mask: np.ndarray, *, ridge: float = 1e-4) -> np.ndarray:
    m = fit_mask & np.isfinite(y)
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    m = m & np.all(np.isfinite(features), axis=1)
    if int(m.sum()) < max(5, features.shape[1] + 1):
        # identity-like fallback: y ~= first feature + bias 0
        coef = np.zeros(features.shape[1], dtype=float)
        coef[0] = 1.0
        return coef
    X = features[m]
    yy = y[m]
    A = X.T @ X + float(ridge) * np.eye(X.shape[1])
    b = X.T @ yy
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(X, yy, rcond=None)[0]
    return np.asarray(coef, dtype=float)


@dataclass
class Candidate:
    name: str
    corrected: np.ndarray
    params: dict[str, Any]
    note: str


def _make_candidates(y: np.ndarray, p: np.ndarray, I: np.ndarray, t: np.ndarray, fit_mask: np.ndarray, current_eps_A: float) -> list[Candidate]:
    n = len(y)
    charge = I > current_eps_A
    discharge = I < -current_eps_A
    rest = np.abs(I) <= current_eps_A
    t_norm = (t - np.nanmin(t)) / max(float(np.nanmax(t) - np.nanmin(t)), 1.0)
    out: list[Candidate] = []
    out.append(Candidate("identity_d9_6_raw", p.copy(), {}, "Original D9.6 battery-8 prediction; no correction."))

    b = _fit_bias(y, p, fit_mask)
    out.append(Candidate("global_bias", p + b, {"bias_V": b}, "Global additive voltage bias."))

    a, c = _fit_affine(y, p, fit_mask)
    out.append(Candidate("global_affine", a * p + c, {"a": a, "b": c}, "Global affine calibration Vcorr=a*Vpred+b."))

    b_dis = _fit_bias(y, p, fit_mask & discharge)
    p_dis_bias = p.copy()
    p_dis_bias[discharge] = p_dis_bias[discharge] + b_dis
    out.append(Candidate("discharge_only_bias", p_dis_bias, {"bias_discharge_V": b_dis}, "Additive correction on I<0 discharge segment only."))

    a_dis, c_dis = _fit_affine(y, p, fit_mask & discharge)
    p_dis_aff = p.copy()
    p_dis_aff[discharge] = a_dis * p_dis_aff[discharge] + c_dis
    out.append(Candidate("discharge_only_affine", p_dis_aff, {"a_discharge": a_dis, "b_discharge": c_dis}, "Affine correction on I<0 discharge segment only; charge/rest unchanged."))

    # Segment affine: charge / discharge / rest handled separately. If a segment is too small, it falls back to identity+bias.
    p_seg = p.copy()
    params_seg: dict[str, Any] = {}
    for name, seg in [("charge", charge), ("discharge", discharge), ("rest", rest)]:
        aa, bb = _fit_affine(y, p, fit_mask & seg)
        params_seg[f"a_{name}"] = aa
        params_seg[f"b_{name}"] = bb
        if int(seg.sum()) >= 3:
            p_seg[seg] = aa * p_seg[seg] + bb
    out.append(Candidate("current_segment_affine", p_seg, params_seg, "Separate affine calibration for charge, discharge and rest current regimes."))

    # Discharge ridge features: light time-aware correction for the problematic discharge regime.
    # Feature matrix includes p, t_norm, p*t_norm and constant. This is still a tiny linear model.
    F_dis = np.column_stack([p, t_norm, p * t_norm, np.ones(n)])
    coef_dis = _fit_linear_features(y, F_dis, fit_mask & discharge, ridge=1e-4)
    p_dis_ridge = p.copy()
    p_dis_ridge[discharge] = F_dis[discharge] @ coef_dis
    out.append(Candidate(
        "discharge_time_affine_ridge",
        p_dis_ridge,
        {"coef_p": float(coef_dis[0]), "coef_tnorm": float(coef_dis[1]), "coef_p_tnorm": float(coef_dis[2]), "bias": float(coef_dis[3])},
        "Tiny discharge-only ridge model using [Vpred, t_norm, Vpred*t_norm, 1]; calibration-only unless validated on holdout.",
    ))
    return out


def _candidate_summary_rows(candidates: list[Candidate], y: np.ndarray, I: np.ndarray, t: np.ndarray, masks: dict[str, np.ndarray], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        base_all_mae = _safe_float(baseline.get("all", {}).get("mae_V"))
        base_dis_mae = _safe_float(baseline.get("discharge_I_neg", {}).get("mae_V"))
        base_charge_mae = _safe_float(baseline.get("charge_I_pos", {}).get("mae_V"))
        base_over = _safe_float(baseline.get("all", {}).get("pred_overshoot_frac_gt_4p35"), 0.0)
        base_upper = _safe_float(baseline.get("all", {}).get("pred_upper_frac_ge_4p269"), 0.0)
        all_m = _metrics("all", y, cand.corrected, I, t, masks["all"])
        dis_m = _metrics("discharge_I_neg", y, cand.corrected, I, t, masks["discharge_I_neg"])
        chg_m = _metrics("charge_I_pos", y, cand.corrected, I, t, masks["charge_I_pos"])
        rest_m = _metrics("rest_I_zero", y, cand.corrected, I, t, masks["rest_I_zero"])
        high_m = _metrics("high_target", y, cand.corrected, I, t, masks["high_target"])
        low_m = _metrics("low_target", y, cand.corrected, I, t, masks["low_target"])
        target_range = _safe_float(all_m.get("target_range_V"), 0.0)
        pred_range = _safe_float(all_m.get("pred_range_V"), 0.0)
        row = {
            "candidate": cand.name,
            "note": cand.note,
            "n_all": all_m.get("n"),
            "mae_V": all_m.get("mae_V"),
            "rmse_V": all_m.get("rmse_V"),
            "corr": all_m.get("corr"),
            "r2": all_m.get("r2"),
            "bias_V": all_m.get("bias_V"),
            "pred_min_V": all_m.get("pred_min_V"),
            "pred_max_V": all_m.get("pred_max_V"),
            "pred_range_V": all_m.get("pred_range_V"),
            "target_range_V": all_m.get("target_range_V"),
            "range_ratio_pred_to_target": (pred_range / target_range) if target_range > 1e-12 else math.nan,
            "pred_upper_frac_ge_4p269": all_m.get("pred_upper_frac_ge_4p269"),
            "pred_overshoot_frac_gt_4p35": all_m.get("pred_overshoot_frac_gt_4p35"),
            "charge_mae_V": chg_m.get("mae_V"),
            "charge_corr": chg_m.get("corr"),
            "charge_bias_V": chg_m.get("bias_V"),
            "discharge_mae_V": dis_m.get("mae_V"),
            "discharge_corr": dis_m.get("corr"),
            "discharge_bias_V": dis_m.get("bias_V"),
            "rest_mae_V": rest_m.get("mae_V"),
            "high_target_mae_V": high_m.get("mae_V"),
            "low_target_mae_V": low_m.get("mae_V"),
            "params_json": json.dumps(cand.params, ensure_ascii=False),
        }
        row["improve_mae_frac_vs_raw"] = (base_all_mae - _safe_float(row["mae_V"])) / base_all_mae if base_all_mae > 0 else math.nan
        row["improve_discharge_mae_frac_vs_raw"] = (base_dis_mae - _safe_float(row["discharge_mae_V"])) / base_dis_mae if base_dis_mae > 0 else math.nan
        row["charge_mae_delta_vs_raw_V"] = _safe_float(row["charge_mae_V"]) - base_charge_mae if base_charge_mae == base_charge_mae else math.nan
        row["overshoot_delta_vs_raw"] = _safe_float(row["pred_overshoot_frac_gt_4p35"], 0.0) - base_over
        row["upper_delta_vs_raw"] = _safe_float(row["pred_upper_frac_ge_4p269"], 0.0) - base_upper
        rows.append(row)
    return rows


def _rank_rows(rows: list[dict[str, Any]], *, baseline_name: str = "identity_d9_6_raw") -> list[dict[str, Any]]:
    if not rows:
        return rows
    raw = next((r for r in rows if r.get("candidate") == baseline_name), rows[0])
    raw_mae = _safe_float(raw.get("mae_V"))
    raw_dis_mae = _safe_float(raw.get("discharge_mae_V"))
    raw_charge_mae = _safe_float(raw.get("charge_mae_V"))
    raw_over = _safe_float(raw.get("pred_overshoot_frac_gt_4p35"), 0.0)
    raw_upper = _safe_float(raw.get("pred_upper_frac_ge_4p269"), 0.0)
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        mae = _safe_float(r.get("mae_V"), 999.0)
        dis = _safe_float(r.get("discharge_mae_V"), 999.0)
        chg = _safe_float(r.get("charge_mae_V"), 999.0)
        corr = _safe_float(r.get("corr"), -999.0)
        over = _safe_float(r.get("pred_overshoot_frac_gt_4p35"), 1.0)
        upper = _safe_float(r.get("pred_upper_frac_ge_4p269"), 1.0)
        range_ratio = _safe_float(r.get("range_ratio_pred_to_target"), 0.0)
        # lower is better; mild preference for better discharge and no overshoot/range collapse.
        score = mae + 0.25 * dis + 0.10 * max(0.0, chg - raw_charge_mae) + 1.5 * max(0.0, over - raw_over) + 0.25 * max(0.0, upper - raw_upper)
        if corr < 0.90:
            score += 0.05 + (0.90 - corr) * 0.1
        if range_ratio < 0.60:
            score += 1.0
        # Avoid choosing the raw candidate unless all corrections fail.
        if r.get("candidate") == baseline_name:
            score += 0.02
        r["selection_score_lower_is_better"] = float(score)
        # Classify safety relative to raw.
        overall_imp = ((raw_mae - mae) / raw_mae) if raw_mae > 0 else math.nan
        dis_imp = ((raw_dis_mae - dis) / raw_dis_mae) if raw_dis_mae > 0 else math.nan
        charge_delta = chg - raw_charge_mae if raw_charge_mae == raw_charge_mae else math.nan
        safe_overshoot = over <= raw_over + 0.002 and upper <= raw_upper + 0.015
        safe_range = range_ratio >= 0.60
        charge_ok = (not math.isfinite(charge_delta)) or charge_delta <= max(0.015, 0.30 * raw_charge_mae)
        if r.get("candidate") == baseline_name:
            cls = "raw_baseline"
        elif overall_imp >= 0.15 and dis_imp >= 0.20 and charge_ok and safe_overshoot and safe_range:
            cls = "safe_lightweight_correction"
        elif overall_imp >= 0.05 and dis_imp >= 0.10 and safe_overshoot and safe_range:
            cls = "weak_or_calibration_only_correction"
        else:
            cls = "not_recommended"
        r["recommendation_class"] = cls
        scored.append((score, r))
    scored.sort(key=lambda x: x[0])
    for i, (_, r) in enumerate(scored, start=1):
        r["score_rank"] = i
    return [r for _, r in scored]


def _make_masks(y: np.ndarray, p: np.ndarray, I: np.ndarray, t: np.ndarray, current_eps_A: float, low_thr: float, high_thr: float, eval_mask: np.ndarray) -> dict[str, np.ndarray]:
    n = len(y)
    return {
        "all": eval_mask.copy(),
        "charge_I_pos": eval_mask & (I > current_eps_A),
        "discharge_I_neg": eval_mask & (I < -current_eps_A),
        "rest_I_zero": eval_mask & (np.abs(I) <= current_eps_A),
        "low_target": eval_mask & (y <= low_thr),
        "high_target": eval_mask & (y >= high_thr),
        "mid_target": eval_mask & (y > low_thr) & (y < high_thr),
    }


def _select_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    safe = [r for r in rows if r.get("recommendation_class") == "safe_lightweight_correction"]
    if safe:
        return sorted(safe, key=lambda r: _safe_float(r.get("selection_score_lower_is_better"), 999))[0]
    weak = [r for r in rows if r.get("recommendation_class") == "weak_or_calibration_only_correction"]
    if weak:
        return sorted(weak, key=lambda r: _safe_float(r.get("selection_score_lower_is_better"), 999))[0]
    return None


def _plot_overlay(path: Path, t: np.ndarray, y: np.ndarray, p_raw: np.ndarray, p_corr: np.ndarray, I: np.ndarray, title: str) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, y, label="target voltage", linewidth=1.2)
    ax.plot(t, p_raw, label="raw D9.6", linewidth=1.0, alpha=0.8)
    ax.plot(t, p_corr, label="D10-P3 corrected", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("time_s")
    ax.set_ylabel("voltage_V")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residual(path: Path, t: np.ndarray, y: np.ndarray, p_raw: np.ndarray, p_corr: np.ndarray, title: str) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, p_raw - y, label="raw residual", linewidth=1.0, alpha=0.8)
    ax.plot(t, p_corr - y, label="corrected residual", linewidth=1.0, alpha=0.8)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("time_s")
    ax.set_ylabel("prediction - target / V")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_npz", default=str(DEFAULT_PREDICTION_NPZ))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--current_eps_A", type=float, default=1e-8)
    ap.add_argument("--low_voltage_threshold_V", type=float, default=2.75)
    ap.add_argument("--high_voltage_threshold_V", type=float, default=4.10)
    ap.add_argument("--holdout_fraction", type=float, default=0.30, help="Chronological tail fraction used only for validation-style scoring.")
    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    pred_path = Path(args.prediction_npz)
    out_dir = Path(args.out_dir)
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(pred_path, allow_pickle=True) as z:
        keys = set(z.files)
        if "voltage_exp" not in keys:
            raise KeyError("prediction.npz must contain voltage_exp")
        y = _as1d(z["voltage_exp"])
        if "voltage_exp_pred" in keys:
            p = _as1d(z["voltage_exp_pred"])
            pred_key = "voltage_exp_pred"
        elif "phis_c_pred" in keys:
            p = _as1d(z["phis_c_pred"])
            pred_key = "phis_c_pred"
        else:
            raise KeyError("prediction.npz must contain voltage_exp_pred or phis_c_pred")
        I = _as1d(z["I_profile"]) if "I_profile" in keys else np.zeros_like(y)
        t = _as1d(z["t_global_s"]) if "t_global_s" in keys else np.arange(len(y), dtype=float)
        original = {k: z[k] for k in z.files}

    n = min(len(y), len(p), len(I), len(t))
    y, p, I, t = y[:n], p[:n], I[:n], t[:n]
    finite = np.isfinite(y) & np.isfinite(p) & np.isfinite(I) & np.isfinite(t)
    if int(finite.sum()) < 10:
        raise ValueError("Too few finite points in prediction.npz")

    # Full-fit candidate benchmark.
    full_fit_mask = finite.copy()
    full_eval_mask = finite.copy()
    masks_full = _make_masks(y, p, I, t, args.current_eps_A, args.low_voltage_threshold_V, args.high_voltage_threshold_V, full_eval_mask)
    baseline_metrics_full = {name: _metrics(name, y, p, I, t, mask) for name, mask in masks_full.items()}
    cands_full = _make_candidates(y, p, I, t, full_fit_mask, args.current_eps_A)
    rows_full = _rank_rows(_candidate_summary_rows(cands_full, y, I, t, masks_full, baseline_metrics_full))

    # Chronological holdout: fit on the visible prefix, score on the hidden tail.
    holdout_fraction = float(np.clip(args.holdout_fraction, 0.05, 0.80))
    finite_idx = np.where(finite)[0]
    split_pos = int(round((1.0 - holdout_fraction) * len(finite_idx)))
    split_pos = min(max(split_pos, 10), len(finite_idx) - 5)
    cutoff_idx = int(finite_idx[split_pos])
    fit_prefix = finite & (np.arange(n) <= cutoff_idx)
    eval_holdout = finite & (np.arange(n) > cutoff_idx)
    masks_holdout = _make_masks(y, p, I, t, args.current_eps_A, args.low_voltage_threshold_V, args.high_voltage_threshold_V, eval_holdout)
    baseline_metrics_holdout = {name: _metrics(name, y, p, I, t, mask) for name, mask in masks_holdout.items()}
    cands_holdout = _make_candidates(y, p, I, t, fit_prefix, args.current_eps_A)
    rows_holdout = _rank_rows(_candidate_summary_rows(cands_holdout, y, I, t, masks_holdout, baseline_metrics_holdout))

    best_full = _select_best(rows_full)
    best_holdout = _select_best(rows_holdout)
    best_name = best_full.get("candidate") if best_full else "identity_d9_6_raw"
    best_cand = next((c for c in cands_full if c.name == best_name), cands_full[0])

    # Determine final verdict.
    if best_full is None:
        verdict = "no_safe_lightweight_correction_keep_battery8_flagged"
        final_action = "Keep battery-8 flagged/excluded. Do not change D9.6 mainline."
    else:
        holdout_same = best_holdout is not None and str(best_holdout.get("candidate")) == str(best_full.get("candidate")) and str(best_holdout.get("recommendation_class")) == "safe_lightweight_correction"
        if str(best_full.get("recommendation_class")) == "safe_lightweight_correction" and holdout_same:
            verdict = "safe_lightweight_correction_supported_by_fullfit_and_holdout"
            final_action = "Use this only as a flagged battery-8 correction wrapper; then run D10-P4 corrected battery-8 report."
        elif str(best_full.get("recommendation_class")) == "safe_lightweight_correction":
            verdict = "fullfit_correction_good_but_holdout_not_confirmed_calibration_only"
            final_action = "May record as battery-8 calibration benchmark, but keep battery-8 flagged for generalization claims."
        else:
            verdict = "weak_correction_only_keep_battery8_flagged"
            final_action = "Do not force battery-8 into the normal 24-profile mainline."

    # Save corrected NPZ.
    corrected_npz = out_dir / "d10_p3_battery8_corrected_prediction.npz"
    save_payload: dict[str, Any] = {}
    for k, v in original.items():
        arr = v
        if hasattr(arr, "shape") and arr.shape and arr.shape[0] == len(original.get("voltage_exp", [])):
            # Avoid trying to resize unrelated arrays; keep original as is unless it is the prediction key.
            pass
        save_payload[k] = arr
    save_payload["voltage_exp_pred_d9_6_raw"] = p
    save_payload["voltage_exp_pred_d10_p3_corrected"] = best_cand.corrected
    save_payload["d10_p3_selected_candidate"] = np.array([best_cand.name], dtype=object)
    save_payload["d10_p3_selected_params_json"] = np.array([json.dumps(best_cand.params, ensure_ascii=False)], dtype=object)
    np.savez_compressed(corrected_npz, **save_payload)

    _write_csv(out_dir / "d10_p3_candidate_metrics_fullfit.csv", rows_full)
    _write_csv(out_dir / "d10_p3_candidate_metrics_holdout.csv", rows_holdout)
    # Convenience combined top table.
    top_rows: list[dict[str, Any]] = []
    for r in rows_full:
        rr = dict(r)
        rr["evaluation_type"] = "fullfit_score_on_full_profile"
        top_rows.append(rr)
    for r in rows_holdout:
        rr = dict(r)
        rr["evaluation_type"] = "prefix_fit_score_on_holdout_tail"
        top_rows.append(rr)
    _write_csv(out_dir / "d10_p3_candidate_metrics.csv", top_rows)

    if args.make_plots:
        _plot_overlay(out_dir / "plots" / "d10_p3_voltage_overlay.png", t, y, p, best_cand.corrected, I, f"D10-P3 {best_cand.name}")
        _plot_residual(out_dir / "plots" / "d10_p3_residual_overlay.png", t, y, p, best_cand.corrected, f"D10-P3 residuals: {best_cand.name}")

    summary = {
        "ok": True,
        "stage": "D10-P3 battery-8 lightweight correction",
        "prediction_npz": str(pred_path),
        "pred_key": pred_key,
        "out_dir": str(out_dir),
        "verdict": verdict,
        "selected_candidate": best_cand.name,
        "selected_params": best_cand.params,
        "corrected_prediction_npz": str(corrected_npz),
        "recommended_next_action": final_action,
        "fit_policy": {
            "fullfit": "calibration benchmark on the whole battery-8 profile",
            "holdout": f"fit chronological prefix and score final {holdout_fraction:.2f} fraction",
            "no_strong_clamp": True,
            "does_not_modify_d9_6_mainline": True,
        },
        "best_fullfit_row": best_full,
        "best_holdout_row": best_holdout,
        "raw_fullfit_baseline": next((r for r in rows_full if r.get("candidate") == "identity_d9_6_raw"), None),
        "raw_holdout_baseline": next((r for r in rows_holdout if r.get("candidate") == "identity_d9_6_raw"), None),
        "outputs": {
            "candidate_metrics_fullfit_csv": str(out_dir / "d10_p3_candidate_metrics_fullfit.csv"),
            "candidate_metrics_holdout_csv": str(out_dir / "d10_p3_candidate_metrics_holdout.csv"),
            "candidate_metrics_combined_csv": str(out_dir / "d10_p3_candidate_metrics.csv"),
            "summary_json": str(out_dir / "d10_p3_lightweight_correction_summary.json"),
            "recommendation_md": str(out_dir / "D10_P3_RECOMMENDATION.md"),
        },
    }
    (out_dir / "d10_p3_lightweight_correction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable recommendation.
    raw_full = summary.get("raw_fullfit_baseline") or {}
    best_full_row = summary.get("best_fullfit_row") or {}
    best_hold_row = summary.get("best_holdout_row") or {}
    md = f"""# D10-P3 Battery-8 Lightweight Correction Recommendation

Verdict:

```text
{verdict}
```

Selected candidate:

```text
{best_cand.name}
```

Recommended next action:

```text
{final_action}
```

Key full-profile comparison:

| item | raw D9.6 | selected D10-P3 |
|---|---:|---:|
| overall MAE / V | {_safe_float(raw_full.get('mae_V')):.6f} | {_safe_float(best_full_row.get('mae_V')):.6f} |
| overall corr | {_safe_float(raw_full.get('corr')):.6f} | {_safe_float(best_full_row.get('corr')):.6f} |
| discharge MAE / V | {_safe_float(raw_full.get('discharge_mae_V')):.6f} | {_safe_float(best_full_row.get('discharge_mae_V')):.6f} |
| charge MAE / V | {_safe_float(raw_full.get('charge_mae_V')):.6f} | {_safe_float(best_full_row.get('charge_mae_V')):.6f} |
| pred max / V | {_safe_float(raw_full.get('pred_max_V')):.6f} | {_safe_float(best_full_row.get('pred_max_V')):.6f} |
| overshoot >4.35 frac | {_safe_float(raw_full.get('pred_overshoot_frac_gt_4p35')):.6f} | {_safe_float(best_full_row.get('pred_overshoot_frac_gt_4p35')):.6f} |

Holdout tail check:

```text
best_holdout_candidate = {best_hold_row.get('candidate') if best_hold_row else None}
best_holdout_class     = {best_hold_row.get('recommendation_class') if best_hold_row else None}
holdout_MAE_V          = {_safe_float(best_hold_row.get('mae_V') if best_hold_row else None):.6f}
holdout_discharge_MAE  = {_safe_float(best_hold_row.get('discharge_mae_V') if best_hold_row else None):.6f}
```

Interpretation:

- This is a battery-8 flagged-profile correction study only.
- It does not change `gv1/model.py`, `gv1/output_transform.py`, `gv1/losses.py`, `gv1/trainer.py`, or `scripts/gv1_train_conditioned_pinn.py`.
- It does not use a strong hard voltage clamp.
- If the full-fit correction is good but the holdout check does not confirm it, record it as a calibration benchmark rather than a generalization result.
"""
    (out_dir / "D10_P3_RECOMMENDATION.md").write_text(md, encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "verdict": verdict,
        "selected_candidate": best_cand.name,
        "out_dir": str(out_dir),
        "recommendation_md": str(out_dir / "D10_P3_RECOMMENDATION.md"),
        "summary_json": str(out_dir / "d10_p3_lightweight_correction_summary.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
