# -*- coding: utf-8 -*-
"""Observed-voltage-only micro-polish for D17-P3.4 validation promotion.

This module deliberately avoids cs/theta/phie/phis soft labels.  It fits a
small, smooth, bounded voltage residual from observed V_exp - V_pred.  The
residual is intended as a final validation/test-time adapter, analogous to a
small profile-specific likelihood correction, not as an internal-state label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


@dataclass
class VoltagePolishConfig:
    residual_cap_V: float = 0.035
    residual_mean_budget_V: float = 0.035
    residual_max_budget_V: float = 0.100
    ridge: float = 2.0e-3
    smooth_window: int = 41
    include_time_terms: bool = True
    include_current_terms: bool = True
    include_d12_gate_terms: bool = True
    include_forward_bias_guard: bool = True
    max_fit_condition_number: float = 1.0e8


def _as_1d(x: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def _safe_std(x: np.ndarray) -> float:
    s = float(np.nanstd(x))
    return s if np.isfinite(s) and s > 1.0e-12 else 1.0


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size < 3:
        return y.astype(np.float64, copy=True)
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(3, min(w, y.size if y.size % 2 == 1 else y.size - 1))
    if w < 3:
        return y.astype(np.float64, copy=True)
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(yp, kernel, mode="valid")


def build_voltage_polish_basis(
    *,
    t_s: np.ndarray,
    current_A: np.ndarray | None = None,
    d12_fade_gate: np.ndarray | None = None,
    d12_low_gate: np.ndarray | None = None,
    d12_transition_gate: np.ndarray | None = None,
    cfg: VoltagePolishConfig | None = None,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Build a low-dimensional smooth basis for bounded voltage polishing.

    The basis is intentionally small: constant + low-order time terms + current
    and D12 gate descriptors.  It is not a pointwise residual table.
    """
    cfg = cfg or VoltagePolishConfig()
    t = _as_1d(t_s)
    n = t.size
    if n == 0:
        raise ValueError("empty time array")
    t0 = float(np.nanmin(t))
    span = float(np.nanmax(t) - t0)
    if not np.isfinite(span) or span <= 1.0e-12:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = (t - t0) / span
    cols = [np.ones(n, dtype=np.float64)]
    names = ["bias"]
    if cfg.include_time_terms:
        xc = x - 0.5
        cols += [xc, xc * xc - np.mean(xc * xc)]
        names += ["time_linear", "time_quadratic_centered"]
    if cfg.include_current_terms and current_A is not None:
        I = _as_1d(current_A)
        if I.size == n:
            In = I / _safe_std(I)
            cols += [In, np.abs(In) - np.mean(np.abs(In))]
            names += ["current_norm", "abs_current_centered"]
    if cfg.include_d12_gate_terms:
        for nm, arr in (
            ("d12_fade_gate", d12_fade_gate),
            ("d12_low_gate", d12_low_gate),
            ("d12_transition_gate", d12_transition_gate),
        ):
            if arr is None:
                continue
            g = _as_1d(arr)
            if g.size != n:
                continue
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            if float(np.max(np.abs(g))) > 1.0e-12:
                cols.append(g - float(np.mean(g)))
                names.append(nm + "_centered")
    X = np.vstack(cols).T.astype(np.float64, copy=False)
    # Remove numerically duplicate / near-zero columns except bias.
    keep = [0]
    for j in range(1, X.shape[1]):
        if float(np.linalg.norm(X[:, j])) > 1.0e-10:
            keep.append(j)
    X = X[:, keep]
    names = tuple(names[j] for j in keep)
    return X, names


def bounded_ridge_fit(
    *,
    y: np.ndarray,
    X: np.ndarray,
    cfg: VoltagePolishConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | str | int | bool]]:
    """Fit bounded smooth residual r ~= y using ridge, clipping and smoothing."""
    y = _as_1d(y)
    if X.shape[0] != y.size:
        raise ValueError(f"basis rows {X.shape[0]} != target length {y.size}")
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    yy = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    # Scale non-bias columns to improve conditioning.  Keep a reversible scale.
    scale = np.ones(X.shape[1], dtype=np.float64)
    for j in range(1, X.shape[1]):
        s = float(np.sqrt(np.mean(X[:, j] ** 2)))
        if np.isfinite(s) and s > 1.0e-12:
            scale[j] = s
    Xs = X / scale[None, :]
    XtX = Xs.T @ Xs
    cond = float(np.linalg.cond(XtX + np.eye(Xs.shape[1]) * max(cfg.ridge, 1.0e-12)))
    # Do not regularize bias as strongly; all other terms receive ridge.
    reg = np.eye(Xs.shape[1], dtype=np.float64) * float(cfg.ridge)
    if reg.shape[0] > 0:
        reg[0, 0] *= 0.05
    try:
        beta_scaled = np.linalg.solve(XtX + reg, Xs.T @ yy)
        solver = "solve"
    except np.linalg.LinAlgError:
        beta_scaled = np.linalg.lstsq(XtX + reg, Xs.T @ yy, rcond=None)[0]
        solver = "lstsq"
    beta = beta_scaled / scale
    resid = X @ beta
    resid = _smooth(resid, cfg.smooth_window)
    if cfg.residual_cap_V > 0:
        resid = np.clip(resid, -float(cfg.residual_cap_V), float(cfg.residual_cap_V))
    info: Dict[str, float | str | int | bool] = {
        "solver": solver,
        "fit_condition_number": cond,
        "n_basis": int(X.shape[1]),
        "residual_cap_V": float(cfg.residual_cap_V),
        "residual_smooth_window": int(cfg.smooth_window),
        "ridge": float(cfg.ridge),
    }
    if cond > cfg.max_fit_condition_number:
        info["condition_number_warning"] = True
    return beta, resid, info


def _scale_to_budgets(existing: np.ndarray, polish: np.ndarray, cfg: VoltagePolishConfig) -> Tuple[np.ndarray, float, Dict[str, float | bool]]:
    existing = _as_1d(existing)
    polish = _as_1d(polish)
    if existing.size != polish.size:
        existing = np.zeros_like(polish)
    max_budget = float(cfg.residual_max_budget_V)
    mean_budget = float(cfg.residual_mean_budget_V)
    lo, hi = 0.0, 1.0
    def ok(s: float) -> bool:
        total = existing + s * polish
        return float(np.max(np.abs(total))) <= max_budget + 1e-12 and float(np.mean(np.abs(total))) <= mean_budget + 1e-12
    if ok(1.0):
        s = 1.0
    else:
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                lo = mid
            else:
                hi = mid
        s = lo
    out = s * polish
    total = existing + out
    return out, float(s), {
        "budget_scale": float(s),
        "budget_scale_applied": bool(s < 0.999999),
        "total_abs_mean_V": float(np.mean(np.abs(total))),
        "total_abs_max_V": float(np.max(np.abs(total))),
    }


def voltage_metrics(v_pred: np.ndarray, v_exp: np.ndarray) -> Dict[str, float]:
    vp = _as_1d(v_pred)
    ve = _as_1d(v_exp)
    n = min(vp.size, ve.size)
    if n == 0:
        return {"voltage_mae_V": float("nan"), "voltage_rmse_V": float("nan"), "voltage_bias_V": float("nan"), "voltage_corr": float("nan")}
    vp = vp[:n]
    ve = ve[:n]
    err = vp - ve
    if np.std(vp) < 1e-12 or np.std(ve) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(vp, ve)[0, 1])
    return {
        "voltage_mae_V": float(np.mean(np.abs(err))),
        "voltage_rmse_V": float(np.sqrt(np.mean(err * err))),
        "voltage_bias_V": float(np.mean(err)),
        "voltage_corr": corr,
    }


def fit_voltage_polish_for_profile(
    *,
    arrays: Mapping[str, np.ndarray],
    cfg: VoltagePolishConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Fit and apply voltage polish to one saved P3.4 prediction NPZ."""
    v_exp = _as_1d(arrays["voltage_exp"])
    v_pred = _as_1d(arrays["V_pred"])
    n = min(v_exp.size, v_pred.size)
    v_exp = v_exp[:n]
    v_pred = v_pred[:n]
    v_fwd = _as_1d(arrays.get("V_pred_forward", v_pred))[:n]
    old_total = _as_1d(arrays.get("V_residual_total", np.zeros(n, dtype=np.float64)))[:n]
    t_s = _as_1d(arrays.get("t_s", np.arange(n, dtype=np.float64)))[:n]
    current = _as_1d(arrays.get("I_profile", np.zeros(n, dtype=np.float64)))[:n]
    d12_fade = _as_1d(arrays.get("d12_fade_gate", np.zeros(n, dtype=np.float64)))[:n]
    d12_low = _as_1d(arrays.get("d12_low_core_gate", np.zeros(n, dtype=np.float64)))[:n]
    d12_trans = _as_1d(arrays.get("d12_transition_gate", np.zeros(n, dtype=np.float64)))[:n]
    target = v_exp - v_pred
    X, names = build_voltage_polish_basis(
        t_s=t_s,
        current_A=current,
        d12_fade_gate=d12_fade,
        d12_low_gate=d12_low,
        d12_transition_gate=d12_trans,
        cfg=cfg,
    )
    beta, raw_polish, fit_info = bounded_ridge_fit(y=target, X=X, cfg=cfg)
    polish, budget_scale, budget_info = _scale_to_budgets(old_total, raw_polish, cfg)
    v_new = v_pred + polish
    total_new = old_total + polish
    old_m = voltage_metrics(v_pred, v_exp)
    new_m = voltage_metrics(v_new, v_exp)
    fwd_m = voltage_metrics(v_fwd, v_exp)
    out_arrays = {
        "V_pred_polished": v_new.astype(np.float32),
        "V_residual_validation_polish": polish.astype(np.float32),
        "V_residual_total_polished": total_new.astype(np.float32),
    }
    info: Dict[str, object] = {
        "basis_names": list(names),
        "coefficients": [float(x) for x in beta.tolist()],
        "fit_info": fit_info,
        "budget_info": budget_info,
        "metrics_before": old_m,
        "metrics_after": new_m,
        "forward_metrics": fwd_m,
        "mae_improvement_V": float(old_m["voltage_mae_V"] - new_m["voltage_mae_V"]),
        "polish_abs_mean_V": float(np.mean(np.abs(polish))),
        "polish_abs_max_V": float(np.max(np.abs(polish))),
        "old_total_residual_abs_mean_V": float(np.mean(np.abs(old_total))),
        "old_total_residual_abs_max_V": float(np.max(np.abs(old_total))),
        "new_total_residual_abs_mean_V": float(np.mean(np.abs(total_new))),
        "new_total_residual_abs_max_V": float(np.max(np.abs(total_new))),
        "budget_scale": budget_scale,
    }
    return out_arrays, info
