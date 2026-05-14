#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ModelFin_107A anode cs_a/theta_a correction wrapper.

Purpose
-------
Use the already validated ModelFin_106 full-cycle raw evaluation arrays and
build a *post-hoc* anode-state correction wrapper:

    ModelFin_107A = ModelFin_106 + linear-cycle common-mode potential gauge
                            + anode cs_a residual correction

This script does not retrain the PINN weights. It only corrects cs_a/theta_a
outputs in the sampled evaluation arrays, leaving phie/phis_c/theta_c/cs_c
unchanged except for the existing ModelFin_106 potential gauge.

Default workflow
----------------
1. Read raw full-cycle evaluation arrays from:
   EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only/
2. Apply ModelFin_106 linear-cycle common-mode gauge to phie/phis_c.
3. Fit a compact ridge-regression residual corrector for cs_a using a chosen
   calibration cycle range.
4. Apply the correction to cycle5-522 and write all six corrected metrics.
5. Create ModelFin_107A as a wrapper directory containing best.pt, config.json,
   gauge_config.json, and anode_correction_config.json.

Important
---------
A full-cycle calibration/evaluation is a calibration benchmark, not an
independent extrapolation test. For stricter validation, run the accompanying
calib5_100_eval5_522 script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_VERSION = "ASSB_ModelFin107A_csA_anodeCorrection_v1"

RAW_EVAL_DEFAULT = Path("EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only")
OUT_EVAL_DEFAULT = Path("EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only")
MODEL106_DEFAULT = Path("ModelFin_106")
MODEL107_DEFAULT = Path("ModelFin_107A")


def _jfloat(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "variable", "cycle_id", "n", "mae", "rmse", "max_abs_error",
        "bias_mean", "corr", "r2", "nmae", "std_ratio_pred_over_label",
    ]
    keys: List[str] = []
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def _find_eval_npz(eval_dir: Path) -> Path:
    candidates = [
        "eval_sampled_arrays_cycles5_522_v2_massclosed_softlabel_only.npz",
        "eval_sampled_arrays_ModelFin106_linearGauge_corrected.npz",
        "eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz",
    ]
    for name in candidates:
        p = eval_dir / name
        if p.exists():
            return p
    hits = sorted(eval_dir.glob("eval_sampled_arrays*.npz"))
    if not hits:
        raise FileNotFoundError(f"No eval_sampled_arrays*.npz found in {eval_dir}")
    return hits[0]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size < 2 or np.std(y) <= 0 or np.std(p) <= 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0:
        return {
            "n": 0, "mae": None, "rmse": None, "max_abs_error": None,
            "bias_mean": None, "corr": None, "r2": None, "nmae": None,
            "std_ratio_pred_over_label": None,
        }
    e = p - y
    ss = float(np.sum((y - np.mean(y)) ** 2))
    rng = float(np.max(y) - np.min(y))
    ys = float(np.std(y)); ps = float(np.std(p))
    return {
        "n": int(y.size),
        "mae": _jfloat(np.mean(np.abs(e))),
        "rmse": _jfloat(np.sqrt(np.mean(e * e))),
        "max_abs_error": _jfloat(np.max(np.abs(e))),
        "bias_mean": _jfloat(np.mean(e)),
        "corr": _jfloat(_corr(y, p)),
        "r2": _jfloat(float("nan") if ss <= 0 else 1.0 - float(np.sum(e * e)) / ss),
        "nmae": _jfloat(float("nan") if rng <= 0 else float(np.mean(np.abs(e))) / rng),
        "std_ratio_pred_over_label": _jfloat(float("nan") if ys <= 0 else ps / ys),
    }


def _repeat_cycle_ids(cycle_time: np.ndarray, flat_len: int, nr: int) -> np.ndarray:
    c = np.asarray(cycle_time).reshape(-1)
    if c.size == flat_len:
        return c
    if c.size * nr == flat_len:
        return np.repeat(c, nr)
    raise ValueError(f"Cannot align cycle IDs: len(cycle_time)={c.size}, flat_len={flat_len}, nr={nr}")


def _align_time_r(arr: Mapping[str, np.ndarray], var: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if var not in ("cs_a", "cs_c", "theta_a", "theta_c"):
        raise ValueError(var)
    electrode = "a" if var.endswith("a") else "c"
    r = np.asarray(arr[f"r_{electrode}"], dtype=np.float64).reshape(-1)
    nr = r.size
    true = np.asarray(arr[f"{var}_true"], dtype=np.float64).reshape(-1)
    pred = np.asarray(arr[f"{var}_pred"], dtype=np.float64).reshape(-1)
    n_time = true.size // nr
    if n_time * nr != true.size or pred.size != true.size:
        raise ValueError(f"{var} size mismatch: true={true.size}, pred={pred.size}, nr={nr}")
    t_cs = np.asarray(arr["t_cs"], dtype=np.float64).reshape(-1)
    cyc_t = np.asarray(arr["cycle_id_cs"], dtype=np.int32).reshape(-1)
    if t_cs.size != n_time or cyc_t.size != n_time:
        # tolerate eval files that saved repeated cycle/time ids
        cyc = _repeat_cycle_ids(cyc_t, true.size, nr)
        t_rep = _repeat_cycle_ids(t_cs, true.size, nr)
    else:
        cyc = np.repeat(cyc_t, nr)
        t_rep = np.repeat(t_cs, nr)
    r_rep = np.tile(r, n_time)
    return true, pred, cyc, t_rep, r_rep


def _sphere_weights(r: np.ndarray) -> np.ndarray:
    """Return normalized trapezoidal r^2 weights for a 1D radial grid."""
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if r.size < 2:
        return np.ones_like(r, dtype=np.float64)
    dr = np.diff(r)
    w = np.zeros_like(r, dtype=np.float64)
    w[0] = dr[0] / 2.0
    w[-1] = dr[-1] / 2.0
    if r.size > 2:
        w[1:-1] = 0.5 * (dr[:-1] + dr[1:])
    w = w * r * r
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(r, dtype=np.float64) / float(r.size)
    return w / s


def _reshape_cs(arr: np.ndarray, nr: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    if a.size % nr != 0:
        raise ValueError(f"Array length {a.size} is not divisible by nr={nr}")
    return a.reshape(a.size // nr, nr)


def _infer_csmax_a(arr: Mapping[str, np.ndarray], default: float = 6.0) -> float:
    if "theta_a_true" in arr and np.asarray(arr["theta_a_true"]).size:
        cs = np.asarray(arr["cs_a_true"], dtype=np.float64).reshape(-1)
        th = np.asarray(arr["theta_a_true"], dtype=np.float64).reshape(-1)
        m = np.isfinite(cs) & np.isfinite(th) & (np.abs(th) > 1e-8)
        if np.any(m):
            ratios = cs[m] / th[m]
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                v = float(np.median(ratios))
                if np.isfinite(v) and v > 0:
                    return v
    return float(default)


def _phase_by_cycle(t_time: np.ndarray, cycle_time: np.ndarray) -> np.ndarray:
    t = np.asarray(t_time, dtype=np.float64).reshape(-1)
    c = np.asarray(cycle_time, dtype=np.int32).reshape(-1)
    out = np.zeros_like(t, dtype=np.float64)
    for cid in np.unique(c):
        m = c == cid
        if not np.any(m):
            continue
        lo = np.nanmin(t[m]); hi = np.nanmax(t[m])
        den = hi - lo
        out[m] = 0.0 if den <= 0 else (t[m] - lo) / den
    return out


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    Z = (X - mu) / sd
    Z[:, 0] = 1.0  # intercept is not standardized
    mu[0] = 0.0; sd[0] = 1.0
    return Z, mu, sd


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sd
    Z[:, 0] = 1.0
    return Z


def _build_features_flat(
    pred_flat: np.ndarray,
    cycle_flat: np.ndarray,
    t_flat: np.ndarray,
    r_flat: np.ndarray,
    r_grid: np.ndarray,
    cbar_pred_time: np.ndarray,
    cycle_from: int,
    cycle_to: int,
) -> Tuple[np.ndarray, List[str]]:
    """Feature matrix for residual correction y_true - y_pred.

    Uses only quantities available at prediction/evaluation time: raw pred,
    cycle_id, t_global_s, r, and cbar(pred). Phase is computed from t inside
    each cycle. This makes the correction physically interpretable as a smooth
    cycle/phase/radial residual map rather than a free per-point lookup table.
    """
    p = np.asarray(pred_flat, dtype=np.float64).reshape(-1)
    cyc = np.asarray(cycle_flat, dtype=np.float64).reshape(-1)
    t = np.asarray(t_flat, dtype=np.float64).reshape(-1)
    r = np.asarray(r_flat, dtype=np.float64).reshape(-1)
    r_grid = np.asarray(r_grid, dtype=np.float64).reshape(-1)
    nr = r_grid.size
    n_time = p.size // nr
    cbar_rep = np.repeat(np.asarray(cbar_pred_time, dtype=np.float64).reshape(-1), nr)
    rR = r / max(float(np.nanmax(r_grid)), 1e-30)
    r2 = rR * rR
    r4 = r2 * r2
    cycle_norm = (cyc - float(cycle_from)) / max(float(cycle_to - cycle_from), 1.0)
    # phase per cycle at time-row level, then repeat radially
    t_time = np.asarray(t[::nr], dtype=np.float64) if t.size == p.size else np.asarray(t[:n_time], dtype=np.float64)
    c_time = np.asarray(cyc[::nr], dtype=np.int32) if cyc.size == p.size else np.asarray(cyc[:n_time], dtype=np.int32)
    phase_time = _phase_by_cycle(t_time, c_time)
    phase = np.repeat(phase_time, nr)
    dev = p - cbar_rep
    s1 = np.sin(2.0 * np.pi * phase)
    c1 = np.cos(2.0 * np.pi * phase)
    s2 = np.sin(4.0 * np.pi * phase)
    c2 = np.cos(4.0 * np.pi * phase)
    cols: List[np.ndarray] = [
        np.ones_like(p),
        cycle_norm, cycle_norm**2, cycle_norm**3,
        phase, phase**2, phase**3,
        s1, c1, s2, c2,
        rR, r2, r4,
        p, cbar_rep, dev,
        cycle_norm * phase,
        cycle_norm * r2,
        cycle_norm**2 * r2,
        phase * r2,
        phase**2 * r2,
        s1 * r2, c1 * r2,
        cycle_norm * dev,
        phase * dev,
        r2 * dev,
        cycle_norm * phase * dev,
        cycle_norm * r2 * dev,
        phase * r2 * dev,
    ]
    names = [
        "1",
        "cycle", "cycle2", "cycle3",
        "phase", "phase2", "phase3",
        "sin_phase", "cos_phase", "sin_2phase", "cos_2phase",
        "rR", "r2", "r4",
        "cs_pred", "cbar_pred", "radial_dev_pred",
        "cycle_phase",
        "cycle_r2",
        "cycle2_r2",
        "phase_r2",
        "phase2_r2",
        "sin_phase_r2", "cos_phase_r2",
        "cycle_dev",
        "phase_dev",
        "r2_dev",
        "cycle_phase_dev",
        "cycle_r2_dev",
        "phase_r2_dev",
    ]
    X = np.column_stack(cols)
    return X, names


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lam = float(ridge)
    A = X.T @ X
    if lam > 0:
        R = np.eye(A.shape[0], dtype=np.float64) * lam
        R[0, 0] = 0.0  # do not penalize intercept
        A = A + R
    b = X.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _subsample_fit_indices(mask: np.ndarray, max_fit_points: int, seed: int) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if max_fit_points is None or int(max_fit_points) <= 0 or idx.size <= int(max_fit_points):
        return idx
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(idx, size=int(max_fit_points), replace=False))


def _apply_linear_cycle_gauge(arr: Mapping[str, np.ndarray], model106_dir: Path) -> Dict[str, np.ndarray]:
    out = {k: np.array(v, copy=True) for k, v in arr.items()}
    gauge_path = model106_dir / "gauge_config.json"
    if not gauge_path.exists():
        # Keep raw potentials if ModelFin_106 gauge does not exist.
        return out
    gauge = _read_json(gauge_path)
    slope = float(gauge.get("linear_bias_slope_V_per_cycle", 0.0))
    intercept = float(gauge.get("linear_bias_intercept_V", 0.0))
    cid = np.asarray(out["cycle_id_potential"], dtype=np.float64).reshape(-1)
    offset = -(slope * cid + intercept)
    for key in ["phie_pred", "phis_c_pred"]:
        if key in out:
            shape = out[key].shape
            out[key] = (out[key].reshape(-1).astype(np.float64) + offset).reshape(shape).astype(np.float32)
    out["potential_common_mode_offset_to_add"] = offset.astype(np.float32)
    return out


def _variables(arr: Mapping[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cid_p = np.asarray(arr["cycle_id_potential"], dtype=np.int32).reshape(-1)
    d: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "phis_c": (arr["phis_c_true"].reshape(-1), arr["phis_c_pred"].reshape(-1), cid_p),
        "phie": (arr["phie_true"].reshape(-1), arr["phie_pred"].reshape(-1), cid_p),
    }
    cid_cs = np.asarray(arr["cycle_id_cs"], dtype=np.int32).reshape(-1)
    nr_a = int(np.asarray(arr["r_a"]).reshape(-1).size)
    nr_c = int(np.asarray(arr["r_c"]).reshape(-1).size)
    for var, nr in [("theta_a", nr_a), ("theta_c", nr_c), ("cs_a", nr_a), ("cs_c", nr_c)]:
        if f"{var}_true" not in arr or np.asarray(arr[f"{var}_true"]).size == 0:
            continue
        yt = np.asarray(arr[f"{var}_true"]).reshape(-1)
        yp = np.asarray(arr[f"{var}_pred"]).reshape(-1)
        d[var] = (yt, yp, _repeat_cycle_ids(cid_cs, yt.size, nr))
    return d


def _compute_metrics(arr: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    glob: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for var, (yt, yp, cid) in _variables(arr).items():
        mask_all = (cid >= cycle_from) & (cid <= cycle_to)
        glob[var] = _metrics(yt[mask_all], yp[mask_all])
        for c in range(cycle_from, cycle_to + 1):
            m = mask_all & (cid == c)
            if not np.any(m):
                continue
            row = {"variable": var, "cycle_id": int(c)}
            row.update(_metrics(yt[m], yp[m]))
            rows.append(row)
    return glob, rows


def _common_diag(arr_before: Mapping[str, np.ndarray], arr_after: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> Dict[str, Any]:
    cid = np.asarray(arr_before["cycle_id_potential"], dtype=np.int32).reshape(-1)
    m = (cid >= cycle_from) & (cid <= cycle_to)

    def stat(x: np.ndarray) -> Dict[str, Any]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)[m]
        return {"n": int(x.size), "mae": _jfloat(np.mean(np.abs(x))), "rmse": _jfloat(np.sqrt(np.mean(x*x))), "bias_mean": _jfloat(np.mean(x)), "std": _jfloat(np.std(x))}

    def errs(a: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phie = np.asarray(a["phie_pred"], dtype=np.float64).reshape(-1) - np.asarray(a["phie_true"], dtype=np.float64).reshape(-1)
        phis = np.asarray(a["phis_c_pred"], dtype=np.float64).reshape(-1) - np.asarray(a["phis_c_true"], dtype=np.float64).reshape(-1)
        cm = 0.5 * (phie + phis)
        diff = phis - phie
        return phie, phis, cm, diff

    labels = ["phie_error", "phis_c_error", "common_mode_error", "differential_phis_minus_phie_error"]
    b = errs(arr_before); a = errs(arr_after)
    return {"before": {k: stat(v) for k, v in zip(labels, b)}, "after": {k: stat(v) for k, v in zip(labels, a)}}


def _make_plots(out_dir: Path, rows_before: List[Dict[str, Any]], rows_after: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    pdir = out_dir / "plots_csA_corrected"
    pdir.mkdir(parents=True, exist_ok=True)
    for var in ["theta_a", "cs_a", "phis_c", "phie", "theta_c", "cs_c"]:
        rb = [r for r in rows_before if r.get("variable") == var]
        ra = [r for r in rows_after if r.get("variable") == var]
        if not rb or not ra:
            continue
        xb = [int(r["cycle_id"]) for r in rb]; yb = [float(r["mae"]) for r in rb]
        xa = [int(r["cycle_id"]) for r in ra]; ya = [float(r["mae"]) for r in ra]
        plt.figure(figsize=(9.5, 4.5))
        plt.plot(xb, yb, label="ModelFin_106")
        plt.plot(xa, ya, label="ModelFin_107A")
        plt.xlabel("cycle_id")
        plt.ylabel(f"{var} MAE")
        plt.title(f"{var} per-cycle MAE: before vs cs_a correction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(pdir / f"per_cycle_mae_{var}_106_vs_107A.png", dpi=170)
        plt.close()


def _build_model107_dir(model106_dir: Path, model107_dir: Path, corr_config: Mapping[str, Any], out_eval_dir: Path) -> None:
    model107_dir.mkdir(parents=True, exist_ok=True)
    for name in ["best.pt", "config.json", "gauge_config.json"]:
        src = model106_dir / name
        if src.exists():
            shutil.copy2(src, model107_dir / name)
    # Update config.json with wrapper metadata when possible.
    config_path = model107_dir / "config.json"
    cfg: Dict[str, Any] = {}
    if config_path.exists():
        try:
            cfg = _read_json(config_path)
        except Exception:
            cfg = {}
    cfg.update({
        "ASSB_MODELFIN_WRAPPER_ID": 107,
        "ASSB_MODELFIN_WRAPPER_NAME": "ModelFin_107A_csA_anodeStateCorrection",
        "ASSB_WRAPPER_BASE_MODEL": "ModelFin_106",
        "ASSB_ANODE_CS_A_CORRECTION_MODEL": corr_config.get("method"),
        "ASSB_ANODE_CS_A_CORRECTION_CONFIG": "anode_correction_config.json",
    })
    _write_json(config_path, cfg)
    _write_json(model107_dir / "anode_correction_config.json", dict(corr_config))
    card = f"""# ModelFin_107A — anode cs_a/theta_a correction wrapper

ModelFin_107A is a post-hoc wrapper around ModelFin_106.

- Base model weights: `ModelFin_106/best.pt`
- Existing potential correction: linear-cycle common-mode gauge from ModelFin_106
- New correction: smooth residual correction for `cs_a`; `theta_a` is recomputed as `cs_a / cs_a,max`
- Unchanged variables: `phis_c`, `phie`, `theta_c`, `cs_c`
- Evaluation output: `{out_eval_dir}`

This wrapper is intended to test whether the remaining full-cycle error is a systematic anode-state residual rather than a failure of the positive-electrode or potential branches.
"""
    (model107_dir / "MODEL_CARD_ModelFin107A_csA_correction.md").write_text(card, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/apply ModelFin_107A cs_a anode-state correction wrapper and evaluate full cycles.")
    p.add_argument("--raw_eval_dir", type=Path, default=RAW_EVAL_DEFAULT, help="ModelFin_106 raw full-cycle eval directory containing eval_sampled_arrays*.npz")
    p.add_argument("--model106_dir", type=Path, default=MODEL106_DEFAULT, help="ModelFin_106 directory with gauge_config.json")
    p.add_argument("--model107_dir", type=Path, default=MODEL107_DEFAULT, help="Output ModelFin_107A wrapper directory")
    p.add_argument("--output_dir", type=Path, default=OUT_EVAL_DEFAULT, help="Output evaluation directory")
    p.add_argument("--calib_cycle_from", type=int, default=5, help="First cycle used to fit cs_a correction")
    p.add_argument("--calib_cycle_to", type=int, default=522, help="Last cycle used to fit cs_a correction")
    p.add_argument("--eval_cycle_from", type=int, default=5, help="First cycle evaluated")
    p.add_argument("--eval_cycle_to", type=int, default=522, help="Last cycle evaluated")
    p.add_argument("--ridge", type=float, default=1e-6, help="Ridge regularization strength for standardized features")
    p.add_argument("--max_fit_points", type=int, default=350000, help="Max flattened cs_a points for fitting; <=0 uses all")
    p.add_argument("--seed", type=int, default=107, help="Random seed for fitting subsample")
    p.add_argument("--csmax_a", type=float, default=0.0, help="cs_a max scale. <=0: infer from theta_a arrays, fallback 6.0")
    p.add_argument("--clip_min", type=float, default=0.0, help="Minimum allowed corrected cs_a")
    p.add_argument("--clip_max", type=float, default=6.0, help="Maximum allowed corrected cs_a; <=0 disables max clipping")
    p.add_argument("--no_model_dir", action="store_true", help="Do not create/update ModelFin_107A directory")
    p.add_argument("--no_plots", action="store_true", help="Skip plots")
    p.add_argument("--save_npz", action="store_true", help="Save corrected sampled arrays npz")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    raw_eval_dir = args.raw_eval_dir.resolve()
    model106_dir = args.model106_dir.resolve()
    model107_dir = args.model107_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = _find_eval_npz(raw_eval_dir)
    raw_arr = _load_npz(npz_path)
    gauge_arr = _apply_linear_cycle_gauge(raw_arr, model106_dir)
    corr_arr = {k: np.array(v, copy=True) for k, v in gauge_arr.items()}

    r_a = np.asarray(raw_arr["r_a"], dtype=np.float64).reshape(-1)
    nr_a = r_a.size
    w_a = _sphere_weights(r_a)
    cs_true_2d = _reshape_cs(raw_arr["cs_a_true"], nr_a)
    cs_pred_2d = _reshape_cs(raw_arr["cs_a_pred"], nr_a)
    n_time = cs_true_2d.shape[0]
    cycle_time = np.asarray(raw_arr["cycle_id_cs"], dtype=np.int32).reshape(-1)
    t_time = np.asarray(raw_arr["t_cs"], dtype=np.float64).reshape(-1)
    if cycle_time.size != n_time or t_time.size != n_time:
        raise ValueError(f"Expected cycle_id_cs/t_cs length {n_time}; got {cycle_time.size}/{t_time.size}")

    cbar_pred = cs_pred_2d @ w_a
    cs_true_flat = cs_true_2d.reshape(-1)
    cs_pred_flat = cs_pred_2d.reshape(-1)
    cycle_flat = np.repeat(cycle_time, nr_a)
    t_flat = np.repeat(t_time, nr_a)
    r_flat = np.tile(r_a, n_time)

    eval_mask = (cycle_flat >= args.eval_cycle_from) & (cycle_flat <= args.eval_cycle_to)
    calib_mask = (cycle_flat >= args.calib_cycle_from) & (cycle_flat <= args.calib_cycle_to)
    finite_mask = np.isfinite(cs_true_flat) & np.isfinite(cs_pred_flat)
    calib_mask = calib_mask & finite_mask

    if not np.any(calib_mask):
        raise RuntimeError("No finite calibration points found. Check calib_cycle_from/to and eval arrays.")
    fit_idx = _subsample_fit_indices(calib_mask, int(args.max_fit_points), int(args.seed))

    X_all, feature_names = _build_features_flat(
        pred_flat=cs_pred_flat,
        cycle_flat=cycle_flat,
        t_flat=t_flat,
        r_flat=r_flat,
        r_grid=r_a,
        cbar_pred_time=cbar_pred,
        cycle_from=args.eval_cycle_from,
        cycle_to=args.eval_cycle_to,
    )
    y_resid = cs_true_flat - cs_pred_flat
    X_fit = X_all[fit_idx]
    y_fit = y_resid[fit_idx]
    Z_fit, mu, sd = _standardize_fit(X_fit)
    coef = _fit_ridge(Z_fit, y_fit, float(args.ridge))

    # Apply in chunks to avoid high memory use if the user evaluates all cs rows.
    correction = np.zeros_like(cs_pred_flat, dtype=np.float64)
    chunk = 500_000
    for lo in range(0, X_all.shape[0], chunk):
        hi = min(lo + chunk, X_all.shape[0])
        Z = _standardize_apply(X_all[lo:hi], mu, sd)
        correction[lo:hi] = Z @ coef

    cs_corr_flat = cs_pred_flat + correction
    if args.clip_min is not None:
        cs_corr_flat = np.maximum(cs_corr_flat, float(args.clip_min))
    if args.clip_max is not None and float(args.clip_max) > 0:
        cs_corr_flat = np.minimum(cs_corr_flat, float(args.clip_max))

    csmax_a = float(args.csmax_a) if float(args.csmax_a) > 0 else _infer_csmax_a(raw_arr, default=6.0)
    theta_corr_flat = cs_corr_flat / csmax_a

    corr_arr["cs_a_pred_before_anodeCorrection"] = np.asarray(corr_arr["cs_a_pred"], dtype=np.float32)
    corr_arr["theta_a_pred_before_anodeCorrection"] = np.asarray(corr_arr.get("theta_a_pred", np.array([], dtype=np.float32)), dtype=np.float32)
    corr_arr["cs_a_pred"] = cs_corr_flat.astype(np.float32)
    corr_arr["theta_a_pred"] = theta_corr_flat.astype(np.float32)
    corr_arr["cs_a_anode_correction_to_add"] = correction.astype(np.float32)

    glob_before, rows_before = _compute_metrics(gauge_arr, args.eval_cycle_from, args.eval_cycle_to)
    glob_after, rows_after = _compute_metrics(corr_arr, args.eval_cycle_from, args.eval_cycle_to)
    common = _common_diag(gauge_arr, corr_arr, args.eval_cycle_from, args.eval_cycle_to)
    calib_before = _metrics(cs_true_flat[calib_mask], cs_pred_flat[calib_mask])
    calib_after = _metrics(cs_true_flat[calib_mask], cs_corr_flat[calib_mask])

    corr_config: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "method": "ridge_residual_cycle_phase_radial_cs_pred",
        "base_model": "ModelFin_106",
        "wrapper_model": "ModelFin_107A",
        "raw_eval_npz": str(npz_path),
        "calib_cycle_from": int(args.calib_cycle_from),
        "calib_cycle_to": int(args.calib_cycle_to),
        "eval_cycle_from": int(args.eval_cycle_from),
        "eval_cycle_to": int(args.eval_cycle_to),
        "max_fit_points": int(args.max_fit_points),
        "fit_points_used": int(fit_idx.size),
        "ridge": float(args.ridge),
        "seed": int(args.seed),
        "csmax_a": csmax_a,
        "clip_min": _jfloat(args.clip_min),
        "clip_max": _jfloat(args.clip_max),
        "feature_names": feature_names,
        "feature_mean": mu.tolist(),
        "feature_scale": sd.tolist(),
        "coefficients": coef.tolist(),
        "calibration_cs_a_before": calib_before,
        "calibration_cs_a_after": calib_after,
        "global_before": glob_before,
        "global_after": glob_after,
        "notes": (
            "The correction is fitted to cs_a residuals only. phie/phis_c use the existing "
            "ModelFin_106 linear-cycle common-mode gauge. theta_c/cs_c are not modified. "
            "theta_a is recomputed from corrected cs_a/csmax_a."
        ),
    }

    _write_json(output_dir / "anode_correction_config.json", corr_config)
    _write_json(output_dir / "metrics_global_before_ModelFin106.json", glob_before)
    _write_json(output_dir / "metrics_global_corrected.json", glob_after)
    _write_json(output_dir / "potential_common_mode_diagnostic_before_after.json", common)
    _write_csv(output_dir / "metrics_by_cycle_before_ModelFin106.csv", rows_before)
    _write_csv(output_dir / "metrics_by_cycle_corrected.csv", rows_after)
    if args.save_npz:
        np.savez_compressed(output_dir / "eval_sampled_arrays_ModelFin107A_csA_corrected.npz", **corr_arr)
    if not args.no_plots:
        _make_plots(output_dir, rows_before, rows_after)
    if not args.no_model_dir:
        _build_model107_dir(model106_dir, model107_dir, corr_config, output_dir)

    print(json.dumps({
        "script_version": SCRIPT_VERSION,
        "raw_eval_npz": str(npz_path),
        "model107_dir": None if args.no_model_dir else str(model107_dir),
        "output_dir": str(output_dir),
        "calib_cycles": [args.calib_cycle_from, args.calib_cycle_to],
        "eval_cycles": [args.eval_cycle_from, args.eval_cycle_to],
        "cs_a_mae_before": glob_before.get("cs_a", {}).get("mae"),
        "cs_a_mae_after": glob_after.get("cs_a", {}).get("mae"),
        "cs_a_r2_before": glob_before.get("cs_a", {}).get("r2"),
        "cs_a_r2_after": glob_after.get("cs_a", {}).get("r2"),
        "theta_a_mae_before": glob_before.get("theta_a", {}).get("mae"),
        "theta_a_mae_after": glob_after.get("theta_a", {}).get("mae"),
        "theta_a_r2_before": glob_before.get("theta_a", {}).get("r2"),
        "theta_a_r2_after": glob_after.get("theta_a", {}).get("r2"),
        "phis_c_r2_after": glob_after.get("phis_c", {}).get("r2"),
        "phie_r2_after": glob_after.get("phie", {}).get("r2"),
        "theta_c_r2_after": glob_after.get("theta_c", {}).get("r2"),
        "cs_c_r2_after": glob_after.get("cs_c", {}).get("r2"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
