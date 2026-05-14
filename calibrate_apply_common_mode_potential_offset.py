#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
D4 potential gauge / common-mode offset calibration for ASSB PINN evaluations.

Purpose
-------
Post-process an existing soft-label-only evaluation npz, estimate the common-mode
potential bias from a calibration cycle slice (default: cycle 5-20), and apply the
same gauge correction to both phie_pred and phis_c_pred for cycle 5-100.

This does NOT retrain the model and does NOT modify ModelFin_* weights.
It is intended as a diagnostic/calibration step after ModelFin_105, where phie and
phis_c have nearly correct differential dynamics but share a common negative bias.

Default input directory:
  EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only

Default output directory:
  EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_commonModeCorrected

Typical command:
  python .\calibrate_apply_common_mode_potential_offset.py ^
    --eval_dir .\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only ^
    --output_dir .\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_commonModeCorrected ^
    --calib_cycle_from 5 --calib_cycle_to 20 ^
    --apply_cycle_from 5 --apply_cycle_to 100 ^
    --method constant_mean
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np


def _np_float(x: Any) -> Optional[float]:
    """Convert a scalar to a JSON-safe float, preserving NaN as None."""
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _find_eval_npz(eval_dir: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = eval_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Explicit eval npz not found: {p}")
        return p

    preferred = [
        "eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz",
        "eval_sampled_arrays_cycles5_100_softlabel_only.npz",
    ]
    for name in preferred:
        p = eval_dir / name
        if p.exists():
            return p

    candidates = sorted(eval_dir.glob("eval_sampled_arrays*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No eval_sampled_arrays*.npz found in {eval_dir}. "
            "Run the cycle5-100 evaluation script first."
        )
    return candidates[0]


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if yt.size < 2:
        return float("nan")
    s1 = float(np.std(yt))
    s2 = float(np.std(yp))
    if s1 <= 0.0 or s2 <= 0.0:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    n = int(yt.size)
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "max_abs_error": None,
            "bias_mean": None,
            "corr": None,
            "r2": None,
            "nmae": None,
            "std_ratio_pred_over_label": None,
        }
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    max_abs = float(np.max(np.abs(err)))
    bias = float(np.mean(err))
    corr = _safe_corr(yt, yp)
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float("nan") if denom <= 0 else float(1.0 - np.sum(err * err) / denom)
    y_range = float(np.max(yt) - np.min(yt))
    nmae = float("nan") if y_range <= 0 else float(mae / y_range)
    y_std = float(np.std(yt))
    p_std = float(np.std(yp))
    std_ratio = float("nan") if y_std <= 0 else float(p_std / y_std)
    return {
        "n": n,
        "mae": _np_float(mae),
        "rmse": _np_float(rmse),
        "max_abs_error": _np_float(max_abs),
        "bias_mean": _np_float(bias),
        "corr": _np_float(corr),
        "r2": _np_float(r2),
        "nmae": _np_float(nmae),
        "std_ratio_pred_over_label": _np_float(std_ratio),
    }


def _repeat_cycle_ids(cycle_id_time: np.ndarray, flat_len: int, nr: int) -> np.ndarray:
    cycle_id_time = np.asarray(cycle_id_time).reshape(-1)
    if cycle_id_time.size * nr == flat_len:
        return np.repeat(cycle_id_time, nr)
    if cycle_id_time.size == flat_len:
        return cycle_id_time
    raise ValueError(
        f"Cannot align cycle ids: len(cycle_id_time)={cycle_id_time.size}, "
        f"nr={nr}, flat_len={flat_len}."
    )


def _variable_arrays(arr: Mapping[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return variable -> (true, pred, cycle_id_flat) for all six variables."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # Potentials are sampled on the potential grid.
    cid_p = np.asarray(arr["cycle_id_potential"]).reshape(-1)
    out["phis_c"] = (arr["phis_c_true"].reshape(-1), arr["phis_c_pred"].reshape(-1), cid_p)
    out["phie"] = (arr["phie_true"].reshape(-1), arr["phie_pred"].reshape(-1), cid_p)

    # Concentrations/thetas are often flattened over (time, radius).
    cid_cs = np.asarray(arr["cycle_id_cs"]).reshape(-1)
    nr_a = int(np.asarray(arr["r_a"]).reshape(-1).size)
    nr_c = int(np.asarray(arr["r_c"]).reshape(-1).size)

    for var, nr in [("cs_a", nr_a), ("theta_a", nr_a), ("cs_c", nr_c), ("theta_c", nr_c)]:
        yt = arr[f"{var}_true"].reshape(-1)
        yp = arr[f"{var}_pred"].reshape(-1)
        cid = _repeat_cycle_ids(cid_cs, yt.size, nr)
        out[var] = (yt, yp, cid)
    return out


def _compute_global_by_cycle(arr: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    variables = _variable_arrays(arr)
    global_metrics: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for var, (yt, yp, cid) in variables.items():
        mask_apply = (cid >= cycle_from) & (cid <= cycle_to)
        global_metrics[var] = _metrics(yt[mask_apply], yp[mask_apply])
        for cyc in range(cycle_from, cycle_to + 1):
            m = mask_apply & (cid == cyc)
            if not np.any(m):
                continue
            met = _metrics(yt[m], yp[m])
            row = {"variable": var, "cycle_id": int(cyc)}
            row.update(met)
            rows.append(row)
    return global_metrics, rows


def _cycle_common_error_stats(arr: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> List[Dict[str, Any]]:
    cid = np.asarray(arr["cycle_id_potential"]).reshape(-1)
    phie_err = arr["phie_pred"].reshape(-1).astype(np.float64) - arr["phie_true"].reshape(-1).astype(np.float64)
    phis_err = arr["phis_c_pred"].reshape(-1).astype(np.float64) - arr["phis_c_true"].reshape(-1).astype(np.float64)
    cm = 0.5 * (phie_err + phis_err)
    diff = phis_err - phie_err
    rows: List[Dict[str, Any]] = []
    for cyc in range(cycle_from, cycle_to + 1):
        m = (cid == cyc)
        if not np.any(m):
            continue
        rows.append({
            "cycle_id": int(cyc),
            "n": int(np.sum(m)),
            "common_mode_error_mean_before": _np_float(np.mean(cm[m])),
            "common_mode_error_median_before": _np_float(np.median(cm[m])),
            "common_mode_error_mae_before": _np_float(np.mean(np.abs(cm[m]))),
            "common_mode_error_std_before": _np_float(np.std(cm[m])),
            "differential_error_mean_before": _np_float(np.mean(diff[m])),
            "differential_error_mae_before": _np_float(np.mean(np.abs(diff[m]))),
        })
    return rows


def _estimate_offset_by_cycle(
    arr: Mapping[str, np.ndarray],
    calib_from: int,
    calib_to: int,
    apply_from: int,
    apply_to: int,
    method: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return offset_to_add for each potential sample and calibration summary."""
    cid = np.asarray(arr["cycle_id_potential"]).reshape(-1).astype(int)
    phie_err = arr["phie_pred"].reshape(-1).astype(np.float64) - arr["phie_true"].reshape(-1).astype(np.float64)
    phis_err = arr["phis_c_pred"].reshape(-1).astype(np.float64) - arr["phis_c_true"].reshape(-1).astype(np.float64)
    cm_err = 0.5 * (phie_err + phis_err)

    calib_mask = (cid >= calib_from) & (cid <= calib_to) & np.isfinite(cm_err)
    if not np.any(calib_mask):
        raise ValueError(f"No calibration samples found for cycles {calib_from}-{calib_to}.")

    summary: Dict[str, Any] = {
        "method": method,
        "calib_cycle_from": int(calib_from),
        "calib_cycle_to": int(calib_to),
        "apply_cycle_from": int(apply_from),
        "apply_cycle_to": int(apply_to),
        "calibration_n": int(np.sum(calib_mask)),
        "calibration_common_mode_error_mean": _np_float(np.mean(cm_err[calib_mask])),
        "calibration_common_mode_error_median": _np_float(np.median(cm_err[calib_mask])),
        "calibration_common_mode_error_std": _np_float(np.std(cm_err[calib_mask])),
    }

    offset = np.zeros_like(cm_err, dtype=np.float64)

    if method == "constant_mean":
        bias = float(np.mean(cm_err[calib_mask]))
        offset_to_add = -bias
        offset[:] = offset_to_add
        summary.update({
            "offset_model": "constant",
            "constant_bias_estimate": _np_float(bias),
            "constant_offset_to_add_V": _np_float(offset_to_add),
        })
    elif method == "constant_median":
        bias = float(np.median(cm_err[calib_mask]))
        offset_to_add = -bias
        offset[:] = offset_to_add
        summary.update({
            "offset_model": "constant",
            "constant_bias_estimate": _np_float(bias),
            "constant_offset_to_add_V": _np_float(offset_to_add),
        })
    elif method == "linear_cycle_mean":
        cycles: List[int] = []
        means: List[float] = []
        for cyc in range(calib_from, calib_to + 1):
            m = (cid == cyc) & np.isfinite(cm_err)
            if np.any(m):
                cycles.append(int(cyc))
                means.append(float(np.mean(cm_err[m])))
        if len(cycles) < 2:
            raise ValueError("linear_cycle_mean requires at least two calibration cycles.")
        coeff = np.polyfit(np.asarray(cycles, dtype=np.float64), np.asarray(means, dtype=np.float64), deg=1)
        slope, intercept = float(coeff[0]), float(coeff[1])
        pred_bias = slope * cid.astype(np.float64) + intercept
        offset[:] = -pred_bias
        summary.update({
            "offset_model": "linear_cycle_mean",
            "linear_bias_slope_V_per_cycle": _np_float(slope),
            "linear_bias_intercept_V": _np_float(intercept),
            "calibration_cycle_mean_biases": [
                {"cycle_id": c, "common_mode_error_mean": _np_float(v)} for c, v in zip(cycles, means)
            ],
        })
    else:
        raise ValueError(f"Unknown method: {method}")

    apply_mask = (cid >= apply_from) & (cid <= apply_to)
    offset[~apply_mask] = 0.0
    summary["offset_to_add_mean_over_apply_samples_V"] = _np_float(np.mean(offset[apply_mask])) if np.any(apply_mask) else None
    summary["offset_to_add_min_over_apply_samples_V"] = _np_float(np.min(offset[apply_mask])) if np.any(apply_mask) else None
    summary["offset_to_add_max_over_apply_samples_V"] = _np_float(np.max(offset[apply_mask])) if np.any(apply_mask) else None
    return offset, summary


def _apply_offset(arr: Mapping[str, np.ndarray], offset: np.ndarray) -> Dict[str, np.ndarray]:
    corrected: Dict[str, np.ndarray] = {k: np.array(v, copy=True) for k, v in arr.items()}
    for key in ["phie_pred", "phis_c_pred"]:
        y = corrected[key].reshape(-1).astype(np.float64)
        y_corr = y + offset
        corrected[key] = y_corr.astype(corrected[key].dtype).reshape(corrected[key].shape)
    return corrected


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    # Stable column order, with any extras appended.
    preferred = [
        "variable", "cycle_id", "n", "mae", "rmse", "max_abs_error", "bias_mean",
        "corr", "r2", "nmae", "std_ratio_pred_over_label",
        "common_mode_error_mean_before", "common_mode_error_median_before",
        "common_mode_error_mae_before", "common_mode_error_std_before",
        "common_mode_error_mean_after", "common_mode_error_median_after",
        "common_mode_error_mae_after", "common_mode_error_std_after",
        "differential_error_mean_before", "differential_error_mae_before",
        "differential_error_mean_after", "differential_error_mae_after",
    ]
    keys = []
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def _common_diag(arr_before: Mapping[str, np.ndarray], arr_after: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> Dict[str, Any]:
    cid = np.asarray(arr_before["cycle_id_potential"]).reshape(-1)
    apply_mask = (cid >= cycle_from) & (cid <= cycle_to)

    def _errors(arr: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        phie_err = arr["phie_pred"].reshape(-1).astype(np.float64) - arr["phie_true"].reshape(-1).astype(np.float64)
        phis_err = arr["phis_c_pred"].reshape(-1).astype(np.float64) - arr["phis_c_true"].reshape(-1).astype(np.float64)
        return {
            "phie_error": phie_err,
            "phis_c_error": phis_err,
            "common_mode_error": 0.5 * (phie_err + phis_err),
            "differential_phis_minus_phie_error": phis_err - phie_err,
        }

    def _one(e: np.ndarray) -> Dict[str, Any]:
        e = e[apply_mask]
        return {
            "n": int(e.size),
            "mae": _np_float(np.mean(np.abs(e))) if e.size else None,
            "rmse": _np_float(np.sqrt(np.mean(e * e))) if e.size else None,
            "bias_mean": _np_float(np.mean(e)) if e.size else None,
            "std": _np_float(np.std(e)) if e.size else None,
        }

    eb = _errors(arr_before)
    ea = _errors(arr_after)
    return {
        "cycle_filter": f"{cycle_from}-{cycle_to}",
        "before": {k: _one(v) for k, v in eb.items()},
        "after": {k: _one(v) for k, v in ea.items()},
        "interpretation_hint": (
            "Gauge correction should reduce phie/phis_c common-mode error while leaving "
            "the differential phis_c-phie error nearly unchanged."
        ),
    }


def _per_cycle_common_before_after(arr_before: Mapping[str, np.ndarray], arr_after: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int) -> List[Dict[str, Any]]:
    cid = np.asarray(arr_before["cycle_id_potential"]).reshape(-1)

    def _cm_diff(arr: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        phie_err = arr["phie_pred"].reshape(-1).astype(np.float64) - arr["phie_true"].reshape(-1).astype(np.float64)
        phis_err = arr["phis_c_pred"].reshape(-1).astype(np.float64) - arr["phis_c_true"].reshape(-1).astype(np.float64)
        return 0.5 * (phie_err + phis_err), phis_err - phie_err

    cm_b, diff_b = _cm_diff(arr_before)
    cm_a, diff_a = _cm_diff(arr_after)
    rows: List[Dict[str, Any]] = []
    for cyc in range(cycle_from, cycle_to + 1):
        m = cid == cyc
        if not np.any(m):
            continue
        rows.append({
            "cycle_id": int(cyc),
            "n": int(np.sum(m)),
            "common_mode_error_mean_before": _np_float(np.mean(cm_b[m])),
            "common_mode_error_median_before": _np_float(np.median(cm_b[m])),
            "common_mode_error_mae_before": _np_float(np.mean(np.abs(cm_b[m]))),
            "common_mode_error_std_before": _np_float(np.std(cm_b[m])),
            "common_mode_error_mean_after": _np_float(np.mean(cm_a[m])),
            "common_mode_error_median_after": _np_float(np.median(cm_a[m])),
            "common_mode_error_mae_after": _np_float(np.mean(np.abs(cm_a[m]))),
            "common_mode_error_std_after": _np_float(np.std(cm_a[m])),
            "differential_error_mean_before": _np_float(np.mean(diff_b[m])),
            "differential_error_mae_before": _np_float(np.mean(np.abs(diff_b[m]))),
            "differential_error_mean_after": _np_float(np.mean(diff_a[m])),
            "differential_error_mae_after": _np_float(np.mean(np.abs(diff_a[m]))),
        })
    return rows


def _try_make_plots(out_dir: Path, rows_metrics_before: List[Mapping[str, Any]], rows_metrics_after: List[Mapping[str, Any]], rows_common: List[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skip plots: {exc}")
        return
    plot_dir = out_dir / "plots_common_mode_corrected"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _metric_series(rows: List[Mapping[str, Any]], variable: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
        rr = [r for r in rows if r.get("variable") == variable and r.get(key) not in (None, "")]
        rr = sorted(rr, key=lambda x: int(x["cycle_id"]))
        return np.array([int(r["cycle_id"]) for r in rr]), np.array([float(r[key]) for r in rr])

    for var in ["phis_c", "phie"]:
        x_b, y_b = _metric_series(rows_metrics_before, var, "mae")
        x_a, y_a = _metric_series(rows_metrics_after, var, "mae")
        if x_b.size and x_a.size:
            plt.figure(figsize=(8, 4.5))
            plt.plot(x_b, y_b, label="before")
            plt.plot(x_a, y_a, label="after")
            plt.xlabel("cycle_id")
            plt.ylabel(f"{var} MAE [V]")
            plt.title(f"Per-cycle MAE before/after common-mode gauge correction: {var}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"per_cycle_mae_{var}_before_after.png", dpi=160)
            plt.close()

    if rows_common:
        rows_common = sorted(rows_common, key=lambda x: int(x["cycle_id"]))
        x = np.array([int(r["cycle_id"]) for r in rows_common])
        yb = np.array([float(r["common_mode_error_mean_before"]) for r in rows_common])
        ya = np.array([float(r["common_mode_error_mean_after"]) for r in rows_common])
        plt.figure(figsize=(8, 4.5))
        plt.plot(x, yb, label="before")
        plt.plot(x, ya, label="after")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("cycle_id")
        plt.ylabel("common-mode error mean [V]")
        plt.title("Per-cycle common-mode error before/after gauge correction")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "per_cycle_common_mode_error_before_after.png", dpi=160)
        plt.close()


def _run_one_method(args: argparse.Namespace, method: str, method_out_dir: Path, arr: Mapping[str, np.ndarray], eval_npz: Path) -> Dict[str, Any]:
    offset, calib_summary = _estimate_offset_by_cycle(
        arr,
        calib_from=args.calib_cycle_from,
        calib_to=args.calib_cycle_to,
        apply_from=args.apply_cycle_from,
        apply_to=args.apply_cycle_to,
        method=method,
    )
    arr_corr = _apply_offset(arr, offset)

    metrics_before, rows_before = _compute_global_by_cycle(arr, args.apply_cycle_from, args.apply_cycle_to)
    metrics_after, rows_after = _compute_global_by_cycle(arr_corr, args.apply_cycle_from, args.apply_cycle_to)
    common_diag = _common_diag(arr, arr_corr, args.apply_cycle_from, args.apply_cycle_to)
    rows_common = _per_cycle_common_before_after(arr, arr_corr, args.apply_cycle_from, args.apply_cycle_to)

    method_out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "script_version": "D4-common-mode-gauge-correction-v1",
        "eval_npz": str(eval_npz),
        "method": method,
        "calibration": calib_summary,
        "metrics_global_before": metrics_before,
        "metrics_global_after": metrics_after,
        "common_mode_diagnostic": common_diag,
        "note": (
            "This is a post-hoc potential gauge calibration. It changes only phie_pred and "
            "phis_c_pred by a shared common-mode offset; concentration/theta arrays are unchanged."
        ),
    }
    _write_json(method_out_dir / "gauge_calibration_summary.json", summary)
    _write_json(method_out_dir / "metrics_global_before.json", metrics_before)
    _write_json(method_out_dir / "metrics_global_corrected.json", metrics_after)
    _write_json(method_out_dir / "potential_common_mode_diagnostic_before_after.json", common_diag)
    _write_csv(method_out_dir / "metrics_by_cycle_before.csv", rows_before)
    _write_csv(method_out_dir / "metrics_by_cycle_corrected.csv", rows_after)
    _write_csv(method_out_dir / "potential_common_mode_by_cycle_before_after.csv", rows_common)

    if args.save_npz:
        # Store corrected potentials and all original arrays. Also store offset per potential sample.
        to_save = dict(arr_corr)
        to_save["potential_common_mode_offset_to_add"] = offset.astype(np.float32)
        to_save["phie_pred_before_gauge_correction"] = arr["phie_pred"]
        to_save["phis_c_pred_before_gauge_correction"] = arr["phis_c_pred"]
        np.savez_compressed(method_out_dir / "eval_sampled_arrays_common_mode_corrected.npz", **to_save)

    if args.plots:
        _try_make_plots(method_out_dir, rows_before, rows_after, rows_common)

    # For method comparison.
    return {
        "method": method,
        "output_dir": str(method_out_dir),
        "offset_summary": calib_summary,
        "phis_c_mae_before": metrics_before.get("phis_c", {}).get("mae"),
        "phis_c_mae_after": metrics_after.get("phis_c", {}).get("mae"),
        "phie_mae_before": metrics_before.get("phie", {}).get("mae"),
        "phie_mae_after": metrics_after.get("phie", {}).get("mae"),
        "theta_c_mae_after": metrics_after.get("theta_c", {}).get("mae"),
        "cs_c_mae_after": metrics_after.get("cs_c", {}).get("mae"),
        "common_mode_mae_before": common_diag.get("before", {}).get("common_mode_error", {}).get("mae"),
        "common_mode_mae_after": common_diag.get("after", {}).get("common_mode_error", {}).get("mae"),
        "differential_mae_after": common_diag.get("after", {}).get("differential_phis_minus_phie_error", {}).get("mae"),
    }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate and apply ASSB potential common-mode gauge correction.")
    p.add_argument("--eval_dir", default="EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only")
    p.add_argument("--eval_npz", default=None, help="Optional explicit npz filename/path. Defaults to eval_sampled_arrays*.npz in eval_dir.")
    p.add_argument("--output_dir", default="EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_commonModeCorrected")
    p.add_argument("--calib_cycle_from", type=int, default=5)
    p.add_argument("--calib_cycle_to", type=int, default=20)
    p.add_argument("--apply_cycle_from", type=int, default=5)
    p.add_argument("--apply_cycle_to", type=int, default=100)
    p.add_argument(
        "--method",
        default="constant_mean",
        choices=["constant_mean", "constant_median", "linear_cycle_mean", "all"],
        help=(
            "constant_mean/median estimate one offset from calibration cycles. "
            "linear_cycle_mean fits common-mode error vs cycle on calibration cycles and extrapolates. "
            "all runs all three methods into subdirectories."
        ),
    )
    p.add_argument("--save_npz", action="store_true", help="Save corrected eval arrays npz.")
    p.add_argument("--no_plots", dest="plots", action="store_false", help="Disable matplotlib plots.")
    p.set_defaults(plots=True)
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    eval_npz = _find_eval_npz(eval_dir, args.eval_npz)
    arr = _load_npz(eval_npz)

    required = [
        "cycle_id_potential", "phis_c_true", "phis_c_pred", "phie_true", "phie_pred",
        "cycle_id_cs", "r_a", "r_c", "cs_a_true", "cs_a_pred", "cs_c_true", "cs_c_pred",
        "theta_a_true", "theta_a_pred", "theta_c_true", "theta_c_pred",
    ]
    missing = [k for k in required if k not in arr]
    if missing:
        raise KeyError(f"Eval npz is missing required arrays: {missing}")

    methods = ["constant_mean", "constant_median", "linear_cycle_mean"] if args.method == "all" else [args.method]
    comparison: List[Dict[str, Any]] = []
    for method in methods:
        method_dir = output_dir / method if args.method == "all" else output_dir
        print(f"[INFO] Running method={method}; output={method_dir}")
        comparison.append(_run_one_method(args, method, method_dir, arr, eval_npz))

    if args.method == "all":
        _write_json(output_dir / "gauge_method_comparison.json", {"methods": comparison})
        _write_csv(output_dir / "gauge_method_comparison.csv", comparison)
        print(f"[INFO] Method comparison written to: {output_dir}")
    else:
        print(f"[INFO] Gauge-corrected outputs written to: {output_dir}")

    # Print compact summary for PowerShell logs.
    print(json.dumps({"methods": comparison}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
