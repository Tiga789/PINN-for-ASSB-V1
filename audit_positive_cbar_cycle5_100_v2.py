#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
audit_positive_cbar_cycle5_100.py

Positive-electrode cbar / mass-closure audit for ModelFin_103 cycle5-100.

Purpose
-------
This script diagnoses whether the positive-electrode concentration error
(theta_c / cs_c) is mainly caused by a mismatch between:

  1) the soft-label positive concentration average cbar_c_true(t),
  2) the hard I(t)-integrated cbar baseline expected by the training output map,
  3) the PINN-predicted positive concentration average cbar_c_pred(t).

It also checks the radial structure via surface-minus-center amplitude.

Recommended placement
---------------------
Put this file at the project root:

  C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\audit_positive_cbar_cycle5_100.py

Typical command on the user's machine
-------------------------------------
D:\Anaconda\envs\torchgpu\python.exe .\audit_positive_cbar_cycle5_100.py `
  --eval_dir .\EvalFin_103_cycles5_100_v1_softlabel_only `
  --soft_label_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1" `
  --cycle_from 5 `
  --cycle_to 100

If solution.npz is not available, the script still audits cbar_pred vs cbar_true
from the EvalFin sampled arrays, but it cannot compute the I(t)-integrated
baseline comparison.

Outputs
-------
By default, writes to:

  <eval_dir>/audit_positive_cbar_cycle5_100/

including:
  - audit_positive_cbar_global.json
  - audit_positive_cbar_by_cycle.csv
  - audit_positive_cbar_timeseries.csv
  - plots/*.png

Notes on units
--------------
The default Faraday constant is 96485.33212 * 1000 C/kmol, consistent with
cs values around 0-50 when interpreted as kmol/m^3. Override with --F if your
local code uses a different convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_EVAL_DIR = "EvalFin_103_cycles5_100_v1_softlabel_only"
DEFAULT_SOFT_LABEL_DIR = str(Path("..") / "assb_soft_lable_cycle5-522_v1")
DEFAULT_OUTPUT_SUBDIR = "audit_positive_cbar_cycle5_100"

# Project defaults used in the D2/D3 workflow.
DEFAULT_CSMAX_C = 51.8
DEFAULT_EPS_S_C = 0.55
DEFAULT_V_C = 1.2566370614359173e-09  # m^3 = pi*(5 mm)^2*16 um
DEFAULT_F_KMOL = 96485.33212 * 1000.0  # C/kmol, if cs is kmol/m^3


def _to_path(x: Any) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(x))))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        y = float(x)
        if math.isfinite(y):
            return y
        return default
    except Exception:
        return default


def _flatten_dict(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else str(k)
        out[kk] = v
        if isinstance(v, Mapping):
            out.update(_flatten_dict(v, kk))
    return out


def _load_first_json(paths: Sequence[Path]) -> Tuple[Dict[str, Any], Optional[Path]]:
    for p in paths:
        if p.exists() and p.is_file():
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data, p
            except Exception:
                continue
    return {}, None


def _find_eval_npz(eval_dir: Path) -> Path:
    candidates = [
        eval_dir / "eval_sampled_arrays_cycles5_100_softlabel_only.npz",
        eval_dir / "eval_sampled_arrays_softlabel_only.npz",
        eval_dir / "eval_sampled_arrays.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = sorted(eval_dir.glob("eval_sampled_arrays*.npz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find eval sampled arrays npz in: {eval_dir}")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _require(data: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required key {key!r}. Available keys: {sorted(data.keys())}")
    return data[key]




def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy-version-safe trapezoidal integration.

    NumPy 2.x removed/changed access to ``np.trapz`` in some builds.
    Do not use ``getattr(np, "trapezoid", np.trapz)`` because the default
    argument is evaluated eagerly and can raise AttributeError before getattr
    returns.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(y, x, axis=axis)

    # Very old / unusual NumPy fallback: manual trapezoid along ``axis``.
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("manual _trapz fallback expects 1D x")
    y_moved = np.moveaxis(y_arr, axis, -1)
    if y_moved.shape[-1] != x_arr.size:
        raise ValueError(
            f"manual _trapz fallback: y axis length {y_moved.shape[-1]} "
            f"does not match x length {x_arr.size}"
        )
    dx = np.diff(x_arr)
    out = np.sum(0.5 * (y_moved[..., :-1] + y_moved[..., 1:]) * dx, axis=-1)
    return out

def _reshape_state(arr: np.ndarray, n_t: int, n_r: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        if arr.shape == (n_t, n_r):
            return arr.astype(np.float64, copy=False)
        if arr.shape == (n_r, n_t):
            return arr.T.astype(np.float64, copy=False)
        raise ValueError(f"{name} has shape {arr.shape}, expected ({n_t},{n_r})")
    if arr.ndim == 1:
        if arr.size != n_t * n_r:
            raise ValueError(
                f"{name} has length {arr.size}, expected n_t*n_r={n_t*n_r}."
            )
        return arr.reshape(n_t, n_r).astype(np.float64, copy=False)
    raise ValueError(f"{name} has unsupported ndim={arr.ndim}, shape={arr.shape}")


def _sphere_average(cs: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Spherical volume average: int c r^2 dr / int r^2 dr."""
    cs = np.asarray(cs, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if cs.ndim != 2:
        raise ValueError(f"cs must be 2D, got {cs.shape}")
    if cs.shape[1] != r.size:
        raise ValueError(f"cs radial size {cs.shape[1]} does not match r size {r.size}")
    if r.size < 2:
        raise ValueError("Need at least two radial points")
    w = r ** 2
    denom = _trapz(w, r)
    if denom <= 0 or not np.isfinite(denom):
        raise ValueError("Invalid radial grid for spherical average")
    return _trapz(cs * w[None, :], r, axis=1) / denom


def _surface_minus_center(cs: np.ndarray) -> np.ndarray:
    return np.asarray(cs[:, -1] - cs[:, 0], dtype=np.float64)


def _cumtrapz_rate(t: np.ndarray, rate: np.ndarray, y0: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    rate = np.asarray(rate, dtype=np.float64).reshape(-1)
    if t.size != rate.size:
        raise ValueError("t and rate lengths do not match")
    y = np.empty_like(t, dtype=np.float64)
    y[0] = float(y0)
    if t.size == 1:
        return y
    dt = np.diff(t)
    incr = 0.5 * (rate[:-1] + rate[1:]) * dt
    y[1:] = y0 + np.cumsum(incr)
    return y


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(np.asarray(arrays[0]).shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(np.asarray(a))
    return mask


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = _finite_mask(y_true, y_pred)
    if mask.sum() == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias_mean": float("nan"),
            "corr": float("nan"),
            "r2": float("nan"),
            "true_min": float("nan"),
            "true_max": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
        }
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if yt.size >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        corr = float("nan")
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n": int(yt.size),
        "mae": mae,
        "rmse": rmse,
        "bias_mean": bias,
        "corr": corr,
        "r2": r2,
        "true_min": float(np.min(yt)),
        "true_max": float(np.max(yt)),
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
    }


def _read_metrics_by_cycle(eval_dir: Path) -> Dict[Tuple[str, int], Dict[str, str]]:
    path = eval_dir / "metrics_by_cycle.csv"
    if not path.exists():
        return {}
    out: Dict[Tuple[str, int], Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                var = str(row.get("variable", ""))
                cyc = int(float(row.get("cycle_id", "nan")))
                out[(var, cyc)] = row
            except Exception:
                continue
    return out


def _lookup_metric(
    metrics_rows: Mapping[Tuple[str, int], Mapping[str, str]],
    variable: str,
    cycle: int,
    field: str,
) -> float:
    row = metrics_rows.get((variable, int(cycle)))
    if not row:
        return float("nan")
    return _safe_float(row.get(field), float("nan"))  # type: ignore[arg-type]


def _maybe_param(
    args_value: Optional[float],
    flat_summary: Mapping[str, Any],
    possible_keys: Sequence[str],
    default: float,
) -> float:
    if args_value is not None:
        return float(args_value)
    lower = {k.lower(): v for k, v in flat_summary.items()}
    for key in possible_keys:
        if key.lower() in lower:
            val = _safe_float(lower[key.lower()], None)
            if val is not None:
                return float(val)
    # Also try suffix match, useful for nested params.V_c etc.
    for k, v in lower.items():
        for key in possible_keys:
            if k.endswith("." + key.lower()) or k == key.lower():
                val = _safe_float(v, None)
                if val is not None:
                    return float(val)
    return float(default)


def _infer_csmax_c(
    args_value: Optional[float],
    flat_summary: Mapping[str, Any],
    soft_data: Optional[Mapping[str, np.ndarray]],
    fallback: float,
) -> float:
    if args_value is not None:
        return float(args_value)
    keys = [
        "cscamax",
        "cs_c_max",
        "csmax_c",
        "c_s_max_c",
        "params.cscamax",
        "params.csmax_c",
    ]
    lower = {k.lower(): v for k, v in flat_summary.items()}
    for key in keys:
        if key.lower() in lower:
            val = _safe_float(lower[key.lower()], None)
            if val is not None and val > 0:
                return float(val)
    for k, v in lower.items():
        for key in keys:
            if k.endswith("." + key.lower()) or k == key.lower():
                val = _safe_float(v, None)
                if val is not None and val > 0:
                    return float(val)
    if soft_data is not None and "cs_c" in soft_data and "theta_c" in soft_data:
        try:
            cs = np.asarray(soft_data["cs_c"], dtype=np.float64).reshape(-1)
            th = np.asarray(soft_data["theta_c"], dtype=np.float64).reshape(-1)
            mask = np.isfinite(cs) & np.isfinite(th) & (np.abs(th) > 1e-9)
            if mask.sum() > 100:
                ratio = cs[mask] / th[mask]
                ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                if ratio.size > 100:
                    return float(np.median(ratio))
        except Exception:
            pass
    return float(fallback)


def _load_soft_solution(soft_label_dir: Path) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Path]]:
    solution_path = soft_label_dir / "solution.npz"
    if not solution_path.exists():
        return None, None
    return _load_npz(solution_path), solution_path


def _filter_solution_cycle_range(
    soft: Mapping[str, np.ndarray], cycle_from: int, cycle_to: int
) -> Dict[str, np.ndarray]:
    if "cycle_id" not in soft:
        return {k: np.asarray(v) for k, v in soft.items()}
    cyc = np.asarray(soft["cycle_id"]).reshape(-1)
    mask = (cyc >= cycle_from) & (cyc <= cycle_to)
    out: Dict[str, np.ndarray] = {}
    for k, v in soft.items():
        arr = np.asarray(v)
        if arr.shape[0] == mask.size:
            out[k] = arr[mask]
        else:
            out[k] = arr
    return out


def _compute_solution_baseline(
    soft: Mapping[str, np.ndarray],
    cycle_from: int,
    cycle_to: int,
    eps_s_c: float,
    F: float,
    V_c: float,
) -> Optional[Dict[str, np.ndarray]]:
    required = ["t_global_s", "cycle_id", "I_profile", "cs_c", "r_c"]
    if not all(k in soft for k in required):
        return None
    s = _filter_solution_cycle_range(soft, cycle_from, cycle_to)
    t = np.asarray(s["t_global_s"], dtype=np.float64).reshape(-1)
    cyc = np.asarray(s["cycle_id"]).reshape(-1).astype(int)
    I = np.asarray(s["I_profile"], dtype=np.float64).reshape(-1)
    r_c = np.asarray(s["r_c"], dtype=np.float64).reshape(-1)
    cs_c = _reshape_state(np.asarray(s["cs_c"]), t.size, r_c.size, "solution.cs_c")

    order = np.argsort(t)
    t = t[order]
    cyc = cyc[order]
    I = I[order]
    cs_c = cs_c[order]

    cbar_true = _sphere_average(cs_c, r_c)
    rate = -I / (float(eps_s_c) * float(F) * float(V_c))
    cbar_from_I = _cumtrapz_rate(t, rate, float(cbar_true[0]))
    return {
        "t": t,
        "cycle_id": cyc,
        "I_profile": I,
        "r_c": r_c,
        "cbar_true": cbar_true,
        "cbar_from_I": cbar_from_I,
        "radial_amp_true": _surface_minus_center(cs_c),
    }


def _interp_like(t_ref: np.ndarray, y_ref: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    t_ref = np.asarray(t_ref, dtype=np.float64).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=np.float64).reshape(-1)
    t_query = np.asarray(t_query, dtype=np.float64).reshape(-1)
    if t_ref.size == 0:
        return np.full_like(t_query, np.nan, dtype=np.float64)
    order = np.argsort(t_ref)
    return np.interp(t_query, t_ref[order], y_ref[order], left=np.nan, right=np.nan)


def _polyfit_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = _finite_mask(x, y)
    if mask.sum() < 2:
        return float("nan")
    try:
        return float(np.polyfit(x[mask], y[mask], 1)[0])
    except Exception:
        return float("nan")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _make_plots(out_dir: Path, ts_rows: Sequence[Mapping[str, Any]], cycle_rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping plots: {exc}")
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Convert rows to arrays.
    def arr_from(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
        vals = []
        for row in rows:
            vals.append(_safe_float(row.get(key), float("nan")))
        return np.asarray(vals, dtype=np.float64)

    t = arr_from(ts_rows, "t_s")
    ctrue = arr_from(ts_rows, "cbar_c_true_eval")
    cpred = arr_from(ts_rows, "cbar_c_pred_eval")
    cI = arr_from(ts_rows, "cbar_c_from_I")
    cyc = arr_from(cycle_rows, "cycle_id")

    # Time series: cbar true / pred / I baseline.
    plt.figure(figsize=(10, 5))
    if np.isfinite(ctrue).any():
        plt.plot(t, ctrue, label="cbar true from Eval cs_c")
    if np.isfinite(cpred).any():
        plt.plot(t, cpred, label="cbar pred from Eval cs_c")
    if np.isfinite(cI).any():
        plt.plot(t, cI, label="cbar from I(t) baseline")
    plt.xlabel("t_global_s")
    plt.ylabel("cbar_c")
    plt.title("Positive-electrode cbar audit: true / pred / I-baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "cbar_c_true_pred_vs_I_by_time.png", dpi=200)
    plt.close()

    # Per-cycle cbar MAE.
    plt.figure(figsize=(10, 5))
    plt.plot(cyc, arr_from(cycle_rows, "mae_pred_cbar_vs_true"), label="pred vs true")
    if np.isfinite(arr_from(cycle_rows, "mae_true_cbar_vs_I")).any():
        plt.plot(cyc, arr_from(cycle_rows, "mae_true_cbar_vs_I"), label="true vs I-baseline")
    plt.xlabel("cycle_id")
    plt.ylabel("MAE(cbar_c)")
    plt.title("Per-cycle cbar_c MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "per_cycle_cbar_mae.png", dpi=200)
    plt.close()

    # Per-cycle cbar bias.
    plt.figure(figsize=(10, 5))
    plt.plot(cyc, arr_from(cycle_rows, "bias_pred_cbar_minus_true"), label="pred - true")
    if np.isfinite(arr_from(cycle_rows, "bias_true_cbar_minus_I")).any():
        plt.plot(cyc, arr_from(cycle_rows, "bias_true_cbar_minus_I"), label="true - I-baseline")
    plt.xlabel("cycle_id")
    plt.ylabel("bias(cbar_c)")
    plt.title("Per-cycle cbar_c bias")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "per_cycle_cbar_bias.png", dpi=200)
    plt.close()

    # Radial amplitude.
    plt.figure(figsize=(10, 5))
    plt.plot(cyc, arr_from(cycle_rows, "radial_amp_true_mean"), label="true surface-center")
    plt.plot(cyc, arr_from(cycle_rows, "radial_amp_pred_mean"), label="pred surface-center")
    plt.xlabel("cycle_id")
    plt.ylabel("surface minus center cs_c")
    plt.title("Positive cs_c radial amplitude by cycle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "per_cycle_radial_amp_surface_minus_center.png", dpi=200)
    plt.close()

    # Relation with phis_c and theta_c bias if available.
    phis_bias = arr_from(cycle_rows, "metrics_phis_c_bias_mean")
    theta_bias = arr_from(cycle_rows, "metrics_theta_c_bias_mean")
    cbar_bias = arr_from(cycle_rows, "bias_pred_cbar_minus_true")
    if np.isfinite(phis_bias).any() or np.isfinite(theta_bias).any():
        plt.figure(figsize=(10, 5))
        if np.isfinite(phis_bias).any():
            plt.plot(cyc, phis_bias, label="phis_c bias from metrics")
        if np.isfinite(theta_bias).any():
            plt.plot(cyc, theta_bias, label="theta_c bias from metrics")
        if np.isfinite(cbar_bias).any():
            plt.plot(cyc, cbar_bias, label="cbar_c pred-true")
        plt.xlabel("cycle_id")
        plt.ylabel("bias")
        plt.title("Bias trends: potential/state/cbar")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "per_cycle_bias_trends.png", dpi=200)
        plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit positive-electrode cbar closure for ModelFin_103 cycle5-100."
    )
    p.add_argument("--eval_dir", type=str, default=DEFAULT_EVAL_DIR)
    p.add_argument("--soft_label_dir", type=str, default=DEFAULT_SOFT_LABEL_DIR)
    p.add_argument("--cycle_from", type=int, default=5)
    p.add_argument("--cycle_to", type=int, default=100)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--csmax_c", type=float, default=None, help="Override positive csmax. Default: infer or 51.8")
    p.add_argument("--eps_s_c", type=float, default=None, help="Override positive active fraction. Default: infer or 0.55")
    p.add_argument("--V_c", type=float, default=None, help="Override positive electrode total volume m^3. Default: infer or 1.2566370614359173e-09")
    p.add_argument("--F", type=float, default=None, help="Override Faraday constant. Default: infer or 96485.33212*1000 C/kmol")
    p.add_argument("--no_plots", action="store_true", help="Skip matplotlib plots")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    eval_dir = _to_path(args.eval_dir)
    soft_label_dir = _to_path(args.soft_label_dir)
    out_dir = _to_path(args.output_dir) if args.output_dir else eval_dir / DEFAULT_OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_npz_path = _find_eval_npz(eval_dir)
    eval_data = _load_npz(eval_npz_path)

    t_cs = np.asarray(_require(eval_data, "t_cs"), dtype=np.float64).reshape(-1)
    cycle_id_cs = np.asarray(_require(eval_data, "cycle_id_cs")).reshape(-1).astype(int)
    r_c = np.asarray(_require(eval_data, "r_c"), dtype=np.float64).reshape(-1)
    n_t, n_r = t_cs.size, r_c.size
    cs_c_true = _reshape_state(_require(eval_data, "cs_c_true"), n_t, n_r, "cs_c_true")
    cs_c_pred = _reshape_state(_require(eval_data, "cs_c_pred"), n_t, n_r, "cs_c_pred")

    soft_data, solution_path = _load_soft_solution(soft_label_dir)

    summary_candidates = [
        soft_label_dir / "soft_label_summary.json",
        soft_label_dir / "summary.json",
        soft_label_dir / "softlabel_integrity_report.json",
        eval_dir / "debug_model_and_data.json",
    ]
    summary, summary_path = _load_first_json(summary_candidates)
    flat_summary = _flatten_dict(summary) if summary else {}

    csmax_c = _infer_csmax_c(args.csmax_c, flat_summary, soft_data, DEFAULT_CSMAX_C)
    eps_s_c = _maybe_param(args.eps_s_c, flat_summary, ["eps_s_c", "eps_c", "eps_solid_c"], DEFAULT_EPS_S_C)
    V_c = _maybe_param(args.V_c, flat_summary, ["V_c", "Vc", "volume_c"], DEFAULT_V_C)
    F = _maybe_param(args.F, flat_summary, ["F", "faraday", "Faraday"], DEFAULT_F_KMOL)

    # Prevent a common silent-unit error: if F is accidentally read as 96485 while
    # csmax is around 50, the cbar_from_I curve will be off by ~1000x.
    F_unit_warning = None
    if F < 1.0e6 and csmax_c < 1000:
        F_unit_warning = (
            "F appears to be in C/mol while csmax_c appears to be in kmol/m^3 scale. "
            "The script used the loaded/argument F value, but you may need --F 96485332.12."
        )

    cbar_true_eval = _sphere_average(cs_c_true, r_c)
    cbar_pred_eval = _sphere_average(cs_c_pred, r_c)
    theta_cbar_true_eval = cbar_true_eval / csmax_c
    theta_cbar_pred_eval = cbar_pred_eval / csmax_c
    radial_amp_true_eval = _surface_minus_center(cs_c_true)
    radial_amp_pred_eval = _surface_minus_center(cs_c_pred)

    solution_baseline = None
    if soft_data is not None:
        try:
            solution_baseline = _compute_solution_baseline(
                soft_data, args.cycle_from, args.cycle_to, eps_s_c, F, V_c
            )
        except Exception as exc:
            print(f"[WARN] Could not compute I(t) baseline from solution.npz: {exc}")
            solution_baseline = None
    else:
        print(f"[WARN] solution.npz not found under soft_label_dir: {soft_label_dir}")

    cbar_from_I_eval = np.full_like(cbar_true_eval, np.nan, dtype=np.float64)
    cbar_true_solution_interp = np.full_like(cbar_true_eval, np.nan, dtype=np.float64)
    I_interp = np.full_like(cbar_true_eval, np.nan, dtype=np.float64)
    if solution_baseline is not None:
        cbar_from_I_eval = _interp_like(solution_baseline["t"], solution_baseline["cbar_from_I"], t_cs)
        cbar_true_solution_interp = _interp_like(solution_baseline["t"], solution_baseline["cbar_true"], t_cs)
        I_interp = _interp_like(solution_baseline["t"], solution_baseline["I_profile"], t_cs)

    # Optional metrics from evaluator metrics_by_cycle.csv.
    metrics_rows = _read_metrics_by_cycle(eval_dir)

    cycles = sorted(int(c) for c in np.unique(cycle_id_cs) if args.cycle_from <= int(c) <= args.cycle_to)
    by_cycle: List[Dict[str, Any]] = []
    for cyc in cycles:
        mask = cycle_id_cs == cyc
        if mask.sum() == 0:
            continue
        m_pred = _metrics(cbar_true_eval[mask], cbar_pred_eval[mask])
        m_pred_theta = _metrics(theta_cbar_true_eval[mask], theta_cbar_pred_eval[mask])
        m_true_i = _metrics(cbar_from_I_eval[mask], cbar_true_eval[mask]) if np.isfinite(cbar_from_I_eval[mask]).any() else {}
        m_eval_vs_solution = (
            _metrics(cbar_true_solution_interp[mask], cbar_true_eval[mask])
            if np.isfinite(cbar_true_solution_interp[mask]).any()
            else {}
        )
        row: Dict[str, Any] = {
            "cycle_id": cyc,
            "n_samples": int(mask.sum()),
            "t_min_s": float(np.nanmin(t_cs[mask])),
            "t_max_s": float(np.nanmax(t_cs[mask])),
            "I_mean_A": float(np.nanmean(I_interp[mask])) if np.isfinite(I_interp[mask]).any() else float("nan"),
            "I_min_A": float(np.nanmin(I_interp[mask])) if np.isfinite(I_interp[mask]).any() else float("nan"),
            "I_max_A": float(np.nanmax(I_interp[mask])) if np.isfinite(I_interp[mask]).any() else float("nan"),
            "cbar_true_mean": float(np.nanmean(cbar_true_eval[mask])),
            "cbar_pred_mean": float(np.nanmean(cbar_pred_eval[mask])),
            "cbar_from_I_mean": float(np.nanmean(cbar_from_I_eval[mask])) if np.isfinite(cbar_from_I_eval[mask]).any() else float("nan"),
            "bias_pred_cbar_minus_true": float(np.nanmean(cbar_pred_eval[mask] - cbar_true_eval[mask])),
            "mae_pred_cbar_vs_true": m_pred.get("mae", float("nan")),
            "rmse_pred_cbar_vs_true": m_pred.get("rmse", float("nan")),
            "corr_pred_cbar_vs_true": m_pred.get("corr", float("nan")),
            "r2_pred_cbar_vs_true": m_pred.get("r2", float("nan")),
            "bias_pred_theta_cbar_minus_true": float(np.nanmean(theta_cbar_pred_eval[mask] - theta_cbar_true_eval[mask])),
            "mae_pred_theta_cbar_vs_true": m_pred_theta.get("mae", float("nan")),
            "bias_true_cbar_minus_I": float(np.nanmean(cbar_true_eval[mask] - cbar_from_I_eval[mask])) if np.isfinite(cbar_from_I_eval[mask]).any() else float("nan"),
            "mae_true_cbar_vs_I": m_true_i.get("mae", float("nan")),
            "rmse_true_cbar_vs_I": m_true_i.get("rmse", float("nan")),
            "corr_true_cbar_vs_I": m_true_i.get("corr", float("nan")),
            "r2_true_cbar_vs_I": m_true_i.get("r2", float("nan")),
            "mae_eval_true_cbar_vs_solution_true": m_eval_vs_solution.get("mae", float("nan")),
            "radial_amp_true_mean": float(np.nanmean(radial_amp_true_eval[mask])),
            "radial_amp_pred_mean": float(np.nanmean(radial_amp_pred_eval[mask])),
            "radial_amp_pred_minus_true_mean": float(np.nanmean(radial_amp_pred_eval[mask] - radial_amp_true_eval[mask])),
            "radial_amp_true_median": float(np.nanmedian(radial_amp_true_eval[mask])),
            "radial_amp_pred_median": float(np.nanmedian(radial_amp_pred_eval[mask])),
            "cbar_true_start": float(cbar_true_eval[mask][0]),
            "cbar_true_end": float(cbar_true_eval[mask][-1]),
            "cbar_pred_start": float(cbar_pred_eval[mask][0]),
            "cbar_pred_end": float(cbar_pred_eval[mask][-1]),
            "cbar_from_I_start": float(cbar_from_I_eval[mask][0]) if np.isfinite(cbar_from_I_eval[mask]).any() else float("nan"),
            "cbar_from_I_end": float(cbar_from_I_eval[mask][-1]) if np.isfinite(cbar_from_I_eval[mask]).any() else float("nan"),
            "metrics_theta_c_mae": _lookup_metric(metrics_rows, "theta_c", cyc, "mae"),
            "metrics_theta_c_bias_mean": _lookup_metric(metrics_rows, "theta_c", cyc, "bias_mean"),
            "metrics_cs_c_mae": _lookup_metric(metrics_rows, "cs_c", cyc, "mae"),
            "metrics_cs_c_bias_mean": _lookup_metric(metrics_rows, "cs_c", cyc, "bias_mean"),
            "metrics_phis_c_mae": _lookup_metric(metrics_rows, "phis_c", cyc, "mae"),
            "metrics_phis_c_bias_mean": _lookup_metric(metrics_rows, "phis_c", cyc, "bias_mean"),
        }
        by_cycle.append(row)

    # Global metrics.
    global_pred = _metrics(cbar_true_eval, cbar_pred_eval)
    global_pred_theta = _metrics(theta_cbar_true_eval, theta_cbar_pred_eval)
    global_true_i = _metrics(cbar_from_I_eval, cbar_true_eval) if np.isfinite(cbar_from_I_eval).any() else {}
    global_eval_solution = (
        _metrics(cbar_true_solution_interp, cbar_true_eval)
        if np.isfinite(cbar_true_solution_interp).any()
        else {}
    )

    cycle_arr = np.asarray([r["cycle_id"] for r in by_cycle], dtype=np.float64)
    bias_pred_arr = np.asarray([r["bias_pred_cbar_minus_true"] for r in by_cycle], dtype=np.float64)
    bias_true_i_arr = np.asarray([r["bias_true_cbar_minus_I"] for r in by_cycle], dtype=np.float64)
    radial_amp_bias_arr = np.asarray([r["radial_amp_pred_minus_true_mean"] for r in by_cycle], dtype=np.float64)

    global_report: Dict[str, Any] = {
        "script": Path(__file__).name,
        "eval_dir": str(eval_dir),
        "eval_npz_path": str(eval_npz_path),
        "soft_label_dir": str(soft_label_dir),
        "solution_path": str(solution_path) if solution_path else None,
        "summary_path": str(summary_path) if summary_path else None,
        "cycle_from": args.cycle_from,
        "cycle_to": args.cycle_to,
        "n_eval_time_points_cs": int(n_t),
        "n_r_c": int(n_r),
        "params_used": {
            "csmax_c": csmax_c,
            "eps_s_c": eps_s_c,
            "V_c_m3": V_c,
            "F": F,
            "F_unit_warning": F_unit_warning,
        },
        "global_cbar_pred_vs_true": global_pred,
        "global_theta_cbar_pred_vs_true": global_pred_theta,
        "global_cbar_true_vs_I_baseline": global_true_i,
        "global_eval_true_cbar_vs_solution_true": global_eval_solution,
        "radial_amplitude_global": {
            "true_mean_surface_minus_center": float(np.nanmean(radial_amp_true_eval)),
            "pred_mean_surface_minus_center": float(np.nanmean(radial_amp_pred_eval)),
            "pred_minus_true_mean": float(np.nanmean(radial_amp_pred_eval - radial_amp_true_eval)),
            "true_median_surface_minus_center": float(np.nanmedian(radial_amp_true_eval)),
            "pred_median_surface_minus_center": float(np.nanmedian(radial_amp_pred_eval)),
        },
        "trend_slopes_per_cycle": {
            "bias_pred_cbar_minus_true_per_cycle": _polyfit_slope(cycle_arr, bias_pred_arr),
            "bias_true_cbar_minus_I_per_cycle": _polyfit_slope(cycle_arr, bias_true_i_arr),
            "radial_amp_pred_minus_true_per_cycle": _polyfit_slope(cycle_arr, radial_amp_bias_arr),
        },
        "interpretation_hint": [
            "If global/per-cycle cbar_true_vs_I_baseline is bad, the continuous soft-label positive cbar does not match the hard I(t)-cbar baseline used by training.",
            "If cbar_true_vs_I_baseline is good but cbar_pred_vs_true is bad, prioritize model/output-map training or radial ansatz issues.",
            "If radial_amp_pred_minus_true is large and sign-opposite, the network may be using radial distortion to compensate for an average-inventory mismatch.",
        ],
    }

    # Timeseries output at eval cs sample points.
    ts_rows: List[Dict[str, Any]] = []
    for i in range(n_t):
        ts_rows.append(
            {
                "idx": i,
                "t_s": float(t_cs[i]),
                "cycle_id": int(cycle_id_cs[i]),
                "I_profile_A": float(I_interp[i]) if np.isfinite(I_interp[i]) else float("nan"),
                "cbar_c_true_eval": float(cbar_true_eval[i]),
                "cbar_c_pred_eval": float(cbar_pred_eval[i]),
                "cbar_c_from_I": float(cbar_from_I_eval[i]) if np.isfinite(cbar_from_I_eval[i]) else float("nan"),
                "cbar_c_true_solution_interp": float(cbar_true_solution_interp[i]) if np.isfinite(cbar_true_solution_interp[i]) else float("nan"),
                "theta_cbar_true_eval": float(theta_cbar_true_eval[i]),
                "theta_cbar_pred_eval": float(theta_cbar_pred_eval[i]),
                "radial_amp_true_surface_minus_center": float(radial_amp_true_eval[i]),
                "radial_amp_pred_surface_minus_center": float(radial_amp_pred_eval[i]),
                "cbar_pred_minus_true": float(cbar_pred_eval[i] - cbar_true_eval[i]),
                "cbar_true_minus_I": float(cbar_true_eval[i] - cbar_from_I_eval[i]) if np.isfinite(cbar_from_I_eval[i]) else float("nan"),
            }
        )

    by_cycle_fields = [
        "cycle_id", "n_samples", "t_min_s", "t_max_s", "I_mean_A", "I_min_A", "I_max_A",
        "cbar_true_mean", "cbar_pred_mean", "cbar_from_I_mean",
        "bias_pred_cbar_minus_true", "mae_pred_cbar_vs_true", "rmse_pred_cbar_vs_true", "corr_pred_cbar_vs_true", "r2_pred_cbar_vs_true",
        "bias_pred_theta_cbar_minus_true", "mae_pred_theta_cbar_vs_true",
        "bias_true_cbar_minus_I", "mae_true_cbar_vs_I", "rmse_true_cbar_vs_I", "corr_true_cbar_vs_I", "r2_true_cbar_vs_I",
        "mae_eval_true_cbar_vs_solution_true",
        "radial_amp_true_mean", "radial_amp_pred_mean", "radial_amp_pred_minus_true_mean", "radial_amp_true_median", "radial_amp_pred_median",
        "cbar_true_start", "cbar_true_end", "cbar_pred_start", "cbar_pred_end", "cbar_from_I_start", "cbar_from_I_end",
        "metrics_theta_c_mae", "metrics_theta_c_bias_mean", "metrics_cs_c_mae", "metrics_cs_c_bias_mean", "metrics_phis_c_mae", "metrics_phis_c_bias_mean",
    ]
    ts_fields = [
        "idx", "t_s", "cycle_id", "I_profile_A",
        "cbar_c_true_eval", "cbar_c_pred_eval", "cbar_c_from_I", "cbar_c_true_solution_interp",
        "theta_cbar_true_eval", "theta_cbar_pred_eval",
        "radial_amp_true_surface_minus_center", "radial_amp_pred_surface_minus_center",
        "cbar_pred_minus_true", "cbar_true_minus_I",
    ]

    with (out_dir / "audit_positive_cbar_global.json").open("w", encoding="utf-8") as f:
        json.dump(global_report, f, indent=2, ensure_ascii=False, default=_json_default)
    _write_csv(out_dir / "audit_positive_cbar_by_cycle.csv", by_cycle, by_cycle_fields)
    _write_csv(out_dir / "audit_positive_cbar_timeseries.csv", ts_rows, ts_fields)

    if not args.no_plots:
        _make_plots(out_dir, ts_rows, by_cycle)

    print("\n=== Positive cbar audit complete ===")
    print(f"eval npz:      {eval_npz_path}")
    print(f"solution npz:  {solution_path if solution_path else 'NOT FOUND / NOT USED'}")
    print(f"output dir:    {out_dir}")
    print("\nGlobal cbar_pred vs cbar_true:")
    print(json.dumps(global_pred, indent=2, default=_json_default))
    if global_true_i:
        print("\nGlobal cbar_true vs I(t)-integrated baseline:")
        print(json.dumps(global_true_i, indent=2, default=_json_default))
    else:
        print("\nGlobal cbar_true vs I(t)-integrated baseline: unavailable because solution.npz / I_profile was not usable.")
    if F_unit_warning:
        print(f"\n[WARN] {F_unit_warning}")
    print("\nKey trend slopes per cycle:")
    print(json.dumps(global_report["trend_slopes_per_cycle"], indent=2, default=_json_default))
    print("\nNext: inspect audit_positive_cbar_by_cycle.csv and plots/*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
