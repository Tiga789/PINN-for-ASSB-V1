#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation-side radial ablation for ASSB cs_c / theta_c.

Purpose
-------
This script does NOT retrain the PINN and does NOT modify model files. It reads an
existing evaluation npz, usually produced by
`evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py`, and rescales only the
positive-electrode radial deviation of `cs_c_pred`:

    cs_c_pred_ablation = cbar_c_pred + scale * (cs_c_pred - cbar_c_pred)

where cbar_c_pred is the spherical volume average of the original predicted
`cs_c_pred`. This preserves the predicted positive average inventory and tests
whether the remaining cs_c/theta_c error mainly comes from the radial shape.

Default input
-------------
EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only/
  eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz

Default output
--------------
<eval_dir>/radial_ablation_cs_c/
  ablation_radial_scale_global.csv
  ablation_radial_scale_by_cycle.csv
  ablation_summary.json
  plots/per_cycle_mae_cs_c_radial_ablation.png
  plots/per_cycle_mae_theta_c_radial_ablation.png

Run example
-----------
python evaluate_assb_radial_ablation_from_eval_npz.py \
  --eval_dir EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only \
  --scales 0 0.05 0.10 0.25 0.50 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_VERSION = "D4-positive-cs-c-radial-ablation-v1"
DEFAULT_EVAL_DIR = Path("EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only")
DEFAULT_NPZ_NAME = "eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz"


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _find_npz(eval_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else eval_dir / explicit
        if not p.exists():
            raise FileNotFoundError(f"Requested npz not found: {p}")
        return p.resolve()
    p = eval_dir / DEFAULT_NPZ_NAME
    if p.exists():
        return p.resolve()
    candidates = sorted(eval_dir.glob("eval_sampled_arrays*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No eval_sampled_arrays*.npz found under {eval_dir}. "
            f"Expected {DEFAULT_NPZ_NAME}."
        )
    return candidates[0].resolve()


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    # NumPy 2.x may remove np.trapz; NumPy 1.x may not have np.trapezoid.
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        # Manual trapezoidal integration fallback.
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        y = np.moveaxis(y, axis, -1)
        dx = np.diff(x)
        out = np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * dx, axis=-1)
        return out
    return fn(y, x, axis=axis)


def _reshape_cs(arr: np.ndarray, n_time: int, n_r: int, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (n_time, n_r):
            raise ValueError(f"{name} shape {arr.shape} != expected {(n_time, n_r)}")
        return arr
    if arr.ndim == 1:
        if arr.size != n_time * n_r:
            raise ValueError(f"{name} length {arr.size} != n_time*n_r={n_time*n_r}")
        return arr.reshape(n_time, n_r)
    raise ValueError(f"{name} must be 1D flattened or 2D; got shape {arr.shape}")


def _sphere_average(cs: np.ndarray, r: np.ndarray) -> np.ndarray:
    cs = np.asarray(cs, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if cs.ndim != 2 or cs.shape[1] != r.size:
        raise ValueError(f"cs shape {cs.shape} incompatible with r length {r.size}")
    w = r ** 2
    denom = float(_trapz(w, r, axis=-1))
    if not np.isfinite(denom) or denom <= 0:
        # Uniform fallback for degenerate radius grid; should not happen here.
        return np.mean(cs, axis=1)
    return _trapz(cs * w[None, :], r, axis=1) / denom


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if int(mask.sum()) == 0:
        return {
            "n": 0,
            "mae": math.nan,
            "rmse": math.nan,
            "maxabs": math.nan,
            "bias_mean": math.nan,
            "corr": math.nan,
            "r2": math.nan,
            "label_min": math.nan,
            "label_max": math.nan,
            "label_range": math.nan,
            "label_std": math.nan,
            "pred_min": math.nan,
            "pred_max": math.nan,
            "pred_std": math.nan,
            "std_ratio_pred_over_label": math.nan,
            "nmae": math.nan,
            "nrmse": math.nan,
        }
    yt = yt[mask]
    yp = yp[mask]
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    label_range = float(np.max(yt) - np.min(yt))
    label_std = float(np.std(yt))
    pred_std = float(np.std(yp))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if yt.size > 1 and label_std > 0 and pred_std > 0 else math.nan
    return {
        "n": int(yt.size),
        "mae": mae,
        "rmse": rmse,
        "maxabs": float(np.max(np.abs(err))),
        "bias_mean": float(np.mean(err)),
        "corr": corr,
        "r2": float(1.0 - sse / sst) if sst > 0 else math.nan,
        "label_min": float(np.min(yt)),
        "label_max": float(np.max(yt)),
        "label_range": label_range,
        "label_std": label_std,
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
        "pred_std": pred_std,
        "std_ratio_pred_over_label": float(pred_std / label_std) if label_std > 0 else math.nan,
        "nmae": float(mae / label_range) if label_range > 0 else math.nan,
        "nrmse": float(rmse / label_range) if label_range > 0 else math.nan,
    }


def _infer_csmax_c(cs_true: np.ndarray, theta_true: np.ndarray) -> float:
    cs = np.asarray(cs_true, dtype=np.float64).reshape(-1)
    th = np.asarray(theta_true, dtype=np.float64).reshape(-1)
    mask = np.isfinite(cs) & np.isfinite(th) & (np.abs(th) > 1e-12)
    if int(mask.sum()) == 0:
        return 51.8
    ratio = cs[mask] / th[mask]
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return 51.8
    return float(np.median(ratio))


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_per_cycle(out_dir: Path, by_cycle_rows: List[Dict[str, object]], variable: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable; skip plots: {exc}")
        return
    rows = [r for r in by_cycle_rows if r.get("variable") == variable]
    if not rows:
        return
    scales = sorted({float(r["radial_scale"]) for r in rows})
    plt.figure(figsize=(10, 5.5))
    for scale in scales:
        sub = [r for r in rows if float(r["radial_scale"]) == scale]
        sub = sorted(sub, key=lambda x: int(x["cycle_id"]))
        plt.plot([int(r["cycle_id"]) for r in sub], [float(r["mae"]) for r in sub], label=f"scale={scale:g}")
    plt.xlabel("cycle_id")
    plt.ylabel(f"MAE({variable})")
    plt.title(f"Per-cycle {variable} MAE under cs_c radial-deviation ablation")
    plt.legend()
    plt.tight_layout()
    plot_path = out_dir / "plots" / f"per_cycle_mae_{variable}_radial_ablation.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=180)
    plt.close()


def run_ablation(eval_dir: Path, npz_path: Optional[Path], out_dir: Optional[Path], scales: Sequence[float]) -> Path:
    eval_dir = eval_dir.resolve()
    src = _find_npz(eval_dir, npz_path)
    out_dir = (out_dir if out_dir is not None else eval_dir / "radial_ablation_cs_c").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(src, allow_pickle=False) as d:
        required = ["t_cs", "cycle_id_cs", "r_c", "cs_c_true", "cs_c_pred", "theta_c_true", "theta_c_pred"]
        missing = [k for k in required if k not in d.files]
        if missing:
            raise KeyError(f"Missing required arrays in {src}: {missing}. Existing keys: {d.files}")
        t_cs = np.asarray(d["t_cs"], dtype=np.float64).reshape(-1)
        cycle_id = np.asarray(d["cycle_id_cs"], dtype=np.int64).reshape(-1)
        r_c = np.asarray(d["r_c"], dtype=np.float64).reshape(-1)
        n_time = t_cs.size
        n_r = r_c.size
        cs_true = _reshape_cs(d["cs_c_true"], n_time, n_r, "cs_c_true")
        cs_pred = _reshape_cs(d["cs_c_pred"], n_time, n_r, "cs_c_pred")
        theta_true = _reshape_cs(d["theta_c_true"], n_time, n_r, "theta_c_true")
        theta_pred_orig = _reshape_cs(d["theta_c_pred"], n_time, n_r, "theta_c_pred")

    if cycle_id.size != n_time:
        raise ValueError(f"cycle_id_cs length {cycle_id.size} != t_cs length {n_time}")

    csmax_c = _infer_csmax_c(cs_true, theta_true)
    cbar_pred = _sphere_average(cs_pred, r_c)
    cbar_true = _sphere_average(cs_true, r_c)
    radial_pred = cs_pred - cbar_pred[:, None]

    global_rows: List[Dict[str, object]] = []
    by_cycle_rows: List[Dict[str, object]] = []
    unique_cycles = np.unique(cycle_id)

    for scale in scales:
        scale_f = float(scale)
        cs_ab = cbar_pred[:, None] + scale_f * radial_pred
        theta_ab = cs_ab / csmax_c
        # Keep original theta_pred for scale=1 numerical comparison, but use inferred csmax for all scales.
        variables = {
            "cs_c": (cs_true, cs_ab),
            "theta_c": (theta_true, theta_ab),
            "cbar_c": (cbar_true, _sphere_average(cs_ab, r_c)),
            "surface_minus_center_cs_c": (cs_true[:, -1] - cs_true[:, 0], cs_ab[:, -1] - cs_ab[:, 0]),
        }
        for var, (yt, yp) in variables.items():
            row = {"radial_scale": scale_f, "variable": var}
            row.update(_metrics(yt, yp))
            global_rows.append(row)
            for cyc in unique_cycles:
                idx = cycle_id == cyc
                crow = {"radial_scale": scale_f, "variable": var, "cycle_id": int(cyc)}
                crow.update(_metrics(np.asarray(yt)[idx], np.asarray(yp)[idx]))
                by_cycle_rows.append(crow)

    field_base = [
        "radial_scale", "variable", "cycle_id", "n", "mae", "rmse", "maxabs", "bias_mean", "corr", "r2",
        "label_min", "label_max", "label_range", "label_std", "pred_min", "pred_max", "pred_std",
        "std_ratio_pred_over_label", "nmae", "nrmse",
    ]
    global_fields = [f for f in field_base if f != "cycle_id"]
    _write_csv(out_dir / "ablation_radial_scale_global.csv", global_rows, global_fields)
    _write_csv(out_dir / "ablation_radial_scale_by_cycle.csv", by_cycle_rows, field_base)

    best_cs = min((r for r in global_rows if r["variable"] == "cs_c"), key=lambda r: float(r["mae"]))
    best_theta = min((r for r in global_rows if r["variable"] == "theta_c"), key=lambda r: float(r["mae"]))
    summary = {
        "script_version": SCRIPT_VERSION,
        "source_npz": str(src),
        "output_dir": str(out_dir),
        "n_time_cs": int(n_time),
        "n_r_c": int(n_r),
        "cycle_min": int(unique_cycles.min()),
        "cycle_max": int(unique_cycles.max()),
        "cycle_count": int(unique_cycles.size),
        "csmax_c_inferred": csmax_c,
        "radial_scales": [float(s) for s in scales],
        "best_global_cs_c_by_mae": best_cs,
        "best_global_theta_c_by_mae": best_theta,
        "method": "cs_c_pred_ablation = cbar_c_pred + scale * (cs_c_pred - cbar_c_pred); cbar_c_pred is spherical r^2 average.",
    }
    _write_json(out_dir / "ablation_summary.json", summary)
    _plot_per_cycle(out_dir, by_cycle_rows, "cs_c")
    _plot_per_cycle(out_dir, by_cycle_rows, "theta_c")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluation-side positive cs_c radial-deviation ablation.")
    parser.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--npz", type=Path, default=None, help="Optional explicit eval_sampled_arrays*.npz path or name.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.25, 0.50, 1.0])
    args = parser.parse_args(argv)
    run_ablation(args.eval_dir, args.npz, args.output_dir, args.scales)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
