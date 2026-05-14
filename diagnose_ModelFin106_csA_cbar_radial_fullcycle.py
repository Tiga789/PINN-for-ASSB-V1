#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose ModelFin_106 cs_a/theta_a errors by cbar and radial components over cycle5-522."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

DEFAULT_RAW_EVAL = Path("EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only")
DEFAULT_OUT = Path("EvalFin_106_cycles5_522_v2_massclosed_candidate_csA_diagnostic")
SCRIPT_VERSION = "ASSB_ModelFin106_csA_cbar_radial_diagnostic_v1"


def jfloat(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    preferred = ["cycle_id", "n_time", "cs_a_mae", "cs_a_r2", "cbar_mae", "cbar_bias", "cbar_r2", "radial_mae", "radial_bias", "surface_center_true_mean", "surface_center_pred_mean", "surface_center_error_mean"]
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


def find_eval_npz(eval_dir: Path) -> Path:
    for name in ["eval_sampled_arrays_cycles5_522_v2_massclosed_softlabel_only.npz", "eval_sampled_arrays_ModelFin106_linearGauge_corrected.npz"]:
        p = eval_dir / name
        if p.exists():
            return p
    hits = sorted(eval_dir.glob("eval_sampled_arrays*.npz"))
    if not hits:
        raise FileNotFoundError(f"No eval_sampled_arrays*.npz found in {eval_dir}")
    return hits[0]


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias_mean": None, "corr": None, "r2": None}
    e = p - y
    ss = float(np.sum((y - np.mean(y)) ** 2))
    corr = float("nan")
    if y.size >= 2 and np.std(y) > 0 and np.std(p) > 0:
        corr = float(np.corrcoef(y, p)[0, 1])
    return {"n": int(y.size), "mae": jfloat(np.mean(np.abs(e))), "rmse": jfloat(np.sqrt(np.mean(e * e))), "bias_mean": jfloat(np.mean(e)), "corr": jfloat(corr), "r2": jfloat(float("nan") if ss <= 0 else 1.0 - np.sum(e * e) / ss)}


def sphere_weights(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    if r.size < 2:
        return np.ones_like(r) / max(r.size, 1)
    dr = np.diff(r)
    w = np.zeros_like(r)
    w[0] = dr[0] / 2.0
    w[-1] = dr[-1] / 2.0
    if r.size > 2:
        w[1:-1] = 0.5 * (dr[:-1] + dr[1:])
    w = w * r * r
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(r) / r.size
    return w / s


def reshape_flat(a: np.ndarray, nr: int) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    if x.size % nr != 0:
        raise ValueError(f"array length {x.size} not divisible by nr={nr}")
    return x.reshape(x.size // nr, nr)


def make_plots(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    pdir = out_dir / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    x = np.array([r["cycle_id"] for r in rows], dtype=float)
    for key, ylabel in [
        ("cs_a_mae", "cs_a MAE"),
        ("cbar_mae", "cbar_a MAE"),
        ("cbar_bias", "cbar_a bias pred-true"),
        ("radial_mae", "radial deviation MAE"),
        ("surface_center_error_mean", "surface-center error mean"),
    ]:
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        plt.figure(figsize=(9, 4.5))
        plt.plot(x, y, marker=".", linewidth=0.8)
        plt.xlabel("cycle_id")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(pdir / f"per_cycle_{key}.png", dpi=170)
        plt.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose anode cs_a error decomposition for ModelFin_106 full cycle evaluation.")
    p.add_argument("--raw_eval_dir", type=Path, default=DEFAULT_RAW_EVAL)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cycle_from", type=int, default=5)
    p.add_argument("--cycle_to", type=int, default=522)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    eval_npz = find_eval_npz(args.raw_eval_dir)
    arr = load_npz(eval_npz)
    r_a = np.asarray(arr["r_a"], dtype=np.float64).reshape(-1)
    nr = r_a.size
    w = sphere_weights(r_a)
    true = reshape_flat(arr["cs_a_true"], nr)
    pred = reshape_flat(arr["cs_a_pred"], nr)
    cyc = np.asarray(arr["cycle_id_cs"], dtype=np.int32).reshape(-1)
    if true.shape[0] != cyc.size:
        raise ValueError(f"cs_a time rows {true.shape[0]} != cycle_id_cs length {cyc.size}")
    cbar_true = true @ w
    cbar_pred = pred @ w
    dev_true = true - cbar_true[:, None]
    dev_pred = pred - cbar_pred[:, None]
    surf_center_true = true[:, -1] - true[:, 0]
    surf_center_pred = pred[:, -1] - pred[:, 0]

    rows: List[Dict[str, Any]] = []
    for c in range(args.cycle_from, args.cycle_to + 1):
        m = cyc == c
        if not np.any(m):
            continue
        cs_m = metrics(true[m, :].reshape(-1), pred[m, :].reshape(-1))
        cb_m = metrics(cbar_true[m], cbar_pred[m])
        rd_m = metrics(dev_true[m, :].reshape(-1), dev_pred[m, :].reshape(-1))
        rows.append({
            "cycle_id": int(c),
            "n_time": int(np.sum(m)),
            "cs_a_mae": cs_m["mae"],
            "cs_a_r2": cs_m["r2"],
            "cbar_mae": cb_m["mae"],
            "cbar_bias": cb_m["bias_mean"],
            "cbar_r2": cb_m["r2"],
            "radial_mae": rd_m["mae"],
            "radial_bias": rd_m["bias_mean"],
            "surface_center_true_mean": jfloat(np.mean(surf_center_true[m])),
            "surface_center_pred_mean": jfloat(np.mean(surf_center_pred[m])),
            "surface_center_error_mean": jfloat(np.mean((surf_center_pred - surf_center_true)[m])),
        })

    global_summary = {
        "script_version": SCRIPT_VERSION,
        "eval_npz": str(eval_npz),
        "cycle_from": args.cycle_from,
        "cycle_to": args.cycle_to,
        "cs_a_global": metrics(true[(cyc >= args.cycle_from) & (cyc <= args.cycle_to), :].reshape(-1), pred[(cyc >= args.cycle_from) & (cyc <= args.cycle_to), :].reshape(-1)),
        "cbar_global": metrics(cbar_true[(cyc >= args.cycle_from) & (cyc <= args.cycle_to)], cbar_pred[(cyc >= args.cycle_from) & (cyc <= args.cycle_to)]),
        "radial_global": metrics(dev_true[(cyc >= args.cycle_from) & (cyc <= args.cycle_to), :].reshape(-1), dev_pred[(cyc >= args.cycle_from) & (cyc <= args.cycle_to), :].reshape(-1)),
        "interpretation": "If cbar error is small but radial error is large, improve anode radial/output residual. If cbar error dominates, check I-cbar baseline or inventory offset.",
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "cs_a_cbar_radial_diagnostic_global.json", global_summary)
    write_csv(out / "cs_a_cbar_radial_diagnostic_by_cycle.csv", rows)
    if not args.no_plots:
        make_plots(out, rows)
    print(json.dumps({
        "output_dir": str(out),
        "cs_a_global_mae": global_summary["cs_a_global"]["mae"],
        "cs_a_global_r2": global_summary["cs_a_global"]["r2"],
        "cbar_global_mae": global_summary["cbar_global"]["mae"],
        "radial_global_mae": global_summary["radial_global"]["mae"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
