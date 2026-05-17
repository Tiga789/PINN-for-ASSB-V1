# -*- coding: utf-8 -*-
r"""Fit a deterministic train-only initializer for ASSB-111 saturating_v2_stable.

The initializer does not use held-out test/partial labels. It fits a simple
floor-aware recurrence on train cycles only:

    SOH_k = floor + (SOH_{k-1} - floor) * exp(-k_base * remaining**gamma * dc)

where ``remaining = (SOH_{k-1}-floor)/(SOH0-floor)``. The fitted floor/soh0/k
are then used to initialize the SOH head so random seeds start from the same
visible-data mechanism before neural rate corrections are learned.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from util.assb111_split import load_manifest, split_for_cycles


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_csv_set(text: str) -> Tuple[str, ...]:
    return tuple(s.strip().lower() for s in str(text).replace(";", ",").split(",") if s.strip())


def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    m = np.isfinite(obs) & np.isfinite(pred)
    if np.sum(m) < 2:
        return float("nan")
    o = obs[m]
    p = pred[m]
    ss_res = float(np.sum((p - o) ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    return float("nan") if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot


def _metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(obs) & np.isfinite(pred)
    if not np.any(m):
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan"), "corr": float("nan")}
    o = obs[m]
    p = pred[m]
    e = p - o
    corr = float("nan")
    if len(o) >= 2 and np.std(o) > 1e-15 and np.std(p) > 1e-15:
        corr = float(np.corrcoef(o, p)[0, 1])
    return {
        "n": int(len(o)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "r2": _r2(o, p),
        "corr": corr,
        "obs_min": float(np.min(o)),
        "obs_max": float(np.max(o)),
        "pred_min": float(np.min(p)),
        "pred_max": float(np.max(p)),
    }


def simulate(cycles: Sequence[int], *, floor: float, soh0: float, k: float, gamma: float) -> np.ndarray:
    cycles_arr = np.asarray(cycles, dtype=float)
    n = len(cycles_arr)
    if n == 0:
        return np.asarray([], dtype=float)
    out = np.empty(n, dtype=float)
    prev = float(soh0)
    denom = max(float(soh0) - float(floor), 1e-12)
    out[0] = prev
    for i in range(1, n):
        dc = max(float(cycles_arr[i] - cycles_arr[i - 1]), 1.0)
        remaining = min(max((prev - floor) / denom, 0.0), 1.0)
        gated = float(k) * (remaining ** float(gamma))
        prev = floor + (prev - floor) * math.exp(-gated * dc)
        out[i] = prev
    return out


def fit_k_grid(cycles: np.ndarray, soh: np.ndarray, *, floor: float, soh0: float, gamma: float, k_min: float, k_max: float, k_grid: int) -> Tuple[float, Dict[str, float]]:
    k_values = np.geomspace(max(k_min, 1e-10), max(k_max, k_min * 1.01), int(max(k_grid, 5)))
    best_k = float(k_values[0])
    best_mae = float("inf")
    best_rmse = float("inf")
    for k in k_values:
        pred = simulate(cycles, floor=floor, soh0=soh0, k=float(k), gamma=gamma)
        m = np.isfinite(soh) & np.isfinite(pred)
        if not np.any(m):
            continue
        err = pred[m] - soh[m]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        if mae < best_mae - 1e-15 or (abs(mae - best_mae) <= 1e-15 and rmse < best_rmse):
            best_mae = mae
            best_rmse = rmse
            best_k = float(k)
    return best_k, {"train_mae": best_mae, "train_rmse": best_rmse}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--split_manifest_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument("--fit_splits", default="train")
    p.add_argument("--select_splits", default="val")
    p.add_argument("--forbidden_splits", default="test,partial")
    p.add_argument("--floor_selection_mode", default="fixed_prior", choices=["fixed_prior", "fixed_floor", "grid_train"])
    p.add_argument("--fixed_floor", type=float, default=None)
    p.add_argument("--floor_prior", type=float, default=0.72)
    p.add_argument("--floor_min", type=float, default=0.68)
    p.add_argument("--floor_max", type=float, default=0.78)
    p.add_argument("--floor_step", type=float, default=0.001)
    p.add_argument("--k_min", type=float, default=1e-5)
    p.add_argument("--k_max", type=float, default=0.08)
    p.add_argument("--k_grid", type=int, default=400)
    p.add_argument("--gate_gamma", type=float, default=1.0)
    p.add_argument("--soh0_mode", default="first_train", choices=["first_train", "max_train", "mean_first3"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.dataset_csv)
    if "cycle_id" not in frame.columns or "SOH_obs" not in frame.columns:
        raise KeyError("dataset_csv must contain cycle_id and SOH_obs")
    frame = frame.copy()
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    manifest = load_manifest(args.split_manifest_json)
    if "split" not in frame.columns:
        frame["split"] = split_for_cycles(frame["cycle_id"].to_numpy(dtype=int), manifest)
    frame["split_l"] = frame["split"].astype(str).str.lower()
    fit_splits = set(_parse_csv_set(args.fit_splits))
    select_splits = set(_parse_csv_set(args.select_splits))
    forbidden_splits = set(_parse_csv_set(args.forbidden_splits))
    if fit_splits & forbidden_splits:
        raise RuntimeError(f"fit_splits overlap forbidden_splits: {sorted(fit_splits & forbidden_splits)}")
    fit = frame[frame["split_l"].isin(fit_splits)].copy().sort_values("cycle_id")
    if fit.empty:
        raise RuntimeError("No fit rows selected")
    fit_soh = pd.to_numeric(fit["SOH_obs"], errors="coerce").to_numpy(dtype=float)
    fit_cycles = fit["cycle_id"].to_numpy(dtype=int)
    if str(args.soh0_mode) == "first_train":
        soh0 = float(fit_soh[np.isfinite(fit_soh)][0])
    elif str(args.soh0_mode) == "max_train":
        soh0 = float(np.nanmax(fit_soh))
    else:
        soh0 = float(np.nanmean(fit_soh[:3]))
    mode = str(args.floor_selection_mode).lower().strip()
    if mode in {"fixed_prior", "fixed_floor"}:
        floor = float(args.fixed_floor if args.fixed_floor is not None else args.floor_prior)
        floors = [min(max(floor, float(args.floor_min)), float(args.floor_max))]
        selected_by = "fixed_floor_train_only_k"
    else:
        floors = list(np.arange(float(args.floor_min), float(args.floor_max) + 0.5 * float(args.floor_step), float(args.floor_step)))
        selected_by = "grid_train_mae"
    rows: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for floor in floors:
        k, train_fit = fit_k_grid(fit_cycles, fit_soh, floor=float(floor), soh0=soh0, gamma=float(args.gate_gamma), k_min=float(args.k_min), k_max=float(args.k_max), k_grid=int(args.k_grid))
        pred_fit = simulate(fit_cycles, floor=float(floor), soh0=soh0, k=k, gamma=float(args.gate_gamma))
        train_metrics = _metrics(fit_soh, pred_fit)
        row = {"floor": float(floor), "soh0": soh0, "k_per_cycle": k, **{f"train_{kk}": vv for kk, vv in train_metrics.items()}}
        rows.append(row)
        if best is None or row["train_mae"] < best["train_mae"]:
            best = row
    assert best is not None
    floor = float(best["floor"])
    k = float(best["k_per_cycle"])

    all_cycles = frame["cycle_id"].to_numpy(dtype=int)
    pred_all = simulate(all_cycles, floor=floor, soh0=soh0, k=k, gamma=float(args.gate_gamma))
    visible = frame[frame["split_l"].isin(fit_splits | select_splits)].copy()
    pred_visible = simulate(visible["cycle_id"].to_numpy(dtype=int), floor=floor, soh0=soh0, k=k, gamma=float(args.gate_gamma))
    if args.output_csv:
        out_vis = visible.copy()
        out_vis["SOH_init_pred"] = pred_visible
        out_vis.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    metrics_by_split: Dict[str, Any] = {}
    for split in sorted(frame["split_l"].unique()):
        sub = frame[frame["split_l"] == split]
        pred = pred_all[sub.index.to_numpy()]
        metrics_by_split[str(split)] = _metrics(pd.to_numeric(sub["SOH_obs"], errors="coerce").to_numpy(dtype=float), pred)
        if split in forbidden_splits:
            metrics_by_split[str(split)]["hidden_forbidden_split"] = True
    payload: Dict[str, Any] = {
        "protocol": "ASSB111_saturating_v2stable_initializer_train_only",
        "dataset_csv": str(args.dataset_csv),
        "split_manifest_json": str(args.split_manifest_json),
        "fit_splits": sorted(fit_splits),
        "select_splits": sorted(select_splits),
        "forbidden_splits": sorted(forbidden_splits),
        "selected_by": selected_by,
        "floor_selection_mode": mode,
        "floor_prior": float(args.floor_prior),
        "fixed_floor": float(args.fixed_floor if args.fixed_floor is not None else args.floor_prior),
        "candidate_count": len(rows),
        "candidate_table_top10": sorted(rows, key=lambda r: r.get("train_mae", float("inf")))[:10],
        "best": best,
        "initializer": {
            "model_variant": "saturating_v2_stable",
            "soh_floor": floor,
            "soh0": soh0,
            "k_per_cycle": k,
            "damage_rate_scale_init": k,
            "cycle0": float(fit_cycles[0]),
            "gate_gamma": float(args.gate_gamma),
            "rate_correction_bound_recommended": 3.0,
            "residual_bound_recommended": 0.006,
            "freeze_floor_recommended": True,
            "freeze_soh0_recommended": False,
            "floor_selection_mode": mode,
            "floor_prior": float(args.floor_prior),
            "fixed_floor": float(args.fixed_floor if args.fixed_floor is not None else args.floor_prior),
        },
        "n_fit_rows": int(len(fit)),
        "n_forbidden_rows": int(frame["split_l"].isin(forbidden_splits).sum()),
        "metrics_by_split": metrics_by_split,
        "leakage_statement": "Only fit_splits rows were used to fit floor/soh0/k. Test/partial metrics are range-only diagnostics and must not be used by training or checkpoint selection.",
    }
    _write_json(Path(args.output_json), payload)
    print(json.dumps(_json_clean({"output_json": args.output_json, "floor": floor, "soh0": soh0, "k_per_cycle": k, "n_fit_rows": len(fit), "selected_by": selected_by}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
