# -*- coding: utf-8 -*-
"""Fit a deterministic train-only saturating SOH initializer for ASSB-111 v3.

This script is intentionally simple and leakage-aware. It reads the ASSB-111
strict30 dataset, fits a one-dimensional saturating curve using only train
cycles, optionally reports validation metrics for visible-cycle selection, and
writes an initializer JSON consumed by the modified v3 training script.

The fitted baseline is:

    SOH_base(c) = floor + (soh0 - floor) * exp(-k_per_cycle * (cycle_id-c0))

Only rows whose split is in --fit_splits are allowed to affect floor/soh0/k.
Rows in --forbidden_splits are never used for fitting or selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        val = float(x)
        return None if not np.isfinite(val) else val
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    return x


def _write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(payload), f, ensure_ascii=False, indent=2, sort_keys=True)


def _parse_split_list(text: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(text, (list, tuple)):
        raw = []
        for item in text:
            raw.extend(str(item).replace(";", ",").split(","))
    else:
        raw = str(text).replace(";", ",").split(",")
    return tuple(s.strip().lower() for s in raw if s.strip())


def _safe_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    out: Dict[str, float] = {"n": int(mask.sum())}
    if mask.sum() == 0:
        out.update(MAE=float("nan"), RMSE=float("nan"), BIAS=float("nan"), R2=float("nan"), corr=float("nan"))
        return out
    o = obs[mask]
    p = pred[mask]
    err = p - o
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    r2 = float("nan") if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot
    corr = float("nan")
    if o.size >= 2 and np.std(o) > 1e-15 and np.std(p) > 1e-15:
        corr = float(np.corrcoef(o, p)[0, 1])
    out.update(
        MAE=float(np.mean(np.abs(err))),
        RMSE=float(np.sqrt(np.mean(err ** 2))),
        BIAS=float(np.mean(err)),
        R2=r2,
        corr=corr,
        SOH_obs_min=float(np.min(o)),
        SOH_obs_max=float(np.max(o)),
        SOH_pred_min=float(np.min(p)),
        SOH_pred_max=float(np.max(p)),
    )
    return out


def _ensure_split(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    if "split" in out.columns:
        out["split"] = out["split"].astype(str).str.lower()
        return out
    train_min, train_max = manifest.get("train_cycles", [5, 139])
    val_min, val_max = manifest.get("val_cycles", [140, 159])
    test_min, test_max = manifest.get("test_cycles", [160, 521])
    partial = set(int(x) for x in manifest.get("partial_cycles", [522]))
    cycle = out["cycle_id"].astype(int).to_numpy()
    split = np.full(len(out), "unknown", dtype=object)
    split[(cycle >= int(train_min)) & (cycle <= int(train_max))] = "train"
    split[(cycle >= int(val_min)) & (cycle <= int(val_max))] = "val"
    split[(cycle >= int(test_min)) & (cycle <= int(test_max))] = "test"
    for pc in partial:
        split[cycle == pc] = "partial"
    out["split"] = split
    return out


def _predict(cycle: np.ndarray, *, c0: float, floor: float, soh0: float, k_per_cycle: float) -> np.ndarray:
    x = np.maximum(0.0, np.asarray(cycle, dtype=float) - float(c0))
    return float(floor) + (float(soh0) - float(floor)) * np.exp(-float(k_per_cycle) * x)


def _fit_for_floor(cycle: np.ndarray, soh: np.ndarray, floor: float, soh0_mode: str = "first_train") -> Optional[Dict[str, float]]:
    mask = np.isfinite(cycle) & np.isfinite(soh)
    cycle = np.asarray(cycle, dtype=float)[mask]
    soh = np.asarray(soh, dtype=float)[mask]
    if cycle.size < 4:
        return None
    order = np.argsort(cycle)
    cycle = cycle[order]
    soh = soh[order]
    c0 = float(cycle[0])
    if str(soh0_mode).lower() in {"mean_first3", "first3_mean"}:
        soh0 = float(np.mean(soh[: min(3, soh.size)]))
    elif str(soh0_mode).lower() in {"max_train", "max"}:
        soh0 = float(np.max(soh))
    else:
        soh0 = float(soh[0])
    # Keep soh0 physically plausible and above floor.
    soh0 = min(max(soh0, floor + 1.0e-5), 1.05)
    y_raw = (soh - floor) / max(soh0 - floor, 1.0e-12)
    valid = (y_raw > 1.0e-8) & np.isfinite(y_raw)
    if valid.sum() < 3:
        return None
    x = np.maximum(0.0, cycle - c0)
    # Fit log(y) = -k*x through origin for x>0.
    mask2 = valid & (x > 0)
    if mask2.sum() < 2:
        return None
    logy = np.log(np.clip(y_raw[mask2], 1.0e-8, 1.0))
    xx = x[mask2]
    denom = float(np.sum(xx * xx))
    if denom <= 1.0e-30:
        return None
    k = max(0.0, float(-np.sum(xx * logy) / denom))
    pred = _predict(cycle, c0=c0, floor=floor, soh0=soh0, k_per_cycle=k)
    m = _safe_metrics(soh, pred)
    return {"floor": float(floor), "soh0": soh0, "k_per_cycle": k, "c0": c0, "train_mae": float(m["MAE"]), "train_rmse": float(m["RMSE"]), "train_r2": float(m["R2"])}


def _grid_floors(floor_min: float, floor_max: float, floor_step: float) -> List[float]:
    step = max(float(floor_step), 1.0e-6)
    n = int(math.floor((float(floor_max) - float(floor_min)) / step + 0.5))
    vals = [float(floor_min) + i * step for i in range(max(n, 0) + 1)]
    vals = [v for v in vals if v <= float(floor_max) + 1.0e-12]
    if not vals:
        vals = [float(floor_min)]
    return vals


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--split_manifest_json", default="")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_csv", default="")
    p.add_argument("--fit_splits", default="train")
    p.add_argument("--select_splits", default="val")
    p.add_argument("--forbidden_splits", default="test,partial")
    p.add_argument("--floor_min", type=float, default=0.68)
    p.add_argument("--floor_max", type=float, default=0.78)
    p.add_argument("--floor_step", type=float, default=0.001)
    p.add_argument("--soh0_mode", default="first_train", choices=["first_train", "mean_first3", "max_train"])
    p.add_argument("--use_val_for_selection", action="store_true")
    p.add_argument(
        "--floor_selection_mode",
        default="grid_visible",
        choices=["grid_visible", "train_mae", "val_mae", "fixed_prior", "fixed_floor", "prior_fixed", "prior_penalized"],
        help=(
            "How to choose the SOH floor for the v3 initializer. grid_visible/train_mae/val_mae "
            "keep grid-search behavior; fixed_prior/fixed_floor/prior_fixed fit k at a single "
            "floor prior; prior_penalized keeps the grid but penalizes high-floor drift."
        ),
    )
    p.add_argument("--fixed_floor", type=float, default=None, help="SOH floor used in fixed_floor/fixed_prior/prior_fixed mode.")
    p.add_argument("--floor_prior", type=float, default=0.72, help="Physical/prior SOH floor. Default 0.72 for v3 floorfix.")
    p.add_argument("--floor_prior_weight", type=float, default=0.0, help="Alias penalty weight for prior_penalized mode.")
    p.add_argument("--w_floor_preference", type=float, default=None, help="Penalty weight alias; overrides --floor_prior_weight if provided.")
    p.add_argument("--include_forbidden_labels_in_output", action="store_true", help="Debug-only: include forbidden split labels in output CSV. Default hides them.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    frame = pd.read_csv(args.dataset_csv)
    if "cycle_id" not in frame.columns or "SOH_obs" not in frame.columns:
        raise ValueError("dataset_csv must contain cycle_id and SOH_obs columns")
    manifest = _load_json(args.split_manifest_json)
    frame = _ensure_split(frame, manifest)
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    frame["SOH_obs"] = pd.to_numeric(frame["SOH_obs"], errors="coerce")

    fit_splits = _parse_split_list(args.fit_splits)
    select_splits = _parse_split_list(args.select_splits)
    forbidden_splits = _parse_split_list(args.forbidden_splits)
    split_lower = frame["split"].astype(str).str.lower()
    fit_mask = split_lower.isin(fit_splits).to_numpy() & np.isfinite(frame["SOH_obs"].to_numpy(dtype=float))
    select_mask = split_lower.isin(select_splits).to_numpy() & np.isfinite(frame["SOH_obs"].to_numpy(dtype=float))
    forbidden_mask = split_lower.isin(forbidden_splits).to_numpy()

    if np.any(fit_mask & forbidden_mask):
        raise RuntimeError("Initializer fit mask overlaps forbidden splits; refusing to fit.")
    if not np.any(fit_mask):
        raise RuntimeError("No rows selected by --fit_splits for initializer fitting")

    cycle_fit = frame.loc[fit_mask, "cycle_id"].to_numpy(dtype=float)
    soh_fit = frame.loc[fit_mask, "SOH_obs"].to_numpy(dtype=float)

    mode = str(args.floor_selection_mode).strip().lower()
    if mode == "prior_fixed":
        mode = "fixed_prior"
    floor_prior = float(args.floor_prior)
    fixed_floor = float(args.fixed_floor) if args.fixed_floor is not None else floor_prior
    if mode in {"fixed_prior", "fixed_floor"}:
        # v3 floorfix path: do not allow visible val/train MAE over a wide grid to push
        # the floor to 0.78. Fit k/soh0 from train split at this single physical prior.
        floors_to_try = [min(max(fixed_floor, float(args.floor_min)), float(args.floor_max))]
    else:
        floors_to_try = _grid_floors(args.floor_min, args.floor_max, args.floor_step)
    penalty_weight = float(args.w_floor_preference) if args.w_floor_preference is not None else float(args.floor_prior_weight)

    candidates: List[Dict[str, float]] = []
    for floor in floors_to_try:
        cand = _fit_for_floor(cycle_fit, soh_fit, floor=floor, soh0_mode=args.soh0_mode)
        if cand is None:
            continue
        pred_all = _predict(frame["cycle_id"].to_numpy(dtype=float), c0=cand["c0"], floor=cand["floor"], soh0=cand["soh0"], k_per_cycle=cand["k_per_cycle"])
        val_metrics = _safe_metrics(frame.loc[select_mask, "SOH_obs"].to_numpy(dtype=float), pred_all[select_mask]) if np.any(select_mask) else {"MAE": float("inf"), "R2": float("nan"), "RMSE": float("inf"), "n": 0}
        cand["val_mae"] = float(val_metrics.get("MAE", float("inf")))
        cand["val_rmse"] = float(val_metrics.get("RMSE", float("inf")))
        cand["val_r2"] = float(val_metrics.get("R2", float("nan")))
        cand["floor_prior"] = floor_prior
        cand["floor_prior_penalty"] = penalty_weight * float((float(cand["floor"]) - floor_prior) ** 2)
        cand["score_prior_penalized"] = float(cand["val_mae"] if args.use_val_for_selection and np.any(select_mask) else cand["train_mae"]) + cand["floor_prior_penalty"]
        candidates.append(cand)
    if not candidates:
        raise RuntimeError("No valid saturating initializer candidate could be fit")

    if mode in {"fixed_prior", "fixed_floor"}:
        key = lambda d: (d["train_mae"], d["val_mae"])
        selected_by = "fixed_floor_train_only_k_soh0"
    elif mode == "prior_penalized":
        key = lambda d: (d["score_prior_penalized"], d["val_mae"], d["train_mae"], abs(d["floor"] - floor_prior))
        selected_by = "visible_mae_plus_floor_prior_penalty"
    elif mode == "val_mae" and args.use_val_for_selection and np.any(select_mask):
        key = lambda d: (d["val_mae"], d["train_mae"])
        selected_by = "val_mae_then_train_mae"
    else:
        key = lambda d: (d["train_mae"], d["val_mae"])
        selected_by = "train_mae_then_val_mae"
    best = sorted(candidates, key=key)[0]
    pred_all = _predict(frame["cycle_id"].to_numpy(dtype=float), c0=best["c0"], floor=best["floor"], soh0=best["soh0"], k_per_cycle=best["k_per_cycle"])

    metrics_by_split: Dict[str, Dict[str, float]] = {}
    visible_splits = tuple(set(fit_splits + select_splits))
    for split in sorted(frame["split"].astype(str).str.lower().unique()):
        m = split_lower.eq(split).to_numpy() & np.isfinite(frame["SOH_obs"].to_numpy(dtype=float))
        if split in forbidden_splits:
            # Do not compute or write held-out label metrics in the initializer.
            # The final evaluator may do that after training; this script must not.
            metrics_by_split[split] = {
                "n": int(np.sum(m)),
                "hidden_forbidden_split": True,
                "SOH_pred_min": float(np.nanmin(pred_all[m])) if np.any(m) else float("nan"),
                "SOH_pred_max": float(np.nanmax(pred_all[m])) if np.any(m) else float("nan"),
            }
        else:
            metrics_by_split[split] = _safe_metrics(frame.loc[m, "SOH_obs"].to_numpy(dtype=float), pred_all[m])
    visible_mask = split_lower.isin(visible_splits).to_numpy()
    metrics_by_split["visible"] = _safe_metrics(frame.loc[visible_mask, "SOH_obs"].to_numpy(dtype=float), pred_all[visible_mask])

    payload: Dict[str, Any] = {
        "protocol": "ASSB111_saturating_v3_initializer_train_only",
        "dataset_csv": str(args.dataset_csv),
        "split_manifest_json": str(args.split_manifest_json) if args.split_manifest_json else None,
        "fit_splits": list(fit_splits),
        "select_splits": list(select_splits),
        "forbidden_splits": list(forbidden_splits),
        "selected_by": selected_by,
        "floor_selection_mode": mode,
        "floor_prior": floor_prior,
        "fixed_floor": fixed_floor if mode in {"fixed_prior", "fixed_floor"} else None,
        "floor_prior_weight": penalty_weight,
        "n_fit_rows": int(fit_mask.sum()),
        "n_select_rows": int(select_mask.sum()),
        "n_forbidden_rows": int(forbidden_mask.sum()),
        "best": best,
        "initializer": {
            "model_variant": "saturating_v3",
            "soh_floor": float(best["floor"]),
            "soh0": float(best["soh0"]),
            "cycle0": float(best["c0"]),
            "k_per_cycle": float(best["k_per_cycle"]),
            "damage_rate_scale_init": float(best["k_per_cycle"]),
            "freeze_floor_recommended": True if mode in {"fixed_prior", "fixed_floor"} else False,
            "floor_regularization_weight_recommended": 1.0 if mode in {"fixed_prior", "fixed_floor"} else 0.5,
            "residual_bound_recommended": 0.004,
            "floor_selection_mode": mode,
            "floor_prior": floor_prior,
            "fixed_floor": fixed_floor if mode in {"fixed_prior", "fixed_floor"} else None,
        },
        "metrics_by_split": metrics_by_split,
        "candidate_count": len(candidates),
        "candidate_table_top10": sorted(candidates, key=key)[:10],
        "leakage_statement": "Only fit_splits rows were used to fit floor/soh0/k. Forbidden splits were not used for fitting or selection.",
    }
    _write_json(payload, args.output_json)

    if args.output_csv:
        out = frame[["cycle_id", "split"]].copy()
        visible_label_mask = np.ones(len(frame), dtype=bool) if args.include_forbidden_labels_in_output else ~forbidden_mask
        if "Q_obs_Ah" in frame.columns:
            out["Q_obs_Ah_visible_only"] = frame["Q_obs_Ah"].where(visible_label_mask, np.nan)
        out["SOH_obs_visible_only"] = frame["SOH_obs"].where(visible_label_mask, np.nan)
        out["SOH_init_pred"] = pred_all
        out["fit_used"] = fit_mask
        out["select_used"] = select_mask
        out["forbidden_split"] = forbidden_mask
        p = Path(args.output_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False, encoding="utf-8-sig")

    print(json.dumps(_json_clean({"output_json": args.output_json, "best": best, "metrics_by_split": metrics_by_split}), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
