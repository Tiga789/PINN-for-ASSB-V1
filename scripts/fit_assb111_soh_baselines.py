# -*- coding: utf-8 -*-
"""Fit simple train-only SOH baselines for ASSB-111 strict30 diagnostics.

The baselines are sanity checks for the saturating SOH head. They are not a
replacement for ModelFin_111 and they do not use test labels for fitting or
model selection. By default, the JSON report contains only train/val/all-visible
metrics. Use ``--include_test_report`` only after the final evaluator stage.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


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


def _as_float(s: Sequence[Any]) -> np.ndarray:
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=np.float64)


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    out = {"n": int(mask.sum()), "MAE": None, "RMSE": None, "BIAS": None, "R2": None, "corr": None, "NMAE": None, "NRMSE": None}
    if not mask.any():
        return out
    yy, pp = y[mask], p[mask]
    e = pp - yy
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    out.update(MAE=mae, RMSE=rmse, BIAS=float(np.mean(e)))
    denom = float(np.nanmax(yy) - np.nanmin(yy))
    if denom > 1e-30:
        out["NMAE"] = float(mae / denom)
        out["NRMSE"] = float(rmse / denom)
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    if ss_tot > 1e-30:
        out["R2"] = float(1.0 - np.sum(e * e) / ss_tot)
    if yy.size >= 2 and float(np.std(yy)) > 1e-30 and float(np.std(pp)) > 1e-30:
        out["corr"] = float(np.corrcoef(yy, pp)[0, 1])
    return out


def _design_poly(x: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        cols.append(x ** d)
    return np.vstack(cols).T


def _fit_lstsq(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    X = _design_poly(x, degree)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _predict_poly(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return _design_poly(x, len(coef) - 1) @ coef


@dataclass
class BaselineResult:
    name: str
    params: Dict[str, Any]
    pred: np.ndarray
    selection_score: float
    selection_metric: str


def _x_from_cycles(cycles: np.ndarray, train_cycles: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    c0 = float(np.nanmin(train_cycles))
    span = float(np.nanmax(train_cycles) - c0)
    if span <= 1e-12:
        span = 1.0
    return (cycles - c0) / span, {"cycle_origin": c0, "cycle_span_train": span}


def _fit_constant(y_train: np.ndarray, n: int, *, soh_min: float, soh_max: float) -> BaselineResult:
    value = float(np.nanmean(y_train))
    return BaselineResult("constant_train_mean", {"value": value}, np.clip(np.full(n, value), soh_min, soh_max), float("nan"), "val_MAE")


def _fit_linear(x_all: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, *, soh_min: float, soh_max: float) -> BaselineResult:
    coef = _fit_lstsq(x_train, y_train, degree=1)
    # SOH should not increase materially with cycle in this strict diagnostic.
    if len(coef) > 1 and coef[1] > 0:
        coef[1] = 0.0
    pred = np.clip(_predict_poly(x_all, coef), soh_min, soh_max)
    return BaselineResult("linear_train_only", {"coef": coef.tolist()}, pred, float("nan"), "val_MAE")


def _fit_quadratic(x_all: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, *, soh_min: float, soh_max: float) -> BaselineResult:
    coef = _fit_lstsq(x_train, y_train, degree=2)
    pred = np.clip(_predict_poly(x_all, coef), soh_min, soh_max)
    return BaselineResult("quadratic_train_only", {"coef": coef.tolist()}, pred, float("nan"), "val_MAE")


def _fit_exp_no_floor(x_all: np.ndarray, x_train: np.ndarray, y_train: np.ndarray, *, soh_min: float, soh_max: float) -> BaselineResult:
    y = np.clip(y_train, max(soh_min, 1e-8), soh_max)
    coef = _fit_lstsq(x_train, np.log(y), degree=1)
    if len(coef) > 1 and coef[1] > 0:
        coef[1] = 0.0
    pred = np.exp(_predict_poly(x_all, coef))
    return BaselineResult("exponential_no_floor_train_only", {"log_coef": coef.tolist()}, np.clip(pred, soh_min, soh_max), float("nan"), "val_MAE")


def _fit_saturating_exp_grid(
    x_all: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    floor_min: float,
    floor_max: float,
    floor_grid: int,
    soh_min: float,
    soh_max: float,
) -> BaselineResult:
    best: Optional[BaselineResult] = None
    floors = np.linspace(float(floor_min), float(floor_max), int(max(floor_grid, 3)))
    for floor in floors:
        if np.nanmin(y_train - floor) <= 1e-8:
            continue
        z_train = np.log(np.maximum(y_train - floor, 1e-12))
        coef = _fit_lstsq(x_train, z_train, degree=1)
        # Positive slope would mean apparent recovery; keep baseline monotone.
        if len(coef) > 1 and coef[1] > 0:
            coef[1] = 0.0
        pred = floor + np.exp(_predict_poly(x_all, coef))
        pred = np.clip(pred, soh_min, soh_max)
        if y_val.size > 0 and np.isfinite(y_val).any():
            # Select using val only; never use test for selection.
            val_pred = floor + np.exp(_predict_poly(x_val, coef))
            score = float(np.nanmean(np.abs(np.clip(val_pred, soh_min, soh_max) - y_val)))
            metric = "val_MAE"
        else:
            train_pred = floor + np.exp(_predict_poly(x_train, coef))
            score = float(np.nanmean(np.abs(np.clip(train_pred, soh_min, soh_max) - y_train)))
            metric = "train_MAE"
        res = BaselineResult("saturating_exponential_train_val_selected", {"floor": float(floor), "log_coef": coef.tolist()}, pred, score, metric)
        if best is None or score < best.selection_score:
            best = res
    if best is None:
        # Fallback to a constant if all candidate floors are invalid.
        best = _fit_constant(y_train, len(x_all), soh_min=soh_min, soh_max=soh_max)
        best.name = "saturating_exponential_fallback_constant"
        best.selection_score = float("nan")
    return best


def _split_mask(df: pd.DataFrame, split_col: str, splits: Iterable[str]) -> np.ndarray:
    wanted = {str(s).strip() for s in splits if str(s).strip()}
    return df[split_col].astype(str).isin(wanted).to_numpy(dtype=bool)


def _metrics_by_splits(df: pd.DataFrame, pred: np.ndarray, *, obs_col: str, split_col: str, include_test: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    y = _as_float(df[obs_col])
    allowed = ["train", "val", "all_visible"]
    for split in sorted(str(s) for s in df[split_col].dropna().unique()):
        if split == "test" and not include_test:
            continue
        mask = df[split_col].astype(str).to_numpy() == split
        out[split] = _metrics(y[mask], pred[mask])
    visible = df[split_col].astype(str).isin(["train", "val"]).to_numpy(dtype=bool)
    out["train_val_visible"] = _metrics(y[visible], pred[visible])
    out["all"] = _metrics(y, pred) if include_test else {"note": "omitted by default; use --include_test_report after final evaluator stage"}
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit train-only SOH baselines for ASSB111 strict30")
    p.add_argument("--dataset_csv", default=r"Data\assb111\dataset_strict30.csv")
    p.add_argument("--output_dir", default=r"EvalFin_111_soh_baselines")
    p.add_argument("--output_json", default="")
    p.add_argument("--output_predictions_csv", default="")
    p.add_argument("--cycle_col", default="cycle_id")
    p.add_argument("--split_col", default="split")
    p.add_argument("--obs_col", default="SOH_obs")
    p.add_argument("--train_splits", default="train")
    p.add_argument("--val_splits", default="val")
    p.add_argument("--floor_min", type=float, default=0.65)
    p.add_argument("--floor_max", type=float, default=0.85)
    p.add_argument("--floor_grid", type=int, default=101)
    p.add_argument("--soh_min", type=float, default=0.60)
    p.add_argument("--soh_max", type=float, default=1.05)
    p.add_argument("--include_test_report", action="store_true", help="Report test metrics after final evaluation. Never used for fitting or selection.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    dataset_csv = Path(args.dataset_csv)
    if not dataset_csv.exists():
        raise FileNotFoundError(dataset_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json) if args.output_json else out_dir / "baseline_train_val_report.json"
    output_predictions_csv = Path(args.output_predictions_csv) if args.output_predictions_csv else out_dir / "baseline_predictions_by_cycle.csv"

    df = pd.read_csv(dataset_csv).copy()
    required = [args.cycle_col, args.split_col, args.obs_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {dataset_csv}: {missing}")
    df[args.cycle_col] = pd.to_numeric(df[args.cycle_col], errors="coerce")
    df[args.obs_col] = pd.to_numeric(df[args.obs_col], errors="coerce")
    df = df[np.isfinite(df[args.cycle_col])].sort_values(args.cycle_col).reset_index(drop=True)

    train_splits = [s.strip() for s in str(args.train_splits).split(",")]
    val_splits = [s.strip() for s in str(args.val_splits).split(",")]
    train_mask = _split_mask(df, args.split_col, train_splits) & np.isfinite(_as_float(df[args.obs_col]))
    val_mask = _split_mask(df, args.split_col, val_splits) & np.isfinite(_as_float(df[args.obs_col]))
    if train_mask.sum() < 3:
        raise RuntimeError(f"Need at least 3 train rows, got {int(train_mask.sum())}")
    cycles = _as_float(df[args.cycle_col])
    y = _as_float(df[args.obs_col])
    x_all, x_meta = _x_from_cycles(cycles, cycles[train_mask])
    x_train = x_all[train_mask]
    y_train = y[train_mask]
    x_val = x_all[val_mask]
    y_val = y[val_mask]

    results: List[BaselineResult] = []
    results.append(_fit_constant(y_train, len(df), soh_min=float(args.soh_min), soh_max=float(args.soh_max)))
    results.append(_fit_linear(x_all, x_train, y_train, soh_min=float(args.soh_min), soh_max=float(args.soh_max)))
    results.append(_fit_quadratic(x_all, x_train, y_train, soh_min=float(args.soh_min), soh_max=float(args.soh_max)))
    results.append(_fit_exp_no_floor(x_all, x_train, y_train, soh_min=float(args.soh_min), soh_max=float(args.soh_max)))
    results.append(_fit_saturating_exp_grid(
        x_all, x_train, y_train, x_val, y_val,
        floor_min=float(args.floor_min), floor_max=float(args.floor_max), floor_grid=int(args.floor_grid),
        soh_min=float(args.soh_min), soh_max=float(args.soh_max),
    ))

    pred_df = df[[args.cycle_col, args.split_col, args.obs_col]].copy()
    report_models: Dict[str, Any] = {}
    for res in results:
        pred_col = f"SOH_pred_{res.name}"
        pred_df[pred_col] = res.pred
        report_models[res.name] = {
            "params": res.params,
            "selection_metric": res.selection_metric,
            "selection_score": res.selection_score,
            "metrics_by_split": _metrics_by_splits(df, res.pred, obs_col=args.obs_col, split_col=args.split_col, include_test=bool(args.include_test_report)),
        }
    pred_df.to_csv(output_predictions_csv, index=False, encoding="utf-8-sig")

    # Pick a visible-only baseline for reference, preferring validation MAE.
    def visible_score(item: Tuple[str, Any]) -> float:
        metrics = item[1]["metrics_by_split"]
        if "val" in metrics and metrics["val"].get("MAE") is not None:
            return float(metrics["val"].get("MAE"))
        if "train_val_visible" in metrics and metrics["train_val_visible"].get("MAE") is not None:
            return float(metrics["train_val_visible"].get("MAE"))
        return float("inf")

    best_name, best_payload = min(report_models.items(), key=visible_score)
    report = {
        "dataset_csv": str(dataset_csv),
        "output_predictions_csv": str(output_predictions_csv),
        "strict_note": "All baselines are fit on train split only. Validation may select the saturating floor. Test metrics are omitted unless --include_test_report is used after final evaluation.",
        "include_test_report": bool(args.include_test_report),
        "x_normalization": x_meta,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "best_visible_baseline": best_name,
        "models": report_models,
    }
    _write_json(output_json, report)
    print(json.dumps(_json_clean({
        "output_json": str(output_json),
        "output_predictions_csv": str(output_predictions_csv),
        "best_visible_baseline": best_name,
        "best_visible_metrics": best_payload.get("metrics_by_split", {}).get("val") or best_payload.get("metrics_by_split", {}).get("train_val_visible"),
    }), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
