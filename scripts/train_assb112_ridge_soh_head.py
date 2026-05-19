# -*- coding: utf-8 -*-
"""Deterministic ASSB-112 strict30 ridge SOH head.

This is a fast, no-test-selection baseline/teacher for the SOH branch.  It uses
exactly the same strict feature schema as the neural SOH head, fits the scaler
on train cycles only, selects the ridge alpha on visible train/val metrics only,
and writes held-out test metrics only after the selected alpha is fixed.
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

from util.assb_soh_feature_schema import (
    audit_feature_frame,
    fit_standard_scaler,
    load_scaler_json,
    select_feature_columns,
    transform_with_scaler,
    write_scaler_json,
    write_schema_json,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train deterministic ASSB-112 ridge SOH head")
    p.add_argument("--dataset_csv", default=r"Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv")
    p.add_argument("--split_manifest_json", default=r"Data\assb111_seed42locked_repro_c00\split_manifest.json")
    p.add_argument("--output_model_dir", "--output_dir", dest="output_model_dir", default=r"ModelFin_112_ridgeSOH_g4")
    p.add_argument("--feature_mode", default="g4_all_strict")
    p.add_argument("--allow_upper_bound", action="store_true")
    p.add_argument("--target_col", default="SOH_obs")
    p.add_argument("--q_col", default="Q_obs_Ah")
    p.add_argument("--scaler_json", default="")
    p.add_argument("--alphas", default="1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10")
    p.add_argument("--selection_metric", default="visible_score")
    p.add_argument("--min_train_r2_for_best", type=float, default=0.990)
    p.add_argument("--max_train_mae_for_best", type=float, default=0.0030)
    p.add_argument("--max_val_mae_for_best", type=float, default=0.00150)
    p.add_argument("--min_val_r2_for_best", type=float, default=0.85)
    p.add_argument("--min_val_corr_for_best", type=float, default=0.95)
    p.add_argument("--max_val_bias_for_best", type=float, default=0.00250)
    p.add_argument("--topk_average", type=int, default=0, help="Average predictions/coefficients of top-k visible alphas. 0 disables.")
    p.add_argument("--candidate_tag", default="ridgeSOH_g4")
    p.add_argument("--protocol_tag", default="ASSB112_ridgeSOH_strict30_trainval_only")
    p.add_argument("--no_test_selection", action="store_true")
    return p.parse_args(argv)


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


def save_json(obj: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(dict(obj)), f, ensure_ascii=False, indent=2, sort_keys=True)


def _load_manifest(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            m = json.load(f)
    else:
        m = {}
    m.setdefault("train_cycle_from", 5)
    m.setdefault("train_cycle_to", 139)
    m.setdefault("val_cycle_from", 140)
    m.setdefault("val_cycle_to", 159)
    m.setdefault("test_cycle_from", 160)
    m.setdefault("test_cycle_to", 521)
    m.setdefault("partial_cycles", [522])
    return m


def _split_from_manifest(cycles: Sequence[int], manifest: Mapping[str, Any]) -> np.ndarray:
    c = np.asarray(cycles, dtype=int)
    out = np.full(c.shape, "out_of_scope", dtype=object)
    out[(c >= int(manifest.get("train_cycle_from", 5))) & (c <= int(manifest.get("train_cycle_to", 139)))] = "train"
    out[(c >= int(manifest.get("val_cycle_from", 140))) & (c <= int(manifest.get("val_cycle_to", 159)))] = "val"
    out[(c >= int(manifest.get("test_cycle_from", 160))) & (c <= int(manifest.get("test_cycle_to", 521)))] = "test"
    partial = manifest.get("partial_cycles", [522])
    if isinstance(partial, (int, float, str)):
        partial = [partial]
    out[np.isin(c, [int(float(x)) for x in partial])] = "partial"
    return out


def _make_masks(split: Sequence[Any]) -> Dict[str, np.ndarray]:
    s = np.asarray([str(x).lower() for x in split])
    return {
        "train": s == "train",
        "val": s == "val",
        "test": s == "test",
        "partial": s == "partial",
        "visible": np.isin(s, ["train", "val"]),
        "fit": s == "train",
    }


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    finite = np.isfinite(y) & np.isfinite(p)
    y = y[finite]
    p = p[finite]
    n = int(y.size)
    if n == 0:
        return {"n": 0, "SOH_MAE": float("nan"), "SOH_RMSE": float("nan"), "SOH_R2": float("nan"), "SOH_corr": float("nan"), "SOH_BIAS": float("nan"), "SOH_NMAE": float("nan"), "SOH_NRMSE": float("nan")}
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(np.mean(err))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / denom) if denom > 1e-15 else float("nan")
    corr = float(np.corrcoef(y, p)[0, 1]) if n >= 2 and np.std(y) > 1e-15 and np.std(p) > 1e-15 else float("nan")
    yrange = float(np.max(y) - np.min(y)) if n else float("nan")
    return {
        "n": n,
        "SOH_MAE": mae,
        "SOH_RMSE": rmse,
        "SOH_R2": r2,
        "SOH_corr": corr,
        "SOH_BIAS": bias,
        "SOH_NMAE": float(mae / yrange) if yrange > 1e-12 else float("nan"),
        "SOH_NRMSE": float(rmse / yrange) if yrange > 1e-12 else float("nan"),
    }


def _metrics_by_split(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    split = frame["split"].astype(str).str.lower()
    for name in ["all", "train", "val", "test", "partial"]:
        if name == "all":
            m = split.isin(["train", "val", "test"])
        else:
            m = split.eq(name)
        if m.any():
            out[name] = _metrics(frame.loc[m, "SOH_obs"].to_numpy(float), frame.loc[m, "SOH_pred"].to_numpy(float))
    return out


def _shape_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)[mask]
    p = np.asarray(p, dtype=float)[mask]
    finite = np.isfinite(y) & np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if y.size == 0:
        return {"val_range_ratio": float("nan"), "val_slope_mae": float("nan"), "val_tail_bias_abs": float("nan")}
    yr = float(np.max(y) - np.min(y))
    pr = float(np.max(p) - np.min(p))
    if y.size >= 3:
        val_slope_mae = float(np.mean(np.abs(np.diff(p) - np.diff(y))))
    else:
        val_slope_mae = float("nan")
    tail_n = min(5, y.size)
    return {
        "val_range_ratio": float(pr / yr) if yr > 1e-12 else float("nan"),
        "val_slope_mae": val_slope_mae,
        "val_tail_bias_abs": float(abs(np.mean(p[-tail_n:] - y[-tail_n:]))),
    }


def _visible_score(train_m: Mapping[str, float], val_m: Mapping[str, float], y: np.ndarray, p: np.ndarray, masks: Mapping[str, np.ndarray], args) -> Tuple[float, Dict[str, float], List[str]]:
    train_mae = float(train_m.get("SOH_MAE", float("inf")))
    train_r2 = float(train_m.get("SOH_R2", -float("inf")))
    val_mae = float(val_m.get("SOH_MAE", float("inf")))
    val_r2 = float(val_m.get("SOH_R2", -float("inf")))
    val_corr = float(val_m.get("SOH_corr", float("nan")))
    val_bias_abs = abs(float(val_m.get("SOH_BIAS", float("inf"))))
    shape = _shape_metrics(y, p, masks["val"])
    score = val_mae + 0.15 * train_mae
    score += 0.10 * max(0.0, float(args.min_train_r2_for_best) - train_r2)
    score += 0.08 * max(0.0, float(args.min_val_r2_for_best) - val_r2)
    if math.isfinite(val_corr):
        score += 0.03 * max(0.0, float(args.min_val_corr_for_best) - val_corr)
    score += 0.05 * val_bias_abs
    if math.isfinite(shape["val_slope_mae"]):
        score += 0.20 * shape["val_slope_mae"]
    vals = {
        "visible_score": float(score),
        "train_mae": train_mae,
        "train_r2": train_r2,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "val_corr": val_corr,
        "val_bias_abs": val_bias_abs,
        **shape,
    }
    guard_reasons: List[str] = []
    if train_r2 < args.min_train_r2_for_best:
        guard_reasons.append(f"train_r2<{args.min_train_r2_for_best}")
    if train_mae > args.max_train_mae_for_best:
        guard_reasons.append(f"train_mae>{args.max_train_mae_for_best}")
    if val_mae > args.max_val_mae_for_best:
        guard_reasons.append(f"val_mae>{args.max_val_mae_for_best}")
    if val_r2 < args.min_val_r2_for_best:
        guard_reasons.append(f"val_r2<{args.min_val_r2_for_best}")
    if (not math.isfinite(val_corr)) or val_corr < args.min_val_corr_for_best:
        guard_reasons.append(f"val_corr<{args.min_val_corr_for_best}")
    if val_bias_abs > args.max_val_bias_for_best:
        guard_reasons.append(f"val_bias_abs>{args.max_val_bias_for_best}")
    return float(score), vals, guard_reasons


def _parse_alphas(s: str) -> List[float]:
    vals: List[float] = []
    for item in str(s).replace(";", ",").split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("No ridge alphas provided")
    return vals


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.column_stack([np.ones(X.shape[0]), X])
    reg = np.eye(Xb.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0
    lhs = Xb.T @ Xb + reg
    rhs = Xb.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _predict_ridge(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X]) @ coef


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.dataset_csv)
    if "cycle_id" not in frame.columns:
        raise KeyError("dataset_csv must contain cycle_id")
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    manifest = _load_manifest(args.split_manifest_json)
    if "split" not in frame.columns:
        frame["split"] = _split_from_manifest(frame["cycle_id"].to_numpy(int), manifest)
    masks = _make_masks(frame["split"].to_numpy())

    feature_columns = select_feature_columns(frame, args.feature_mode, allow_upper_bound=bool(args.allow_upper_bound), allow_missing=False)
    faudit = audit_feature_frame(frame, feature_columns, allow_upper_bound=bool(args.allow_upper_bound))
    save_json(faudit, out_dir / "feature_audit_train_pre.json")
    if not faudit.get("ok", False):
        raise RuntimeError("Feature audit failed: " + "; ".join(faudit.get("failures", [])))

    if args.scaler_json and Path(args.scaler_json).exists():
        scaler = load_scaler_json(args.scaler_json)
    else:
        scaler = fit_standard_scaler(frame, feature_columns, fit_mask=masks["fit"])
    write_scaler_json(scaler, out_dir / "feature_scaler.json")
    write_schema_json(out_dir / "feature_schema.json", args.feature_mode, allow_upper_bound=bool(args.allow_upper_bound))

    X = np.asarray(transform_with_scaler(frame, scaler), dtype=float)
    y = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y[masks["fit"]])):
        raise RuntimeError("Non-finite target values in train split")

    rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[float, float, np.ndarray, np.ndarray, Dict[str, float], List[str]]] = []
    for alpha in _parse_alphas(args.alphas):
        coef = _fit_ridge(X[masks["fit"]], y[masks["fit"]], alpha=alpha)
        pred = _predict_ridge(X, coef)
        train_m = _metrics(y[masks["train"]], pred[masks["train"]])
        val_m = _metrics(y[masks["val"]], pred[masks["val"]])
        score, vals, guard_reasons = _visible_score(train_m, val_m, y, pred, masks, args)
        rows.append({"alpha": alpha, "visible_score": score, "guard_ok": len(guard_reasons) == 0, "guard_reasons": ";".join(guard_reasons), **vals})
        candidates.append((score, alpha, coef, pred, vals, guard_reasons))

    candidates.sort(key=lambda z: z[0])
    topk = int(args.topk_average)
    if topk and topk > 1:
        chosen = candidates[:topk]
        coef = np.mean([c[2] for c in chosen], axis=0)
        pred = _predict_ridge(X, coef)
        selected_alpha: Any = [float(c[1]) for c in chosen]
        selected_status = f"top{topk}_visible_score_average"
    else:
        score, alpha, coef, pred, vals, guard_reasons = candidates[0]
        selected_alpha = float(alpha)
        selected_status = "visible_score_selected"

    train_m = _metrics(y[masks["train"]], pred[masks["train"]])
    val_m = _metrics(y[masks["val"]], pred[masks["val"]])
    score, vals, guard_reasons = _visible_score(train_m, val_m, y, pred, masks, args)
    hard_guard_ok = len(guard_reasons) == 0

    pred_frame = frame[["cycle_id", "split"]].copy()
    pred_frame["SOH_obs"] = y
    pred_frame["SOH_pred"] = pred
    pred_frame["SOH_err"] = pred - y
    if args.q_col in frame.columns:
        pred_frame[args.q_col] = pd.to_numeric(frame[args.q_col], errors="coerce")
    pred_frame.to_csv(out_dir / "soh_pred_by_cycle.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out_dir / "alpha_selection_visible_only.csv", index=False, encoding="utf-8-sig")

    visible_metrics = _metrics_by_split(pred_frame[pred_frame["split"].astype(str).str.lower().isin(["train", "val"])].copy())
    final_metrics = _metrics_by_split(pred_frame)
    save_json({"metrics_by_split_visible_only": visible_metrics, "note": "No held-out test metrics were used for alpha selection."}, out_dir / "metrics_soh_by_split_train_eval.json")
    save_json({"metrics_by_split_after_selection": final_metrics, "test_metrics_used_for_selection": False}, out_dir / "metrics_soh_by_split_final_report.json")

    model_json = {
        "model_type": "ridge_soh_head",
        "feature_mode": args.feature_mode,
        "feature_columns": feature_columns,
        "coef_intercept_first": [float(v) for v in coef.tolist()],
        "selected_alpha": selected_alpha,
        "selected_status": selected_status,
        "target_col": args.target_col,
        "scaler_json": "feature_scaler.json",
    }
    save_json(model_json, out_dir / "ridge_model.json")

    selected_audit = {
        "ok": True,
        "hard_visible_guard_ok": bool(hard_guard_ok),
        "selected_alpha": selected_alpha,
        "best_selection_status": selected_status,
        "visible_guard": {
            "final_visible_metrics": vals,
            "guard_reasons": guard_reasons,
            "min_train_r2_for_best": args.min_train_r2_for_best,
            "max_train_mae_for_best": args.max_train_mae_for_best,
            "max_val_mae_for_best": args.max_val_mae_for_best,
            "min_val_r2_for_best": args.min_val_r2_for_best,
            "min_val_corr_for_best": args.min_val_corr_for_best,
            "max_val_bias_for_best": args.max_val_bias_for_best,
        },
        "test_metrics_used_for_selection": False,
        "no_test_metrics_in_training_history": True,
        "selected_model_json": "ridge_model.json",
    }
    save_json(selected_audit, out_dir / "selected_checkpoint_audit.json")

    summary = {
        "output_model_dir": str(out_dir),
        "candidate_tag": args.candidate_tag,
        "protocol_tag": args.protocol_tag,
        "model_variant": "ridge_soh_head",
        "feature_mode": args.feature_mode,
        "n_features": len(feature_columns),
        "selected_alpha": selected_alpha,
        "best_selection_status": selected_status,
        "best_visible_score": float(score),
        "final_visible_metrics": vals,
        "metrics_by_split_visible_only": visible_metrics,
        "final_report_metrics_available": True,
        "final_report_metrics_file": "metrics_soh_by_split_final_report.json",
        "selected_checkpoint_audit_ok": True,
        "hard_visible_guard_ok": bool(hard_guard_ok),
        "selected_checkpoint_guard_reasons": guard_reasons,
        "feature_audit_ok": bool(faudit.get("ok", False)),
        "feature_audit_failures": faudit.get("failures", []),
        "no_test_metrics_in_training_history": True,
        "test_metrics_used_for_selection": False,
    }
    save_json(summary, out_dir / "train_summary.json")
    save_json(summary, out_dir / "training_summary.json")
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[OK] ridge selected={selected_alpha} final_report={out_dir / 'metrics_soh_by_split_final_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
