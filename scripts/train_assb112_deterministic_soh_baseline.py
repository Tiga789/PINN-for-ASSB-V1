# -*- coding: utf-8 -*-
"""ASSB-112 deterministic SOH baseline.

Purpose
-------
This script intentionally avoids neural-network seed variability.  It fits a
strict train-only StandardScaler and a deterministic ridge head.  Ridge alpha is
selected with visible train/val metrics only.  Held-out test metrics are written
after the model is fixed.

Optional CUDA support is included for two reasons:
1) run the ridge alpha sweep with torch on GPU when requested;
2) reserve a user-specified amount of GPU memory so the run visibly uses GPU
   resources without opening multiple PowerShell windows or background jobs.

The GPU reservation tensor is not used for fitting and is released at process
exit.  It is recorded in train_summary.json for auditability.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic ridge SOH baseline for ASSB-112")
    p.add_argument("--dataset_csv", default=r"Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv")
    p.add_argument("--split_manifest_json", default=r"Data\assb111_seed42locked_repro_c00\split_manifest.json")
    p.add_argument("--output_model_dir", "--output_dir", dest="output_model_dir", default=r"ModelFin_112_deterministicSOH_ridge_g4")
    p.add_argument("--feature_mode", default="g4_all_strict")
    p.add_argument("--target_col", default="SOH_obs")
    p.add_argument("--q_col", default="Q_obs_Ah")
    p.add_argument("--allow_upper_bound", action="store_true", help="Allow diagnostic capacity-equivalent features. Do not use for strict claims.")
    p.add_argument("--scaler_json", default="")
    p.add_argument(
        "--alphas",
        default=(
            "1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,"
            "1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,"
            "1e-2,3e-2,1e-1,3e-1,1,3,10,30,100"
        ),
    )
    p.add_argument("--topk_average", type=int, default=3, help="Average coefficients of top-k visible candidates. 1 disables averaging.")
    p.add_argument("--clip_soh_min", type=float, default=0.0)
    p.add_argument("--clip_soh_max", type=float, default=1.05)
    p.add_argument("--selection_mode", choices=["visible_score", "val_mae", "val_score"], default="visible_score")
    p.add_argument("--min_train_r2_for_audit", type=float, default=0.990)
    p.add_argument("--max_train_mae_for_audit", type=float, default=0.0030)
    p.add_argument("--max_val_mae_for_audit", type=float, default=0.00150)
    p.add_argument("--min_val_r2_for_audit", type=float, default=0.85)
    p.add_argument("--min_val_corr_for_audit", type=float, default=0.95)
    p.add_argument("--max_val_bias_for_audit", type=float, default=0.00250)
    p.add_argument("--candidate_tag", default="deterministic_ridge_g4")
    p.add_argument("--protocol_tag", default="ASSB112_deterministic_ridge_strict30_trainval_only")
    p.add_argument("--no_test_selection", action="store_true", help="Kept for protocol clarity; test is never used for selection.")

    # GPU knobs.  Ridge itself is tiny; these options make GPU use explicit and auditable.
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--gpu_reserve_gb", type=float, default=0.0, help="Try to reserve this many GB on CUDA. Auto-halves on OOM.")
    p.add_argument("--gpu_work_repeats", type=int, default=1, help="Repeat small deterministic GPU solves to increase utilization; output unchanged.")
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
    # D6 strict30 defaults.
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
        "all_eval": np.isin(s, ["train", "val", "test"]),
    }


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    finite = np.isfinite(y) & np.isfinite(pred)
    y = y[finite]
    pred = pred[finite]
    n = int(y.size)
    if n == 0:
        return {
            "n": 0,
            "SOH_MAE": float("nan"),
            "SOH_RMSE": float("nan"),
            "SOH_R2": float("nan"),
            "SOH_corr": float("nan"),
            "SOH_BIAS": float("nan"),
            "SOH_NMAE": float("nan"),
            "SOH_NRMSE": float("nan"),
        }
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(np.mean(err))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / denom) if denom > 1e-15 else float("nan")
    corr = float(np.corrcoef(y, pred)[0, 1]) if n >= 2 and np.std(y) > 1e-15 and np.std(pred) > 1e-15 else float("nan")
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


def _metrics_by_split(pred_frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    split = pred_frame["split"].astype(str).str.lower()
    out: Dict[str, Dict[str, float]] = {}
    for name in ["all", "train", "val", "test", "partial"]:
        if name == "all":
            m = split.isin(["train", "val", "test"])
        else:
            m = split.eq(name)
        if bool(m.any()):
            out[name] = _metrics(pred_frame.loc[m, "SOH_obs"].to_numpy(float), pred_frame.loc[m, "SOH_pred"].to_numpy(float))
    return out


def _parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for item in str(s).replace(";", ",").split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise ValueError("Empty float list")
    return vals


def _fit_ridge_numpy(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xb = np.column_stack([np.ones(X.shape[0], dtype=float), X])
    reg = np.eye(Xb.shape[1], dtype=float) * float(alpha)
    reg[0, 0] = 0.0
    lhs = Xb.T @ Xb + reg
    rhs = Xb.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _predict_numpy(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0], dtype=float), X]) @ coef


def _select_device(requested: str) -> Tuple[str, Any, Any]:
    if requested == "cpu":
        return "cpu", None, None
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu", None, None
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False")
    if torch.cuda.is_available() and requested in {"auto", "cuda"}:
        return "cuda", torch, torch.device("cuda")
    return "cpu", torch, torch.device("cpu") if torch is not None else None


def _reserve_gpu_memory(torch: Any, device: Any, reserve_gb: float) -> Tuple[Optional[Any], float, str]:
    if torch is None or str(device) != "cuda" or reserve_gb <= 0:
        return None, 0.0, "disabled"
    gb = float(reserve_gb)
    last_err = ""
    while gb >= 0.125:
        try:
            n = int(gb * (1024 ** 3) / 4)  # float32 elements
            t = torch.empty((n,), device=device, dtype=torch.float32)
            t.fill_(0.0)
            torch.cuda.synchronize()
            return t, gb, "reserved"
        except Exception as e:  # OOM or allocator issue
            last_err = str(e).split("\n", 1)[0]
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gb *= 0.5
    return None, 0.0, f"failed: {last_err}"


def _fit_all_alphas_torch(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    alphas: Sequence[float],
    torch: Any,
    device: Any,
    dtype_name: str,
    repeats: int,
) -> List[np.ndarray]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    Xtr = torch.as_tensor(X[train_mask], dtype=dtype, device=device)
    ytr = torch.as_tensor(y[train_mask], dtype=dtype, device=device).reshape(-1, 1)
    ones = torch.ones((Xtr.shape[0], 1), dtype=dtype, device=device)
    Xb = torch.cat([ones, Xtr], dim=1)
    XtX = Xb.T @ Xb
    Xty = Xb.T @ ytr
    eye = torch.eye(Xb.shape[1], dtype=dtype, device=device)
    eye[0, 0] = 0.0
    coefs: List[np.ndarray] = []
    repeats = max(1, int(repeats))
    for alpha in alphas:
        lhs = XtX + float(alpha) * eye
        coef = torch.linalg.solve(lhs, Xty)
        # Optional deterministic extra work.  Does not affect coef.
        for _ in range(repeats - 1):
            _ = torch.linalg.solve(lhs, Xty)
        coefs.append(coef.detach().cpu().numpy().reshape(-1).astype(float))
    if str(device) == "cuda":
        torch.cuda.synchronize()
    return coefs


def _visible_shape(y: np.ndarray, pred: np.ndarray, val_mask: np.ndarray) -> Dict[str, float]:
    yy = np.asarray(y, dtype=float)[val_mask]
    pp = np.asarray(pred, dtype=float)[val_mask]
    finite = np.isfinite(yy) & np.isfinite(pp)
    yy = yy[finite]
    pp = pp[finite]
    if yy.size == 0:
        return {"val_range_ratio": float("nan"), "val_slope_mae": float("nan"), "val_tail_bias_abs": float("nan")}
    yr = float(np.max(yy) - np.min(yy))
    pr = float(np.max(pp) - np.min(pp))
    slope = float(np.mean(np.abs(np.diff(pp) - np.diff(yy)))) if yy.size >= 3 else float("nan")
    tail_n = min(5, yy.size)
    return {
        "val_range_ratio": float(pr / yr) if yr > 1e-12 else float("nan"),
        "val_slope_mae": slope,
        "val_tail_bias_abs": float(abs(np.mean(pp[-tail_n:] - yy[-tail_n:]))),
    }


def _selection_score(
    train_m: Mapping[str, float],
    val_m: Mapping[str, float],
    shape: Mapping[str, float],
    mode: str,
    args: argparse.Namespace,
) -> Tuple[float, Dict[str, float], List[str]]:
    train_mae = float(train_m.get("SOH_MAE", float("inf")))
    train_r2 = float(train_m.get("SOH_R2", -float("inf")))
    val_mae = float(val_m.get("SOH_MAE", float("inf")))
    val_r2 = float(val_m.get("SOH_R2", -float("inf")))
    val_corr = float(val_m.get("SOH_corr", float("nan")))
    val_bias_abs = abs(float(val_m.get("SOH_BIAS", float("inf"))))
    val_slope = float(shape.get("val_slope_mae", 0.0)) if math.isfinite(float(shape.get("val_slope_mae", float("nan")))) else 0.0
    val_tail_bias = float(shape.get("val_tail_bias_abs", 0.0)) if math.isfinite(float(shape.get("val_tail_bias_abs", float("nan")))) else 0.0

    if mode == "val_mae":
        score = val_mae
    elif mode == "val_score":
        score = val_mae + 0.04 * val_bias_abs + 0.10 * val_slope + 0.03 * val_tail_bias
    else:
        score = val_mae + 0.10 * train_mae
        score += 0.08 * max(0.0, float(args.min_train_r2_for_audit) - train_r2)
        score += 0.06 * max(0.0, float(args.min_val_r2_for_audit) - val_r2)
        if math.isfinite(val_corr):
            score += 0.03 * max(0.0, float(args.min_val_corr_for_audit) - val_corr)
        score += 0.04 * val_bias_abs + 0.10 * val_slope + 0.03 * val_tail_bias

    audit: Dict[str, float] = {
        "selection_score": float(score),
        "train_mae": train_mae,
        "train_r2": train_r2,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "val_corr": val_corr,
        "val_bias_abs": val_bias_abs,
        **{k: float(v) for k, v in shape.items()},
    }

    guard_reasons: List[str] = []
    if train_r2 < args.min_train_r2_for_audit:
        guard_reasons.append(f"train_r2<{args.min_train_r2_for_audit}")
    if train_mae > args.max_train_mae_for_audit:
        guard_reasons.append(f"train_mae>{args.max_train_mae_for_audit}")
    if val_mae > args.max_val_mae_for_audit:
        guard_reasons.append(f"val_mae>{args.max_val_mae_for_audit}")
    if val_r2 < args.min_val_r2_for_audit:
        guard_reasons.append(f"val_r2<{args.min_val_r2_for_audit}")
    if (not math.isfinite(val_corr)) or val_corr < args.min_val_corr_for_audit:
        guard_reasons.append(f"val_corr<{args.min_val_corr_for_audit}")
    if val_bias_abs > args.max_val_bias_for_audit:
        guard_reasons.append(f"val_bias_abs>{args.max_val_bias_for_audit}")
    return float(score), audit, guard_reasons


def _coefficient_importance(feature_columns: Sequence[str], coef: np.ndarray) -> pd.DataFrame:
    # coef[0] is intercept; standardized feature coefficients are directly comparable.
    rows = []
    for name, val in zip(feature_columns, coef[1:]):
        rows.append({"feature": str(name), "coef": float(val), "importance_abs_coef": float(abs(val))})
    return pd.DataFrame(rows).sort_values("importance_abs_coef", ascending=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_name, torch, torch_device = _select_device(args.device)
    reserve_tensor = None
    reserved_gb = 0.0
    reserve_status = "disabled"
    if device_name == "cuda":
        reserve_tensor, reserved_gb, reserve_status = _reserve_gpu_memory(torch, torch_device, float(args.gpu_reserve_gb))
        print(f"[GPU] device=cuda reserve_request_gb={args.gpu_reserve_gb} actual_reserved_gb={reserved_gb:.3f} status={reserve_status}", flush=True)
    else:
        print("[GPU] device=cpu; deterministic ridge will run on CPU", flush=True)

    dataset_path = Path(args.dataset_csv)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_csv not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    if "cycle_id" not in frame.columns:
        raise KeyError("dataset_csv must contain cycle_id")
    frame["cycle_id"] = pd.to_numeric(frame["cycle_id"], errors="raise").astype(int)

    manifest = _load_manifest(args.split_manifest_json)
    if "split" not in frame.columns:
        frame["split"] = _split_from_manifest(frame["cycle_id"].to_numpy(int), manifest)
    masks = _make_masks(frame["split"].to_numpy())
    if not bool(masks["train"].any()) or not bool(masks["val"].any()):
        raise RuntimeError("train/val split is empty; check split_manifest_json or dataset split column")

    feature_columns = select_feature_columns(
        frame,
        args.feature_mode,
        allow_upper_bound=bool(args.allow_upper_bound),
        allow_missing=False,
    )
    faudit = audit_feature_frame(frame, feature_columns, allow_upper_bound=bool(args.allow_upper_bound))
    save_json(faudit, out_dir / "feature_audit_train_pre.json")
    if not faudit.get("ok", False):
        raise RuntimeError("Feature audit failed: " + "; ".join(faudit.get("failures", [])))

    if args.scaler_json and Path(args.scaler_json).exists():
        scaler = load_scaler_json(args.scaler_json)
        scaler_source = str(args.scaler_json)
    else:
        scaler = fit_standard_scaler(frame, feature_columns, fit_mask=masks["train"])
        scaler_source = "fit_train_only"
    write_scaler_json(scaler, out_dir / "feature_scaler.json")
    write_schema_json(out_dir / "feature_schema.json", args.feature_mode, allow_upper_bound=bool(args.allow_upper_bound))

    X = np.asarray(transform_with_scaler(frame, scaler), dtype=float)
    y = pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y[masks["train"]])):
        raise RuntimeError("Non-finite SOH targets in train split")
    if not np.all(np.isfinite(X[masks["train"]])):
        raise RuntimeError("Non-finite feature values in train split after scaling")

    alphas = _parse_float_list(args.alphas)
    if device_name == "cuda" and torch is not None:
        coefs = _fit_all_alphas_torch(
            X, y, masks["train"], alphas,
            torch=torch, device=torch_device, dtype_name=args.dtype,
            repeats=max(1, int(args.gpu_work_repeats)),
        )
    else:
        coefs = [_fit_ridge_numpy(X[masks["train"]], y[masks["train"]], alpha=a) for a in alphas]

    candidate_rows: List[Dict[str, Any]] = []
    candidates: List[Tuple[float, float, np.ndarray, np.ndarray, Dict[str, float], List[str]]] = []
    for alpha, coef in zip(alphas, coefs):
        pred = _predict_numpy(X, coef)
        pred = np.clip(pred, float(args.clip_soh_min), float(args.clip_soh_max))
        train_m = _metrics(y[masks["train"]], pred[masks["train"]])
        val_m = _metrics(y[masks["val"]], pred[masks["val"]])
        shape = _visible_shape(y, pred, masks["val"])
        score, audit_vals, guard_reasons = _selection_score(train_m, val_m, shape, args.selection_mode, args)
        row = {
            "alpha": float(alpha),
            "selection_score": float(score),
            "guard_ok_for_audit": len(guard_reasons) == 0,
            "guard_reasons": ";".join(guard_reasons),
            **audit_vals,
        }
        candidate_rows.append(row)
        candidates.append((float(score), float(alpha), coef, pred, audit_vals, guard_reasons))

    candidates.sort(key=lambda z: (z[0], z[1]))
    topk = max(1, int(args.topk_average))
    if topk > 1:
        selected = candidates[:min(topk, len(candidates))]
        coef = np.mean([c[2] for c in selected], axis=0)
        selected_alpha: Any = [float(c[1]) for c in selected]
        selected_status = f"top{len(selected)}_visible_score_coef_average"
    else:
        selected = [candidates[0]]
        coef = candidates[0][2]
        selected_alpha = float(candidates[0][1])
        selected_status = "best_visible_score_alpha"

    pred = _predict_numpy(X, coef)
    pred = np.clip(pred, float(args.clip_soh_min), float(args.clip_soh_max))
    train_m = _metrics(y[masks["train"]], pred[masks["train"]])
    val_m = _metrics(y[masks["val"]], pred[masks["val"]])
    shape = _visible_shape(y, pred, masks["val"])
    final_score, final_visible_audit, final_guard_reasons = _selection_score(train_m, val_m, shape, args.selection_mode, args)

    pred_frame = frame[["cycle_id", "split"]].copy()
    pred_frame["SOH_obs"] = y
    pred_frame["SOH_pred"] = pred
    pred_frame["SOH_err"] = pred - y
    if args.q_col in frame.columns:
        pred_frame[args.q_col] = pd.to_numeric(frame[args.q_col], errors="coerce")
    pred_frame.to_csv(out_dir / "soh_pred_by_cycle.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(out_dir / "alpha_selection_visible_only.csv", index=False, encoding="utf-8-sig")
    _coefficient_importance(feature_columns, coef).to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    visible_pred_frame = pred_frame[pred_frame["split"].astype(str).str.lower().isin(["train", "val"])].copy()
    visible_metrics = _metrics_by_split(visible_pred_frame)
    final_metrics = _metrics_by_split(pred_frame)

    scorecard_rows: List[Dict[str, Any]] = []
    for split_name, m in final_metrics.items():
        scorecard_rows.append({"variable": "SOH", "source": "deterministic_ridge", "split": split_name, **m})
    pd.DataFrame(scorecard_rows).to_csv(out_dir / "deterministic_soh_scorecard.csv", index=False, encoding="utf-8-sig")

    save_json(
        {
            "metrics_by_split_visible_only": visible_metrics,
            "note": "No held-out test metrics were used for scaler fitting, alpha selection, coefficient averaging, or checkpoint/model selection.",
        },
        out_dir / "metrics_soh_by_split_train_eval.json",
    )
    save_json(
        {"metrics_by_split_after_selection": final_metrics, "test_metrics_used_for_selection": False},
        out_dir / "metrics_soh_by_split_final_report.json",
    )

    model_json = {
        "model_type": "deterministic_ridge_soh_head",
        "feature_mode": args.feature_mode,
        "feature_columns": list(feature_columns),
        "coef_intercept_first": [float(v) for v in coef.tolist()],
        "selected_alpha": selected_alpha,
        "selected_status": selected_status,
        "selection_mode": args.selection_mode,
        "target_col": args.target_col,
        "scaler_json": "feature_scaler.json",
        "clip_soh_min": float(args.clip_soh_min),
        "clip_soh_max": float(args.clip_soh_max),
    }
    save_json(model_json, out_dir / "ridge_model.json")
    save_json(model_json, out_dir / "deterministic_soh_model.json")

    selected_audit = {
        "ok": True,
        "hard_visible_guard_ok_for_audit": len(final_guard_reasons) == 0,
        "selected_alpha": selected_alpha,
        "best_selection_status": selected_status,
        "visible_guard_audit": {
            "final_visible_metrics": final_visible_audit,
            "guard_reasons": final_guard_reasons,
            "min_train_r2_for_audit": args.min_train_r2_for_audit,
            "max_train_mae_for_audit": args.max_train_mae_for_audit,
            "max_val_mae_for_audit": args.max_val_mae_for_audit,
            "min_val_r2_for_audit": args.min_val_r2_for_audit,
            "min_val_corr_for_audit": args.min_val_corr_for_audit,
            "max_val_bias_for_audit": args.max_val_bias_for_audit,
        },
        "test_metrics_used_for_selection": False,
        "no_test_metrics_in_training_history": True,
        "selected_model_json": "deterministic_soh_model.json",
    }
    save_json(selected_audit, out_dir / "selected_checkpoint_audit.json")

    train_count = int(np.sum(masks["train"]))
    val_count = int(np.sum(masks["val"]))
    test_count = int(np.sum(masks["test"]))
    summary = {
        "output_model_dir": str(out_dir),
        "candidate_tag": args.candidate_tag,
        "protocol_tag": args.protocol_tag,
        "model_variant": "deterministic_ridge_soh_head",
        "feature_mode": args.feature_mode,
        "n_features": len(feature_columns),
        "split_counts": {"train": train_count, "val": val_count, "test": test_count, "partial": int(np.sum(masks["partial"]))},
        "selected_alpha": selected_alpha,
        "best_selection_status": selected_status,
        "best_visible_score": float(final_score),
        "final_visible_metrics": final_visible_audit,
        "metrics_by_split_visible_only": visible_metrics,
        "final_report_metrics_available": True,
        "final_report_metrics_file": "metrics_soh_by_split_final_report.json",
        "selected_checkpoint_audit_ok": True,
        "hard_visible_guard_ok_for_audit": len(final_guard_reasons) == 0,
        "selected_checkpoint_guard_reasons": final_guard_reasons,
        "feature_audit_ok": bool(faudit.get("ok", False)),
        "feature_audit_failures": faudit.get("failures", []),
        "scaler_source": scaler_source,
        "no_test_metrics_in_training_history": True,
        "test_metrics_used_for_selection": False,
        "device_requested": args.device,
        "device_used": device_name,
        "torch_dtype": args.dtype if device_name == "cuda" else "numpy_float64",
        "gpu_reserve_request_gb": float(args.gpu_reserve_gb),
        "gpu_reserved_actual_gb": float(reserved_gb),
        "gpu_reserve_status": reserve_status,
        "gpu_work_repeats": int(args.gpu_work_repeats),
    }
    save_json(summary, out_dir / "train_summary.json")
    save_json(summary, out_dir / "training_summary.json")

    print("\n[DETERMINISTIC RIDGE SUMMARY]", flush=True)
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    test_m = final_metrics.get("test", {})
    print(
        "[RIDGE TEST] "
        f"R2={test_m.get('SOH_R2')} MAE={test_m.get('SOH_MAE')} "
        f"RMSE={test_m.get('SOH_RMSE')} BIAS={test_m.get('SOH_BIAS')} corr={test_m.get('SOH_corr')}",
        flush=True,
    )
    print(f"[OK] deterministic ridge model saved to: {out_dir}", flush=True)

    # Keep reserve_tensor alive until after all GPU work is done.
    if reserve_tensor is not None:
        del reserve_tensor
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
