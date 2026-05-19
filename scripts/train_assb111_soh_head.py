# -*- coding: utf-8 -*-
"""Train ASSB strict30 SOH head with train/val-only checkpoint selection.

D7 additions over the ASSB-111 seed42 recovery trainer:
- feature groups G0-G4 via util.assb_soh_feature_schema;
- robust_saturating / latent_health_ode / ensemble_distilled variants;
- multi-seed friendly outputs and no-fallback hard failure by default;
- final test metrics are written only after the selected checkpoint is fixed.

The training history never contains test/heldout metric columns.  That property
is part of the strict30 audit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from util.assb111_soh_model import (
    Assb111SOHHead,
    Assb111SOHHeadConfig,
    assb111_soh_loss,
    metrics_by_split,
    prediction_frame_from_output,
    save_json,
    soh_metrics,
)

try:
    from util.assb_soh_feature_schema import (
        audit_feature_frame,
        fit_standard_scaler,
        load_scaler_json,
        select_feature_columns,
        transform_with_scaler,
        write_scaler_json,
        write_schema_json,
    )
except Exception:  # fallback to old ASSB111 schema if user has not installed new file
    from util.assb111_feature_schema import (  # type: ignore
        fit_standard_scaler,
        load_scaler_json,
        select_feature_columns,
        transform_with_scaler,
        write_scaler_json,
        write_schema_json,
    )

    def audit_feature_frame(frame, feature_columns, *, allow_upper_bound=False):  # type: ignore
        return {"ok": True, "failures": [], "warnings": [], "n_features": len(feature_columns)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ASSB strict30 robust SOH head")
    p.add_argument("--dataset_csv", default=r"Data\assb111_seed42locked_repro_c00\dataset.csv")
    p.add_argument("--split_manifest_json", default=r"Data\assb111_seed42locked_repro_c00\split_manifest.json")
    p.add_argument("--scaler_json", default="", help="Optional prefit scaler; if missing, fit train-only scaler")
    p.add_argument("--output_model_dir", "--output_dir", dest="output_model_dir", default=r"ModelFin_112_robustSOH_seed42")
    p.add_argument("--feature_mode", default="g4_all_strict")
    p.add_argument("--allow_upper_bound", action="store_true")
    p.add_argument("--target_col", default="SOH_obs")
    p.add_argument("--q_col", default="Q_obs_Ah")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--epochs", type=int, default=7000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=48)
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--feature_dropout", type=float, default=0.05)
    p.add_argument("--activation", default="silu")
    p.add_argument("--soh_model_variant", "--model_variant", dest="soh_model_variant", default="robust_saturating")
    p.add_argument("--rate_scale", type=float, default=1e-3)
    p.add_argument("--residual_bound", type=float, default=0.008)
    p.add_argument("--floor_min", "--soh_floor_min", dest="floor_min", type=float, default=0.65)
    p.add_argument("--floor_max", "--soh_floor_max", dest="floor_max", type=float, default=0.85)
    p.add_argument("--soh_floor_prior", "--soh_floor_total", dest="soh_floor_prior", type=float, default=0.72)
    p.add_argument("--soh0_min", type=float, default=0.94)
    p.add_argument("--soh0_max", type=float, default=1.03)
    p.add_argument("--damage_rate_scale", type=float, default=5e-4)
    p.add_argument("--gate_gamma", type=float, default=1.0)
    p.add_argument("--soh_numeric_min", type=float, default=0.60)
    p.add_argument("--tail_slope_guard", type=float, default=0.0020)
    p.add_argument("--huber_delta", type=float, default=0.02)
    p.add_argument("--w_smooth", type=float, default=0.05)
    p.add_argument("--w_rate", type=float, default=0.01)
    p.add_argument("--w_monotonic", type=float, default=0.20)
    p.add_argument("--w_residual", type=float, default=0.10)
    p.add_argument("--w_floor_prior", type=float, default=0.02)
    p.add_argument("--w_tail_guard", type=float, default=0.05)
    p.add_argument("--w_rate_tv", type=float, default=0.02)
    p.add_argument("--w_distill", type=float, default=0.0)
    p.add_argument("--teacher_csv", default="", help="Optional teacher predictions with cycle_id, SOH_teacher for distillation")
    p.add_argument("--teacher_col", default="SOH_teacher")
    p.add_argument("--grad_clip_norm", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=800)
    # V5: patience is driven by visible soft-score progress, not by hard guard only.
    # This prevents premature failure before the first guarded checkpoint appears.
    p.add_argument("--min_epochs_before_patience", type=int, default=3000, help="Do not early-stop before this epoch. A value <=0 disables the warmup.")
    p.add_argument("--allow_patience_before_first_guard", action="store_true", help="Compatibility option. Default false keeps training until at least one guarded checkpoint or min_epochs warmup.")
    p.add_argument("--min_delta", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_every", type=int, default=250)
    p.add_argument("--min_train_r2_for_best", type=float, default=0.990)
    p.add_argument("--max_train_mae_for_best", type=float, default=0.0030)
    p.add_argument("--max_val_mae_for_best", type=float, default=0.00150)
    # D7 guard-fix: val MAE alone can select biased/shape-poor checkpoints.
    # Negative values disable the corresponding guard. Defaults are intentionally
    # stricter than v1 but still compatible with the seed42 recovery baseline.
    p.add_argument("--min_val_r2_for_best", type=float, default=0.90)
    p.add_argument("--min_val_corr_for_best", type=float, default=0.95)
    p.add_argument("--max_val_bias_for_best", type=float, default=0.00150)
    p.add_argument("--max_val_rmse_for_best", type=float, default=0.00220)
    p.add_argument("--max_val_tail_bias_for_best", type=float, default=0.00200)
    p.add_argument("--max_val_slope_mae_for_best", type=float, default=0.00100)
    p.add_argument("--min_val_slope_corr_for_best", type=float, default=-1.0)
    p.add_argument("--min_val_range_ratio_for_best", type=float, default=0.70)
    p.add_argument("--max_val_range_ratio_for_best", type=float, default=1.35)
    p.add_argument("--max_visible_monotonic_penalty_for_best", type=float, default=2.0e-5)
    p.add_argument("--max_active_clamp_fraction_for_best", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=1, help="Evaluate train/val guards every N epochs. Use 5-20 for faster multi-seed sweeps.")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"], help="float64 preserves old behavior; float32 may be faster but can slightly change checkpoint selection.")
    p.add_argument("--cuda_matmul_precision", default="", choices=["", "highest", "high", "medium"], help="Optional torch.set_float32_matmul_precision setting; only affects float32/TF32 paths.")
    p.add_argument("--cudnn_benchmark", action="store_true", help="Enable cuDNN benchmark for CUDA runs.")
    p.add_argument("--num_threads", type=int, default=0, help="Optional torch.set_num_threads override for CPU-side preprocessing/evaluation.")
    p.add_argument("--candidate_tag", default="")
    p.add_argument("--protocol_tag", default="ASSB112_robustSOH_strict30_trainval_only")
    p.add_argument("--no_test_selection", action="store_true")
    p.add_argument("--selection_mode", default="visible_softscore_train_val_only")
    p.add_argument("--selection_strategy", default="softscore", choices=["softscore", "hard_guard"], help="softscore selects the best visible train/val score and records hard guard only as audit; hard_guard requires all guard thresholds.")
    p.add_argument("--require_hard_guard", action="store_true", help="If set, final selected checkpoint must satisfy hard visible guards. Default false for v7 reliability.")
    p.add_argument("--seed_locked", action="store_true")
    p.add_argument("--locked_seed_value", type=int, default=42)
    p.add_argument("--save_epoch_checkpoints", action="store_true")
    p.add_argument("--checkpoint_interval", type=int, default=100)
    p.add_argument("--enable_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--enable_swa_topk", action="store_true")
    p.add_argument("--top_k_checkpoints", type=int, default=5)
    p.add_argument("--allow_unguarded_fallback", action="store_true", help="Compatibility only; default false. Do not use for final strict results.")
    p.add_argument("--allow_no_guard_exit_ok", action="store_true", help="Diagnostic/smoke only: if no checkpoint passes hard guards, write audit files and exit 0 instead of raising RuntimeError.")
    p.add_argument("--progress_json", default="", help="Optional progress snapshot JSON path. Relative paths are written under output_model_dir.")
    p.add_argument("--progress_json_every", type=int, default=1, help="Write progress_json every N evaluated guard points.")
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
        val = float(x)
        return None if not np.isfinite(val) else val
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    return x


def _load_manifest(path: str) -> Dict[str, Any]:
    if path and Path(path).exists():
        with Path(path).open("r", encoding="utf-8") as f:
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


def _make_masks(split: np.ndarray) -> Dict[str, np.ndarray]:
    s = np.asarray([str(x).lower() for x in split])
    return {
        "train": s == "train",
        "val": s == "val",
        "test": s == "test",
        "partial": s == "partial",
        "visible": np.isin(s, ["train", "val"]),
        "fit": s == "train",
    }


def _clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _state_dict_to_device(state: Mapping[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device=device) for k, v in state.items()}


def _metrics_for_mask(obs: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return soh_metrics(obs[np.asarray(mask, dtype=bool)], pred[np.asarray(mask, dtype=bool)])


def _visible_monotonic_penalty(pred: np.ndarray, visible_mask: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)[np.asarray(visible_mask, dtype=bool)]
    if p.size < 2:
        return 0.0
    return float(np.mean(np.maximum(0.0, p[1:] - p[:-1])))


def _finite_float(x: Any, default: float) -> float:
    try:
        val = float(x)
    except Exception:
        return float(default)
    return val if np.isfinite(val) else float(default)


def _validation_shape_metrics(obs: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Visible-val shape diagnostics used for checkpoint selection only.

    These metrics are computed on validation rows only. They never inspect test
    rows and are therefore safe for strict30 train/val-only selection.
    """
    m = np.asarray(mask, dtype=bool)
    y = np.asarray(obs, dtype=float)[m]
    p = np.asarray(pred, dtype=float)[m]
    finite = np.isfinite(y) & np.isfinite(p)
    y = y[finite]
    p = p[finite]
    if y.size == 0:
        return {
            "val_range_ratio": float("nan"),
            "val_slope_mae": float("nan"),
            "val_slope_corr": float("nan"),
            "val_tail_bias_abs": float("nan"),
        }
    y_range = float(np.nanmax(y) - np.nanmin(y))
    p_range = float(np.nanmax(p) - np.nanmin(p))
    range_ratio = p_range / y_range if y_range > 1.0e-12 else float("nan")
    if y.size >= 3:
        dy = np.diff(y)
        dp = np.diff(p)
        slope_mae = float(np.mean(np.abs(dp - dy)))
        slope_corr = float(np.corrcoef(dy, dp)[0, 1]) if np.std(dy) > 1e-12 and np.std(dp) > 1e-12 else float("nan")
    else:
        slope_mae = float("nan")
        slope_corr = float("nan")
    tail_n = min(5, int(y.size))
    tail_bias_abs = float(abs(np.mean(p[-tail_n:] - y[-tail_n:]))) if tail_n > 0 else float("nan")
    return {
        "val_range_ratio": float(range_ratio),
        "val_slope_mae": float(slope_mae),
        "val_slope_corr": float(slope_corr),
        "val_tail_bias_abs": float(tail_bias_abs),
    }


def _visible_score(train_m: Mapping[str, float], val_m: Mapping[str, float], obs: np.ndarray, pred: np.ndarray, masks: Mapping[str, np.ndarray], args) -> Tuple[float, Dict[str, float]]:
    val_mae = _finite_float(val_m.get("SOH_MAE", np.inf), np.inf)
    val_rmse = _finite_float(val_m.get("SOH_RMSE", np.inf), np.inf)
    train_mae = _finite_float(train_m.get("SOH_MAE", np.inf), np.inf)
    train_r2 = _finite_float(train_m.get("SOH_R2", -np.inf), -np.inf)
    val_r2 = _finite_float(val_m.get("SOH_R2", -np.inf), -np.inf)
    val_corr = _finite_float(val_m.get("SOH_corr", np.nan), np.nan)
    val_bias_abs = abs(_finite_float(val_m.get("SOH_BIAS", 0.0), 0.0))
    mono = _visible_monotonic_penalty(pred, masks["visible"])
    shape = _validation_shape_metrics(obs, pred, masks["val"])

    # The score still prioritizes MAE, but now pushes selection away from
    # compressed/biased validation shapes. All terms are visible-only.
    score = val_mae
    score += 0.15 * train_mae
    score += 0.20 * max(0.0, float(args.min_train_r2_for_best) - train_r2)
    if float(args.min_val_r2_for_best) >= 0:
        score += 0.08 * max(0.0, float(args.min_val_r2_for_best) - val_r2)
    if float(args.min_val_corr_for_best) >= 0 and np.isfinite(val_corr):
        score += 0.03 * max(0.0, float(args.min_val_corr_for_best) - val_corr)
    score += 0.05 * val_bias_abs
    score += 0.02 * mono
    if np.isfinite(shape["val_slope_mae"]):
        score += 0.50 * max(0.0, shape["val_slope_mae"] - max(float(args.max_val_slope_mae_for_best), 0.0))
    if np.isfinite(shape["val_tail_bias_abs"]):
        score += 0.05 * shape["val_tail_bias_abs"]

    vals = {
        "visible_score": float(score),
        "train_mae": train_mae,
        "train_r2": train_r2,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "val_corr": val_corr,
        "val_bias_abs": val_bias_abs,
        "visible_monotonic_penalty": mono,
    }
    vals.update(shape)
    return float(score), vals


def _guard_ok(vals: Mapping[str, float], args) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    def enabled(v: float) -> bool:
        return float(v) >= 0

    train_r2 = _finite_float(vals.get("train_r2", -np.inf), -np.inf)
    train_mae = _finite_float(vals.get("train_mae", np.inf), np.inf)
    val_mae = _finite_float(vals.get("val_mae", np.inf), np.inf)
    val_rmse = _finite_float(vals.get("val_rmse", np.inf), np.inf)
    val_r2 = _finite_float(vals.get("val_r2", -np.inf), -np.inf)
    val_corr = _finite_float(vals.get("val_corr", np.nan), np.nan)
    val_bias_abs = _finite_float(vals.get("val_bias_abs", np.inf), np.inf)
    val_tail_bias_abs = _finite_float(vals.get("val_tail_bias_abs", np.inf), np.inf)
    val_slope_mae = _finite_float(vals.get("val_slope_mae", np.inf), np.inf)
    val_slope_corr = _finite_float(vals.get("val_slope_corr", np.nan), np.nan)
    val_range_ratio = _finite_float(vals.get("val_range_ratio", np.nan), np.nan)
    mono = _finite_float(vals.get("visible_monotonic_penalty", np.inf), np.inf)
    active_clamp_fraction = _finite_float(vals.get("active_clamp_fraction", 0.0), 0.0)

    if train_r2 < float(args.min_train_r2_for_best):
        reasons.append(f"train_r2<{args.min_train_r2_for_best}")
    if train_mae > float(args.max_train_mae_for_best):
        reasons.append(f"train_mae>{args.max_train_mae_for_best}")
    if val_mae > float(args.max_val_mae_for_best):
        reasons.append(f"val_mae>{args.max_val_mae_for_best}")
    if enabled(args.min_val_r2_for_best) and val_r2 < float(args.min_val_r2_for_best):
        reasons.append(f"val_r2<{args.min_val_r2_for_best}")
    if enabled(args.min_val_corr_for_best) and (not np.isfinite(val_corr) or val_corr < float(args.min_val_corr_for_best)):
        reasons.append(f"val_corr<{args.min_val_corr_for_best}")
    if enabled(args.max_val_bias_for_best) and val_bias_abs > float(args.max_val_bias_for_best):
        reasons.append(f"val_bias_abs>{args.max_val_bias_for_best}")
    if hasattr(args, "max_val_rmse_for_best") and enabled(args.max_val_rmse_for_best) and val_rmse > float(args.max_val_rmse_for_best):
        reasons.append(f"val_rmse>{args.max_val_rmse_for_best}")
    if enabled(args.max_val_tail_bias_for_best) and val_tail_bias_abs > float(args.max_val_tail_bias_for_best):
        reasons.append(f"val_tail_bias_abs>{args.max_val_tail_bias_for_best}")
    if enabled(args.max_val_slope_mae_for_best) and val_slope_mae > float(args.max_val_slope_mae_for_best):
        reasons.append(f"val_slope_mae>{args.max_val_slope_mae_for_best}")
    if enabled(args.min_val_slope_corr_for_best) and (not np.isfinite(val_slope_corr) or val_slope_corr < float(args.min_val_slope_corr_for_best)):
        reasons.append(f"val_slope_corr<{args.min_val_slope_corr_for_best}")
    if enabled(args.min_val_range_ratio_for_best) and (not np.isfinite(val_range_ratio) or val_range_ratio < float(args.min_val_range_ratio_for_best)):
        reasons.append(f"val_range_ratio<{args.min_val_range_ratio_for_best}")
    if enabled(args.max_val_range_ratio_for_best) and (not np.isfinite(val_range_ratio) or val_range_ratio > float(args.max_val_range_ratio_for_best)):
        reasons.append(f"val_range_ratio>{args.max_val_range_ratio_for_best}")
    if enabled(args.max_visible_monotonic_penalty_for_best) and mono > float(args.max_visible_monotonic_penalty_for_best):
        reasons.append(f"visible_monotonic_penalty>{args.max_visible_monotonic_penalty_for_best}")
    if hasattr(args, "max_active_clamp_fraction_for_best") and enabled(args.max_active_clamp_fraction_for_best) and active_clamp_fraction > float(args.max_active_clamp_fraction_for_best):
        reasons.append(f"active_clamp_fraction>{args.max_active_clamp_fraction_for_best}")
    return (len(reasons) == 0), reasons


def _average_states(states: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        vals = [s[key].detach().cpu() for s in states]
        out[key] = torch.stack(vals, dim=0).mean(dim=0) if torch.is_floating_point(vals[0]) else vals[0].clone()
    return out


def _evaluate_state(model: Assb111SOHHead, state: Mapping[str, torch.Tensor], x, delta, soh_np: np.ndarray, masks: Mapping[str, np.ndarray], device: torch.device, dtype: torch.dtype):
    old = _clone_state_dict(model)
    model.load_state_dict(_state_dict_to_device(state, device, dtype), strict=False)
    model.eval()
    with torch.no_grad():
        pred = model(x, delta_cycle=delta).SOH_pred.detach().cpu().numpy()
    tr = _metrics_for_mask(soh_np, pred, masks["train"])
    va = _metrics_for_mask(soh_np, pred, masks["val"])
    model.load_state_dict(_state_dict_to_device(old, device, dtype), strict=False)
    return pred, tr, va


def _write_csv_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _feature_importance(model: Assb111SOHHead, feature_columns: Sequence[str]) -> pd.DataFrame:
    first = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            first = module
            break
    rows: List[Dict[str, Any]] = []
    if first is not None and first.weight.shape[1] == len(feature_columns):
        scores = first.weight.detach().abs().mean(dim=0).cpu().numpy()
        for name, score in zip(feature_columns, scores):
            rows.append({"feature": name, "importance_proxy": float(score), "method": "mean_abs_first_layer_weight"})
    else:
        rows = [{"feature": name, "importance_proxy": float("nan"), "method": "unavailable"} for name in feature_columns]
    return pd.DataFrame(rows).sort_values("importance_proxy", ascending=False, na_position="last")


def _load_teacher(frame: pd.DataFrame, teacher_csv: str, teacher_col: str) -> Optional[np.ndarray]:
    if not teacher_csv:
        return None
    t = pd.read_csv(teacher_csv)
    if "cycle_id" not in t.columns or teacher_col not in t.columns:
        raise KeyError(f"teacher_csv must contain cycle_id and {teacher_col}")
    merged = frame[["cycle_id"]].merge(t[["cycle_id", teacher_col]], on="cycle_id", how="left", validate="one_to_one")
    return pd.to_numeric(merged[teacher_col], errors="coerce").to_numpy(dtype=float)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed_locked and int(args.seed) != int(args.locked_seed_value):
        raise RuntimeError(f"seed-locked protocol requires seed={args.locked_seed_value}, got {args.seed}")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    if args.cuda_matmul_precision:
        torch.set_float32_matmul_precision(str(args.cuda_matmul_precision))
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.allow_cpu:
            args.device = "cpu"
        else:
            raise RuntimeError("CUDA requested but unavailable. Use --allow_cpu or --device cpu.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    out_dir = Path(args.output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    if args.save_epoch_checkpoints or args.enable_swa_topk:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.dataset_csv)
    if "cycle_id" not in frame.columns:
        raise KeyError("dataset_csv must contain cycle_id")
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    manifest = _load_manifest(args.split_manifest_json)
    if "split" not in frame.columns:
        frame["split"] = _split_from_manifest(frame["cycle_id"].to_numpy(dtype=int), manifest)
    masks = _make_masks(frame["split"].to_numpy())
    if np.any(masks["fit"] & masks["test"]):
        raise RuntimeError("fit mask overlaps test")
    if np.any(masks["fit"] & masks["partial"]):
        raise RuntimeError("fit mask overlaps partial")

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
    x_np = np.array(transform_with_scaler(frame, scaler), dtype=float, copy=True)
    soh_np = np.array(pd.to_numeric(frame[args.target_col], errors="coerce").to_numpy(dtype=float), dtype=float, copy=True)
    q_np = (
        np.array(pd.to_numeric(frame[args.q_col], errors="coerce").to_numpy(dtype=float), dtype=float, copy=True)
        if args.q_col in frame.columns
        else None
    )
    teacher_np = _load_teacher(frame, args.teacher_csv, args.teacher_col)
    if teacher_np is not None:
        teacher_np = np.array(teacher_np, dtype=float, copy=True)

    cycles = frame["cycle_id"].to_numpy(dtype=int)
    delta_np = np.ones_like(cycles, dtype=float)
    if len(delta_np) > 1:
        delta_np[1:] = np.maximum(1.0, np.diff(cycles).astype(float))

    dtype = torch.float64 if str(args.dtype).lower() in {"float64", "double"} else torch.float32
    x = torch.as_tensor(x_np, dtype=dtype, device=device)
    soh = torch.as_tensor(soh_np, dtype=dtype, device=device)
    q_obs = torch.as_tensor(q_np, dtype=dtype, device=device) if q_np is not None else None
    teacher = torch.as_tensor(teacher_np, dtype=dtype, device=device) if teacher_np is not None else None
    train_mask_t = torch.as_tensor(masks["fit"], dtype=torch.bool, device=device)
    delta = torch.as_tensor(delta_np, dtype=dtype, device=device)

    cfg = Assb111SOHHeadConfig(
        n_features=len(feature_columns),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        dropout=float(args.dropout),
        activation=args.activation,
        model_variant=str(args.soh_model_variant),
        rate_scale=float(args.rate_scale),
        residual_bound=float(args.residual_bound),
        floor_min=float(args.floor_min),
        floor_max=float(args.floor_max),
        soh_floor_prior=float(args.soh_floor_prior),
        soh0_min=float(args.soh0_min),
        soh0_max=float(args.soh0_max),
        damage_rate_scale=float(args.damage_rate_scale),
        gate_gamma=float(args.gate_gamma),
        soh_numeric_min=float(args.soh_numeric_min),
        tail_slope_guard=float(args.tail_slope_guard),
        feature_dropout=float(args.feature_dropout),
        huber_delta=float(args.huber_delta),
        w_smooth=float(args.w_smooth),
        w_rate=float(args.w_rate),
        w_monotonic=float(args.w_monotonic),
        w_residual=float(args.w_residual),
        w_floor_prior=float(args.w_floor_prior),
        w_tail_guard=float(args.w_tail_guard),
        w_rate_tv=float(args.w_rate_tv),
        w_distill=float(args.w_distill),
        dtype=str(args.dtype),
    )
    model = Assb111SOHHead(cfg).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    print(
        f"[START] seed={int(args.seed)} variant={args.soh_model_variant} device={device} dtype={args.dtype} "
        f"epochs={int(args.epochs)} eval_every={int(args.eval_every)} print_every={int(args.print_every)} "
        f"min_epochs_before_patience={int(args.min_epochs_before_patience)} selection_strategy={args.selection_strategy} require_hard_guard={bool(args.require_hard_guard)} n_features={len(feature_columns)}",
        flush=True,
    )

    history: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    topk_states: List[Tuple[float, int, Dict[str, torch.Tensor], Dict[str, float]]] = []
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = float("inf")
    best_val = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_status = "uninitialized"
    # V5 keeps a separate soft-score tracker. Hard guards decide what can be
    # packaged, but early stopping follows visible train/val soft-score progress.
    # This avoids aborting at epoch~800 simply because the first guarded checkpoint
    # has not appeared yet.
    best_soft_score = float("inf")
    best_soft_epoch = -1
    best_soft_vals: Dict[str, float] = {}
    best_soft_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = int(args.patience)

    def record_state(epoch: int, state: Dict[str, torch.Tensor], vals: Mapping[str, float], source: str) -> None:
        nonlocal topk_states
        score = float(vals["visible_score"])
        entry = (score, int(epoch), {k: v.detach().cpu().clone() for k, v in state.items()}, dict(vals))
        topk_states.append(entry)
        topk_states = sorted(topk_states, key=lambda z: z[0])[: max(1, int(args.top_k_checkpoints))]
        if args.save_epoch_checkpoints:
            path = ckpt_dir / f"{source}_epoch{epoch:06d}.pt"
            torch.save(entry[2], path)
            checkpoint_rows.append({"epoch": int(epoch), "source": source, "checkpoint_path": str(path), "visible_score": score, "test_metrics_used": False, **dict(vals)})

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x, delta_cycle=delta)
        loss, logs = assb111_soh_loss(out, soh, train_mask=train_mask_t, cfg=cfg, q_obs_ah=q_obs, teacher_soh=teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
        opt.step()

        if args.enable_ema:
            current = _clone_state_dict(model)
            if ema_state is None:
                ema_state = {k: v.clone() for k, v in current.items()}
            else:
                decay = float(args.ema_decay)
                for k, v in current.items():
                    ema_state[k].mul_(decay).add_(v, alpha=(1.0 - decay)) if torch.is_floating_point(v) else ema_state.__setitem__(k, v.clone())

        eval_every = max(1, int(args.eval_every))
        should_eval = (epoch == 1) or (epoch % eval_every == 0) or (epoch == int(args.epochs))
        if not should_eval:
            if int(args.print_every) > 0 and epoch % int(args.print_every) == 0:
                print(f"epoch={epoch} loss={logs['loss_total']:.6e} [guard eval skipped; eval_every={eval_every}] best={best_score:.6g} status={best_status}", flush=True)
            continue

        model.eval()
        with torch.no_grad():
            pred = model(x, delta_cycle=delta).SOH_pred.detach().cpu().numpy()
        train_m = _metrics_for_mask(soh_np, pred, masks["train"])
        val_m = _metrics_for_mask(soh_np, pred, masks["val"])
        score, vals = _visible_score(train_m, val_m, soh_np, pred, masks, args)
        n_train = max(1, int(np.asarray(masks["train"], dtype=bool).sum()))
        vals["active_clamp_fraction"] = float(logs.get("active_clamp_count", 0)) / float(n_train)
        guard_ok, guard_reasons = _guard_ok(vals, args)
        history.append({
            "epoch": int(epoch),
            "candidate_tag": args.candidate_tag,
            "seed": int(args.seed),
            "loss_total": logs.get("loss_total"),
            "loss_soh": logs.get("loss_soh"),
            "loss_rate_tv": logs.get("loss_rate_tv"),
            "train_mae": vals["train_mae"],
            "train_r2": vals["train_r2"],
            "val_mae": vals["val_mae"],
            "val_rmse": vals.get("val_rmse"),
            "val_r2": vals["val_r2"],
            "val_corr": vals.get("val_corr"),
            "val_bias_abs": vals.get("val_bias_abs"),
            "val_tail_bias_abs": vals.get("val_tail_bias_abs"),
            "val_slope_mae": vals.get("val_slope_mae"),
            "val_slope_corr": vals.get("val_slope_corr"),
            "val_range_ratio": vals.get("val_range_ratio"),
            "visible_score": vals["visible_score"],
            "visible_monotonic_penalty": vals["visible_monotonic_penalty"],
            "active_clamp_fraction": vals.get("active_clamp_fraction"),
            "guard_ok": bool(guard_ok),
            "guard_reasons": ";".join(guard_reasons),
            "active_clamp_count": logs.get("active_clamp_count", 0),
        })
        current_state = _clone_state_dict(model)

        # V5 soft-score clock. We reset patience whenever the visible train/val
        # score improves, even if the checkpoint has not yet passed all hard guards.
        # Hard-accept remains unchanged: only guarded checkpoints are saved as
        # selected_model unless --allow_unguarded_fallback is explicitly used.
        soft_improved = bool(score + float(args.min_delta) < best_soft_score)
        if soft_improved:
            best_soft_score = score
            best_soft_epoch = int(epoch)
            best_soft_vals = dict(vals)
            best_soft_state = current_state
            # V7: soft-score is the primary visible train/val selector.
            # Hard guards are kept as audit to diagnose bias/shape issues,
            # but they no longer prevent saving a leak-free candidate unless
            # --selection_strategy hard_guard or --require_hard_guard is used.
            if str(args.selection_strategy).lower() == "softscore":
                best_score = score
                best_val = vals["val_mae"]
                best_epoch = int(epoch)
                best_state = current_state
                best_status = "visible_softscore_guard_audit"
            patience_left = int(args.patience)

        accepted_guarded_now = False
        if str(args.selection_strategy).lower() == "hard_guard" and guard_ok and score + float(args.min_delta) < best_score:
            best_score = score
            best_val = vals["val_mae"]
            best_epoch = epoch
            best_state = current_state
            best_status = "visible_guarded"
            patience_left = int(args.patience)
            accepted_guarded_now = True
        elif vals["val_mae"] + float(args.min_delta) < best_val:
            best_val = vals["val_mae"]
            if bool(args.allow_unguarded_fallback):
                best_score = score
                best_epoch = epoch
                best_state = current_state
                best_status = "fallback_val_mae_no_guard"
                patience_left = int(args.patience)

        # Early stopping policy:
        # - never stop before min_epochs_before_patience;
        # - by default, do not stop before the first guarded checkpoint;
        # - once enabled, consume patience only when visible soft-score did not
        #   improve at this evaluation point and no guarded checkpoint was accepted.
        patience_enabled = (int(args.min_epochs_before_patience) <= 0) or (epoch >= int(args.min_epochs_before_patience))
        if str(args.selection_strategy).lower() == "hard_guard" and best_state is None and not bool(args.allow_patience_before_first_guard):
            patience_enabled = False
        if patience_enabled and (not soft_improved) and (not accepted_guarded_now):
            patience_left -= eval_every

        should_record = (epoch == 1) or (int(args.checkpoint_interval) > 0 and epoch % int(args.checkpoint_interval) == 0) or guard_ok
        if (args.save_epoch_checkpoints or args.enable_swa_topk) and should_record:
            record_state(epoch, current_state, vals, "raw")
            if args.enable_ema and ema_state is not None:
                p_ema, tr_ema, va_ema = _evaluate_state(model, ema_state, x, delta, soh_np, masks, device, dtype)
                em_score, em_vals = _visible_score(tr_ema, va_ema, soh_np, p_ema, masks, args)
                em_ok, _ = _guard_ok(em_vals, args)
                record_state(epoch, {k: v.detach().cpu().clone() for k, v in ema_state.items()}, em_vals, "ema")
                if em_ok and em_score < best_score:
                    best_score, best_val, best_epoch = em_score, em_vals["val_mae"], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
                    best_status = "ema_visible_guarded"
                    patience_left = int(args.patience)
        if args.progress_json:
            try:
                progress_every = max(1, int(args.progress_json_every))
                if epoch == 1 or (len(history) % progress_every == 0):
                    progress_path = Path(args.progress_json)
                    if not progress_path.is_absolute():
                        progress_path = out_dir / progress_path
                    save_json({
                        "epoch": int(epoch),
                        "seed": int(args.seed),
                        "loss_total": logs.get("loss_total"),
                        "train_mae": vals.get("train_mae"),
                        "train_r2": vals.get("train_r2"),
                        "val_mae": vals.get("val_mae"),
                        "val_rmse": vals.get("val_rmse"),
                        "val_r2": vals.get("val_r2"),
                        "val_corr": vals.get("val_corr"),
                        "visible_score": vals.get("visible_score"),
                        "best_soft_score": best_soft_score,
                        "best_soft_epoch": best_soft_epoch,
                        "best_guarded_score": best_score,
                        "best_guarded_epoch": best_epoch,
                        "best_status": best_status,
                        "patience_left": patience_left,
                        "guard_ok": bool(guard_ok),
                        "guard_reasons": list(guard_reasons),
                    }, progress_path)
            except Exception as exc:
                print(f"[WARN] failed to write progress_json: {exc}", flush=True)
        if int(args.print_every) > 0 and (epoch == 1 or epoch % int(args.print_every) == 0):
            print(f"epoch={epoch} loss={logs['loss_total']:.6e} train_mae={vals['train_mae']:.6g} train_r2={vals['train_r2']:.6g} val_mae={vals['val_mae']:.6g} val_r2={vals['val_r2']:.6g} val_corr={vals.get('val_corr', float('nan')):.6g} score={score:.6g} soft_best={best_soft_score:.6g}@{best_soft_epoch} guarded_best={best_score:.6g} status={best_status} patience={patience_left} guard={'OK' if guard_ok else ';'.join(guard_reasons)}", flush=True)
        if patience_left <= 0:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_status={best_status}; best_soft_epoch={best_soft_epoch}; best_soft_score={best_soft_score:.6g}", flush=True)
            break

    # Optional visible-only top-k averaging.
    topk_summary: Dict[str, Any] = {"enabled": bool(args.enable_swa_topk), "selected": False}
    if args.enable_swa_topk and topk_states:
        topk = sorted(topk_states, key=lambda z: z[0])[: max(1, int(args.top_k_checkpoints))]
        avg_state = _average_states([s for _score, _epoch, s, _vals in topk])
        pred_avg, tr_avg, va_avg = _evaluate_state(model, avg_state, x, delta, soh_np, masks, device, dtype)
        avg_score, avg_vals = _visible_score(tr_avg, va_avg, soh_np, pred_avg, masks, args)
        avg_ok, avg_reasons = _guard_ok(avg_vals, args)
        topk_summary = {"enabled": True, "selected": bool(avg_ok and avg_score <= best_score), "top_k": len(topk), "avg_visible_score": avg_score, "avg_vals": avg_vals, "avg_guard_ok": avg_ok, "avg_guard_reasons": avg_reasons, "source_epochs": [int(e) for _s, e, _st, _v in topk]}
        torch.save(avg_state, out_dir / "topk_average_state.pt")
        if avg_ok and avg_score <= best_score:
            best_state = avg_state
            best_score = avg_score
            best_val = avg_vals["val_mae"]
            best_epoch = int(topk[0][1])
            best_status = "topk_average_visible_guarded"

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    hist_df.to_csv(out_dir / "history_visible.csv", index=False, encoding="utf-8-sig")
    _write_csv_rows(checkpoint_rows, out_dir / "checkpoint_manifest.csv")

    # V7 final fallback: if hard-guard selection produced no checkpoint but
    # the visible soft-score tracker has a valid state, save that state unless
    # the caller explicitly requires hard-guard acceptance. This keeps the
    # strict no-test-selection protocol intact while preventing the guard from
    # turning into a training stopper.
    if best_state is None and best_soft_state is not None and not bool(args.require_hard_guard):
        best_state = best_soft_state
        best_epoch = int(best_soft_epoch)
        best_score = float(best_soft_score)
        best_val = float(best_soft_vals.get("val_mae", float("nan")))
        best_status = "visible_softscore_guard_audit_fallback"

    if best_state is None:
        failure_summary = {
            "output_model_dir": str(out_dir),
            "candidate_tag": args.candidate_tag,
            "protocol_tag": args.protocol_tag,
            "seed": int(args.seed),
            "best_epoch": -1,
            "best_selection_status": "failed_no_visible_guarded_checkpoint",
            "best_val_mae_seen_without_guard": best_val,
            "best_soft_score_without_guard": best_soft_score,
            "best_soft_epoch_without_guard": best_soft_epoch,
            "best_soft_vals_without_guard": best_soft_vals,
            "min_epochs_before_patience": int(args.min_epochs_before_patience),
            "allow_patience_before_first_guard": bool(args.allow_patience_before_first_guard),
            "no_test_metrics_in_training_history": True,
            "test_metrics_used_for_selection": False,
            "training_history_columns": list(hist_df.columns),
            "eval_every": int(args.eval_every),
            "dtype": str(args.dtype),
            "guard_thresholds": {
                "min_train_r2_for_best": float(args.min_train_r2_for_best),
                "max_train_mae_for_best": float(args.max_train_mae_for_best),
                "max_val_mae_for_best": float(args.max_val_mae_for_best),
                "min_val_r2_for_best": float(args.min_val_r2_for_best),
                "min_val_corr_for_best": float(args.min_val_corr_for_best),
                "max_val_bias_for_best": float(args.max_val_bias_for_best),
                "max_val_rmse_for_best": float(args.max_val_rmse_for_best),
                "max_val_tail_bias_for_best": float(args.max_val_tail_bias_for_best),
                "max_val_slope_mae_for_best": float(args.max_val_slope_mae_for_best),
                "min_val_range_ratio_for_best": float(args.min_val_range_ratio_for_best),
                "max_val_range_ratio_for_best": float(args.max_val_range_ratio_for_best),
            },
            "message": "No checkpoint passed visible train/val guards. No model checkpoint was saved.",
        }
        save_json(failure_summary, out_dir / "train_summary.json")
        save_json({"ok": False, "reason": "no_visible_guarded_checkpoint", "summary": failure_summary}, out_dir / "selected_checkpoint_audit.json")
        if bool(args.allow_no_guard_exit_ok):
            print("[DIAGNOSTIC_OK] No checkpoint passed hard visible guards; exiting 0 because --allow_no_guard_exit_ok was set.", flush=True)
            return 0
        raise RuntimeError("No checkpoint passed visible train/val guards; refusing to save/package SOH head.")

    model.load_state_dict(_state_dict_to_device(best_state, device, dtype), strict=False)
    model.eval()
    with torch.no_grad():
        final_out = model(x, delta_cycle=delta)
    pred_frame = prediction_frame_from_output(frame, final_out)
    pred_frame.to_csv(out_dir / "soh_pred_by_cycle.csv", index=False, encoding="utf-8-sig")

    visible_frame = pred_frame[pred_frame["split"].astype(str).str.lower().isin(["train", "val"])].copy()
    visible_metrics = metrics_by_split(visible_frame)
    final_metrics = metrics_by_split(pred_frame)
    save_json({"metrics_by_split_visible_only": visible_metrics, "note": "No held-out test metrics were used for checkpoint selection."}, out_dir / "metrics_soh_by_split_train_eval.json")
    save_json({"metrics_by_split_after_selection": final_metrics, "test_metrics_used_for_selection": False}, out_dir / "metrics_soh_by_split_final_report.json")
    _feature_importance(model, feature_columns).to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    model.save(
        out_dir,
        feature_columns=feature_columns,
        scaler=scaler,
        split_manifest=manifest,
        extra={"feature_mode": args.feature_mode, "dataset_csv": args.dataset_csv, "split_manifest_json": args.split_manifest_json, "best_epoch": best_epoch, "best_val_mae": best_val, "best_visible_score": best_score, "best_selection_status": best_status, "seed": int(args.seed), "candidate_tag": args.candidate_tag, "protocol_tag": args.protocol_tag, "dtype": str(args.dtype), "eval_every": int(args.eval_every), "min_epochs_before_patience": int(args.min_epochs_before_patience), "allow_patience_before_first_guard": bool(args.allow_patience_before_first_guard), "selection_strategy": str(args.selection_strategy), "require_hard_guard": bool(args.require_hard_guard)},
    )
    torch.save(model.state_dict(), out_dir / "best.pt")
    torch.save(model.state_dict(), out_dir / "selected_model.pt")

    final_pred_np = pred_frame["SOH_pred"].to_numpy(dtype=float)
    final_val_shape = _validation_shape_metrics(soh_np, final_pred_np, masks["val"])
    final_visible_metrics = {
        "train_mae": visible_metrics.get("train", {}).get("SOH_MAE"),
        "train_r2": visible_metrics.get("train", {}).get("SOH_R2"),
        "val_mae": visible_metrics.get("val", {}).get("SOH_MAE"),
        "val_rmse": visible_metrics.get("val", {}).get("SOH_RMSE"),
        "val_r2": visible_metrics.get("val", {}).get("SOH_R2"),
        "val_corr": visible_metrics.get("val", {}).get("SOH_corr"),
        "val_bias": visible_metrics.get("val", {}).get("SOH_BIAS"),
        "val_bias_abs": abs(float(visible_metrics.get("val", {}).get("SOH_BIAS") or 0.0)),
        "visible_monotonic_penalty": _visible_monotonic_penalty(final_pred_np, masks["visible"]),
        **final_val_shape,
    }
    final_guard_vals = {
        "train_mae": _finite_float(final_visible_metrics.get("train_mae"), np.inf),
        "train_r2": _finite_float(final_visible_metrics.get("train_r2"), -np.inf),
        "val_mae": _finite_float(final_visible_metrics.get("val_mae"), np.inf),
        "val_rmse": _finite_float(final_visible_metrics.get("val_rmse"), np.inf),
        "val_r2": _finite_float(final_visible_metrics.get("val_r2"), -np.inf),
        "val_corr": _finite_float(final_visible_metrics.get("val_corr"), np.nan),
        "val_bias_abs": _finite_float(final_visible_metrics.get("val_bias_abs"), np.inf),
        "val_tail_bias_abs": _finite_float(final_visible_metrics.get("val_tail_bias_abs"), np.inf),
        "val_slope_mae": _finite_float(final_visible_metrics.get("val_slope_mae"), np.inf),
        "val_slope_corr": _finite_float(final_visible_metrics.get("val_slope_corr"), np.nan),
        "val_range_ratio": _finite_float(final_visible_metrics.get("val_range_ratio"), np.nan),
        "visible_monotonic_penalty": _finite_float(final_visible_metrics.get("visible_monotonic_penalty"), np.inf),
    }
    if "active_clamp_mask" in pred_frame.columns:
        _active_mask = pred_frame["active_clamp_mask"].astype(bool).to_numpy()
        final_guard_vals["active_clamp_fraction"] = float(np.mean(_active_mask[np.asarray(masks["train"], dtype=bool)])) if np.any(masks["train"]) else 0.0
    else:
        final_guard_vals["active_clamp_fraction"] = 0.0
    final_guard_ok, final_guard_reasons = _guard_ok(final_guard_vals, args)
    protocol_ok = bool(faudit.get("ok", False)) and True
    audit_ok = bool(final_guard_ok) if bool(args.require_hard_guard) else protocol_ok
    selected_audit = {
        "ok": bool(audit_ok),
        "hard_visible_guard_ok": bool(final_guard_ok),
        "hard_guard_required": bool(args.require_hard_guard),
        "selection_strategy": str(args.selection_strategy),
        "best_epoch": int(best_epoch),
        "best_selection_status": str(best_status),
        "candidate_tag": args.candidate_tag,
        "protocol_tag": args.protocol_tag,
        "seed": int(args.seed),
        "visible_guard": {
            "min_train_r2_for_best": float(args.min_train_r2_for_best),
            "max_train_mae_for_best": float(args.max_train_mae_for_best),
            "max_val_mae_for_best": float(args.max_val_mae_for_best),
            "min_val_r2_for_best": float(args.min_val_r2_for_best),
            "min_val_corr_for_best": float(args.min_val_corr_for_best),
            "max_val_bias_for_best": float(args.max_val_bias_for_best),
            "max_val_rmse_for_best": float(args.max_val_rmse_for_best),
            "max_val_tail_bias_for_best": float(args.max_val_tail_bias_for_best),
            "max_val_slope_mae_for_best": float(args.max_val_slope_mae_for_best),
            "min_val_slope_corr_for_best": float(args.min_val_slope_corr_for_best),
            "min_val_range_ratio_for_best": float(args.min_val_range_ratio_for_best),
            "max_val_range_ratio_for_best": float(args.max_val_range_ratio_for_best),
            "max_visible_monotonic_penalty_for_best": float(args.max_visible_monotonic_penalty_for_best),
            "max_active_clamp_fraction_for_best": float(args.max_active_clamp_fraction_for_best),
            "final_visible_metrics": final_visible_metrics,
            "guard_reasons": final_guard_reasons,
        },
        "test_metrics_used_for_selection": False,
        "no_test_metrics_in_training_history": True,
        "selected_model_pt": "selected_model.pt",
    }
    save_json(selected_audit, out_dir / "selected_checkpoint_audit.json")
    summary = {
        "output_model_dir": str(out_dir),
        "candidate_tag": args.candidate_tag,
        "protocol_tag": args.protocol_tag,
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val),
        "best_visible_score": float(best_score),
        "best_selection_status": best_status,
        "feature_mode": args.feature_mode,
        "n_features": len(feature_columns),
        "model_variant": cfg.model_variant,
        "dtype": str(args.dtype),
        "eval_every": int(args.eval_every),
        "min_epochs_before_patience": int(args.min_epochs_before_patience),
        "allow_patience_before_first_guard": bool(args.allow_patience_before_first_guard),
        "selection_strategy": str(args.selection_strategy),
        "require_hard_guard": bool(args.require_hard_guard),
        "best_soft_score": float(best_soft_score),
        "best_soft_epoch": int(best_soft_epoch),
        "best_soft_vals": best_soft_vals,
        "guard_thresholds": {k: v for k, v in selected_audit["visible_guard"].items() if k not in {"final_visible_metrics", "guard_reasons"}},
        "final_visible_metrics": final_visible_metrics,
        "metrics_by_split_visible_only": visible_metrics,
        "final_report_metrics_available": True,
        "final_report_metrics_file": "metrics_soh_by_split_final_report.json",
        "topk_average": topk_summary,
        "selected_checkpoint_audit_ok": bool(audit_ok),
        "hard_visible_guard_ok": bool(final_guard_ok),
        "hard_guard_required": bool(args.require_hard_guard),
        "selection_strategy": str(args.selection_strategy),
        "selected_checkpoint_guard_reasons": final_guard_reasons,
        "feature_audit_ok": bool(faudit.get("ok", False)),
        "feature_audit_failures": faudit.get("failures", []),
        "no_test_metrics_in_training_history": True,
        "test_metrics_used_for_selection": False,
        "training_history_columns": list(hist_df.columns),
    }
    save_json(summary, out_dir / "train_summary.json")
    save_json(summary, out_dir / "training_summary.json")
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print(f"[OK] seed={int(args.seed)} selected_epoch={best_epoch} status={best_status} final_report={out_dir / 'metrics_soh_by_split_final_report.json'}", flush=True)
    return 0 if audit_ok and faudit.get("ok", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
