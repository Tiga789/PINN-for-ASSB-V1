# -*- coding: utf-8 -*-
"""Train ASSB ModelFin_111 strict30 SOH head.

This seed42-compatible trainer keeps the original ASSB-111 strict30 rule:
SOH labels in test/partial cycles may exist in the dataset for final evaluation,
but the loss, scaler, early stopping and checkpoint selection are train/val only.

New seed42-locked options add visible-only checkpoint metadata, optional EMA, and
optional top-k weight averaging. No test metrics are written to the training
history or training summary.
"""
from __future__ import annotations

import argparse
import copy
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

from util.assb111_feature_schema import select_feature_columns, load_scaler_json, write_scaler_json, write_schema_json
from util.assb111_leakage_guard import (
    audit_assb111_dataset,
    audit_seed42locked_protocol_metadata,
    make_supervised_masks,
    transform_features_checked,
    write_audit_json,
)
from util.assb111_split import load_manifest, split_for_cycles
from util.assb111_soh_model import (
    Assb111SOHHead,
    Assb111SOHHeadConfig,
    assb111_soh_loss,
    metrics_by_split,
    prediction_frame_from_output,
    save_json,
    soh_metrics,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train ASSB111 strict30 SOH head")
    p.add_argument("--dataset_csv", default=r"Data\assb111\dataset.csv")
    p.add_argument("--split_manifest_json", default=r"Data\assb111\split_manifest.json")
    p.add_argument("--scaler_json", default=r"Data\assb111\feature_scaler.json")
    p.add_argument("--output_model_dir", "--output_dir", dest="output_model_dir", default=r"ModelFin_111")
    p.add_argument("--feature_mode", default="p1_107a_strict")
    p.add_argument("--allow_upper_bound", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--activation", default="silu")

    # Legacy and common parameters.
    p.add_argument("--rate_scale", type=float, default=1e-3)
    p.add_argument("--residual_bound", type=float, default=0.008)
    p.add_argument("--w_smooth", type=float, default=0.05)
    p.add_argument("--w_rate", type=float, default=0.01)
    p.add_argument("--w_monotonic", type=float, default=0.20)
    p.add_argument("--w_residual", type=float, default=0.10)

    # Saturating_v2 parameters.
    p.add_argument("--soh_model_variant", "--model_variant", dest="soh_model_variant", default="saturating_v2")
    p.add_argument("--floor_min", "--soh_floor_min", dest="floor_min", type=float, default=0.65)
    p.add_argument("--floor_max", "--soh_floor_max", dest="floor_max", type=float, default=0.85)
    p.add_argument("--soh_floor_prior", "--soh_floor_total", dest="soh_floor_prior", type=float, default=0.72)
    p.add_argument("--soh0_min", type=float, default=0.94)
    p.add_argument("--soh0_max", type=float, default=1.03)
    p.add_argument("--damage_rate_scale", "--damage_scale", dest="damage_rate_scale", type=float, default=5e-4)
    p.add_argument("--gate_gamma", type=float, default=1.0)
    p.add_argument("--soh_numeric_min", type=float, default=0.60)
    p.add_argument("--tail_slope_guard", "--tail_guard_max_drop_per_cycle", dest="tail_slope_guard", type=float, default=0.0020)
    p.add_argument("--w_floor_prior", type=float, default=0.02)
    p.add_argument("--w_tail_guard", "--w_tail_slope", dest="w_tail_guard", type=float, default=0.05)
    p.add_argument("--w_deceleration", type=float, default=0.0)
    p.add_argument("--w_clamp_hit", "--w_clamp", dest="w_clamp_hit", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=5.0)

    # Early stopping and visible-only selection.
    p.add_argument("--patience", type=int, default=600)
    p.add_argument("--min_delta", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_every", type=int, default=250)
    p.add_argument("--min_train_r2_for_best", type=float, default=-1e9)
    p.add_argument("--max_train_mae_for_best", type=float, default=1e9)
    p.add_argument("--max_val_mae_for_best", type=float, default=1e9)
    p.add_argument("--best_selection_mode", default="val_mae")
    p.add_argument("--visible_score_mode", default="val_mae")
    p.add_argument("--write_training_summary", action="store_true")
    p.add_argument("--no_test_selection", action="store_true")
    p.add_argument("--selection_mode", default="")
    p.add_argument("--protocol_tag", default="")
    p.add_argument("--candidate_tag", default="")

    # Seed42-locked engineering mode.
    p.add_argument("--seed_locked", action="store_true")
    p.add_argument("--locked_seed_value", type=int, default=42)

    # Checkpoint / EMA / top-k support. These are visible-only; they do not read final eval files.
    p.add_argument("--save_epoch_checkpoints", action="store_true")
    p.add_argument("--checkpoint_interval", type=int, default=50)
    p.add_argument("--enable_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--enable_swa_topk", action="store_true")
    p.add_argument("--top_k_checkpoints", type=int, default=5)
    p.add_argument("--allow_unguarded_fallback", action="store_true",
                   help="Compatibility only: allow packaging a val-MAE checkpoint even if visible train/val guards fail. Default is false for ASSB-111 seed42 recovery.")
    return p.parse_args(argv)


def _json_clean(x: Any) -> Any:
    if isinstance(x, dict):
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


def _metrics_for_mask(obs: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return soh_metrics(obs[np.asarray(mask, dtype=bool)], pred[np.asarray(mask, dtype=bool)])


def _visible_monotonic_penalty(pred: np.ndarray, visible_mask: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float)[np.asarray(visible_mask, dtype=bool)]
    if p.size < 2:
        return 0.0
    return float(np.mean(np.maximum(0.0, p[1:] - p[:-1])))


def _visible_score(train_m: Mapping[str, float], val_m: Mapping[str, float], pred: np.ndarray, masks: Mapping[str, np.ndarray], args) -> Tuple[float, Dict[str, float]]:
    val_mae = float(val_m.get("SOH_MAE", np.inf))
    train_mae = float(train_m.get("SOH_MAE", np.inf))
    train_r2 = float(train_m.get("SOH_R2", -np.inf))
    val_bias_abs = abs(float(val_m.get("SOH_BIAS", 0.0))) if np.isfinite(float(val_m.get("SOH_BIAS", 0.0))) else 0.0
    mono = _visible_monotonic_penalty(pred, masks["visible"])
    score = val_mae + 0.15 * train_mae + 0.20 * max(0.0, float(args.min_train_r2_for_best) - train_r2) + 0.05 * val_bias_abs + 0.02 * mono
    values = {
        "visible_score": float(score),
        "train_mae": train_mae,
        "train_r2": train_r2,
        "val_mae": val_mae,
        "val_r2": float(val_m.get("SOH_R2", np.nan)),
        "val_bias_abs": val_bias_abs,
        "visible_monotonic_penalty": mono,
    }
    return float(score), values


def _guard_ok(vals: Mapping[str, float], args) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if float(vals.get("train_r2", -np.inf)) < float(args.min_train_r2_for_best):
        reasons.append(f"train_r2<{args.min_train_r2_for_best}")
    if float(vals.get("train_mae", np.inf)) > float(args.max_train_mae_for_best):
        reasons.append(f"train_mae>{args.max_train_mae_for_best}")
    if float(vals.get("val_mae", np.inf)) > float(args.max_val_mae_for_best):
        reasons.append(f"val_mae>{args.max_val_mae_for_best}")
    return (len(reasons) == 0), reasons


def _clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _state_dict_to_device(state: Mapping[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    return {k: v.to(device=device, dtype=dtype) if torch.is_floating_point(v) else v.to(device=device) for k, v in state.items()}


def _write_csv_rows(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
    else:
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
        for name in feature_columns:
            rows.append({"feature": name, "importance_proxy": float("nan"), "method": "unavailable"})
    return pd.DataFrame(rows).sort_values("importance_proxy", ascending=False, na_position="last")


def _temporary_state_eval(model: Assb111SOHHead, state: Mapping[str, torch.Tensor], x, delta, soh_np: np.ndarray, masks: Mapping[str, np.ndarray], device: torch.device, dtype: torch.dtype) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float], Dict[str, float]]:
    old = _clone_state_dict(model)
    model.load_state_dict(_state_dict_to_device(state, device, dtype), strict=False)
    model.eval()
    with torch.no_grad():
        pred = model(x, delta_cycle=delta).SOH_pred.detach().cpu().numpy()
    train_m = _metrics_for_mask(soh_np, pred, masks["train"])
    val_m = _metrics_for_mask(soh_np, pred, masks["val"])
    visible = _metrics_for_mask(soh_np, pred, masks["visible"])
    model.load_state_dict(_state_dict_to_device(old, device, dtype), strict=False)
    return pred, train_m, val_m, visible


def _average_states(states: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No states to average")
    out: Dict[str, torch.Tensor] = {}
    for key in states[0].keys():
        vals = [s[key].detach().cpu() for s in states]
        if torch.is_floating_point(vals[0]):
            out[key] = torch.stack(vals, dim=0).mean(dim=0)
        else:
            out[key] = vals[0].clone()
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed_locked and int(args.seed) != int(args.locked_seed_value):
        raise RuntimeError(f"seed-locked protocol requires seed={args.locked_seed_value}, got {args.seed}")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.allow_cpu:
            args.device = "cpu"
        else:
            raise RuntimeError("CUDA requested but unavailable. Use --allow_cpu or --device cpu.")
    device = torch.device(args.device)
    out_dir = Path(args.output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    if args.save_epoch_checkpoints or args.enable_swa_topk:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.dataset_csv)
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    manifest = load_manifest(args.split_manifest_json)
    if "split" not in frame.columns:
        frame["split"] = split_for_cycles(frame["cycle_id"].to_numpy(dtype=int), manifest)
    scaler = load_scaler_json(args.scaler_json)
    feature_columns = list(scaler.get("feature_columns", [])) or select_feature_columns(frame, args.feature_mode, allow_upper_bound=bool(args.allow_upper_bound))

    audit = audit_assb111_dataset(
        frame,
        manifest=manifest,
        feature_columns=feature_columns,
        feature_mode=args.feature_mode,
        allow_upper_bound=bool(args.allow_upper_bound),
        scaler=scaler,
        fit_splits=("train",),
    )
    write_audit_json(audit, out_dir / "leakage_audit_train_pre.json")
    if not audit.ok:
        raise RuntimeError("Pre-training leakage audit failed: " + "; ".join(audit.failures))

    x_np = transform_features_checked(frame, scaler, manifest)
    soh_np = pd.to_numeric(frame["SOH_obs"], errors="coerce").to_numpy(dtype=float)
    q_np = pd.to_numeric(frame["Q_obs_Ah"], errors="coerce").to_numpy(dtype=float) if "Q_obs_Ah" in frame.columns else None
    masks = make_supervised_masks(frame, manifest)
    if np.any(masks["fit"] & masks["test"]):
        raise RuntimeError("fit mask overlaps test")
    if np.any(masks["fit"] & masks["partial"]):
        raise RuntimeError("fit mask overlaps partial")

    cycle_np = frame["cycle_id"].to_numpy(dtype=int)
    delta_np = np.ones_like(cycle_np, dtype=float)
    if len(delta_np) > 1:
        delta_np[1:] = np.maximum(1.0, np.diff(cycle_np).astype(float))

    dtype = torch.float64
    x = torch.as_tensor(x_np, dtype=dtype, device=device)
    soh = torch.as_tensor(soh_np, dtype=dtype, device=device)
    q_obs = torch.as_tensor(q_np, dtype=dtype, device=device) if q_np is not None else None
    train_mask = torch.as_tensor(masks["fit"], dtype=torch.bool, device=device)
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
        w_smooth=float(args.w_smooth),
        w_rate=float(args.w_rate),
        w_monotonic=float(args.w_monotonic),
        w_residual=float(args.w_residual),
        w_floor_prior=float(args.w_floor_prior),
        w_tail_guard=float(args.w_tail_guard),
        dtype="float64",
    )
    model = Assb111SOHHead(cfg).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []
    topk_states: List[Tuple[float, int, Dict[str, torch.Tensor], Dict[str, float]]] = []
    best_score = float("inf")
    best_val = float("inf")
    best_epoch = -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_status = "uninitialized"
    patience_left = int(args.patience)
    ema_state: Optional[Dict[str, torch.Tensor]] = None

    def record_candidate_state(epoch: int, state: Dict[str, torch.Tensor], vals: Mapping[str, float], source: str) -> None:
        nonlocal topk_states
        score = float(vals["visible_score"])
        entry = (score, int(epoch), {k: v.detach().cpu().clone() for k, v in state.items()}, dict(vals))
        topk_states.append(entry)
        topk_states = sorted(topk_states, key=lambda z: z[0])[: max(1, int(args.top_k_checkpoints))]
        if args.save_epoch_checkpoints:
            path = ckpt_dir / f"{source}_epoch{epoch:06d}.pt"
            torch.save(entry[2], path)
            checkpoint_rows.append({
                "epoch": int(epoch),
                "source": source,
                "checkpoint_path": str(path),
                "visible_score": score,
                **{k: vals.get(k) for k in ["train_mae", "train_r2", "val_mae", "val_r2", "val_bias_abs", "visible_monotonic_penalty"]},
                "candidate_tag": args.candidate_tag,
                "seed": int(args.seed),
                "test_metrics_used": False,
            })

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        out = model(x, delta_cycle=delta)
        loss, logs = assb111_soh_loss(out, soh, train_mask=train_mask, cfg=cfg, q_obs_ah=q_obs)
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
                    if torch.is_floating_point(v):
                        ema_state[k].mul_(decay).add_(v, alpha=(1.0 - decay))
                    else:
                        ema_state[k] = v.clone()

        model.eval()
        with torch.no_grad():
            pred = model(x, delta_cycle=delta).SOH_pred.detach().cpu().numpy()
        train_m = _metrics_for_mask(soh_np, pred, masks["train"])
        val_m = _metrics_for_mask(soh_np, pred, masks["val"])
        visible_m = _metrics_for_mask(soh_np, pred, masks["visible"])
        score, vals = _visible_score(train_m, val_m, pred, masks, args)
        guard_ok, guard_reasons = _guard_ok(vals, args)

        row = {
            "epoch": epoch,
            "candidate_tag": args.candidate_tag,
            "seed": int(args.seed),
            "loss_total": logs.get("loss_total"),
            "loss_soh": logs.get("loss_soh"),
            "train_mae": vals["train_mae"],
            "train_r2": vals["train_r2"],
            "val_mae": vals["val_mae"],
            "val_r2": vals["val_r2"],
            "visible_score": vals["visible_score"],
            "visible_monotonic_penalty": vals["visible_monotonic_penalty"],
            "guard_ok": bool(guard_ok),
            "guard_reasons": ";".join(guard_reasons),
            "active_clamp_count": logs.get("active_clamp_count", 0),
        }
        # Do not add test_* columns here. leakage_guard will fail if they appear.
        history.append(row)

        current_state = _clone_state_dict(model)
        if guard_ok and (score + float(args.min_delta) < best_score):
            best_score = score
            best_val = vals["val_mae"]
            best_epoch = epoch
            best_state = current_state
            best_status = "visible_guarded"
            patience_left = int(args.patience)
        elif vals["val_mae"] + float(args.min_delta) < best_val and best_state is None:
            # ASSB-111 seed42 recovery rule:
            # Do NOT silently package an unguarded checkpoint.  The previous
            # seed42-locked smoke failed because epoch-1 fallback was selected
            # even though train/val metrics were invalid.  By default we only
            # remember the best unguarded val value for diagnostics and keep
            # training; packaging will hard-fail if no guarded checkpoint appears.
            best_val = vals["val_mae"]
            if bool(args.allow_unguarded_fallback):
                best_score = score
                best_epoch = epoch
                best_state = current_state
                best_status = "fallback_val_mae_no_guard"
                patience_left = int(args.patience)
            else:
                patience_left -= 1
        else:
            patience_left -= 1

        should_record = (epoch == 1) or (int(args.checkpoint_interval) > 0 and epoch % int(args.checkpoint_interval) == 0) or guard_ok
        if (args.save_epoch_checkpoints or args.enable_swa_topk) and should_record:
            record_candidate_state(epoch, current_state, vals, "raw")
        if args.enable_ema and ema_state is not None and should_record:
            _p, em_train, em_val, _em_vis = _temporary_state_eval(model, ema_state, x, delta, soh_np, masks, device, dtype)
            em_score, em_vals = _visible_score(em_train, em_val, _p, masks, args)
            em_ok, _em_reasons = _guard_ok(em_vals, args)
            if em_ok and em_score < best_score:
                best_score = em_score
                best_val = em_vals["val_mae"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in ema_state.items()}
                best_status = "ema_visible_guarded"
                patience_left = int(args.patience)
            record_candidate_state(epoch, {k: v.detach().cpu().clone() for k, v in ema_state.items()}, em_vals, "ema")

        if int(args.print_every) > 0 and (epoch == 1 or epoch % int(args.print_every) == 0):
            print(
                f"epoch={epoch} loss={logs['loss_total']:.6e} train_mae={vals['train_mae']:.6g} "
                f"train_r2={vals['train_r2']:.6g} val_mae={vals['val_mae']:.6g} "
                f"score={score:.6g} best_score={best_score:.6g} status={best_status} clamp={logs.get('active_clamp_count',0)}"
            )
        if patience_left <= 0:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_status={best_status}")
            break

    # Optional top-k averaging selected solely by visible score.
    topk_summary: Dict[str, Any] = {"enabled": bool(args.enable_swa_topk), "selected": False}
    if args.enable_swa_topk and topk_states:
        topk = sorted(topk_states, key=lambda z: z[0])[: max(1, int(args.top_k_checkpoints))]
        avg_state = _average_states([s for _score, _epoch, s, _vals in topk])
        pred_avg, tr_avg, va_avg, _vis_avg = _temporary_state_eval(model, avg_state, x, delta, soh_np, masks, device, dtype)
        avg_score, avg_vals = _visible_score(tr_avg, va_avg, pred_avg, masks, args)
        avg_ok, avg_reasons = _guard_ok(avg_vals, args)
        topk_summary = {
            "enabled": True,
            "selected": bool(avg_ok and avg_score <= best_score),
            "top_k": len(topk),
            "avg_visible_score": avg_score,
            "avg_vals": avg_vals,
            "avg_guard_ok": avg_ok,
            "avg_guard_reasons": avg_reasons,
            "source_epochs": [int(e) for _s, e, _st, _v in topk],
        }
        torch.save(avg_state, out_dir / "topk_average_state.pt")
        if avg_ok and avg_score <= best_score:
            best_state = avg_state
            best_score = avg_score
            best_val = avg_vals["val_mae"]
            best_epoch = int(topk[0][1])
            best_status = "topk_average_visible_guarded"

    if best_state is None:
        # Hard failure: no checkpoint satisfied the visible train/val guards.
        # Save diagnostics, but do not create soh_head.pt / best.pt /
        # selected_model.pt.  This prevents a failed epoch-1/last-epoch model
        # from being packaged and evaluated as if it were valid.
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "train_history.csv", index=False, encoding="utf-8-sig")
        hist_df.to_csv(out_dir / "history_visible.csv", index=False, encoding="utf-8-sig")
        failure_summary = {
            "output_model_dir": str(out_dir),
            "candidate_tag": args.candidate_tag,
            "protocol_tag": args.protocol_tag or "ASSB111_seed42_locked_trainval_only_small_optimization",
            "seed": int(args.seed),
            "seed_locked": bool(args.seed_locked),
            "best_epoch": -1,
            "best_val_mae_seen_without_guard": best_val,
            "best_selection_status": "failed_no_visible_guarded_checkpoint",
            "required_guards": {
                "min_train_r2_for_best": float(args.min_train_r2_for_best),
                "max_train_mae_for_best": float(args.max_train_mae_for_best),
                "max_val_mae_for_best": float(args.max_val_mae_for_best),
            },
            "no_test_metrics_in_training_history": True,
            "test_metrics_used_for_selection": False,
            "training_history_columns": list(hist_df.columns),
            "message": "No checkpoint passed the visible train/val guard. Training stopped before packaging; rerun with safer hyperparameters or inspect train_history.csv.",
        }
        save_json(failure_summary, out_dir / "train_summary.json")
        save_json(failure_summary, out_dir / "training_summary.json")
        save_json({"ok": False, "reason": "no_visible_guarded_checkpoint", "summary": failure_summary}, out_dir / "selected_checkpoint_audit.json")
        raise RuntimeError("No checkpoint passed visible train/val guards; refusing to save/package ASSB-111 SOH head. See train_history.csv and train_summary.json.")

    model.load_state_dict(_state_dict_to_device(best_state, device, dtype), strict=False)

    model.eval()
    with torch.no_grad():
        final_out = model(x, delta_cycle=delta)

    pred_frame = prediction_frame_from_output(frame, final_out)
    pred_frame.to_csv(out_dir / "soh_pred_by_cycle.csv", index=False, encoding="utf-8-sig")
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    hist_df.to_csv(out_dir / "history_visible.csv", index=False, encoding="utf-8-sig")
    _write_csv_rows(checkpoint_rows, out_dir / "checkpoint_manifest.csv")

    visible_frame = pred_frame[pred_frame["split"].astype(str).str.lower().isin(["train", "val"])].copy()
    metrics = metrics_by_split(visible_frame)
    final_visible_metrics = {
        "train_mae": metrics.get("train", {}).get("SOH_MAE"),
        "train_r2": metrics.get("train", {}).get("SOH_R2"),
        "val_mae": metrics.get("val", {}).get("SOH_MAE"),
        "val_r2": metrics.get("val", {}).get("SOH_R2"),
    }
    final_guard_vals = {
        "train_mae": float(final_visible_metrics.get("train_mae") if final_visible_metrics.get("train_mae") is not None else np.inf),
        "train_r2": float(final_visible_metrics.get("train_r2") if final_visible_metrics.get("train_r2") is not None else -np.inf),
        "val_mae": float(final_visible_metrics.get("val_mae") if final_visible_metrics.get("val_mae") is not None else np.inf),
        "val_r2": float(final_visible_metrics.get("val_r2") if final_visible_metrics.get("val_r2") is not None else np.nan),
    }
    final_guard_ok, final_guard_reasons = _guard_ok(final_guard_vals, args)
    if not final_guard_ok and not bool(args.allow_unguarded_fallback):
        save_json({
            "ok": False,
            "reason": "selected_checkpoint_failed_guard_after_reload",
            "best_selection_status": best_status,
            "final_visible_metrics": final_visible_metrics,
            "guard_reasons": final_guard_reasons,
        }, out_dir / "selected_checkpoint_audit.json")
        raise RuntimeError("Selected checkpoint failed visible guard after reload: " + "; ".join(final_guard_reasons))

    save_json(
        {
            "metrics_by_split_visible_only": metrics,
            "final_visible_metrics": final_visible_metrics,
            "best_epoch": best_epoch,
            "best_visible_score": best_score,
            "best_val_mae": best_val,
            "best_selection_status": best_status,
            "model_variant": cfg.model_variant,
            "soh_head_config": cfg.to_dict(),
            "note": "No held-out test metrics are used or written by the training script.",
        },
        out_dir / "metrics_soh_by_split_train_eval.json",
    )

    _feature_importance(model, feature_columns).to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    write_scaler_json(scaler, out_dir / "feature_scaler.json")
    write_schema_json(out_dir / "feature_schema.json", args.feature_mode, allow_upper_bound=bool(args.allow_upper_bound))
    model.save(
        out_dir,
        feature_columns=feature_columns,
        scaler=scaler,
        split_manifest=manifest,
        extra={
            "feature_mode": args.feature_mode,
            "dataset_csv": args.dataset_csv,
            "split_manifest_json": args.split_manifest_json,
            "scaler_json": args.scaler_json,
            "best_epoch": best_epoch,
            "best_val_mae": best_val,
            "best_visible_score": best_score,
            "best_selection_status": best_status,
            "seed": int(args.seed),
            "seed_locked": bool(args.seed_locked),
            "candidate_tag": args.candidate_tag,
            "protocol_tag": args.protocol_tag,
        },
    )
    torch.save(model.state_dict(), out_dir / "best.pt")
    torch.save(model.state_dict(), out_dir / "selected_model.pt")
    selected_audit = {
        "ok": bool(final_guard_ok),
        "best_epoch": int(best_epoch),
        "best_selection_status": str(best_status),
        "candidate_tag": args.candidate_tag,
        "protocol_tag": args.protocol_tag,
        "seed": int(args.seed),
        "seed_locked": bool(args.seed_locked),
        "visible_guard": {
            "min_train_r2_for_best": float(args.min_train_r2_for_best),
            "max_train_mae_for_best": float(args.max_train_mae_for_best),
            "max_val_mae_for_best": float(args.max_val_mae_for_best),
            "final_visible_metrics": final_visible_metrics,
            "guard_reasons": final_guard_reasons,
        },
        "test_metrics_used_for_selection": False,
        "no_test_metrics_in_training_history": True,
        "selected_model_pt": "selected_model.pt",
    }
    save_json(selected_audit, out_dir / "selected_checkpoint_audit.json")

    audit_post = audit_assb111_dataset(
        pred_frame,
        manifest=manifest,
        feature_columns=feature_columns,
        feature_mode=args.feature_mode,
        allow_upper_bound=bool(args.allow_upper_bound),
        scaler=scaler,
        fit_splits=("train",),
        train_history=hist_df,
    )
    write_audit_json(audit_post, out_dir / "leakage_audit_train_post.json")
    protocol_audit = audit_seed42locked_protocol_metadata(
        seed=int(args.seed),
        fit_splits=("train",),
        select_splits=("val",),
        required_seed=int(args.locked_seed_value) if args.seed_locked else int(args.seed),
        train_history=hist_df,
    )
    write_audit_json(protocol_audit, out_dir / "seed42_locked_protocol_audit.json")

    summary = {
        "output_model_dir": str(out_dir),
        "candidate_tag": args.candidate_tag,
        "protocol_tag": args.protocol_tag or "ASSB111_seed42_locked_trainval_only_small_optimization",
        "seed": int(args.seed),
        "seed_locked": bool(args.seed_locked),
        "best_epoch": best_epoch,
        "best_val_mae": best_val,
        "best_visible_score": best_score,
        "best_selection_status": best_status,
        "feature_mode": args.feature_mode,
        "n_features": len(feature_columns),
        "model_variant": cfg.model_variant,
        "final_visible_metrics": final_visible_metrics,
        "metrics_by_split_visible_only": metrics,
        "topk_average": topk_summary,
        "selected_checkpoint_audit_ok": bool(final_guard_ok),
        "selected_checkpoint_guard_reasons": final_guard_reasons,
        "leakage_ok": bool(audit_post.ok),
        "leakage_failures": audit_post.failures,
        "protocol_audit_ok": bool(protocol_audit.ok),
        "protocol_audit_failures": protocol_audit.failures,
        "no_test_metrics_in_training_history": True,
        "test_metrics_used_for_selection": False,
        "training_history_columns": list(hist_df.columns),
    }
    save_json(summary, out_dir / "train_summary.json")
    save_json(summary, out_dir / "training_summary.json")
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit_post.ok and protocol_audit.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
