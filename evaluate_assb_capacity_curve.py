#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate ModelFin_108 cycle-level capacity/SOH curve.

This evaluator reads the shared capacity target CSV and a trained
AgingCapacityHead from --model_dir. For the first landing step, it can also run
--fit_standalone to train only the capacity head and prove the target/evaluator
loop without opening the original PINN data loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    pd = None

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None

# Make util imports work when this script is placed in the project root.
ROOT = Path(__file__).resolve().parent
UTIL = ROOT / "util"
if str(UTIL) not in sys.path:
    sys.path.insert(0, str(UTIL))

from assb_capacity_targets import load_capacity_targets, summarize_capacity_targets
from aging_assb_capacity import AgingCapacityHead, capacity_physics_loss, load_capacity_head, save_capacity_head


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1.0e-15 or sy < 1.0e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    if denom <= 1.0e-24:
        return float("nan")
    return float(1.0 - np.sum((yt - yp) ** 2) / denom)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float = 1.0) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    err = y_pred[mask] - y_true[mask]
    if err.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "max_abs": float("nan"), "corr": float("nan"), "r2": float("nan")}
    return {
        "mae": float(np.mean(np.abs(err)) * scale),
        "rmse": float(np.sqrt(np.mean(err**2)) * scale),
        "max_abs": float(np.max(np.abs(err)) * scale),
        "corr": _pearson(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
    }


def _plot_capacity_curves(out_dir: Path, by_cycle: "pd.DataFrame") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] matplotlib unavailable; skip plots: {exc}")
        return

    cycle = by_cycle["cycle_id"].to_numpy()
    train = by_cycle["train_mask"].astype(bool).to_numpy()

    plt.figure(figsize=(9, 4.8))
    plt.plot(cycle, by_cycle["Q_obs_mAh"], label="Q_obs raw", linewidth=1.4)
    plt.plot(cycle, by_cycle["Q_pred_mAh"], label="Q_pred", linewidth=1.4)
    if (~train).any():
        plt.scatter(cycle[~train], by_cycle.loc[~train, "Q_obs_mAh"], label="excluded/partial", s=18)
    plt.xlabel("Cycle")
    plt.ylabel("Capacity (mAh)")
    plt.title("ASSB cycle-level discharge capacity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "capacity_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(cycle, by_cycle["SOH_obs"], label="SOH_obs", linewidth=1.4)
    plt.plot(cycle, by_cycle["SOH_pred"], label="SOH_pred", linewidth=1.4)
    if (~train).any():
        plt.scatter(cycle[~train], by_cycle.loc[~train, "SOH_obs"], label="excluded/partial", s=18)
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.title("ASSB cycle-level SOH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soh_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.axhline(0.0, linewidth=1.0)
    plt.plot(cycle, by_cycle["residual_mAh"], linewidth=1.2)
    if (~train).any():
        plt.scatter(cycle[~train], by_cycle.loc[~train, "residual_mAh"], s=18)
    plt.xlabel("Cycle")
    plt.ylabel("Q_pred - Q_obs (mAh)")
    plt.title("Capacity residual by cycle")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_by_cycle.png", dpi=180)
    plt.close()


def _fit_standalone_head(bundle, *, out_model_dir: Path, epochs: int, lr: float, hidden: int, device: str, seed: int) -> AgingCapacityHead:
    if torch is None:
        raise RuntimeError("PyTorch is required for standalone fitting.")
    torch.manual_seed(int(seed))
    dev = torch.device(device)
    dtype = torch.float64
    batch = bundle.as_torch(device=dev, dtype=dtype)
    head = AgingCapacityHead(n_features=bundle.features_np.shape[1], hidden=hidden, soh_min=0.45).to(device=dev, dtype=dtype)
    opt = torch.optim.Adam(head.parameters(), lr=float(lr))
    best_state = None
    best_loss = float("inf")
    for epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        loss, info = capacity_physics_loss(
            head,
            batch,
            w_capacity=1.0,
            w_monotone=0.1,
            w_smooth=0.05,
            w_prior=0.01,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
        opt.step()
        val = float(loss.detach().cpu())
        if val < best_loss:
            best_loss = val
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if epoch in {0, 1, 2, 5, 10, 50, 100, int(epochs) - 1}:
            print(
                f"[standalone capacity fit] epoch={epoch:05d} loss={val:.6e} "
                f"cap_mae_mAh={float(info['cap_mae_mAh']):.6f} soh_mae={float(info['soh_mae']):.6f}"
            )
    if best_state is not None:
        head.load_state_dict(best_state)
    save_capacity_head(head, out_model_dir)
    return head


def evaluate_capacity_head(head: AgingCapacityHead, bundle, *, device: str = "cpu") -> Tuple[Dict[str, Any], "pd.DataFrame"]:
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate capacity head.")
    dev = torch.device(device)
    dtype = torch.float64
    head = head.to(device=dev, dtype=dtype)
    head.eval()
    batch = bundle.as_torch(device=dev, dtype=dtype)
    with torch.no_grad():
        pred = head(batch["features"], batch["d_tau"], batch["Q_ref_Ah"])
        q_pred = pred["Q_pred_Ah"].detach().cpu().numpy()
        soh_pred = pred["SOH_pred"].detach().cpu().numpy()
        q_loss = pred["Q_loss_frac"].detach().cpu().numpy()

    frame = bundle.frame.copy()
    q_obs = bundle.q_dis_ah_np
    soh_obs = bundle.soh_np
    train = bundle.train_mask_np.astype(bool)
    by_cycle = pd.DataFrame(
        {
            "cycle_id": bundle.cycle_id,
            "Q_obs_Ah": q_obs,
            "Q_obs_mAh": q_obs * 1000.0,
            "Q_pred_Ah": q_pred,
            "Q_pred_mAh": q_pred * 1000.0,
            "SOH_obs": soh_obs,
            "SOH_pred": soh_pred,
            "Q_loss_frac_pred": q_loss,
            "residual_Ah": q_pred - q_obs,
            "residual_mAh": (q_pred - q_obs) * 1000.0,
            "train_mask": train,
            "complete_cycle": bundle.complete_cycle_np.astype(bool),
        }
    )

    q_m = _metrics(q_obs[train], q_pred[train], scale=1000.0)
    soh_m = _metrics(soh_obs[train], soh_pred[train], scale=1.0)
    q_all = _metrics(q_obs, q_pred, scale=1000.0)
    soh_all = _metrics(soh_obs, soh_pred, scale=1.0)
    global_metrics: Dict[str, Any] = {
        "q_ref_Ah": float(bundle.q_ref_ah),
        "q_ref_mAh": float(bundle.q_ref_ah * 1000.0),
        "n_cycles": int(len(q_obs)),
        "n_train_cycles": int(train.sum()),
        "cycle_min": int(np.min(bundle.cycle_id)),
        "cycle_max": int(np.max(bundle.cycle_id)),
        "train_Q_MAE_mAh": q_m["mae"],
        "train_Q_RMSE_mAh": q_m["rmse"],
        "train_Q_MAX_mAh": q_m["max_abs"],
        "train_Q_corr": q_m["corr"],
        "train_Q_R2": q_m["r2"],
        "train_SOH_MAE": soh_m["mae"],
        "train_SOH_RMSE": soh_m["rmse"],
        "train_SOH_corr": soh_m["corr"],
        "train_SOH_R2": soh_m["r2"],
        "all_Q_MAE_mAh": q_all["mae"],
        "all_Q_RMSE_mAh": q_all["rmse"],
        "all_Q_corr": q_all["corr"],
        "all_Q_R2": q_all["r2"],
        "all_SOH_MAE": soh_all["mae"],
        "all_SOH_RMSE": soh_all["rmse"],
        "all_SOH_corr": soh_all["corr"],
        "all_SOH_R2": soh_all["r2"],
        "feature_columns": list(bundle.feature_columns),
        "target_csv": bundle.csv_path,
    }
    return global_metrics, by_cycle


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ASSB ModelFin_108 capacity/SOH curve.")
    p.add_argument("--model_dir", default="ModelFin_108_capacityPhysicsOnly", help="Model directory containing capacity_head.pt/config. Used or created by --fit_standalone.")
    p.add_argument("--capacity_target_csv", required=True)
    p.add_argument("--output_dir", default="EvalFin_108_capacity_curve_physicsOnly")
    p.add_argument("--cycle_from", type=int, default=None)
    p.add_argument("--cycle_to", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fit_standalone", action="store_true", help="Train only the capacity head from the target CSV before evaluation. This does not use PINN data loss.")
    p.add_argument("--fit_epochs", type=int, default=3000)
    p.add_argument("--fit_lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    if pd is None:
        raise SystemExit("pandas is required for evaluate_assb_capacity_curve.py")
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    bundle = load_capacity_targets(args.capacity_target_csv, cycle_from=args.cycle_from, cycle_to=args.cycle_to)
    summary = summarize_capacity_targets(bundle)
    print("[ASSB capacity evaluator] target summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.fit_standalone:
        print("[ASSB capacity evaluator] fitting standalone capacity head (capacity-only scalar loss; original data loss remains unused).")
        head = _fit_standalone_head(bundle, out_model_dir=model_dir, epochs=args.fit_epochs, lr=args.fit_lr, hidden=args.hidden, device=args.device, seed=args.seed)
    else:
        head = load_capacity_head(model_dir, map_location=args.device)

    metrics, by_cycle = evaluate_capacity_head(head, bundle, device=args.device)
    (out_dir / "metrics_capacity_global.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    by_cycle.to_csv(out_dir / "metrics_by_cycle_capacity.csv", index=False, encoding="utf-8-sig")
    _plot_capacity_curves(out_dir, by_cycle)

    cfg = {
        "model_dir": str(model_dir),
        "capacity_target_csv": str(args.capacity_target_csv),
        "output_dir": str(out_dir),
        "fit_standalone": bool(args.fit_standalone),
        "device": args.device,
        "data_loss_used": False,
        "note": "Capacity/SOH scalar residual only; original PINN soft-label state data loss is not used by this evaluator.",
    }
    (out_dir / "config_snapshot.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ASSB capacity evaluator] OK")
    print(f"  output_dir       : {out_dir}")
    print(f"  train_Q_MAE_mAh : {metrics['train_Q_MAE_mAh']:.6f}")
    print(f"  train_SOH_MAE   : {metrics['train_SOH_MAE']:.6f}")
    print(f"  train_Q_R2      : {metrics['train_Q_R2']:.6f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
