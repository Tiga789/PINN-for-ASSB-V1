# -*- coding: utf-8 -*-
r"""Train Stage-B ASSB aging mechanism head.

This script freezes the PINN core by not loading it at all.  It trains only the
cycle-level ``AgingMechanismHead`` against capacity/SOH observations.  The goal
is to prove that the low-dimensional aging mechanism can learn the observed
capacity fade before it is injected into effective-SPM closure.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb_aging_capacity import capacity_loss, capacity_metrics_by_split, q_ref_from_targets, save_json
from util.assb_aging_fix1_config import AgingFix1Config, save_aging_config, validate_aging_config
from util.assb_aging_mechanism import AgingMechanismHead, save_profiles_csv


def _load_feature_columns(frame: pd.DataFrame, explicit: Optional[str] = None) -> List[str]:
    if explicit:
        cols = [c.strip() for c in explicit.split(",") if c.strip()]
    elif "feature_columns" in frame.columns and frame["feature_columns"].notna().any():
        first = str(frame["feature_columns"].dropna().iloc[0])
        cols = [c.strip() for c in first.replace(",", ";").split(";") if c.strip()]
    else:
        cols = [
            "cycle_norm",
            "throughput_norm",
            "duration_norm",
            "I_abs_mean_norm",
            "I_abs_max_norm",
            "q_charge_norm",
            "q_discharge_norm",
            "rest_fraction_norm",
        ]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns in cycle table: {missing}")
    return cols


def _as_tensor(values, *, device, dtype=torch.float64):
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)


def _make_training_masks(frame: pd.DataFrame, fit_splits: Sequence[str], device) -> Dict[str, torch.Tensor]:
    split = frame["split"].astype(str).to_numpy() if "split" in frame.columns else np.array(["train"] * len(frame))
    fit = np.zeros(len(frame), dtype=bool)
    wanted = {s.strip().lower() for s in fit_splits if s.strip()}
    if "all" in wanted:
        fit[:] = True
    else:
        for name in wanted:
            fit |= np.char.lower(split.astype(str)) == name
    if not fit.any():
        raise RuntimeError(f"No training rows selected by fit_splits={fit_splits}")
    return {
        "fit": torch.as_tensor(fit, dtype=torch.bool, device=device),
        "complete": torch.as_tensor(frame.get("complete_cycle", pd.Series([True] * len(frame))).astype(bool).to_numpy(), dtype=torch.bool, device=device),
    }


def _write_training_history(history: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False, encoding="utf-8-sig")


def _evaluate_frame(frame: pd.DataFrame, model: AgingMechanismHead, features: torch.Tensor, cycle_id: torch.Tensor, q_ref_ah: float, device: str) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        prof = model(features, cycle_id=cycle_id, q_ref_ah=q_ref_ah)
    out = frame.copy()
    def cpu(x):
        return x.detach().cpu().numpy()
    out["Q_pred_Ah"] = cpu(prof.Q_pred_Ah)
    out["Q_pred_mAh"] = out["Q_pred_Ah"] * 1000.0
    out["SOH_pred"] = cpu(prof.SOH_struct)
    out["f_LAM_c"] = cpu(prof.f_LAM_c)
    out["theta_window_scale_c"] = cpu(prof.theta_window_scale_c)
    out["R_ohm_eff"] = cpu(prof.R_ohm_eff)
    out["lam_damage"] = cpu(prof.lam_damage)
    out["window_damage"] = cpu(prof.window_damage)
    out["r_ohm_growth"] = cpu(prof.r_ohm_growth)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train ASSB aging-fix1 Stage-B mechanism head")
    p.add_argument("--cycle_table_csv", required=True)
    p.add_argument("--capacity_target_csv", default="")  # retained for explicit logging; table already contains targets
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-6)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--hidden_layers", type=int, default=2)
    p.add_argument("--fit_splits", default="train,val,test", help="comma-separated split names or all")
    p.add_argument("--feature_columns", default="")
    p.add_argument("--lam_max", type=float, default=0.60)
    p.add_argument("--window_loss_max", type=float, default=0.45)
    p.add_argument("--r_ohm_delta_max", type=float, default=250.0)
    p.add_argument("--w_soh", type=float, default=5.0)
    p.add_argument("--w_q", type=float, default=1.0)
    p.add_argument("--w_final", type=float, default=10.0)
    p.add_argument("--w_smooth", type=float, default=0.05)
    p.add_argument("--w_rate", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_every", type=int, default=250)
    p.add_argument("--allow_cpu", action="store_true")
    args = p.parse_args(argv)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.allow_cpu:
            args.device = "cpu"
        else:
            raise RuntimeError("CUDA requested but unavailable. Use --device cpu or --allow_cpu.")
    device = torch.device(args.device)
    dtype = torch.float64

    frame = pd.read_csv(args.cycle_table_csv)
    frame = frame.sort_values("cycle_id").reset_index(drop=True)
    feature_cols = _load_feature_columns(frame, explicit=args.feature_columns or None)
    q_ref_ah = q_ref_from_targets(frame)
    features = _as_tensor(frame[feature_cols].to_numpy(dtype=float), device=device, dtype=dtype)
    cycle_id = torch.as_tensor(frame["cycle_id"].to_numpy(dtype=int), dtype=torch.long, device=device)
    q_obs = _as_tensor(frame["Q_obs_Ah"].to_numpy(dtype=float), device=device, dtype=dtype)
    soh_obs = _as_tensor(frame["SOH_obs"].to_numpy(dtype=float), device=device, dtype=dtype)
    masks = _make_training_masks(frame, [s.strip() for s in args.fit_splits.split(",")], device=device)

    cfg = AgingFix1Config(
        stage="B_MECHANISM",
        feature_dim=len(feature_cols),
        hidden_dim=int(args.hidden_dim),
        hidden_layers=int(args.hidden_layers),
        lam_max=float(args.lam_max),
        window_loss_max=float(args.window_loss_max),
        r_ohm_delta_max=float(args.r_ohm_delta_max),
        w_soh=float(args.w_soh),
        w_q=float(args.w_q),
        w_final=float(args.w_final),
        w_smooth=float(args.w_smooth),
        w_rate=float(args.w_rate),
        data_loss=False,
        alpha_data=0.0,
        max_batch_size_data=0,
    )
    cfg = validate_aging_config(cfg)
    model = AgingMechanismHead(cfg).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        prof = model(features, cycle_id=cycle_id, q_ref_ah=q_ref_ah)
        loss, logs = capacity_loss(
            q_obs,
            prof.Q_pred_Ah,
            soh_obs,
            prof.SOH_struct,
            cfg,
            train_mask=masks["fit"],
            complete_mask=masks["complete"],
            lam_rate=prof.lam_rate,
            window_rate=prof.window_rate,
            f_lam=prof.f_LAM_c,
            window_scale=prof.theta_window_scale_c,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()
        logs["epoch"] = epoch
        history.append(logs)
        if logs["loss_total"] < best_loss:
            best_loss = logs["loss_total"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % int(args.print_every) == 0 or epoch == int(args.epochs):
            print(f"[StageB] epoch={epoch} loss={logs['loss_total']:.6e} q={logs['loss_q']:.3e} soh={logs['loss_soh']:.3e} final={logs['loss_final']:.3e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.save(output_dir, extra={"feature_columns": feature_cols, "q_ref_ah": q_ref_ah, "fit_splits": args.fit_splits})
    save_aging_config(cfg, output_dir / "aging_config.json")
    _write_training_history(history, output_dir / "training_history.csv")

    pred_frame = _evaluate_frame(frame, model, features, cycle_id, q_ref_ah, str(device))
    pred_frame.to_csv(output_dir / "mechanism_by_cycle.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "capacity_by_split": capacity_metrics_by_split(pred_frame, complete_only=False),
        "capacity_by_split_complete_only": capacity_metrics_by_split(pred_frame, complete_only=True),
        "q_ref_Ah": float(q_ref_ah),
        "feature_columns": feature_cols,
        "fit_splits": args.fit_splits,
        "best_loss": float(best_loss),
        "n_cycles": int(len(pred_frame)),
    }
    save_json(metrics, output_dir / "metrics_capacity_by_split.json")
    save_json({"available": False, "reason": "Stage B trains mechanism only; four-state PINN core is not evaluated here."}, output_dir / "metrics_states_global.json")
    print("[StageB] wrote", output_dir)
    print(json.dumps(metrics["capacity_by_split"].get("all", {}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
