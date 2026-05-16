# -*- coding: utf-8 -*-
r"""Evaluate ASSB aging-fix1 Stage-B/Stage-C outputs.

Outputs:
- metrics_capacity_by_split.json
- metrics_states_global.json
- mechanism_by_cycle.csv
- soh_pred_vs_obs.png
- mechanism_profiles.png
- state_error_by_cycle.png (when state prediction/reference npz are provided)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb_aging_capacity import capacity_metrics_by_split, q_ref_from_targets, save_json
from util.assb_aging_mechanism import AgingMechanismHead
from util.assb_model_integrity import compare_npz_states


def _load_feature_columns(frame: pd.DataFrame, explicit: Optional[str] = None):
    if explicit:
        return [c.strip() for c in explicit.split(",") if c.strip()]
    if "feature_columns" in frame.columns and frame["feature_columns"].notna().any():
        first = str(frame["feature_columns"].dropna().iloc[0])
        cols = [c.strip() for c in first.replace(",", ";").split(";") if c.strip()]
        if cols:
            return cols
    return [
        "cycle_norm",
        "throughput_norm",
        "duration_norm",
        "I_abs_mean_norm",
        "I_abs_max_norm",
        "q_charge_norm",
        "q_discharge_norm",
        "rest_fraction_norm",
    ]


def _predict_mechanism(model_dir: Path, cycle_table: pd.DataFrame, *, feature_columns, device: str) -> pd.DataFrame:
    if torch is None:
        raise RuntimeError("PyTorch is required to evaluate aging mechanism")
    model = AgingMechanismHead.load(model_dir, map_location=device)
    model.eval()
    dtype = torch.float64 if str(model.cfg.dtype).lower() in {"float64", "double", "torch.float64"} else torch.float32
    features = torch.as_tensor(cycle_table[feature_columns].to_numpy(dtype=float), dtype=dtype, device=device)
    cycle_id = torch.as_tensor(cycle_table["cycle_id"].to_numpy(dtype=int), dtype=torch.long, device=device)
    q_ref = q_ref_from_targets(cycle_table)
    with torch.no_grad():
        prof = model(features, cycle_id=cycle_id, q_ref_ah=q_ref)
    def cpu(x):
        return x.detach().cpu().numpy()
    out = cycle_table.copy()
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


def _plot_capacity(frame: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = frame["cycle_id"].to_numpy(dtype=float)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, frame["SOH_obs"].to_numpy(dtype=float), label="SOH obs")
    plt.plot(x, frame["SOH_pred"].to_numpy(dtype=float), label="SOH pred")
    plt.xlabel("cycle")
    plt.ylabel("SOH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "soh_pred_vs_obs.png", dpi=180)
    plt.close()


def _plot_mechanism(frame: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    x = frame["cycle_id"].to_numpy(dtype=float)
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x, frame["f_LAM_c"], label="f_LAM_c")
    ax1.plot(x, frame["theta_window_scale_c"], label="theta window scale")
    ax1.set_xlabel("cycle")
    ax1.set_ylabel("fraction / scale")
    ax2 = ax1.twinx()
    ax2.plot(x, frame["R_ohm_eff"], label="R_ohm_eff", linestyle="--")
    ax2.set_ylabel("R_ohm_eff")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "mechanism_profiles.png", dpi=180)
    plt.close(fig)


def _plot_state_error_by_cycle(metrics_csv: Optional[Path], out_dir: Path) -> None:
    if metrics_csv is None or not metrics_csv.exists():
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    frame = pd.read_csv(metrics_csv)
    if "cycle_id" not in frame.columns:
        return
    plt.figure(figsize=(8, 4.5))
    for col in frame.columns:
        if col.endswith("_MAE") and col != "cycle_id":
            plt.plot(frame["cycle_id"], frame[col], label=col)
    if len(plt.gca().lines) == 0:
        plt.close()
        return
    plt.xlabel("cycle")
    plt.ylabel("MAE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "state_error_by_cycle.png", dpi=180)
    plt.close()


def _state_metrics_from_npz(reference_npz: Optional[str], prediction_npz: Optional[str]) -> Dict[str, object]:
    if not reference_npz or not prediction_npz:
        return {"available": False, "reason": "prediction_npz and reference_npz not provided"}
    try:
        return compare_npz_states(reference_npz, prediction_npz, state_keys=("cs_a", "cs_c", "phie", "phis_c"))
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate ASSB aging-fix1 mechanism and state metrics")
    p.add_argument("--aging_model_dir", default="ModelFin_110_stageB")
    p.add_argument("--model_dir", default="", help="Alias for --aging_model_dir when evaluating Stage C")
    p.add_argument("--cycle_table_csv", required=True)
    p.add_argument("--capacity_target_csv", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--feature_columns", default="")
    p.add_argument("--prediction_npz", default="")
    p.add_argument("--reference_npz", default="")
    args = p.parse_args(argv)

    if args.model_dir:
        args.aging_model_dir = args.model_dir
    if args.device == "cuda" and (torch is None or not torch.cuda.is_available()):
        args.device = "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle_table = pd.read_csv(args.cycle_table_csv).sort_values("cycle_id").reset_index(drop=True)
    feature_columns = _load_feature_columns(cycle_table, explicit=args.feature_columns or None)
    pred_frame = _predict_mechanism(Path(args.aging_model_dir), cycle_table, feature_columns=feature_columns, device=args.device)
    pred_frame.to_csv(out_dir / "mechanism_by_cycle.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "capacity_by_split": capacity_metrics_by_split(pred_frame, complete_only=False),
        "capacity_by_split_complete_only": capacity_metrics_by_split(pred_frame, complete_only=True),
        "q_ref_Ah": float(q_ref_from_targets(pred_frame)),
        "n_cycles": int(len(pred_frame)),
        "feature_columns": feature_columns,
        "aging_model_dir": str(args.aging_model_dir),
    }
    save_json(metrics, out_dir / "metrics_capacity_by_split.json")
    state_metrics = _state_metrics_from_npz(args.reference_npz or None, args.prediction_npz or None)
    save_json(state_metrics, out_dir / "metrics_states_global.json")
    _plot_capacity(pred_frame, out_dir)
    _plot_mechanism(pred_frame, out_dir)
    print("[evaluate_assb_aging_fix1] wrote", out_dir)
    print(json.dumps(metrics["capacity_by_split"].get("all", {}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
