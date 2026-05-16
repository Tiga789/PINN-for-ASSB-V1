# -*- coding: utf-8 -*-
"""
Evaluate ASSB ModelFin_109 aging mechanism outputs.

This evaluator focuses on the new mechanism variables and cycle-level SOH/Q
metrics.  State-output metrics for cs_a/cs_c/phie/phis_c are saved when a
prediction npz is available; otherwise the evaluator still works for mechanism
validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for evaluate_assb_ModelFin109_aging.py") from exc

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb_aging_state import AgingConfig, AgingMechanismHead, aging_profiles_to_numpy
from util.assb_capacity_from_states import (
    capacity_from_mechanism,
    capacity_metrics,
    load_capacity_targets_simple,
    save_json,
    split_capacity_metrics,
)
from util.assb_cycle_table import load_cycle_table, summarize_cycle_table


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_state_file(model_dir: Path) -> Optional[Path]:
    for name in ["aging_state.pt", "aging_head.pt", "aging_state.pth", "best_aging_state.pt"]:
        p = model_dir / name
        if p.exists():
            return p
    return None


def _load_aging_head(model_dir: Path, *, feature_dim: int, device: str = "cpu") -> AgingMechanismHead:
    if torch is None:
        raise RuntimeError("PyTorch is required to load ModelFin_109 aging state.")
    cfg_dict = _load_json(model_dir / "aging_config.json")
    if not cfg_dict:
        cfg_dict = _load_json(model_dir / "config.json")
    cfg_dict = dict(cfg_dict)
    cfg_dict.setdefault("AGING_FEATURE_DIM", int(feature_dim))
    cfg = AgingConfig.from_params(cfg_dict)
    cfg.feature_dim = int(feature_dim)
    head = AgingMechanismHead(cfg).to(device=device)
    state_file = _find_state_file(model_dir)
    if state_file is None:
        raise FileNotFoundError(
            f"No aging_state.pt found in {model_dir}. Train ModelFin_109 first, or use --fit_mechanism_standalone."
        )
    state = torch.load(state_file, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "aging_head" in state:
        state = state["aging_head"]
    head.load_state_dict(state, strict=False)
    head.eval()
    return head


def _fit_standalone_head(
    cycle_table,
    cap_frame: pd.DataFrame,
    *,
    epochs: int,
    lr: float,
    device: str,
    seed: int = 42,
) -> AgingMechanismHead:
    if torch is None:
        raise RuntimeError("PyTorch is required for --fit_mechanism_standalone.")
    torch.manual_seed(int(seed))
    features = torch.as_tensor(cycle_table.features_np, dtype=torch.float64, device=device)
    cycle = cycle_table.cycle_id
    merged = pd.DataFrame({"cycle_id": cycle}).merge(cap_frame, on="cycle_id", how="left")
    if merged["Q_dis_Ah"].isna().any():
        raise RuntimeError("Standalone fitting requires capacity targets for all cycle-table rows.")
    q_obs = torch.as_tensor(merged["Q_dis_Ah"].to_numpy(dtype=float), dtype=torch.float64, device=device)
    q_ref = float(merged["Q_ref_Ah"].dropna().iloc[0]) if "Q_ref_Ah" in merged.columns else float(q_obs.max().detach().cpu())
    cfg = AgingConfig(feature_dim=features.shape[1])
    head = AgingMechanismHead(cfg).to(device=device)
    opt = torch.optim.Adam(head.parameters(), lr=float(lr))
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        profiles = head(features, cycle_id=torch.as_tensor(cycle, dtype=torch.long, device=device))
        q_pred, _ = capacity_from_mechanism(profiles, q_ref)
        loss = torch.nn.functional.smooth_l1_loss(q_pred, q_obs)
        # Small monotone/smooth priors are already structural, but add a weak curvature penalty.
        if profiles.soh_mech.numel() > 2:
            loss = loss + 0.01 * torch.mean((profiles.soh_mech[2:] - 2.0 * profiles.soh_mech[1:-1] + profiles.soh_mech[:-2]) ** 2)
        loss.backward()
        opt.step()
    head.eval()
    return head


def _merge_mechanism_capacity(cycle_table, cap_frame: pd.DataFrame, profiles_np: Dict[str, object], q_ref_ah: float) -> pd.DataFrame:
    q_pred = np.asarray(q_ref_ah * np.asarray(profiles_np["SOH_mech"], dtype=float), dtype=float)
    out = cycle_table.frame.copy()
    out["z"] = np.asarray(profiles_np["z"], dtype=float)
    out["z_norm"] = np.asarray(profiles_np["z_norm"], dtype=float)
    out["f_LAM_c"] = np.asarray(profiles_np["f_lam_c"], dtype=float)
    out["R_ohm"] = np.asarray(profiles_np["R_ohm"], dtype=float)
    out["theta_window_c"] = np.asarray(profiles_np["theta_window_c"], dtype=float)
    out["SOH_pred"] = np.asarray(profiles_np["SOH_mech"], dtype=float)
    out["Q_pred_Ah"] = q_pred
    if cap_frame is not None:
        cap_small = cap_frame[["cycle_id", "Q_dis_Ah", "SOH", "Q_ref_Ah"]].copy()
        cap_small = cap_small.rename(columns={"Q_dis_Ah": "Q_obs_Ah", "SOH": "SOH_obs"})
        # Avoid duplicate *_x/*_y columns if the prepared cycle table already
        # contains capacity labels.
        for col in ["Q_obs_Ah", "SOH_obs", "Q_ref_Ah"]:
            if col in out.columns:
                out = out.drop(columns=[col])
        out = out.merge(cap_small, on="cycle_id", how="left")
    return out


def _state_metrics_from_prediction_npz(prediction_npz: Optional[Path], output_dir: Path) -> Dict[str, object]:
    if prediction_npz is None or not prediction_npz.exists():
        return {"available": False, "reason": "prediction_npz not provided"}
    with np.load(prediction_npz, allow_pickle=True) as z:
        keys = set(z.files)
        metrics: Dict[str, object] = {"available": True, "prediction_npz": str(prediction_npz)}
        for name in ["cs_a", "cs_c", "phie", "phis_c"]:
            pred_key = f"{name}_pred"
            true_key = f"{name}_true"
            if pred_key in keys and true_key in keys:
                pred = np.asarray(z[pred_key], dtype=float)
                true = np.asarray(z[true_key], dtype=float)
                mask = np.isfinite(pred) & np.isfinite(true)
                if mask.any():
                    resid = pred[mask] - true[mask]
                    metrics[name] = {
                        "MAE": float(np.mean(np.abs(resid))),
                        "RMSE": float(np.sqrt(np.mean(resid ** 2))),
                        "MAX": float(np.max(np.abs(resid))),
                    }
        return metrics


def _save_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    x = frame["cycle_id"].to_numpy(dtype=float)
    if "SOH_obs" in frame.columns:
        plt.figure(figsize=(8, 4.5))
        for split, part in frame.groupby("split"):
            plt.scatter(part["cycle_id"], part["SOH_obs"], s=8, label=f"obs {split}")
        plt.plot(x, frame["SOH_pred"].to_numpy(dtype=float), linewidth=1.5, label="pred")
        plt.xlabel("cycle")
        plt.ylabel("SOH")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "soh_pred_vs_obs.png", dpi=180)
        plt.close()

    plt.figure(figsize=(8, 5.0))
    plt.plot(x, frame["f_LAM_c"].to_numpy(dtype=float), label="f_LAM_c")
    r = frame["R_ohm"].to_numpy(dtype=float)
    if np.nanmax(r) > np.nanmin(r):
        r_norm = (r - np.nanmin(r)) / max(np.nanmax(r) - np.nanmin(r), 1.0e-12)
    else:
        r_norm = np.zeros_like(r)
    w = frame["theta_window_c"].to_numpy(dtype=float)
    if np.nanmax(w) > np.nanmin(w):
        w_norm = (w - np.nanmin(w)) / max(np.nanmax(w) - np.nanmin(w), 1.0e-12)
    else:
        w_norm = np.zeros_like(w)
    plt.plot(x, r_norm, label="R_ohm normalized")
    plt.plot(x, w_norm, label="theta_window normalized")
    plt.xlabel("cycle")
    plt.ylabel("mechanism profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mechanism_profiles.png", dpi=180)
    plt.close()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ASSB ModelFin_109 aging mechanism.")
    p.add_argument("--model_dir", default="ModelFin_109")
    p.add_argument("--solution_npz", default=None, help="Reserved for state prediction integration; currently optional.")
    p.add_argument("--capacity_target_csv", required=True)
    p.add_argument("--cycle_table_csv", required=True)
    p.add_argument("--output_dir", default="EvalFin_109_aging_mechanism")
    p.add_argument("--device", default="cuda")
    p.add_argument("--prediction_npz", default=None, help="Optional evaluator npz containing *_pred and *_true arrays for state metrics.")
    p.add_argument("--fit_mechanism_standalone", action="store_true", help="Fit an aging head to capacity labels only for script/data validation.")
    p.add_argument("--fit_epochs", type=int, default=2000)
    p.add_argument("--fit_lr", type=float, default=2.0e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device = args.device
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        device = "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle_table = load_cycle_table(args.cycle_table_csv)
    cap_frame = load_capacity_targets_simple(args.capacity_target_csv)
    q_ref = float(cap_frame["Q_ref_Ah"].dropna().iloc[0]) if "Q_ref_Ah" in cap_frame.columns else float(cap_frame["Q_dis_Ah"].max())

    if args.fit_mechanism_standalone:
        head = _fit_standalone_head(cycle_table, cap_frame, epochs=args.fit_epochs, lr=args.fit_lr, device=device, seed=args.seed)
    else:
        head = _load_aging_head(Path(args.model_dir), feature_dim=cycle_table.features_np.shape[1], device=device)

    with torch.no_grad():
        features = torch.as_tensor(cycle_table.features_np, dtype=torch.float64, device=device)
        cycle_id = torch.as_tensor(cycle_table.cycle_id, dtype=torch.long, device=device)
        profiles = head(features, cycle_id=cycle_id)
    profiles_np = aging_profiles_to_numpy(profiles)
    mech_frame = _merge_mechanism_capacity(cycle_table, cap_frame, profiles_np, q_ref)
    mech_frame.to_csv(output_dir / "mechanism_by_cycle.csv", index=False, encoding="utf-8-sig")

    metrics_capacity = split_capacity_metrics(mech_frame)
    metrics_states = _state_metrics_from_prediction_npz(Path(args.prediction_npz) if args.prediction_npz else None, output_dir)
    save_json(metrics_capacity, output_dir / "metrics_capacity_by_split.json")
    save_json(metrics_states, output_dir / "metrics_states_global.json")
    save_json(summarize_cycle_table(cycle_table), output_dir / "cycle_table_summary.json")
    save_json(
        {
            "model_dir": args.model_dir,
            "capacity_target_csv": args.capacity_target_csv,
            "cycle_table_csv": args.cycle_table_csv,
            "device": device,
            "fit_mechanism_standalone": bool(args.fit_mechanism_standalone),
        },
        output_dir / "config_snapshot.json",
    )
    _save_plots(mech_frame, output_dir)
    print("[evaluate_assb_ModelFin109_aging] wrote:")
    print(f"  {output_dir / 'mechanism_by_cycle.csv'}")
    print(f"  {output_dir / 'metrics_capacity_by_split.json'}")
    print(json.dumps(metrics_capacity, ensure_ascii=False, indent=2))
    if not metrics_states.get("available", False):
        print("[state metrics] prediction_npz not provided; cs/phie/phis_c state metrics skipped.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
