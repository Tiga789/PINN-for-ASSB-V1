# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_pinn.config import cfg_get, load_config
from gv1.d17_pinn.p33_trainer import train_p33_forward_core_reliability


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P3.3 forward-core reliability audit + D12-S1K transition-fade formula migration")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p33_forward_core_reliability.json")
    ap.add_argument("--split_manifest", default=None)
    ap.add_argument("--resolved_spec", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--validation_split", default=None)
    ap.add_argument("--profile_count", type=int, default=None)
    ap.add_argument("--validation_profile_count", type=int, default=None)
    ap.add_argument("--time_window_s", type=float, default=None)
    ap.add_argument("--max_time_points", type=int, default=None)
    ap.add_argument("--n_r", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--warmup_epochs", type=int, default=None)
    ap.add_argument("--voltage_recovery_until_epoch", type=int, default=None)
    ap.add_argument("--validation_adaptation_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["d17_protocol_version"] = 33
    cfg["experiment_name"] = "d17_p33_forward_core_reliability_audit"
    if args.split_manifest:
        cfg.setdefault("paths", {})["split_manifest"] = args.split_manifest
    if args.resolved_spec:
        cfg.setdefault("paths", {})["resolved_spec"] = args.resolved_spec
    if args.split:
        cfg.setdefault("train", {})["split"] = args.split
    if args.validation_split:
        cfg.setdefault("validation", {})["split"] = args.validation_split
    train_arg_names = ["profile_count", "time_window_s", "max_time_points", "n_r", "epochs", "warmup_epochs", "voltage_recovery_until_epoch", "lr", "device"]
    for name in train_arg_names:
        v = getattr(args, name)
        if v is not None:
            cfg.setdefault("train", {})[name] = v
    if args.validation_profile_count is not None:
        cfg.setdefault("validation", {})["profile_count"] = args.validation_profile_count
    if args.validation_adaptation_steps is not None:
        cfg.setdefault("validation", {})["adaptation_steps"] = args.validation_adaptation_steps
    out_dir = args.out_dir or str(Path(str(cfg_get(cfg, "paths.output_root"))) / "p33_forward_core_reliability_audit")
    summary = train_p33_forward_core_reliability(cfg, out_dir)
    print(json.dumps({
        "status": summary.get("status"),
        "reasons": summary.get("reasons"),
        "promotion_status": summary.get("promotion_status"),
        "promotion_reasons": summary.get("promotion_reasons"),
        "out_dir": str(out_dir),
        "train_profile_count": summary.get("train_profile_count"),
        "validation_profile_count": summary.get("validation_profile_count"),
        "best_epoch": summary.get("best_epoch"),
        "train_corrected_voltage_mae_mean_V": summary.get("voltage_recovery", {}).get("train_corrected_voltage_mae_mean_V"),
        "train_forward_voltage_mae_mean_V": summary.get("voltage_recovery", {}).get("train_forward_voltage_mae_mean_V"),
        "validation_corrected_voltage_mae_mean_V": summary.get("voltage_recovery", {}).get("validation_corrected_voltage_mae_mean_V"),
        "validation_forward_voltage_mae_mean_V": summary.get("voltage_recovery", {}).get("validation_forward_voltage_mae_mean_V"),
        "corrected_target_met": summary.get("voltage_recovery", {}).get("corrected_target_met"),
        "forward_core_reliability_status": summary.get("residual_budget_audit", {}).get("forward_core_reliability_status"),
        "residual_budget_status": summary.get("residual_budget_audit", {}).get("residual_budget_status"),
        "summary_json": str(Path(out_dir) / "D17_P33_FORWARD_CORE_RELIABILITY_SUMMARY.json"),
        "formula_alignment_audit_json": str(Path(out_dir) / "D17_P33_FORMULA_ALIGNMENT_AUDIT.json"),
        "residual_budget_audit_json": str(Path(out_dir) / "D17_P33_RESIDUAL_BUDGET_AUDIT.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
