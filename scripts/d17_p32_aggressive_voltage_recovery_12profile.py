# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_pinn.config import cfg_get, load_config
from gv1.d17_pinn.p32_trainer import train_p32_mechanism_smoke


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P3.2 aggressive 12-profile voltage recovery smoke without state labels")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p32_12profile_voltage_recovery.json")
    ap.add_argument("--split_manifest", default=None)
    ap.add_argument("--resolved_spec", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--profile_count", type=int, default=None)
    ap.add_argument("--time_window_s", type=float, default=None)
    ap.add_argument("--max_time_points", type=int, default=None)
    ap.add_argument("--n_r", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--warmup_epochs", type=int, default=None)
    ap.add_argument("--voltage_recovery_until_epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["d17_protocol_version"] = 32
    cfg["experiment_name"] = "d17_p32_12profile_aggressive_voltage_recovery"
    if args.split_manifest:
        cfg.setdefault("paths", {})["split_manifest"] = args.split_manifest
    if args.resolved_spec:
        cfg.setdefault("paths", {})["resolved_spec"] = args.resolved_spec
    if args.split:
        cfg.setdefault("train", {})["split"] = args.split
    for name in ["profile_count", "time_window_s", "max_time_points", "n_r", "epochs", "warmup_epochs", "voltage_recovery_until_epoch", "lr", "device"]:
        v = getattr(args, name)
        if v is not None:
            cfg.setdefault("train", {})[name] = v
    out_dir = args.out_dir or str(Path(str(cfg_get(cfg, "paths.output_root"))) / "p32_12profile_aggressive_voltage_recovery")
    summary = train_p32_mechanism_smoke(cfg, out_dir)
    print(json.dumps({
        "status": summary.get("status"),
        "reasons": summary.get("reasons"),
        "out_dir": str(out_dir),
        "profile_count": summary.get("profile_count"),
        "best_epoch": summary.get("best_epoch"),
        "initial_voltage_mae_mean_V": summary.get("voltage_recovery", {}).get("initial_voltage_mae_mean_V"),
        "final_voltage_mae_mean_V": summary.get("voltage_recovery", {}).get("final_voltage_mae_mean_V"),
        "final_voltage_corr_mean": summary.get("voltage_recovery", {}).get("final_voltage_corr_mean"),
        "voltage_target_met": summary.get("voltage_recovery", {}).get("target_met"),
        "zero_mean_a_max": summary.get("final_aggregate", {}).get("zero_mean_max_abs_a_mol_m3_max"),
        "zero_mean_c_max": summary.get("final_aggregate", {}).get("zero_mean_max_abs_c_mol_m3_max"),
        "summary_json": str(Path(out_dir) / "D17_P32_12PROFILE_VOLTAGE_RECOVERY_SUMMARY.json"),
        "loss_scale_audit_json": str(Path(out_dir) / "D17_P32_LOSS_SCALE_AUDIT.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
