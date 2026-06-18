# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from gv1.d17_pinn.config import cfg_get, load_config
from gv1.d17_pinn.p3_trainer import train_p3_mechanism_smoke

def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P3 6-profile no-state-label mechanism smoke")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p3_6profile_smoke.json")
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
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["d17_protocol_version"] = 3
    cfg["experiment_name"] = "d17_p3_6profile_mechanism_smoke"
    if args.split_manifest: cfg.setdefault("paths", {})["split_manifest"] = args.split_manifest
    if args.resolved_spec: cfg.setdefault("paths", {})["resolved_spec"] = args.resolved_spec
    if args.split: cfg.setdefault("train", {})["split"] = args.split
    for name in ["profile_count", "time_window_s", "max_time_points", "n_r", "epochs", "warmup_epochs", "lr", "device"]:
        v = getattr(args, name)
        if v is not None: cfg.setdefault("train", {})[name] = v
    out_dir = args.out_dir or str(Path(str(cfg_get(cfg, "paths.output_root"))) / "p3_6profile_mechanism_smoke")
    summary = train_p3_mechanism_smoke(cfg, out_dir)
    print(json.dumps({"status": summary.get("status"), "reasons": summary.get("reasons"), "out_dir": str(out_dir), "profile_count": summary.get("profile_count"), "best_epoch": summary.get("best_epoch"), "final_voltage_mae_mean_V": summary.get("final_aggregate", {}).get("voltage_mae_V_mean"), "final_voltage_corr_mean": summary.get("final_aggregate", {}).get("voltage_corr_mean"), "summary_json": str(Path(out_dir) / "D17_P3_6PROFILE_SMOKE_SUMMARY.json")}, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()
