# -*- coding: utf-8 -*-
"""D17-P2 1-profile forward/backward smoke trainer.

This script trains only against observed voltage and physics residuals.  It does
not read cs/theta/phie/phis soft-label arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Dict

from gv1.d17_pinn.config import cfg_get, load_config
from gv1.d17_pinn.trainer import train_smoke


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P2 no-state-label mechanism smoke trainer")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p2_smoke.yaml")
    ap.add_argument("--split_manifest", default=None)
    ap.add_argument("--resolved_spec", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--profile_index", type=int, default=None)
    ap.add_argument("--time_window_s", type=float, default=None)
    ap.add_argument("--max_time_points", type=int, default=None)
    ap.add_argument("--n_r", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)
    if args.split_manifest:
        cfg.setdefault("paths", {})["split_manifest"] = args.split_manifest
    if args.resolved_spec:
        cfg.setdefault("paths", {})["resolved_spec"] = args.resolved_spec
    if args.split:
        cfg.setdefault("train", {})["split"] = args.split
    for name in ["profile_index", "time_window_s", "max_time_points", "n_r", "epochs", "lr", "device"]:
        v = getattr(args, name)
        if v is not None:
            cfg.setdefault("train", {})[name] = v
    out_dir = args.out_dir or str(Path(str(cfg_get(cfg, "paths.output_root"))) / "smoke_1profile_p2")
    summary = train_smoke(cfg, out_dir)
    print(json.dumps({
        "status": summary.get("status"),
        "out_dir": str(out_dir),
        "best_epoch": summary.get("best_epoch"),
        "best_loss": summary.get("best_loss"),
        "final_voltage_mae_V": summary.get("final_metrics", {}).get("voltage_mae_V"),
        "zero_mean_max_abs_a_mol_m3": summary.get("final_metrics", {}).get("zero_mean_max_abs_a_mol_m3"),
        "zero_mean_max_abs_c_mol_m3": summary.get("final_metrics", {}).get("zero_mean_max_abs_c_mol_m3"),
        "summary_json": str(Path(out_dir) / "D17_P2_SMOKE_SUMMARY.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
