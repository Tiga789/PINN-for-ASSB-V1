from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_g.g1_data import build_g1_dataset, json_load
from gv1.d17_g.g1_trainer import train_g1_smoke


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D17-G1 supervised generator-surrogate smoke. Train-cell soft labels are allowed by design.")
    p.add_argument("--config", default="configs/d17_g1_supervised_surrogate_smoke.json")
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--g0_profile_semantics_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_profile_count", type=int, default=None)
    p.add_argument("--validation_profile_count", type=int, default=None)
    p.add_argument("--max_time_points", type=int, default=None)
    p.add_argument("--time_window_s", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg: Dict[str, Any] = json_load(config_path, default={}) or {}
    train_count = int(args.train_profile_count if args.train_profile_count is not None else cfg.get("train_profile_count", 12))
    val_count = int(args.validation_profile_count if args.validation_profile_count is not None else cfg.get("validation_profile_count", 3))
    max_points = int(args.max_time_points if args.max_time_points is not None else cfg.get("max_time_points", 512))
    time_window = float(args.time_window_s if args.time_window_s is not None else cfg.get("time_window_s", 40000.0))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 180))
    lr = float(args.lr if args.lr is not None else cfg.get("lr", 0.001))
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 2048))
    device = str(args.device if args.device is not None else cfg.get("device", "auto"))
    ds = build_g1_dataset(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        train_profile_count=train_count,
        validation_profile_count=val_count,
        max_time_points=max_points,
        time_window_s=time_window,
    )
    summary = train_g1_smoke(
        dataset=ds,
        out_dir=args.out_dir,
        config=cfg,
        device_arg=device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )
    print(json.dumps({
        "status": summary.get("status"),
        "promotion_status": summary.get("promotion_status"),
        "g2_ready": summary.get("g2_ready"),
        "reasons": summary.get("reasons", []),
        "promotion_reasons": summary.get("promotion_reasons", []),
        "out_dir": str(args.out_dir),
        "best_epoch": summary.get("best_epoch"),
        "train_r2_mean": summary.get("train_profile_aggregate", {}).get("r2_mean_mean"),
        "validation_r2_mean_report_only": summary.get("validation_profile_aggregate_report_only", {}).get("r2_mean_mean"),
        "summary_json": summary.get("files", {}).get("summary_json"),
    }, ensure_ascii=False, indent=2))
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
