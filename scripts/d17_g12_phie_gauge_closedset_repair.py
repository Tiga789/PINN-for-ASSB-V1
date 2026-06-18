from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.d17_g.g12_trainer import build_and_train_g12


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D17-G1.2 phie/gauge/target-scaling closed-set repair for generator surrogate.")
    p.add_argument("--config", required=True)
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
    cfg = load_json(args.config)
    summary = build_and_train_g12(
        split_manifest=args.split_manifest,
        g0_profile_semantics_csv=args.g0_profile_semantics_csv,
        out_dir=args.out_dir,
        config=cfg,
        train_profile_count=int(args.train_profile_count if args.train_profile_count is not None else cfg.get("train_profile_count", 12)),
        validation_profile_count=int(args.validation_profile_count if args.validation_profile_count is not None else cfg.get("validation_profile_count", 3)),
        max_time_points=int(args.max_time_points if args.max_time_points is not None else cfg.get("max_time_points", 512)),
        time_window_s=float(args.time_window_s if args.time_window_s is not None else cfg.get("time_window_s", 40000.0)),
        device_arg=str(args.device if args.device is not None else cfg.get("device", "auto")),
        epochs=int(args.epochs if args.epochs is not None else cfg.get("epochs", 700)),
        lr=float(args.lr if args.lr is not None else cfg.get("lr", 8e-4)),
        batch_size=int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1024)),
    )
    print(json.dumps({
        "status": summary.get("status"),
        "recommendation": summary.get("recommendation"),
        "reasons": summary.get("reasons", []),
        "g2_ready": summary.get("g2_ready"),
        "train_closedset_mean_r2": summary.get("train_closedset_per_target_aggregate", {}).get("all_target_profile_r2_mean"),
        "train_closedset_min_r2": summary.get("train_closedset_per_target_aggregate", {}).get("all_target_profile_r2_min"),
        "phie_r2_mean": summary.get("train_closedset_per_target_aggregate", {}).get("phie_r2_mean"),
        "phie_r2_min": summary.get("train_closedset_per_target_aggregate", {}).get("phie_r2_min"),
        "best_epoch": summary.get("best_epoch"),
        "summary_json": summary.get("files", {}).get("summary_json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
