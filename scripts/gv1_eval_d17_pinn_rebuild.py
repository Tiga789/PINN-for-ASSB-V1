# -*- coding: utf-8 -*-
"""D17-P2 observed-only evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Dict

from gv1.d17_pinn.config import load_config
from gv1.d17_pinn.evaluator import eval_one_profile


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P2 observed-only eval smoke")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p2_smoke.yaml")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--profile_index", type=int, default=0)
    ap.add_argument("--split_manifest", default=None)
    ap.add_argument("--resolved_spec", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    cfg: Dict[str, Any] = load_config(args.config)
    if args.split_manifest:
        cfg.setdefault("paths", {})["split_manifest"] = args.split_manifest
    if args.resolved_spec:
        cfg.setdefault("paths", {})["resolved_spec"] = args.resolved_spec
    summary = eval_one_profile(cfg, args.model_path, args.out_dir, split=args.split, profile_index=args.profile_index)
    print(json.dumps({
        "status": summary.get("status"),
        "out_dir": args.out_dir,
        "split": args.split,
        "metrics": summary.get("metrics"),
        "summary_json": str(Path(args.out_dir) / "D17_P2_EVAL_SUMMARY.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
