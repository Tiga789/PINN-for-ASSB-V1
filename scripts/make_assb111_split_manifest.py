# -*- coding: utf-8 -*-
"""Create the strict 30/70 split manifest for ASSB ModelFin_111.

This script is the first step of the ASSB-111 workflow. It writes a hard
split manifest that allows SOH supervision only in the visible 30% cycles and
marks cycle 160..521 as held-out test. Cycle 522 is treated as partial/report
only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb111_split import (
    Strict30SplitConfig,
    load_capacity_targets_strict30,
    make_strict30_manifest,
    manifest_to_frame,
    validate_split_manifest,
    write_manifest,
)


def _json_clean(x: Any) -> Any:
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return [_json_clean(v) for v in x.tolist()]
    except Exception:
        pass
    if isinstance(x, dict):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    return x


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ASSB-111 strict30 split manifest generator")
    p.add_argument("--capacity_target_csv", default="Data/assb_capacity_soh_targets/capacity_soh_targets.csv")
    p.add_argument("--output_dir", default="Data/assb111")
    p.add_argument("--output_json", default="", help="Default: <output_dir>/split_manifest.json")
    p.add_argument("--output_csv", default="", help="Default: <output_dir>/split_manifest.csv")
    p.add_argument("--complete_cycle_min", type=int, default=5)
    p.add_argument("--complete_cycle_max", type=int, default=521)
    p.add_argument("--visible_label_cycle_min", type=int, default=5)
    p.add_argument("--visible_label_cycle_max", type=int, default=159)
    p.add_argument("--train_cycle_min", type=int, default=5)
    p.add_argument("--train_cycle_max", type=int, default=139)
    p.add_argument("--val_cycle_min", type=int, default=140)
    p.add_argument("--val_cycle_max", type=int, default=159)
    p.add_argument("--test_cycle_min", type=int, default=160)
    p.add_argument("--test_cycle_max", type=int, default=521)
    p.add_argument("--partial_cycles", default="522", help="comma separated partial/report-only cycles")
    p.add_argument("--model_id", type=int, default=111)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_json = Path(args.output_json) if args.output_json else out_dir / "split_manifest.json"
    out_csv = Path(args.output_csv) if args.output_csv else out_dir / "split_manifest.csv"
    partial = tuple(int(x.strip()) for x in str(args.partial_cycles).split(",") if x.strip())
    cfg = Strict30SplitConfig(
        model_id=int(args.model_id),
        complete_cycle_min=int(args.complete_cycle_min),
        complete_cycle_max=int(args.complete_cycle_max),
        visible_label_cycle_min=int(args.visible_label_cycle_min),
        visible_label_cycle_max=int(args.visible_label_cycle_max),
        train_cycle_min=int(args.train_cycle_min),
        train_cycle_max=int(args.train_cycle_max),
        val_cycle_min=int(args.val_cycle_min),
        val_cycle_max=int(args.val_cycle_max),
        test_cycle_min=int(args.test_cycle_min),
        test_cycle_max=int(args.test_cycle_max),
        partial_cycles=partial,
    )
    cap_csv = Path(args.capacity_target_csv)
    manifest: Dict[str, Any] = make_strict30_manifest(cap_csv, cfg=cfg)
    validate_split_manifest(manifest)
    write_manifest(manifest, out_json)

    targets = load_capacity_targets_strict30(cap_csv, cycle_from=cfg.complete_cycle_min, cycle_to=max(cfg.complete_cycle_max, *cfg.partial_cycles))
    split_frame = manifest_to_frame(manifest, targets)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "output_json": str(out_json),
        "output_csv": str(out_csv),
        "split_counts": manifest.get("split_counts", {}),
        "capacity_target_csv": str(cap_csv),
        "n_rows_csv": int(len(split_frame)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "split_manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_clean(summary), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
