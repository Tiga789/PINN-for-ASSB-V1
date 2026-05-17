#!/usr/bin/env python3
"""Train/val-only helper for ASSB-111 seed42-locked small optimization.

This script intentionally does not read final test evaluation files such as
five_state_scorecard.csv, soh_pred_by_cycle.csv, or soh_overdecay_diagnostic.json.
It can either write a candidate grid JSON, or score already-trained candidate
model directories from their train_summary.json and leakage_audit.json files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

FORBIDDEN_SELECTION_STEMS = {
    "five_state_scorecard.csv",
    "soh_pred_by_cycle.csv",
    "soh_overdecay_diagnostic.json",
    "soh_overdecay_diagnostic_segments.csv",
}

@dataclass
class CandidateSpec:
    tag: str
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    use_ema: bool = False
    topk_checkpoint_avg: bool = False
    dropout: float = 0.05
    seed: int = 42
    soh_model_variant: str = "saturating_v2"


def default_grid() -> List[CandidateSpec]:
    return [
        CandidateSpec("c00_base_lr5e4_e5000", 5e-4, 1e-5, 5000, 900, False, False, 0.05),
        CandidateSpec("c01_lr3e4_e7000", 3e-4, 1e-5, 7000, 1200, False, False, 0.05),
        CandidateSpec("c02_lr2e4_e7000_ema", 2e-4, 1e-5, 7000, 1200, True, False, 0.05),
        CandidateSpec("c03_lr3e4_e7000_topk", 3e-4, 5e-6, 7000, 1200, False, True, 0.05),
        CandidateSpec("c04_lr2e4_e7000_ema_topk", 2e-4, 5e-6, 7000, 1400, True, True, 0.02),
    ]


def read_json(path: Path, files_read: List[str]) -> Dict[str, Any]:
    files_read.append(str(path))
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(d: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def safe_float(x: Any, default: float = math.nan) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def score_candidate(model_dir: Path, *, max_val_mae: float, min_train_r2: float, require_no_test_history: bool) -> Dict[str, Any]:
    files_read: List[str] = []
    summary_path = model_dir / "train_summary.json"
    if not summary_path.exists():
        summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        return {
            "model_dir": str(model_dir),
            "candidate_tag": model_dir.name,
            "available": False,
            "visible_ok": False,
            "reason": "missing train_summary.json/training_summary.json",
            "files_read": [],
            "selection_used_test_metrics": False,
        }

    summary = read_json(summary_path, files_read)
    leakage_path = model_dir / "leakage_audit.json"
    leakage = read_json(leakage_path, files_read) if leakage_path.exists() else {"ok": None, "missing": True}

    final_visible = summary.get("final_visible_metrics", {}) or {}
    split_visible = summary.get("metrics_by_split_visible_only", {}) or {}
    train_metrics = split_visible.get("train", {}) if isinstance(split_visible, dict) else {}
    val_metrics = split_visible.get("val", {}) if isinstance(split_visible, dict) else {}

    train_mae = safe_float(final_visible.get("train_mae", train_metrics.get("SOH_MAE")))
    train_r2 = safe_float(final_visible.get("train_r2", train_metrics.get("SOH_R2")))
    val_mae = safe_float(final_visible.get("val_mae", val_metrics.get("SOH_MAE", summary.get("best_val_mae"))))
    val_r2 = safe_float(final_visible.get("val_r2", val_metrics.get("SOH_R2")))
    monotonic_penalty = safe_float(summary.get("visible_monotonic_penalty", 0.0), 0.0)

    no_test_hist = bool(summary.get("no_test_metrics_in_training_history", False))
    leakage_ok = bool(leakage.get("ok", False)) if leakage.get("ok", None) is not None else False
    visible_ok = True
    reasons: List[str] = []
    if math.isnan(val_mae):
        visible_ok = False; reasons.append("val_mae_nan")
    if math.isnan(train_mae):
        visible_ok = False; reasons.append("train_mae_nan")
    if math.isnan(train_r2):
        visible_ok = False; reasons.append("train_r2_nan")
    if not math.isnan(val_mae) and val_mae > max_val_mae:
        visible_ok = False; reasons.append(f"val_mae>{max_val_mae}")
    if not math.isnan(train_r2) and train_r2 < min_train_r2:
        visible_ok = False; reasons.append(f"train_r2<{min_train_r2}")
    if not leakage_ok:
        visible_ok = False; reasons.append("leakage_not_ok")
    if require_no_test_history and not no_test_hist:
        visible_ok = False; reasons.append("test_metrics_may_be_in_history")

    visible_score = val_mae + 0.15 * train_mae + 0.02 * monotonic_penalty
    tag = summary.get("candidate_tag") or summary.get("config", {}).get("candidate_tag") or model_dir.name.replace("Model_", "")
    return {
        "model_dir": str(model_dir),
        "candidate_tag": str(tag),
        "available": True,
        "visible_ok": visible_ok,
        "reason": ";".join(reasons) if reasons else "ok",
        "visible_score": visible_score,
        "val_mae": val_mae,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "visible_monotonic_penalty": monotonic_penalty,
        "leakage_ok": leakage_ok,
        "no_test_metrics_in_training_history": no_test_hist,
        "files_read": files_read,
        "selection_used_test_metrics": any(Path(p).name in FORBIDDEN_SELECTION_STEMS for p in files_read),
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "candidate_tag", "model_dir", "available", "visible_ok", "visible_score",
        "val_mae", "train_mae", "train_r2", "val_r2", "visible_monotonic_penalty",
        "leakage_ok", "no_test_metrics_in_training_history", "reason",
        "selection_used_test_metrics",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate_model_dirs", nargs="*", default=[], help="Candidate model directories to score from visible train/val summaries.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--write_grid_json", action="store_true", help="Write the default candidate grid JSON into output_dir.")
    p.add_argument("--max_val_mae", type=float, default=0.00150)
    p.add_argument("--min_train_r2", type=float, default=0.990)
    p.add_argument("--require_no_test_history", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.write_grid_json:
        grid = [asdict(c) for c in default_grid()]
        (out / "seed42locked_candidate_grid.json").write_text(json.dumps({
            "protocol": "ASSB111_seed42_locked_trainval_only_small_optimization",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": 42,
            "soh_model_variant": "saturating_v2",
            "selection_rule": "visible_train_val_only",
            "candidate_grid": grid,
        }, indent=2), encoding="utf-8")

    rows = [score_candidate(Path(d), max_val_mae=args.max_val_mae, min_train_r2=args.min_train_r2, require_no_test_history=args.require_no_test_history) for d in args.candidate_model_dirs]
    rows_sorted = sorted(rows, key=lambda r: (not bool(r.get("visible_ok")), safe_float(r.get("visible_score"), 1e99)))
    write_csv(rows_sorted, out / "candidate_visible_score.csv")

    eligible = [r for r in rows_sorted if r.get("visible_ok") and not r.get("selection_used_test_metrics")]
    selected = eligible[0] if eligible else (rows_sorted[0] if rows_sorted else None)
    result = {
        "protocol": "ASSB111_seed42_locked_trainval_only_small_optimization",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_mode": "visible_train_val_only",
        "selection_rule": "minimize val_mae + 0.15*train_mae + 0.02*visible_monotonic_penalty subject to visible guards",
        "test_metrics_used_for_selection": False,
        "forbidden_selection_files": sorted(FORBIDDEN_SELECTION_STEMS),
        "n_candidates": len(rows),
        "n_eligible": len(eligible),
        "selected": selected,
    }
    (out / "selected_candidate.json").write_text(json.dumps(result if selected is None else {**selected, **{
        "protocol": result["protocol"],
        "created_at": result["created_at"],
        "selection_mode": result["selection_mode"],
        "selection_rule": result["selection_rule"],
        "test_metrics_used_for_selection": False,
        "selection_used_test_metrics": False,
        "forbidden_selection_files": sorted(FORBIDDEN_SELECTION_STEMS),
        "all_candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
    }}, indent=2), encoding="utf-8")

    print(json.dumps({"output_dir": str(out), "n_candidates": len(rows), "selected": None if selected is None else selected.get("candidate_tag"), "test_metrics_used_for_selection": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
