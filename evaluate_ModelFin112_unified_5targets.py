# -*- coding: utf-8 -*-
"""Evaluate ModelFin_112 deterministic five-target wrapper.

Outputs:
- five_state_scorecard.csv: cs_a/cs_c/phie/phis_c from frozen 107A state source,
  SOH from deterministic ridge head.
- soh_pred_by_cycle.csv: deterministic SOH predictions.
- unified_eval_audit.json: provenance and leakage boundary.

No model selection is performed here.  Test metrics are evaluated only after the
ridge model has already been fixed by train/val-visible selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from util.assb112_deterministic_wrapper import (
    default_state_npz_candidates,
    default_state_scorecard_candidates,
    load_deterministic_soh_from_wrapper,
    load_json,
    load_state_metrics_from_npz,
    load_state_rows_from_scorecard,
    save_json,
    soh_score_rows_from_prediction,
    write_scorecard,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ModelFin_112 deterministic wrapper")
    p.add_argument("--model_dir", default="ModelFin_112_deterministic_wrapper")
    p.add_argument("--dataset_csv", default=r"Data\assb112_feature_audit_v1\dataset_with_voltage_features.csv")
    p.add_argument("--output_dir", default="EvalFin_112_deterministic_wrapper")
    p.add_argument("--state_scorecard_csv", default="", help="Optional override for state metric scorecard")
    p.add_argument("--state_eval_npz", default="", help="Optional override for paired state NPZ")
    p.add_argument("--prefer_scorecard", action="store_true", help="Prefer CSV state scorecard over NPZ when both exist")
    return p.parse_args(argv)


def _resolve_ref(ref: str, model_dir: Path) -> Path:
    if not ref:
        return Path("")
    p = Path(ref)
    if p.is_absolute():
        return p
    if (model_dir / p).exists():
        return model_dir / p
    if (ROOT / p).exists():
        return ROOT / p
    return p


def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _load_state_rows(args: argparse.Namespace, model_dir: Path, cfg: Dict[str, Any]) -> tuple[list, dict]:
    # Explicit CLI override first.
    scorecard = Path(args.state_scorecard_csv) if args.state_scorecard_csv else None
    npz = Path(args.state_eval_npz) if args.state_eval_npz else None

    # Wrapper config references.
    if not scorecard or not scorecard.exists():
        ref = str(cfg.get("state_scorecard_csv", ""))
        cand = _resolve_ref(ref, model_dir) if ref else Path("")
        if cand.exists():
            scorecard = cand
    if not npz or not npz.exists():
        ref = str(cfg.get("state_eval_npz", ""))
        cand = _resolve_ref(ref, model_dir) if ref else Path("")
        if cand.exists():
            npz = cand

    # Autodiscovery fallback.
    if not scorecard or not scorecard.exists():
        scorecard = _first_existing(default_state_scorecard_candidates(ROOT))
    if not npz or not npz.exists():
        npz = _first_existing(default_state_npz_candidates(ROOT))

    errors: List[str] = []
    if args.prefer_scorecard:
        order = [("scorecard", scorecard), ("npz", npz)]
    else:
        order = [("npz", npz), ("scorecard", scorecard)]

    for kind, path in order:
        if not path or not Path(path).exists():
            continue
        try:
            if kind == "npz":
                rows, audit = load_state_metrics_from_npz(path)
            else:
                rows, audit = load_state_rows_from_scorecard(path)
            if len(rows) >= 4:
                audit["state_source_kind"] = kind
                return rows, audit
            errors.append(f"{kind}:{path} produced only {len(rows)} state rows")
        except Exception as e:
            errors.append(f"{kind}:{path}: {type(e).__name__}: {e}")

    return [], {"state_source_kind": "missing", "errors": errors, "message": "No usable frozen 107A state scorecard/NPZ was found. SOH evaluation can still be valid, but five-target scorecard is incomplete."}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = model_dir / "unified_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"unified_config.json not found. Run scripts/build_ModelFin112_single_model.py first: {cfg_path}")
    cfg = load_json(cfg_path)

    dataset_csv = Path(args.dataset_csv)
    if not dataset_csv.exists():
        # Try config reference if CLI default is missing.
        cfg_dataset = _resolve_ref(str(cfg.get("dataset_csv", "")), model_dir)
        if cfg_dataset.exists():
            dataset_csv = cfg_dataset
        else:
            raise FileNotFoundError(f"dataset_csv not found: {args.dataset_csv}")
    frame = pd.read_csv(dataset_csv)

    soh_model = load_deterministic_soh_from_wrapper(model_dir)
    soh_pred = soh_model.predict_frame(frame)
    soh_pred.to_csv(output_dir / "soh_pred_by_cycle.csv", index=False, encoding="utf-8-sig")

    state_rows, state_audit = _load_state_rows(args, model_dir, cfg)
    soh_rows = soh_score_rows_from_prediction(soh_pred)

    rows = []
    rows.extend(state_rows)
    # Place test SOH first among SOH rows if available, then other splits.
    soh_rows_sorted = sorted(soh_rows, key=lambda r: {"test": 0, "all_eval": 1, "train": 2, "val": 3, "partial": 4, "all_rows": 5}.get(str(r.get("split", "")), 9))
    rows.extend(soh_rows_sorted)

    scorecard = write_scorecard(rows, output_dir / "five_state_scorecard.csv")
    scorecard.to_csv(output_dir / "five_target_scorecard.csv", index=False, encoding="utf-8-sig")

    # Convenience: compact one-row state/SOH summary for quick terminal inspection.
    compact = []
    for var in ["cs_a", "cs_c", "phie", "phis_c"]:
        sub = scorecard[scorecard["variable"].astype(str).eq(var)]
        if not sub.empty:
            r = sub.iloc[0].to_dict()
            compact.append({"variable": var, "split": r.get("split", ""), "MAE": r.get("MAE"), "RMSE": r.get("RMSE"), "R2": r.get("R2"), "corr": r.get("corr"), "n": r.get("n")})
    soh_test = scorecard[(scorecard["variable"].astype(str).eq("SOH")) & (scorecard.get("split", pd.Series(dtype=str)).astype(str).eq("test"))]
    if not soh_test.empty:
        r = soh_test.iloc[0].to_dict()
        compact.append({"variable": "SOH", "split": "test", "MAE": r.get("MAE"), "RMSE": r.get("RMSE"), "R2": r.get("R2"), "corr": r.get("corr"), "n": r.get("n")})
    pd.DataFrame(compact).to_csv(output_dir / "five_target_compact_summary.csv", index=False, encoding="utf-8-sig")

    audit = {
        "ok": True,
        "model_dir": str(model_dir),
        "dataset_csv": str(dataset_csv),
        "output_dir": str(output_dir),
        "scorecard_csv": "five_state_scorecard.csv",
        "compact_summary_csv": "five_target_compact_summary.csv",
        "model_level": cfg.get("model_level"),
        "boundary_note": cfg.get("boundary_note"),
        "state_audit": state_audit,
        "soh_model_type": cfg.get("soh_model_type", "deterministic_ridge_soh_head"),
        "soh_no_test_selection": True,
        "test_metrics_used_for_selection": False,
        "note": "This evaluator reports a deterministic engineering wrapper: frozen 107A states + deterministic ridge SOH. It does not claim end-to-end jointly trained coupling.",
    }
    save_json(audit, output_dir / "unified_eval_audit.json")

    print(f"[OK] wrote {output_dir / 'five_state_scorecard.csv'}")
    if not state_rows:
        print("[WARN] no state rows found; provide --state_scorecard_csv or --state_eval_npz for complete five-target evaluation")
    soh_test = scorecard[(scorecard["variable"].astype(str).eq("SOH")) & (scorecard.get("split", pd.Series(dtype=str)).astype(str).eq("test"))]
    if not soh_test.empty:
        r = soh_test.iloc[0]
        print(f"[SOH TEST] R2={r.get('R2')} MAE={r.get('MAE')} RMSE={r.get('RMSE')} BIAS={r.get('BIAS')} corr={r.get('corr')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
