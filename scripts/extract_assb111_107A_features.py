# -*- coding: utf-8 -*-
"""Extract strict non-label cycle features for ASSB ModelFin_111.

The extractor uses the 107A corrected evaluation NPZ when available and falls
back to the continuous v2 mass-closed solution for missing fields. It writes one
row per cycle and does not merge SOH labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from util.assb111_feature_extractor import extract_assb111_cycle_features, feature_summary, write_feature_outputs
from util.assb111_feature_schema import select_feature_columns, write_schema_json

PRED_NPZ_PREFERRED_NAMES: Tuple[str, ...] = (
    "eval_sampled_arrays_ModelFin107A_csA_corrected.npz",
    "eval_sampled_arrays_corrected.npz",
    "eval_sampled_arrays.npz",
    "states_corrected.npz",
    "predictions_corrected.npz",
    "eval_predictions_corrected.npz",
    "prediction_corrected.npz",
    "predictions.npz",
    "prediction.npz",
    "eval_predictions.npz",
    "state_prediction.npz",
    "pinn_predictions.npz",
    "results.npz",
)
PRED_ALIASES = {
    "cs_a": ["cs_a_pred", "cs_a_prediction", "pred_cs_a", "prediction_cs_a", "csa_pred", "cs_a_pred_corrected", "cs_a_corrected_pred", "cs_a_hat"],
    "cs_c": ["cs_c_pred", "cs_c_prediction", "pred_cs_c", "prediction_cs_c", "csc_pred", "cs_c_pred_corrected", "cs_c_corrected_pred", "cs_c_hat"],
    "phie": ["phie_pred", "phie_prediction", "pred_phie", "prediction_phie", "phi_e_pred", "phie_hat"],
    "phis_c": ["phis_c_pred", "phis_c_prediction", "pred_phis_c", "prediction_phis_c", "phi_s_c_pred", "phis_pred", "phis_c_hat"],
}
TRUE_ALIASES = {
    "cs_a": ["cs_a_true", "true_cs_a", "cs_a_ref", "ref_cs_a", "cs_a_reference", "reference_cs_a", "cs_a_label", "label_cs_a", "csa_true", "csa_ref"],
    "cs_c": ["cs_c_true", "true_cs_c", "cs_c_ref", "ref_cs_c", "cs_c_reference", "reference_cs_c", "cs_c_label", "label_cs_c", "csc_true", "csc_ref"],
    "phie": ["phie_true", "true_phie", "phie_ref", "ref_phie", "phie_reference", "reference_phie", "phie_label", "label_phie"],
    "phis_c": ["phis_c_true", "true_phis_c", "phis_c_ref", "ref_phis_c", "phis_c_reference", "reference_phis_c", "phis_c_label", "label_phis_c"],
}


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _find_key(files: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    direct = {str(k).lower(): str(k) for k in files}
    for alias in aliases:
        hit = direct.get(str(alias).lower())
        if hit is not None:
            return hit
    relaxed = {_norm(k): str(k) for k in files}
    for alias in aliases:
        hit = relaxed.get(_norm(alias))
        if hit is not None:
            return hit
    return None


def _score_npz(path: Path) -> Dict[str, Any]:
    try:
        with np.load(path, allow_pickle=True) as z:
            files = list(z.files)
            score = 0
            paired = 0
            details: Dict[str, Dict[str, Optional[str]]] = {}
            for var in ("cs_a", "cs_c", "phie", "phis_c"):
                pred = _find_key(files, PRED_ALIASES[var])
                true = _find_key(files, TRUE_ALIASES[var])
                if pred:
                    score += 3
                if true:
                    score += 3
                if pred and true:
                    score += 5
                    paired += 1
                details[var] = {"pred_key": pred, "true_key": true}
            if path.name in PRED_NPZ_PREFERRED_NAMES:
                score += 3
            return {"path": str(path), "score": score, "n_paired_variables": paired, "details": details, "keys_sample": files[:50]}
    except Exception as exc:
        return {"path": str(path), "score": -1, "error": str(exc)}


def discover_state_npz(state_eval_dir: Optional[Path], explicit: Optional[Path], output_dir: Path) -> Optional[Path]:
    if explicit is not None and str(explicit).strip() and explicit.exists():
        selected = explicit
        candidates = [_score_npz(explicit)]
    elif state_eval_dir is not None and state_eval_dir.exists():
        candidates_paths: List[Path] = []
        for name in PRED_NPZ_PREFERRED_NAMES:
            p = state_eval_dir / name
            if p.exists():
                candidates_paths.append(p)
        candidates_paths += [p for p in sorted(state_eval_dir.rglob("*.npz")) if p not in candidates_paths]
        candidates = [_score_npz(p) for p in candidates_paths]
        valid = [d for d in candidates if int(d.get("score", -1)) >= 8]
        selected_dict = max(valid, key=lambda d: (int(d.get("n_paired_variables", 0)), int(d.get("score", -1)))) if valid else None
        selected = Path(selected_dict["path"]) if selected_dict else None
    else:
        selected = None
        candidates = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "state_npz_discovery_assb111.json").open("w", encoding="utf-8") as f:
        json.dump({"state_eval_dir": str(state_eval_dir) if state_eval_dir else "", "explicit": str(explicit) if explicit else "", "selected": str(selected) if selected else None, "candidates": candidates}, f, ensure_ascii=False, indent=2, sort_keys=True)
    return selected


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract ASSB111 107A cycle-level features")
    p.add_argument("--solution_npz", default="../assb_soft_labels_cycle5_522_v2_massclosed_candidate/solution.npz")
    p.add_argument("--state_eval_dir", default=r"EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only")
    p.add_argument("--state_eval_npz", default="")
    p.add_argument("--cycle_table_csv", default="")
    p.add_argument("--split_manifest_json", default="Data/assb111/split_manifest.json")
    p.add_argument("--output_dir", default="Data/assb111")
    p.add_argument("--output_csv", default="", help="Default: <output_dir>/features_107A_cycle.csv")
    p.add_argument("--output_json", default="", help="Default: <output_dir>/feature_summary.json")
    p.add_argument("--feature_mode", default="p1_107a_strict")
    p.add_argument("--allow_missing_features", action="store_true")
    p.add_argument("--cycle_from", type=int, default=5)
    p.add_argument("--cycle_to", type=int, default=521)
    p.add_argument("--cs_a_max", type=float, default=6.0)
    p.add_argument("--cs_c_max", type=float, default=51.8)
    p.add_argument("--allow_solution_state_fallback", action="store_true", help="diagnostic only; strict P1 should provide a 107A prediction NPZ")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_csv = Path(args.output_csv) if args.output_csv else out_dir / "features_107A_cycle.csv"
    out_json = Path(args.output_json) if args.output_json else out_dir / "feature_summary.json"
    explicit = Path(args.state_eval_npz) if args.state_eval_npz else None
    selected_npz = discover_state_npz(Path(args.state_eval_dir) if args.state_eval_dir else None, explicit, out_dir)
    frame = extract_assb111_cycle_features(
        solution_npz=args.solution_npz,
        state_eval_npz=selected_npz,
        cycle_table_csv=args.cycle_table_csv or None,
        split_manifest_json=args.split_manifest_json or None,
        cycle_from=int(args.cycle_from),
        cycle_to=int(args.cycle_to),
        cs_a_max=float(args.cs_a_max),
        cs_c_max=float(args.cs_c_max),
    )
    cols = select_feature_columns(frame, args.feature_mode, allow_missing=bool(args.allow_missing_features))
    frame.attrs["assb111_feature_columns"] = cols
    write_feature_outputs(frame, out_csv, out_json)
    write_schema_json(out_dir / "feature_schema.json", args.feature_mode, allow_upper_bound=False)
    summary = feature_summary(frame)
    summary["selected_state_npz"] = str(selected_npz) if selected_npz else ""
    summary["selected_feature_columns"] = cols
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
