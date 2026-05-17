# -*- coding: utf-8 -*-
"""Evaluate ASSB ModelFin_111 as a five-output package.

Four electrochemical state metrics are read from the frozen ModelFin_107A
corrected evaluation NPZ using paired true/pred arrays. SOH is produced by the
ASSB111 SOH head and evaluated under the strict 30/70 split.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from util.assb111_feature_schema import load_scaler_json
from util.assb111_leakage_guard import audit_assb111_dataset, transform_features_checked, write_audit_json
from util.assb111_split import load_manifest, split_for_cycles
from util.assb111_soh_model import Assb111SOHHead, metrics_by_split, prediction_frame_from_output, save_json, soh_metrics

STATE_KEYS = ("cs_a", "cs_c", "phie", "phis_c")
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
PRED_NPZ_PREFERRED_NAMES = (
    "eval_sampled_arrays_ModelFin107A_csA_corrected.npz",
    "eval_sampled_arrays_corrected.npz",
    "state_predictions_corrected.npz",
    "state_prediction_npz_internal_true_pred.npz",
    "eval_arrays.npz",
)


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


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


def _metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    if obs.shape != pred.shape:
        raise ValueError(f"shape mismatch: obs {obs.shape} vs pred {pred.shape}")
    mask = np.isfinite(obs) & np.isfinite(pred)
    out = {"n": int(np.sum(mask)), "MAE": float("nan"), "RMSE": float("nan"), "BIAS": float("nan"), "MAX": float("nan"), "R2": float("nan"), "corr": float("nan"), "NMAE": float("nan"), "NRMSE": float("nan")}
    if not mask.any():
        return out
    y = obs[mask]
    p = pred[mask]
    e = p - y
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e * e)))
    out.update(MAE=mae, RMSE=rmse, BIAS=float(np.mean(e)), MAX=float(np.max(np.abs(e))))
    denom = float(np.nanmax(y) - np.nanmin(y))
    if denom > 1e-30:
        out["NMAE"] = float(mae / denom)
        out["NRMSE"] = float(rmse / denom)
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot > 1e-30:
        out["R2"] = float(1.0 - np.sum(e * e) / ss_tot)
    if y.size >= 2 and np.std(y) > 1e-30 and np.std(p) > 1e-30:
        out["corr"] = float(np.corrcoef(y, p)[0, 1])
    return out


def _score_npz(path: Path) -> Dict[str, Any]:
    try:
        with np.load(path, allow_pickle=True) as z:
            files = list(z.files)
            score = 0
            paired = 0
            details = {}
            for var in STATE_KEYS:
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


def discover_state_npz(state_eval_dir: Path, explicit: Optional[Path], output_dir: Path) -> Path:
    candidates: List[Path] = []
    if explicit and explicit.exists():
        candidates = [explicit]
    elif state_eval_dir.exists():
        for name in PRED_NPZ_PREFERRED_NAMES:
            p = state_eval_dir / name
            if p.exists():
                candidates.append(p)
        candidates += [p for p in sorted(state_eval_dir.rglob("*.npz")) if p not in candidates]
    scored = [_score_npz(p) for p in candidates]
    valid = [d for d in scored if int(d.get("score", -1)) >= 8]
    selected = max(valid, key=lambda d: (int(d.get("n_paired_variables", 0)), int(d.get("score", -1)))) if valid else None
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json({"state_eval_dir": str(state_eval_dir), "explicit": str(explicit) if explicit else "", "selected": selected, "candidates": scored}, output_dir / "state_npz_discovery_assb111_eval.json")
    if not selected:
        raise FileNotFoundError(f"Cannot find paired 107A state npz in {state_eval_dir}")
    return Path(selected["path"])


def evaluate_states(state_npz_path: Path, output_dir: Path) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    prov: Dict[str, Any] = {"state_npz": str(state_npz_path), "variables": {}}
    with np.load(state_npz_path, allow_pickle=True) as z:
        files = list(z.files)
        for var in STATE_KEYS:
            pred_key = _find_key(files, PRED_ALIASES[var])
            true_key = _find_key(files, TRUE_ALIASES[var])
            if pred_key is None or true_key is None:
                metrics[var] = {"available": False, "reason": "missing paired true/pred arrays"}
                prov["variables"][var] = {"pred_key": pred_key, "true_key": true_key, "available": False}
                continue
            pred = np.asarray(z[pred_key])
            true = np.asarray(z[true_key])
            if pred.shape != true.shape:
                metrics[var] = {"available": False, "reason": f"shape mismatch {pred.shape} vs {true.shape}"}
                prov["variables"][var] = {"pred_key": pred_key, "true_key": true_key, "pred_shape": list(pred.shape), "true_shape": list(true.shape), "available": False}
                continue
            metrics[var] = _metrics(true, pred)
            metrics[var]["available"] = True
            prov["variables"][var] = {"pred_key": pred_key, "true_key": true_key, "shape": list(pred.shape), "available": True}
    save_json(metrics, output_dir / "metrics_state_guard.json")
    save_json(prov, output_dir / "debug_state_array_provenance_assb111.json")
    return metrics


def _segment_metrics(frame: pd.DataFrame, cycle_min: int, cycle_max: int) -> Dict[str, Any]:
    sub = frame[(pd.to_numeric(frame["cycle_id"], errors="coerce") >= cycle_min) & (pd.to_numeric(frame["cycle_id"], errors="coerce") <= cycle_max)]
    if sub.empty:
        return {"cycle_min": cycle_min, "cycle_max": cycle_max, "n": 0}
    m = soh_metrics(sub["SOH_obs"], sub["SOH_pred"])
    m.update(cycle_min=cycle_min, cycle_max=cycle_max)
    return m


def build_overdecay_diagnostic(pred_frame: pd.DataFrame) -> Dict[str, Any]:
    df = pred_frame.copy()
    diag: Dict[str, Any] = {}
    if "active_clamp_mask" in df.columns:
        active = df["active_clamp_mask"].astype(str).str.lower().isin(["true", "1", "yes"])
        diag["active_clamp_count"] = int(active.sum())
        diag["active_clamp_test_count"] = int((active & df["split"].astype(str).str.lower().eq("test")).sum()) if "split" in df.columns else int(active.sum())
    for split in sorted(str(s) for s in df.get("split", pd.Series(["all"])).dropna().unique()):
        sub = df[df["split"].astype(str) == split] if "split" in df.columns else df
        if sub.empty:
            continue
        prefix = f"{split}_"
        pred = pd.to_numeric(sub["SOH_pred"], errors="coerce").to_numpy(dtype=float)
        obs = pd.to_numeric(sub["SOH_obs"], errors="coerce").to_numpy(dtype=float) if "SOH_obs" in sub.columns else np.full_like(pred, np.nan)
        diag[prefix + "n"] = int(len(sub))
        diag[prefix + "SOH_pred_min"] = float(np.nanmin(pred)) if pred.size else None
        diag[prefix + "SOH_pred_max"] = float(np.nanmax(pred)) if pred.size else None
        diag[prefix + "SOH_obs_min"] = float(np.nanmin(obs)) if np.isfinite(obs).any() else None
        diag[prefix + "SOH_obs_max"] = float(np.nanmax(obs)) if np.isfinite(obs).any() else None
    # Tail cycles are especially important for the previous 0.4 clamp failure.
    test = df[df["split"].astype(str).str.lower().eq("test")] if "split" in df.columns else df
    if not test.empty:
        tail = test.sort_values("cycle_id").tail(20)
        diag["test_tail_cycles"] = [int(x) for x in tail["cycle_id"].tolist()]
        diag["test_tail_SOH_pred_min"] = float(pd.to_numeric(tail["SOH_pred"], errors="coerce").min())
        diag["test_tail_SOH_pred_max"] = float(pd.to_numeric(tail["SOH_pred"], errors="coerce").max())
        if "remaining_degradable" in tail.columns:
            diag["test_tail_remaining_min"] = float(pd.to_numeric(tail["remaining_degradable"], errors="coerce").min())
            diag["test_tail_remaining_max"] = float(pd.to_numeric(tail["remaining_degradable"], errors="coerce").max())
    diag["segments"] = [
        _segment_metrics(df, 160, 250),
        _segment_metrics(df, 251, 400),
        _segment_metrics(df, 401, 521),
    ]
    return diag


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate ASSB111 five-state package")
    p.add_argument("--model111_dir", "--model_dir", dest="model111_dir", default=r"ModelFin_111")
    p.add_argument("--dataset_csv", default=r"Data\assb111\dataset.csv")
    p.add_argument("--split_manifest_json", default=r"Data\assb111\split_manifest.json")
    p.add_argument("--scaler_json", default="")
    p.add_argument("--state_eval_dir", default=r"EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only")
    p.add_argument("--state_eval_npz", default="")
    p.add_argument("--output_dir", default=r"EvalFin_111_strict30_test70")
    p.add_argument("--feature_mode", default="p1_107a_strict")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--soft_fail", action="store_true", help="accepted for run-script compatibility; evaluator still writes outputs")
    return p.parse_args(argv)


def _load_scaler(model_dir: Path, explicit: str):
    if explicit:
        return load_scaler_json(explicit)
    p = model_dir / "feature_scaler.json"
    if p.exists():
        return load_scaler_json(p)
    cfg_path = model_dir / "soh_head_config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("scaler"):
            return payload["scaler"]
    raise FileNotFoundError("No scaler found. Provide --scaler_json or include feature_scaler.json in ModelFin_111.")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.allow_cpu:
            args.device = "cpu"
        else:
            raise RuntimeError("CUDA requested but unavailable. Use --allow_cpu for CPU evaluation.")
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model111_dir)
    manifest = load_manifest(args.split_manifest_json)
    frame = pd.read_csv(args.dataset_csv)
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    if "split" not in frame.columns:
        frame["split"] = split_for_cycles(frame["cycle_id"].to_numpy(dtype=int), manifest)
    scaler = _load_scaler(model_dir, args.scaler_json)
    feature_columns = list(scaler["feature_columns"])
    audit = audit_assb111_dataset(frame, manifest=manifest, feature_columns=feature_columns, feature_mode=args.feature_mode, scaler=scaler, fit_splits=("train",))
    write_audit_json(audit, out_dir / "leakage_audit.json")
    if not audit.ok:
        raise RuntimeError("Evaluation leakage audit failed: " + "; ".join(audit.failures))

    x_np = transform_features_checked(frame, scaler, manifest)
    cycle_np = frame["cycle_id"].to_numpy(dtype=int)
    delta = np.ones_like(cycle_np, dtype=float)
    if len(cycle_np) >= 2:
        delta[1:] = np.maximum(1.0, np.diff(cycle_np).astype(float))
    model = Assb111SOHHead.load(model_dir, map_location=device).to(device=device, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        out = model(torch.as_tensor(x_np, dtype=torch.float64, device=device), delta_cycle=torch.as_tensor(delta, dtype=torch.float64, device=device))
    pred_frame = prediction_frame_from_output(frame, out)
    pred_frame.to_csv(out_dir / "soh_pred_by_cycle.csv", index=False, encoding="utf-8-sig")
    soh_by_split = metrics_by_split(pred_frame)
    save_json({"metrics_by_split": soh_by_split}, out_dir / "metrics_soh_by_split.json")
    over = build_overdecay_diagnostic(pred_frame)
    save_json(over, out_dir / "soh_overdecay_diagnostic.json")

    state_npz = discover_state_npz(Path(args.state_eval_dir), Path(args.state_eval_npz) if args.state_eval_npz else None, out_dir)
    state_metrics = evaluate_states(state_npz, out_dir)
    variant = getattr(model.cfg, "model_variant", "unknown")
    soh_source = f"ModelFin_111_{variant}_strict30_test_cycles_160_521"

    rows = []
    for var in STATE_KEYS:
        m = dict(state_metrics.get(var, {}))
        rows.append({"variable": var, "source": "frozen_ModelFin107A_state_eval_npz", "n": m.get("n", 0), "MAE": m.get("MAE"), "RMSE": m.get("RMSE"), "NMAE": m.get("NMAE"), "NRMSE": m.get("NRMSE"), "R2": m.get("R2"), "corr": m.get("corr"), "available": m.get("available", False)})
    tm = soh_by_split.get("test", {})
    rows.append({"variable": "SOH", "source": soh_source, "n": tm.get("n", 0), "MAE": tm.get("SOH_MAE"), "RMSE": tm.get("SOH_RMSE"), "NMAE": tm.get("NMAE"), "NRMSE": tm.get("NRMSE"), "R2": tm.get("SOH_R2"), "corr": tm.get("SOH_corr"), "available": True})
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(out_dir / "five_state_scorecard_111.csv", index=False, encoding="utf-8-sig")
    scorecard.to_csv(out_dir / "five_state_scorecard.csv", index=False, encoding="utf-8-sig")
    save_json({
        "model111_dir": str(model_dir),
        "model_variant": variant,
        "dataset_csv": str(args.dataset_csv),
        "split_manifest_json": str(args.split_manifest_json),
        "state_eval_dir": str(args.state_eval_dir),
        "state_npz": str(state_npz),
        "feature_columns": feature_columns,
        "soh_overdecay_diagnostic_json": "soh_overdecay_diagnostic.json",
        "strict30_note": "Only SOH test metrics are new prediction metrics; four states are guarded frozen 107A outputs.",
    }, out_dir / "debug_model111_provenance.json")
    print(scorecard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
