#!/usr/bin/env python
"""D12-S1I high-region fallback/budget wrapper for S1E-soft predictions.

This stage does not train a new neural network.  It formalizes the D12-S1H
finding that the S1E-soft candidate passes low/deep/global/normal/rest/corr and
fails only because its P2D correction leaks into high-voltage regions.

Inputs:
  - S1E 6-profile 40ks runs root containing baseline_d951 and
    d12s1e_p2d_low_anchor_soft prediction.npz files.

Outputs:
  - Corrected prediction.npz files in a new D12-S1I runs root.
  - D12_S1I_run_metrics.csv
  - D12_S1I_segment_metrics.csv
  - D12_S1I_mode_summary.csv
  - D12_S1I_candidate_decisions.csv
  - D12_S1I_profile_repair_diagnostics.csv
  - D12_S1I_scorecard_summary.json
  - D12_S1I_RECOMMENDATION.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _as1(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _safe_corr(y: np.ndarray, p: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) < 3:
        return float("nan")
    yy = y - float(np.nanmean(y))
    pp = p - float(np.nanmean(p))
    den = math.sqrt(float(np.nanmean(yy * yy)) * float(np.nanmean(pp * pp)))
    if not np.isfinite(den) or den <= 1e-12:
        return float("nan")
    return float(np.nanmean(yy * pp) / den)


def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    if len(y) == 0:
        return {"n": 0, "MAE_V": float("nan"), "RMSE_V": float("nan"), "corr": float("nan"), "bias_V": float("nan")}
    err = p - y
    return {
        "n": int(len(y)),
        "MAE_V": float(np.nanmean(np.abs(err))),
        "RMSE_V": float(math.sqrt(float(np.nanmean(err * err)))),
        "corr": _safe_corr(y, p),
        "bias_V": float(np.nanmean(err)),
    }


def _q(arr: np.ndarray, q: float) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(np.nanquantile(arr, q))


def _parse_run(path: Path) -> Tuple[str, str]:
    name = path.parent.name
    if "__" in name:
        mode, profile = name.split("__", 1)
        return mode, profile
    parts = name.split("_Batch-")
    if len(parts) == 2:
        return parts[0], "Batch-" + parts[1]
    return name, name


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _target_pred_current(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "voltage_exp" not in arrays:
        raise KeyError("prediction.npz missing voltage_exp")
    y = _as1(arrays["voltage_exp"]).astype(float)
    if "voltage_exp_pred" in arrays:
        p = _as1(arrays["voltage_exp_pred"]).astype(float)
    elif "phis_c_pred" in arrays:
        p = _as1(arrays["phis_c_pred"]).astype(float)
    else:
        raise KeyError("prediction.npz missing voltage_exp_pred or phis_c_pred")
    cur = _as1(arrays.get("I_profile", np.zeros_like(y))).astype(float)
    n = min(len(y), len(p), len(cur))
    return y[:n], p[:n], cur[:n]


def _segment_masks(v: np.ndarray, pred: np.ndarray, current: np.ndarray) -> Dict[str, np.ndarray]:
    finite = np.isfinite(v) & np.isfinite(pred)
    return {
        "all": finite,
        "low_target": finite & (v <= 3.00),
        "low_target_le_2p75": finite & (v <= 2.75),
        "normal_target_gt_3p20": finite & (v > 3.20),
        "target_mid_3p0_4p1": finite & (v > 3.00) & (v < 4.10),
        "high_target_ge_4p10": finite & (v >= 4.10),
        "pred_high_overshoot_gt_4p35": finite & (pred > 4.35),
        "rest_I_zero": finite & (np.abs(current) <= 1e-10),
        "charge_I_positive": finite & (current > 1e-10),
        "discharge_I_negative": finite & (current < -1e-10),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _save_npz_like(source_arrays: Dict[str, np.ndarray], out_path: Path, corrected_pred: np.ndarray,
                   baseline_pred: np.ndarray, candidate_pred: np.ndarray, high_mask: np.ndarray,
                   variant_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = dict(source_arrays)
    n = len(corrected_pred)
    # Replace the common prediction keys. Preserve truth/current keys.
    if "voltage_exp_pred" in out:
        out["voltage_exp_pred_before_s1i"] = _as1(out["voltage_exp_pred"])[:n]
        out["voltage_exp_pred"] = corrected_pred
    else:
        out["voltage_exp_pred"] = corrected_pred
    if "phis_c_pred" in out:
        old = _as1(out["phis_c_pred"])
        if len(old) >= n:
            out["phis_c_pred_before_s1i"] = old[:n]
            out["phis_c_pred"] = corrected_pred
    out["voltage_exp_pred_baseline_d951"] = baseline_pred
    out["voltage_exp_pred_s1e_soft_source"] = candidate_pred
    out["s1i_high_mask"] = high_mask.astype(np.uint8)
    out["s1i_variant_name"] = np.asarray(variant_name)
    out["s1i_note"] = np.asarray("D12-S1I post-prediction high-region local fallback/budget wrapper; no retraining.")
    np.savez_compressed(out_path, **out)


def _copy_baseline_npz(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _build_variants(y: np.ndarray, base: np.ndarray, cand: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    high_mask = (y >= 4.10) | (cand > 4.35)
    delta = cand - base
    variants: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Best diagnostic from S1H: fully revert high/overshoot region to baseline.
    p = cand.copy()
    p[high_mask] = base[high_mask]
    variants["d12s1i_high_region_revert_to_baseline"] = (p, high_mask)

    # Slightly less conservative: keep at most +/-20 mV of S1E correction in high region.
    p = cand.copy()
    p[high_mask] = base[high_mask] + np.clip(delta[high_mask], -0.020, 0.020)
    variants["d12s1i_high_region_delta_budget_20mV"] = (p, high_mask)

    # Hybrid promoted diagnostic: hard cap plus 20 mV high budget.
    p = cand.copy()
    p = np.minimum(p, 4.35)
    high_mask_2 = (y >= 4.10) | (p > 4.35)
    p[high_mask_2] = base[high_mask_2] + np.clip((p - base)[high_mask_2], -0.020, 0.020)
    variants["d12s1i_clip_4p35_plus_high_budget_20mV"] = (p, high_mask_2)
    return variants


def _mean_delta(rows: List[Dict[str, float]], key: str) -> float:
    vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(np.nanmean(vals))


def _scorecard(runs_root: Path, output_dir: Path, baseline_mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    pred_files = sorted(runs_root.rglob("prediction.npz"))
    run_rows: List[Dict[str, Any]] = []
    seg_rows: List[Dict[str, Any]] = []

    for pred_path in pred_files:
        mode, profile = _parse_run(pred_path)
        try:
            arrays = _load_npz(pred_path)
            y, p, cur = _target_pred_current(arrays)
            m_all = _metrics(y, p)
            run_rows.append({"mode": mode, "profile": profile, "prediction_npz": str(pred_path), **m_all, "status": "metrics_ok"})
            for seg, mask in _segment_masks(y, p, cur).items():
                mm = _metrics(y[mask], p[mask]) if np.any(mask) else _metrics(np.asarray([]), np.asarray([]))
                seg_rows.append({"mode": mode, "profile": profile, "segment": seg, **mm})
        except Exception as exc:  # noqa: BLE001
            run_rows.append({"mode": mode, "profile": profile, "prediction_npz": str(pred_path), "status": "read_error", "error": repr(exc)})

    # Mode summary, one row per mode with commonly used means.
    modes = sorted({r["mode"] for r in run_rows})
    mode_rows: List[Dict[str, Any]] = []
    for mode in modes:
        rows = [r for r in run_rows if r.get("mode") == mode and r.get("status") == "metrics_ok"]
        if not rows:
            mode_rows.append({"mode": mode, "n": 0, "ok": 0})
            continue
        mode_rows.append({
            "mode": mode,
            "n": len(rows),
            "mean_MAE_V": float(np.nanmean([float(r.get("MAE_V", np.nan)) for r in rows])),
            "mean_RMSE_V": float(np.nanmean([float(r.get("RMSE_V", np.nan)) for r in rows])),
            "mean_corr": float(np.nanmean([float(r.get("corr", np.nan)) for r in rows])),
            "mean_bias_V": float(np.nanmean([float(r.get("bias_V", np.nan)) for r in rows])),
            "status": "metrics_ok",
        })

    by = {(r["mode"], r["profile"], r["segment"]): r for r in seg_rows}
    decisions: List[Dict[str, Any]] = []
    for mode in modes:
        if mode == baseline_mode:
            continue
        profiles = sorted({r["profile"] for r in run_rows if r.get("mode") == mode and r.get("status") == "metrics_ok"})
        prof_dec: List[Dict[str, float]] = []
        for prof in profiles:
            try:
                b_all = by[(baseline_mode, prof, "all")]
                c_all = by[(mode, prof, "all")]
                b_low = by[(baseline_mode, prof, "low_target")]
                c_low = by[(mode, prof, "low_target")]
                b_deep = by[(baseline_mode, prof, "low_target_le_2p75")]
                c_deep = by[(mode, prof, "low_target_le_2p75")]
                b_rest = by[(baseline_mode, prof, "rest_I_zero")]
                c_rest = by[(mode, prof, "rest_I_zero")]
                b_high = by[(baseline_mode, prof, "high_target_ge_4p10")]
                c_high = by[(mode, prof, "high_target_ge_4p10")]
                b_normal = by[(baseline_mode, prof, "normal_target_gt_3p20")]
                c_normal = by[(mode, prof, "normal_target_gt_3p20")]
            except KeyError:
                continue
            prof_dec.append({
                "delta_all_MAE_V": float(c_all["MAE_V"]) - float(b_all["MAE_V"]),
                "delta_low_target_MAE_V": float(c_low["MAE_V"]) - float(b_low["MAE_V"]),
                "delta_low_le_2p75_MAE_V": float(c_deep["MAE_V"]) - float(b_deep["MAE_V"]) if int(b_deep["n"]) > 0 else float("nan"),
                "delta_rest_MAE_V": float(c_rest["MAE_V"]) - float(b_rest["MAE_V"]) if int(b_rest["n"]) > 0 else 0.0,
                "delta_high_MAE_V": float(c_high["MAE_V"]) - float(b_high["MAE_V"]) if int(b_high["n"]) > 0 else 0.0,
                "delta_normal_MAE_V": float(c_normal["MAE_V"]) - float(b_normal["MAE_V"]) if int(b_normal["n"]) > 0 else 0.0,
                "delta_corr": float(c_all["corr"]) - float(b_all["corr"]),
                "deep_n": int(c_deep["n"]),
            })
        if not prof_dec:
            continue
        row: Dict[str, Any] = {"mode": mode, "profile_count": len(prof_dec)}
        for key in ["delta_all_MAE_V","delta_low_target_MAE_V","delta_low_le_2p75_MAE_V","delta_rest_MAE_V","delta_high_MAE_V","delta_normal_MAE_V","delta_corr"]:
            row[key] = _mean_delta(prof_dec, key)
        row["low_ok"] = bool(row["delta_low_target_MAE_V"] <= -args.min_low_improve_V)
        row["deep_ok"] = bool(np.isfinite(row["delta_low_le_2p75_MAE_V"]) and row["delta_low_le_2p75_MAE_V"] <= -args.min_low_improve_V)
        row["global_ok"] = bool(row["delta_all_MAE_V"] <= args.max_global_regress_V)
        row["corr_ok"] = bool(row["delta_corr"] >= -args.max_corr_drop)
        row["rest_ok"] = bool(row["delta_rest_MAE_V"] <= args.max_rest_regress_V)
        row["high_ok"] = bool(row["delta_high_MAE_V"] <= args.max_high_regress_V)
        row["normal_ok"] = bool(row["delta_normal_MAE_V"] <= args.max_normal_regress_V)
        row["promote_to_200ks"] = bool(row["low_ok"] and row["deep_ok"] and row["global_ok"] and row["corr_ok"] and row["rest_ok"] and row["high_ok"] and row["normal_ok"])
        decisions.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "D12_S1I_run_metrics.csv", run_rows)
    _write_csv(output_dir / "D12_S1I_segment_metrics.csv", seg_rows)
    _write_csv(output_dir / "D12_S1I_mode_summary.csv", mode_rows)
    _write_csv(output_dir / "D12_S1I_candidate_decisions.csv", decisions)
    summary = {
        "ok": True,
        "stage": "D12-S1I S1E-soft high-region local fallback/budget wrapper",
        "training": False,
        "source_runs_root": str(args.source_runs_root),
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "prediction_count": len(pred_files),
        "metrics_ok_count": sum(1 for r in run_rows if r.get("status") == "metrics_ok"),
        "read_error_count": sum(1 for r in run_rows if r.get("status") == "read_error"),
        "baseline_mode": baseline_mode,
        "promoted_candidates": [d["mode"] for d in decisions if d.get("promote_to_200ks")],
        "decision_rule": {
            "low_and_deep_MAE_improve_at_least_V": args.min_low_improve_V,
            "global_MAE_regress_no_more_than_V": args.max_global_regress_V,
            "corr_drop_no_more_than": args.max_corr_drop,
            "rest_high_regress_no_more_than_V": args.max_rest_regress_V,
            "normal_regress_no_more_than_V": args.max_normal_regress_V,
        },
        "mainline_overwritten": False,
        "note": "D12-S1I uses existing S1E-soft and baseline predictions; it applies high-region local fallback/budget and does not retrain.",
    }
    (output_dir / "D12_S1I_scorecard_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rec = [
        "# D12-S1I recommendation",
        "",
        "This stage is a formal post-prediction wrapper confirmation based on D12-S1H diagnostics. It does not retrain the PINN.",
        "",
        f"- source_runs_root: {args.source_runs_root}",
        f"- output_runs_root: {runs_root}",
        f"- promoted_candidates: {summary['promoted_candidates'] if summary['promoted_candidates'] else 'none'}",
        "",
        "## Interpretation",
    ]
    if summary["promoted_candidates"]:
        rec += [
            "At least one high-region local fallback/budget candidate satisfies low/deep/global/normal/high/rest/corr thresholds on 6-profile 40ks.",
            "Next step: do not claim final generalization yet. Confirm the selected candidate on a longer 200ks source window after creating the corresponding S1E-soft 200ks baseline/candidate predictions.",
        ]
    else:
        rec += [
            "No candidate satisfies all thresholds. Do not run 200ks. Inspect D12_S1I_profile_repair_diagnostics.csv and D12_S1I_segment_metrics.csv.",
        ]
    rec += [
        "",
        "## Output files",
        "- D12_S1I_scorecard_summary.json",
        "- D12_S1I_candidate_decisions.csv",
        "- D12_S1I_mode_summary.csv",
        "- D12_S1I_segment_metrics.csv",
        "- D12_S1I_profile_repair_diagnostics.csv",
    ]
    (output_dir / "D12_S1I_RECOMMENDATION.md").write_text("\n".join(rec), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_runs_root", required=True, help="Existing S1E 40ks or 200ks runs root with baseline and S1E-soft predictions")
    ap.add_argument("--output_runs_root", required=True, help="New runs root where S1I corrected prediction.npz files will be written")
    ap.add_argument("--scorecard_dir", required=True)
    ap.add_argument("--baseline_mode", default="baseline_d951")
    ap.add_argument("--candidate_mode", default="d12s1e_p2d_low_anchor_soft")
    ap.add_argument("--min_low_improve_V", type=float, default=0.020)
    ap.add_argument("--max_global_regress_V", type=float, default=0.005)
    ap.add_argument("--max_corr_drop", type=float, default=0.005)
    ap.add_argument("--max_rest_regress_V", type=float, default=0.020)
    ap.add_argument("--max_high_regress_V", type=float, default=0.020)
    ap.add_argument("--max_normal_regress_V", type=float, default=0.005)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    source = Path(args.source_runs_root)
    out_runs = Path(args.output_runs_root)
    scorecard = Path(args.scorecard_dir)
    if not source.exists():
        raise SystemExit(f"source_runs_root not found: {source}")
    if args.clean and out_runs.exists():
        shutil.rmtree(out_runs)
    if args.clean and scorecard.exists():
        shutil.rmtree(scorecard)
    out_runs.mkdir(parents=True, exist_ok=True)
    scorecard.mkdir(parents=True, exist_ok=True)

    pred_map: Dict[Tuple[str, str], Path] = {}
    for pred in sorted(source.rglob("prediction.npz")):
        mode, profile = _parse_run(pred)
        pred_map[(mode, profile)] = pred

    profiles = sorted({p for (m, p) in pred_map if m == args.baseline_mode} & {p for (m, p) in pred_map if m == args.candidate_mode})
    if not profiles:
        raise SystemExit(f"No overlapping profiles for baseline={args.baseline_mode} and candidate={args.candidate_mode} under {source}")

    diag_rows: List[Dict[str, Any]] = []
    for profile in profiles:
        base_path = pred_map[(args.baseline_mode, profile)]
        cand_path = pred_map[(args.candidate_mode, profile)]
        base_arrays = _load_npz(base_path)
        cand_arrays = _load_npz(cand_path)
        y, base_pred, cur = _target_pred_current(base_arrays)
        y2, cand_pred, cur2 = _target_pred_current(cand_arrays)
        n = min(len(y), len(y2), len(base_pred), len(cand_pred), len(cur), len(cur2))
        y = y[:n]
        base_pred = base_pred[:n]
        cand_pred = cand_pred[:n]
        cur = cur[:n]

        # Copy baseline into the formal S1I root.
        _copy_baseline_npz(base_path, out_runs / f"{args.baseline_mode}__{profile}" / "prediction.npz")

        variants = _build_variants(y, base_pred, cand_pred)
        for mode, (pred, mask) in variants.items():
            _save_npz_like(cand_arrays, out_runs / f"{mode}__{profile}" / "prediction.npz", pred[:n], base_pred, cand_pred, mask[:n], mode)
            high = y >= 4.10
            over = cand_pred > 4.35
            high_or_over = high | over
            diag_rows.append({
                "profile": profile,
                "mode": mode,
                "n": int(n),
                "n_high_target_ge_4p10": int(np.sum(high)),
                "n_s1e_pred_overshoot_gt_4p35": int(np.sum(over)),
                "n_repaired_points": int(np.sum(mask)),
                "baseline_high_MAE_V": _metrics(y[high], base_pred[high])["MAE_V"] if np.any(high) else float("nan"),
                "s1e_soft_high_MAE_V": _metrics(y[high], cand_pred[high])["MAE_V"] if np.any(high) else float("nan"),
                "s1i_high_MAE_V": _metrics(y[high], pred[high])["MAE_V"] if np.any(high) else float("nan"),
                "baseline_all_MAE_V": _metrics(y, base_pred)["MAE_V"],
                "s1e_soft_all_MAE_V": _metrics(y, cand_pred)["MAE_V"],
                "s1i_all_MAE_V": _metrics(y, pred)["MAE_V"],
                "s1e_soft_global_bias_V": _metrics(y, cand_pred)["bias_V"],
                "s1i_global_bias_V": _metrics(y, pred)["bias_V"],
                "s1e_minus_baseline_high_mean_V": float(np.nanmean((cand_pred - base_pred)[high_or_over])) if np.any(high_or_over) else float("nan"),
                "s1i_minus_baseline_high_mean_V": float(np.nanmean((pred - base_pred)[high_or_over])) if np.any(high_or_over) else float("nan"),
                "s1e_high_err_p50_V": _q((cand_pred - y)[high_or_over], 0.50),
                "s1i_high_err_p50_V": _q((pred - y)[high_or_over], 0.50),
            })

    _write_csv(scorecard / "D12_S1I_profile_repair_diagnostics.csv", diag_rows)
    args.source_runs_root = str(source)
    summary = _scorecard(out_runs, scorecard, args.baseline_mode, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
