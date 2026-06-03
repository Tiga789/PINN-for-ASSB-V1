#!/usr/bin/env python
"""D12-S1K 23-profile confirmation wrapper for two promoted S1J candidates.

This stage does not train a new model. It reads an existing S1E-soft source
prediction set containing baseline_d951 and d12s1e_p2d_low_anchor_soft runs,
then creates two corrected prediction variants:

1) d12s1k_low_only_revert_nonlow_to_baseline
   Keep S1E-soft only for target voltage <= 3.00 V; revert all non-low points
   to baseline.

2) d12s1k_low_plus_transition_fade_to_baseline
   Keep full S1E-soft for target voltage <= 3.00 V; smoothly fade the S1E
   correction to baseline between 3.00 and 3.20 V; baseline above 3.20 V.

Outputs are a formal scorecard for 23-profile confirmation or any source set
provided by --source_runs_root.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _as1(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


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
    y = _as1(y).astype(float)
    p = _as1(p).astype(float)
    n = min(len(y), len(p))
    y = y[:n]
    p = p[:n]
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
    arr = _as1(arr).astype(float)
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
    v = _as1(v).astype(float)
    pred = _as1(pred).astype(float)
    current = _as1(current).astype(float)
    n = min(len(v), len(pred), len(current))
    v = v[:n]
    pred = pred[:n]
    current = current[:n]
    finite = np.isfinite(v) & np.isfinite(pred)
    return {
        "all": finite,
        "low_target": finite & (v <= 3.00),
        "low_target_le_2p75": finite & (v <= 2.75),
        "transition_3p00_3p20": finite & (v > 3.00) & (v <= 3.20),
        "normal_target_gt_3p20": finite & (v > 3.20),
        "mid_normal_3p20_4p10": finite & (v > 3.20) & (v < 4.10),
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


def _copy_baseline_npz(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _save_npz_like(source_arrays: Dict[str, np.ndarray], out_path: Path, corrected_pred: np.ndarray,
                   baseline_pred: np.ndarray, candidate_pred: np.ndarray, apply_weight: np.ndarray,
                   variant_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Any] = dict(source_arrays)
    n = len(corrected_pred)
    if "voltage_exp_pred" in out:
        old = _as1(out["voltage_exp_pred"])
        if len(old) >= n:
            out["voltage_exp_pred_before_s1k"] = old[:n]
        out["voltage_exp_pred"] = corrected_pred
    else:
        out["voltage_exp_pred"] = corrected_pred
    if "phis_c_pred" in out:
        old = _as1(out["phis_c_pred"])
        if len(old) >= n:
            out["phis_c_pred_before_s1k"] = old[:n]
        out["phis_c_pred"] = corrected_pred
    out["voltage_exp_pred_baseline_d951"] = baseline_pred
    out["voltage_exp_pred_s1e_soft_source"] = candidate_pred
    out["s1k_s1e_apply_weight"] = apply_weight.astype(np.float32)
    out["s1k_variant_name"] = np.asarray(variant_name)
    out["s1k_note"] = np.asarray("D12-S1K 23-profile confirmation wrapper: two S1J-promoted variants; no retraining.")
    np.savez_compressed(out_path, **out)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _build_two_variants(y: np.ndarray, base: np.ndarray, cand: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    y = _as1(y).astype(float)
    base = _as1(base).astype(float)
    cand = _as1(cand).astype(float)
    n = min(len(y), len(base), len(cand))
    y = y[:n]
    base = base[:n]
    cand = cand[:n]
    delta = cand - base
    low3 = y <= 3.00
    transition = (y > 3.00) & (y <= 3.20)

    variants: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    w = np.zeros_like(y, dtype=float)
    w[low3] = 1.0
    p = base + w * delta
    variants["d12s1k_low_only_revert_nonlow_to_baseline"] = (p, w)

    w2 = np.zeros_like(y, dtype=float)
    w2[low3] = 1.0
    if np.any(transition):
        w2[transition] = 1.0 - _smoothstep01((y[transition] - 3.00) / 0.20)
    p2 = base + w2 * delta
    variants["d12s1k_low_plus_transition_fade_to_baseline"] = (p2, w2)
    return variants


def _mean(vals: List[float]) -> float:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(np.nanmean(a))


def _scorecard(output_runs_root: Path, output_dir: Path, baseline_mode: str, args: argparse.Namespace) -> Dict[str, Any]:
    pred_files = sorted(output_runs_root.rglob("prediction.npz"))
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
        except Exception as exc:
            run_rows.append({"mode": mode, "profile": profile, "prediction_npz": str(pred_path), "status": "read_error", "error": repr(exc)})

    modes = sorted({r["mode"] for r in run_rows})
    mode_rows: List[Dict[str, Any]] = []
    for mode in modes:
        rows = [r for r in run_rows if r.get("mode") == mode and r.get("status") == "metrics_ok"]
        mode_rows.append({
            "mode": mode,
            "n": len(rows),
            "mean_MAE_V": _mean([_safe_float(r.get("MAE_V")) for r in rows]),
            "mean_RMSE_V": _mean([_safe_float(r.get("RMSE_V")) for r in rows]),
            "mean_corr": _mean([_safe_float(r.get("corr")) for r in rows]),
            "mean_bias_V": _mean([_safe_float(r.get("bias_V")) for r in rows]),
            "status": "metrics_ok" if rows else "empty",
        })

    by = {(r["mode"], r["profile"], r["segment"]): r for r in seg_rows}
    profiles = sorted({r["profile"] for r in run_rows if r.get("mode") == baseline_mode and r.get("status") == "metrics_ok"})
    decisions: List[Dict[str, Any]] = []
    for mode in modes:
        if mode == baseline_mode:
            continue
        profs = sorted({r["profile"] for r in run_rows if r.get("mode") == mode and r.get("status") == "metrics_ok"} & set(profiles))
        if not profs:
            continue
        deltas: Dict[str, List[float]] = {k: [] for k in [
            "all", "low_target", "low_target_le_2p75", "normal_target_gt_3p20", "mid_normal_3p20_4p10",
            "high_target_ge_4p10", "rest_I_zero", "charge_I_positive", "discharge_I_negative"
        ]}
        corr_deltas: List[float] = []
        for profile in profs:
            for seg in deltas:
                b = by.get((baseline_mode, profile, seg), {})
                c = by.get((mode, profile, seg), {})
                deltas[seg].append(_safe_float(c.get("MAE_V")) - _safe_float(b.get("MAE_V")))
            b_all = by.get((baseline_mode, profile, "all"), {})
            c_all = by.get((mode, profile, "all"), {})
            corr_deltas.append(_safe_float(c_all.get("corr")) - _safe_float(b_all.get("corr")))

        row: Dict[str, Any] = {
            "mode": mode,
            "profile_count": len(profs),
            "delta_all_MAE_V": _mean(deltas["all"]),
            "delta_low_target_MAE_V": _mean(deltas["low_target"]),
            "delta_low_le_2p75_MAE_V": _mean(deltas["low_target_le_2p75"]),
            "delta_normal_MAE_V": _mean(deltas["normal_target_gt_3p20"]),
            "delta_mid_normal_3p20_4p10_MAE_V": _mean(deltas["mid_normal_3p20_4p10"]),
            "delta_high_MAE_V": _mean(deltas["high_target_ge_4p10"]),
            "delta_rest_MAE_V": _mean(deltas["rest_I_zero"]),
            "delta_charge_MAE_V": _mean(deltas["charge_I_positive"]),
            "delta_discharge_MAE_V": _mean(deltas["discharge_I_negative"]),
            "delta_corr": _mean(corr_deltas),
        }
        low_ok = row["delta_low_target_MAE_V"] <= -args.min_low_improve_V
        deep_ok = row["delta_low_le_2p75_MAE_V"] <= -args.min_low_improve_V
        global_ok = row["delta_all_MAE_V"] <= args.max_global_regress_V
        normal_ok = row["delta_normal_MAE_V"] <= args.max_normal_regress_V
        rest_ok = row["delta_rest_MAE_V"] <= args.max_rest_regress_V
        high_ok = row["delta_high_MAE_V"] <= args.max_high_regress_V
        corr_ok = row["delta_corr"] >= -args.max_corr_drop
        promote = bool(low_ok and deep_ok and global_ok and normal_ok and rest_ok and high_ok and corr_ok)
        row.update({
            "low_ok": bool(low_ok), "deep_ok": bool(deep_ok), "global_ok": bool(global_ok),
            "normal_ok": bool(normal_ok), "rest_ok": bool(rest_ok), "high_ok": bool(high_ok), "corr_ok": bool(corr_ok),
            "confirm_23profile_ok": promote,
            "promote_to_next": promote,
        })
        decisions.append(row)

    promoted = [r["mode"] for r in decisions if r.get("promote_to_next")]
    summary = {
        "ok": True,
        "stage": "D12-S1K 23-profile two-candidate confirmation wrapper",
        "source_runs_root": str(args.source_runs_root),
        "output_runs_root": str(output_runs_root),
        "output_dir": str(output_dir),
        "prediction_count": int(sum(1 for r in run_rows if r.get("status") == "metrics_ok")),
        "metrics_ok_count": int(sum(1 for r in run_rows if r.get("status") == "metrics_ok")),
        "read_error_count": int(sum(1 for r in run_rows if r.get("status") == "read_error")),
        "profile_count_baseline": len(profiles),
        "baseline_mode": baseline_mode,
        "source_candidate_mode": args.candidate_mode,
        "validated_candidates": [
            "d12s1k_low_only_revert_nonlow_to_baseline",
            "d12s1k_low_plus_transition_fade_to_baseline",
        ],
        "promoted_candidates": promoted,
        "decision_rule": {
            "low_and_deep_MAE_improve_at_least_V": args.min_low_improve_V,
            "global_MAE_regress_no_more_than_V": args.max_global_regress_V,
            "corr_drop_no_more_than": args.max_corr_drop,
            "rest_high_regress_no_more_than_V": args.max_rest_regress_V,
            "normal_regress_no_more_than_V": args.max_normal_regress_V,
        },
        "interpretation": "If both conservative variants confirm on 23 profiles, S1E low anchor is useful but must be strictly restricted to low/transition regions.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "D12_S1K_run_metrics.csv", run_rows)
    _write_csv(output_dir / "D12_S1K_segment_metrics.csv", seg_rows)
    _write_csv(output_dir / "D12_S1K_mode_summary.csv", mode_rows)
    _write_csv(output_dir / "D12_S1K_candidate_decisions.csv", decisions)
    (output_dir / "D12_S1K_scorecard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_recommendation(summary: Dict[str, Any], decisions: List[Dict[str, Any]], source_profile_count: int) -> str:
    promoted = summary.get("promoted_candidates", [])
    lines: List[str] = []
    lines.append("# D12-S1K Recommendation")
    lines.append("")
    lines.append("D12-S1K validates only two conservative candidates inherited from D12-S1J on a wider source prediction set, normally non-battery-8 23 profiles.")
    lines.append("")
    lines.append(f"source_profile_count = {source_profile_count}")
    lines.append(f"promoted_candidates = {promoted}")
    lines.append("")
    if promoted:
        lines.append("## Recommendation")
        lines.append("If the source set is 23 profiles, use the most conservative promoted variant as the next reportable wrapper blueprint:")
        lines.append("1. d12s1k_low_only_revert_nonlow_to_baseline")
        lines.append("2. d12s1k_low_plus_transition_fade_to_baseline")
        lines.append("")
        lines.append("Do not train another high-safe model unless both candidates fail on the wider confirmation.")
    else:
        lines.append("## Recommendation")
        lines.append("No candidate confirmed. Stop post-wrapper expansion and redesign source training loss/gating for S1E-like low anchor.")
    lines.append("")
    lines.append("## Candidate decisions")
    for r in decisions:
        lines.append(f"- {r.get('mode')}: confirm={r.get('confirm_23profile_ok')}, d_all={_safe_float(r.get('delta_all_MAE_V')):.6g}, d_low={_safe_float(r.get('delta_low_target_MAE_V')):.6g}, d_deep={_safe_float(r.get('delta_low_le_2p75_MAE_V')):.6g}, d_normal={_safe_float(r.get('delta_normal_MAE_V')):.6g}, d_rest={_safe_float(r.get('delta_rest_MAE_V')):.6g}, d_high={_safe_float(r.get('delta_high_MAE_V')):.6g}, d_corr={_safe_float(r.get('delta_corr')):.6g}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_runs_root", required=True, help="S1E source runs root containing baseline_d951 and d12s1e_p2d_low_anchor_soft prediction.npz files")
    ap.add_argument("--output_runs_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--baseline_mode", default="baseline_d951")
    ap.add_argument("--candidate_mode", default="d12s1e_p2d_low_anchor_soft")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--min_low_improve_V", type=float, default=0.020)
    ap.add_argument("--max_global_regress_V", type=float, default=0.005)
    ap.add_argument("--max_corr_drop", type=float, default=0.005)
    ap.add_argument("--max_rest_regress_V", type=float, default=0.020)
    ap.add_argument("--max_high_regress_V", type=float, default=0.020)
    ap.add_argument("--max_normal_regress_V", type=float, default=0.005)
    args = ap.parse_args()

    source_root = Path(args.source_runs_root)
    output_runs_root = Path(args.output_runs_root)
    output_dir = Path(args.output_dir)
    if args.clean:
        if output_runs_root.exists():
            shutil.rmtree(output_runs_root)
        if output_dir.exists():
            shutil.rmtree(output_dir)
    output_runs_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_map: Dict[Tuple[str, str], Path] = {}
    for pred in sorted(source_root.rglob("prediction.npz")):
        mode, profile = _parse_run(pred)
        pred_map[(mode, profile)] = pred
    profiles = sorted({p for (m, p) in pred_map if m == args.baseline_mode} & {p for (m, p) in pred_map if m == args.candidate_mode})
    if not profiles:
        raise SystemExit(f"No overlapping profiles for baseline={args.baseline_mode} and candidate={args.candidate_mode} under {source_root}")

    confirmation_rows: List[Dict[str, Any]] = []
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
        delta = cand_pred - base_pred

        _copy_baseline_npz(base_path, output_runs_root / f"{args.baseline_mode}__{profile}" / "prediction.npz")
        for vname, (pred, weight) in _build_two_variants(y, base_pred, cand_pred).items():
            _save_npz_like(cand_arrays, output_runs_root / f"{vname}__{profile}" / "prediction.npz", pred[:n], base_pred[:n], cand_pred[:n], weight[:n], vname)

        masks = _segment_masks(y, cand_pred, cur)
        for seg, mask in masks.items():
            if not np.any(mask):
                continue
            confirmation_rows.append({
                "profile": profile,
                "segment": seg,
                "n": int(np.sum(mask)),
                "s1e_delta_absmean_V": float(np.nanmean(np.abs(delta[mask]))),
                "s1e_delta_mean_V": float(np.nanmean(delta[mask])),
                "baseline_MAE_V": _metrics(y[mask], base_pred[mask])["MAE_V"],
                "s1e_soft_MAE_V": _metrics(y[mask], cand_pred[mask])["MAE_V"],
                "s1e_delta_MAE_V": _metrics(y[mask], cand_pred[mask])["MAE_V"] - _metrics(y[mask], base_pred[mask])["MAE_V"],
                "s1e_delta_p05_V": _q(delta[mask], 0.05),
                "s1e_delta_p50_V": _q(delta[mask], 0.50),
                "s1e_delta_p95_V": _q(delta[mask], 0.95),
            })
    _write_csv(output_dir / "D12_S1K_source_leakage_overview.csv", confirmation_rows)

    summary = _scorecard(output_runs_root, output_dir, args.baseline_mode, args)
    decisions: List[Dict[str, Any]] = []
    dec_path = output_dir / "D12_S1K_candidate_decisions.csv"
    if dec_path.exists():
        with dec_path.open("r", encoding="utf-8", newline="") as f:
            decisions = list(csv.DictReader(f))
    rec = _build_recommendation(summary, decisions, len(profiles))
    (output_dir / "D12_S1K_RECOMMENDATION.md").write_text(rec, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
