#!/usr/bin/env python
"""Collect D11-S4 scorecard from prediction.npz files.

This script does not launch training. It reads completed prediction files and
summarizes global and segment-level voltage metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PRED_KEYS = ["voltage_exp_pred", "voltage_pred", "pred_voltage", "phis_c_pred", "y_pred", "v_pred"]
TRUE_KEYS = ["voltage_exp", "voltage_true", "target_voltage", "voltage_target", "phis_c_true", "y_true", "v_true"]
TIME_KEYS = ["t_global_s", "time_s", "t_s", "t"]
CURRENT_KEYS = ["I_profile", "current_A", "I", "current"]


def _to_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr[:, 0]
    return np.asarray(arr, dtype=float).reshape(-1)


def pick_key(npz, keys: Iterable[str]) -> Optional[str]:
    names = set(npz.files)
    for k in keys:
        if k in names:
            return k
    # suffix-based fallback
    for k in npz.files:
        lk = k.lower()
        if any(x.lower() in lk for x in keys):
            return k
    return None


def finite_pair(y_true: np.ndarray, y_pred: np.ndarray, *extra: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    n = min(len(y_true), len(y_pred), *[len(e) for e in extra] if extra else [len(y_true)])
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    extras = [e[:n] for e in extra]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    for e in extras:
        mask &= np.isfinite(e)
    return y_true[mask], y_pred[mask], [e[mask] for e in extras]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred, _ = finite_pair(y_true, y_pred)
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "MAE_V": math.nan, "RMSE_V": math.nan, "corr": math.nan, "bias_V": math.nan, "target_min_V": math.nan, "target_max_V": math.nan, "pred_min_V": math.nan, "pred_max_V": math.nan}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if n >= 2 and float(np.std(y_true)) > 1e-12 and float(np.std(y_pred)) > 1e-12:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = math.nan
    return {
        "n": n,
        "MAE_V": mae,
        "RMSE_V": rmse,
        "corr": corr,
        "bias_V": bias,
        "target_min_V": float(np.min(y_true)),
        "target_max_V": float(np.max(y_true)),
        "pred_min_V": float(np.min(y_pred)),
        "pred_max_V": float(np.max(y_pred)),
    }


def infer_protocol(profile: str) -> str:
    if "R2.5" in profile or "R25" in profile:
        return "R2.5"
    if "R3" in profile:
        return "R3"
    if "2C" in profile:
        return "2C"
    return "unknown"


def load_prediction(path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        npz = np.load(path, allow_pickle=True)
    except Exception:
        return None
    pred_key = pick_key(npz, PRED_KEYS)
    true_key = pick_key(npz, TRUE_KEYS)
    if pred_key is None or true_key is None:
        return None
    out = {
        "pred": _to_1d(npz[pred_key]),
        "true": _to_1d(npz[true_key]),
    }
    t_key = pick_key(npz, TIME_KEYS)
    i_key = pick_key(npz, CURRENT_KEYS)
    if t_key is not None:
        out["time"] = _to_1d(npz[t_key])
    else:
        out["time"] = np.arange(len(out["true"]), dtype=float)
    if i_key is not None:
        out["current"] = _to_1d(npz[i_key])
    else:
        out["current"] = np.full(len(out["true"]), np.nan)
    return out


def segment_masks(y_true: np.ndarray, y_pred: np.ndarray, time: np.ndarray, current: np.ndarray) -> Dict[str, np.ndarray]:
    n = min(len(y_true), len(y_pred), len(time), len(current))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    time = time[:n]
    current = current[:n]
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.isfinite(current).any():
        finite &= np.isfinite(current)
    else:
        current = np.zeros(n)
    q1, q2 = np.nanquantile(time[np.isfinite(time)], [1/3, 2/3]) if np.isfinite(time).any() else (n/3, 2*n/3)
    masks = {
        "all": finite,
        "charge_I_positive": finite & (current > 1e-8),
        "discharge_I_negative": finite & (current < -1e-8),
        "rest_I_zero": finite & (np.abs(current) <= 1e-8),
        "low_target": finite & (y_true <= 2.90),
        "low_target_le_2p75": finite & (y_true <= 2.75),
        "high_target_ge_4p10": finite & (y_true >= 4.10),
        "pred_high_overshoot_gt_4p35": finite & (y_pred > 4.35),
        "early_time_third": finite & (time <= q1),
        "middle_time_third": finite & (time > q1) & (time <= q2),
        "late_time_third": finite & (time > q2),
    }
    return masks


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k); seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_mean(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, vals in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n_rows"] = len(vals)
        for m in ["MAE_V", "RMSE_V", "corr", "bias_V"]:
            arr = np.array([float(v.get(m, np.nan)) for v in vals], dtype=float)
            if np.isfinite(arr).any():
                rec[f"mean_{m}"] = float(np.nanmean(arr))
            else:
                rec[f"mean_{m}"] = math.nan
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    pred_root = cache_root / "xjtu_batch134_d11_s4_lowtail_correction_smoke"
    out_dir = cache_root / "xjtu_batch134_d11_s4_lowtail_correction_smoke_scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_root.rglob("prediction.npz"))
    run_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []

    for p in pred_paths:
        rel = p.relative_to(pred_root)
        parts = rel.parts
        mode = parts[0] if len(parts) >= 3 else "unknown"
        profile = parts[1] if len(parts) >= 3 else p.parent.name
        protocol = infer_protocol(profile)
        data = load_prediction(p)
        status = "strict_completed_metrics_ok" if data is not None else "read_error_or_missing_keys"
        if data is None:
            run_rows.append({"mode": mode, "profile": profile, "protocol": protocol, "prediction_npz": str(p), "status": status})
            continue
        y_true, y_pred, extras = finite_pair(data["true"], data["pred"], data["time"], data["current"])
        time, current = extras
        m = metrics(y_true, y_pred)
        rr = {"mode": mode, "profile": profile, "protocol": protocol, "prediction_npz": str(p), "status": status}
        rr.update(m)
        run_rows.append(rr)
        masks = segment_masks(y_true, y_pred, time, current)
        for seg, mask in masks.items():
            if int(np.sum(mask)) == 0:
                sm = metrics(np.array([]), np.array([]))
            else:
                sm = metrics(y_true[mask], y_pred[mask])
            sr = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg, "prediction_npz": str(p), "status": status}
            sr.update(sm)
            seg_rows.append(sr)

    write_csv(out_dir / "D11_S4_run_metrics.csv", run_rows)
    write_csv(out_dir / "D11_S4_segment_metrics.csv", seg_rows)

    mode_summary = group_mean([r for r in run_rows if r.get("status") == "strict_completed_metrics_ok"], ["mode"])
    mode_protocol_summary = group_mean([r for r in run_rows if r.get("status") == "strict_completed_metrics_ok"], ["mode", "protocol"])
    mode_segment_summary = group_mean([r for r in seg_rows if r.get("status") == "strict_completed_metrics_ok" and r.get("segment") != "all"], ["mode", "segment"])
    mode_protocol_segment_summary = group_mean([r for r in seg_rows if r.get("status") == "strict_completed_metrics_ok" and r.get("segment") != "all"], ["mode", "protocol", "segment"])

    write_csv(out_dir / "D11_S4_mode_summary.csv", mode_summary)
    write_csv(out_dir / "D11_S4_mode_protocol_summary.csv", mode_protocol_summary)
    write_csv(out_dir / "D11_S4_mode_segment_summary.csv", mode_segment_summary)
    write_csv(out_dir / "D11_S4_mode_protocol_segment_summary.csv", mode_protocol_segment_summary)

    worst_segments = sorted([r for r in seg_rows if r.get("status") == "strict_completed_metrics_ok" and np.isfinite(float(r.get("MAE_V", np.nan)))], key=lambda r: float(r.get("MAE_V", 0)), reverse=True)[:30]
    write_csv(out_dir / "D11_S4_worst_segments.csv", worst_segments)

    def find_mode(mode: str) -> Optional[Dict[str, object]]:
        for r in mode_summary:
            if r.get("mode") == mode:
                return r
        return None

    baseline = find_mode("baseline_d951")
    comparisons = []
    for candidate in ["lowtail_mild", "lowtail_strong_safe"]:
        cand = find_mode(candidate)
        if baseline and cand:
            comparisons.append({
                "candidate": candidate,
                "baseline": "baseline_d951",
                "candidate_mean_MAE_V": cand.get("mean_MAE_V"),
                "baseline_mean_MAE_V": baseline.get("mean_MAE_V"),
                "candidate_minus_baseline_MAE_V": float(cand.get("mean_MAE_V", math.nan)) - float(baseline.get("mean_MAE_V", math.nan)),
                "candidate_mean_corr": cand.get("mean_corr"),
                "baseline_mean_corr": baseline.get("mean_corr"),
                "candidate_minus_baseline_corr": float(cand.get("mean_corr", math.nan)) - float(baseline.get("mean_corr", math.nan)),
            })
    write_csv(out_dir / "D11_S4_lowtail_comparison.csv", comparisons)

    status_counts: Dict[str, int] = {}
    for r in run_rows:
        status_counts[str(r.get("status"))] = status_counts.get(str(r.get("status")), 0) + 1

    expected_runs = 18
    completed = status_counts.get("strict_completed_metrics_ok", 0)
    verdict = "d11_s4_all_runs_completed_metrics_ok" if completed == expected_runs else "d11_s4_incomplete_or_read_errors"

    recommendation_lines = [
        "# D11-S4 low-voltage tail correction smoke recommendation",
        "",
        f"- Verdict: `{verdict}`",
        f"- Run count: `{len(run_rows)}`",
        f"- Completed metrics OK: `{completed}`",
        "",
        "## Mode summary",
        "",
        "| mode | n_rows | mean_MAE_V | mean_corr | mean_bias_V |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in mode_summary:
        recommendation_lines.append(f"| {r.get('mode')} | {r.get('n_rows')} | {r.get('mean_MAE_V')} | {r.get('mean_corr')} | {r.get('mean_bias_V')} |")
    recommendation_lines += ["", "## Low-tail candidate comparison", ""]
    if comparisons:
        recommendation_lines += ["| candidate | ΔMAE vs baseline | Δcorr vs baseline |", "|---|---:|---:|"]
        for c in comparisons:
            recommendation_lines.append(f"| {c['candidate']} | {c['candidate_minus_baseline_MAE_V']} | {c['candidate_minus_baseline_corr']} |")
    else:
        recommendation_lines.append("Comparison unavailable because baseline or candidate mode is missing.")
    recommendation_lines += [
        "",
        "## Decision rule",
        "",
        "Promote a candidate only if global MAE does not increase, global corr does not drop materially, and low-target segments improve.",
        "If not, keep D9.6/D9.5.1 as mainline and treat D11-S4 as an ablation audit.",
        "",
        "## Next action",
        "",
        "Open `D11_S4_mode_segment_summary.csv` and inspect `low_target` / `low_target_le_2p75`. If a low-tail mode improves tails without damaging all-segment metrics, run a 6-profile 200ks confirmation before any 23-profile expansion.",
    ]
    (out_dir / "D11_S4_RECOMMENDATION.md").write_text("\n".join(recommendation_lines), encoding="utf-8")

    summary = {
        "ok": True,
        "stage": "D11-S4 low-voltage tail correction smoke scorecard from predictions",
        "prediction_root": str(pred_root),
        "out_dir": str(out_dir),
        "run_count": len(run_rows),
        "expected_run_count": expected_runs,
        "counts": status_counts,
        "mode_summary": mode_summary,
        "comparisons": comparisons,
        "verdict": verdict,
        "note": "No training launched by this collector. Metrics computed from existing prediction.npz files.",
    }
    (out_dir / "D11_S4_scorecard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
