#!/usr/bin/env python
"""Collect D11-S5C low-target amplitude repair scorecard from predictions.

No training is launched.  This script reads D11-S5C prediction.npz files and
checks whether the amplitude-repair candidates actually improve low_target and
low_target_le_2p75 versus baseline while preserving global trend quality.
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
COMPONENT_KEYS = {
    "low_gate": ["voltage_low_gate", "low_gate", "low_voltage_gate"],
    "low_tail_correction": ["voltage_low_tail_correction", "low_tail_correction", "v_low_tail_scaled"],
    "event_correction": ["voltage_event_correction", "event_correction", "v_event_scaled"],
    "softsign_correction": ["voltage_softsign_correction", "voltage_correction", "v_correction"],
    "ocv_baseline": ["voltage_ocv_baseline", "v_ocv"],
    "direct_head": ["voltage_direct_head", "v_direct"],
    "base_branch": ["voltage_base_branch", "v_base"],
    "event_branch_delta": ["voltage_event_branch_delta", "v_event_total"],
}
SEGMENT_ORDER = [
    "all", "charge_I_positive", "discharge_I_negative", "rest_I_zero",
    "low_target", "low_target_le_2p75", "high_target_ge_4p10",
    "pred_high_overshoot_gt_4p35", "early_time_third", "middle_time_third", "late_time_third",
]
BASELINE_MODE = "baseline_d951"


def _to_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return np.asarray(arr, dtype=float).reshape(-1)


def pick_key(npz, keys: Iterable[str]) -> Optional[str]:
    names = set(npz.files)
    for k in keys:
        if k in names:
            return k
    lower_map = {k.lower(): k for k in npz.files}
    for k in keys:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]; y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "MAE_V": math.nan, "RMSE_V": math.nan, "corr": math.nan, "bias_V": math.nan,
                "pred_over_frac": math.nan, "pred_under_frac": math.nan,
                "target_min_V": math.nan, "target_max_V": math.nan, "pred_min_V": math.nan, "pred_max_V": math.nan}
    err = y_pred - y_true
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if n >= 2 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12 else math.nan
    return {
        "n": n,
        "MAE_V": float(np.mean(np.abs(err))),
        "RMSE_V": float(np.sqrt(np.mean(err ** 2))),
        "corr": corr,
        "bias_V": float(np.mean(err)),
        "pred_over_frac": float(np.mean(err > 0)),
        "pred_under_frac": float(np.mean(err < 0)),
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
    out: Dict[str, np.ndarray] = {"pred": _to_1d(npz[pred_key]), "true": _to_1d(npz[true_key])}
    t_key = pick_key(npz, TIME_KEYS)
    i_key = pick_key(npz, CURRENT_KEYS)
    out["time"] = _to_1d(npz[t_key]) if t_key else np.arange(len(out["true"]), dtype=float)
    out["current"] = _to_1d(npz[i_key]) if i_key else np.full(len(out["true"]), np.nan)
    for cname, keys in COMPONENT_KEYS.items():
        k = pick_key(npz, keys)
        if k is not None:
            out[cname] = _to_1d(npz[k])
    return out


def segment_masks(y_true: np.ndarray, y_pred: np.ndarray, time: np.ndarray, current: np.ndarray) -> Dict[str, np.ndarray]:
    n = min(len(y_true), len(y_pred), len(time), len(current))
    y_true = y_true[:n]; y_pred = y_pred[:n]; time = time[:n]; current = current[:n]
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.isfinite(current).any():
        finite &= np.isfinite(current)
    else:
        current = np.zeros(n)
    if np.isfinite(time).any():
        q1, q2 = np.nanquantile(time[np.isfinite(time)], [1/3, 2/3])
    else:
        q1, q2 = n / 3.0, 2.0 * n / 3.0
    return {
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


def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    n = min(len(arr), len(mask))
    if n == 0:
        return math.nan
    a = arr[:n]; m = mask[:n] & np.isfinite(a)
    return float(np.mean(a[m])) if np.any(m) else math.nan


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k); seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_mean(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k, "") for k in group_keys), []).append(r)
    out = []
    for key, vals in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n_rows"] = len(vals)
        for col in ["MAE_V", "RMSE_V", "corr", "bias_V", "n", "pred_over_frac", "pred_under_frac"]:
            arr = np.array([float(v.get(col, np.nan)) for v in vals], dtype=float)
            rec[f"mean_{col}"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan
        for c in COMPONENT_KEYS.keys():
            for suffix in ["mean", "negative_frac", "positive_frac"]:
                col = f"{c}_{suffix}"
                arr = np.array([float(v.get(col, np.nan)) for v in vals], dtype=float)
                if np.isfinite(arr).any():
                    rec[f"mean_{col}"] = float(np.nanmean(arr))
        out.append(rec)
    return out


def extract_mode_profile(path: Path) -> Tuple[str, str]:
    profile = path.parent.name
    mode = path.parent.parent.name if path.parent.parent.name else "unknown"
    return mode, profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    pred_root = cache_root / "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair"
    out_dir = cache_root / "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_root.rglob("prediction.npz"))
    run_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []
    comp_rows: List[Dict[str, object]] = []

    for p in pred_paths:
        mode, profile = extract_mode_profile(p)
        protocol = infer_protocol(profile)
        data = load_prediction(p)
        if data is None:
            run_rows.append({"mode": mode, "profile": profile, "protocol": protocol, "status": "prediction_load_failed", "prediction_path": str(p)})
            continue
        y_true = data["true"]; y_pred = data["pred"]; time = data["time"]; current = data["current"]
        n = min(len(y_true), len(y_pred), len(time), len(current))
        y_true = y_true[:n]; y_pred = y_pred[:n]; time = time[:n]; current = current[:n]
        run_m = metrics(y_true, y_pred)
        rr = {"mode": mode, "profile": profile, "protocol": protocol, "status": "strict_completed_metrics_ok", "prediction_path": str(p)}
        rr.update(run_m)
        run_rows.append(rr)
        masks = segment_masks(y_true, y_pred, time, current)
        for seg in SEGMENT_ORDER:
            mask = masks.get(seg)
            if mask is None:
                continue
            m = metrics(y_true[:len(mask)][mask], y_pred[:len(mask)][mask])
            rec = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg}
            rec.update(m)
            seg_rows.append(rec)
            comp = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg, "n": m["n"]}
            for cname in COMPONENT_KEYS.keys():
                if cname in data:
                    arr = data[cname]
                    comp[f"{cname}_mean"] = safe_mean(arr, mask)
                    val = arr[:min(len(arr), len(mask))]
                    mk = mask[:len(val)] & np.isfinite(val)
                    if np.any(mk):
                        comp[f"{cname}_negative_frac"] = float(np.mean(val[mk] < 0))
                        comp[f"{cname}_positive_frac"] = float(np.mean(val[mk] > 0))
            comp_rows.append(comp)

    mode_summary = group_mean([r for r in run_rows if r.get("status") == "strict_completed_metrics_ok"], ["mode"])
    mode_seg_summary = group_mean(seg_rows, ["mode", "segment"])
    protocol_summary = group_mean([r for r in run_rows if r.get("status") == "strict_completed_metrics_ok"], ["protocol"])
    comp_summary = group_mean(comp_rows, ["mode", "segment"])

    write_csv(out_dir / "D11_S5C_run_metrics.csv", run_rows)
    write_csv(out_dir / "D11_S5C_segment_metrics.csv", seg_rows)
    write_csv(out_dir / "D11_S5C_mode_summary.csv", mode_summary)
    write_csv(out_dir / "D11_S5C_mode_segment_summary.csv", mode_seg_summary)
    write_csv(out_dir / "D11_S5C_protocol_summary.csv", protocol_summary)
    write_csv(out_dir / "D11_S5C_component_summary.csv", comp_summary)

    # Compare candidates against baseline by segment.
    by_mode_seg = {(r["mode"], r["segment"]): r for r in mode_seg_summary}
    compare_rows = []
    candidate_modes = [r["mode"] for r in mode_summary if r.get("mode") != BASELINE_MODE]
    for mode in sorted(candidate_modes):
        for seg in ["all", "low_target", "low_target_le_2p75", "rest_I_zero", "high_target_ge_4p10", "pred_high_overshoot_gt_4p35"]:
            b = by_mode_seg.get((BASELINE_MODE, seg), {})
            c = by_mode_seg.get((mode, seg), {})
            compare_rows.append({
                "candidate": mode,
                "segment": seg,
                "candidate_mean_MAE_V": c.get("mean_MAE_V", math.nan),
                "baseline_mean_MAE_V": b.get("mean_MAE_V", math.nan),
                "candidate_minus_baseline_MAE_V": float(c.get("mean_MAE_V", math.nan)) - float(b.get("mean_MAE_V", math.nan)) if c and b else math.nan,
                "candidate_mean_corr": c.get("mean_corr", math.nan),
                "baseline_mean_corr": b.get("mean_corr", math.nan),
                "candidate_minus_baseline_corr": float(c.get("mean_corr", math.nan)) - float(b.get("mean_corr", math.nan)) if c and b else math.nan,
                "candidate_mean_bias_V": c.get("mean_bias_V", math.nan),
                "baseline_mean_bias_V": b.get("mean_bias_V", math.nan),
                "candidate_minus_baseline_bias_V": float(c.get("mean_bias_V", math.nan)) - float(b.get("mean_bias_V", math.nan)) if c and b else math.nan,
            })
    write_csv(out_dir / "D11_S5C_global_vs_lowtarget_tradeoff.csv", compare_rows)

    ok_runs = sum(1 for r in run_rows if r.get("status") == "strict_completed_metrics_ok")
    expected = 24

    # Candidate promotion logic.
    decisions = []
    for mode in sorted(candidate_modes):
        rows = [r for r in compare_rows if r["candidate"] == mode]
        d = {r["segment"]: r for r in rows}
        low_ok = (
            float(d.get("low_target", {}).get("candidate_minus_baseline_MAE_V", math.inf)) < -0.02 and
            float(d.get("low_target_le_2p75", {}).get("candidate_minus_baseline_MAE_V", math.inf)) < -0.02
        )
        global_ok = float(d.get("all", {}).get("candidate_minus_baseline_MAE_V", math.inf)) < 0.005
        corr_ok = float(d.get("all", {}).get("candidate_minus_baseline_corr", -math.inf)) > -0.01
        rest_ok = float(d.get("rest_I_zero", {}).get("candidate_minus_baseline_MAE_V", math.inf)) < 0.03
        high_overshoot = d.get("pred_high_overshoot_gt_4p35", {}).get("candidate_mean_MAE_V", math.nan)
        high_ok = (not np.isfinite(float(high_overshoot))) or float(high_overshoot) < 0.20
        promote = bool(low_ok and global_ok and corr_ok and rest_ok and high_ok)
        decisions.append({"mode": mode, "low_ok": low_ok, "global_ok": global_ok, "corr_ok": corr_ok, "rest_ok": rest_ok, "high_ok": high_ok, "promote_candidate": promote})

    promoted = [d["mode"] for d in decisions if d["promote_candidate"]]
    next_action = "run_6profile_200ks_confirmation_for_promoted_candidate" if promoted else "do_not_expand_redesign_lowtarget_amplitude_or_model_capacity"
    verdict = "d11_s5c_all_runs_completed_metrics_ok" if ok_runs == expected else "d11_s5c_incomplete_or_failed_runs"
    summary = {
        "ok": ok_runs == expected,
        "stage": "D11-S5C low-target correction amplitude repair scorecard from predictions",
        "prediction_root": str(pred_root),
        "out_dir": str(out_dir),
        "run_count": len(run_rows),
        "expected_run_count": expected,
        "counts": {"strict_completed_metrics_ok": ok_runs},
        "mode_summary": mode_summary,
        "candidate_decisions": decisions,
        "promoted_candidates": promoted,
        "verdict": verdict,
        "next_action": next_action,
        "promotion_rule": "Candidate must reduce low_target and low_target_le_2p75 MAE by at least 20 mV versus baseline while preserving all/rest/high-tail metrics.",
    }
    (out_dir / "D11_S5C_scorecard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# D11-S5C low-target correction amplitude repair recommendation",
        "",
        f"- Verdict: `{verdict}`",
        f"- Run count: `{len(run_rows)}`",
        f"- Completed metrics OK: `{ok_runs}`",
        "",
        "## Mode summary",
        "",
        "| mode | n_rows | mean_MAE_V | mean_corr | mean_bias_V |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in mode_summary:
        lines.append(f"| {r.get('mode')} | {r.get('n_rows')} | {r.get('mean_MAE_V')} | {r.get('mean_corr')} | {r.get('mean_bias_V')} |")
    lines += ["", "## Candidate decisions", "", "| mode | low_ok | global_ok | corr_ok | rest_ok | high_ok | promote |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in decisions:
        lines.append(f"| {d['mode']} | {d['low_ok']} | {d['global_ok']} | {d['corr_ok']} | {d['rest_ok']} | {d['high_ok']} | {d['promote_candidate']} |")
    lines += ["", "## Decision", ""]
    if promoted:
        lines.append("At least one candidate passed the low-target criterion. Run a 6-profile 200 ks confirmation only for the promoted candidate(s), not 23-profile expansion.")
    else:
        lines.append("No candidate passed the low-target criterion. Do not expand to 200 ks; redesign low-target amplitude or consider a model-capacity / P2D-like correction.")
    lines += ["", "## Files", "", "- `D11_S5C_mode_segment_summary.csv`", "- `D11_S5C_global_vs_lowtarget_tradeoff.csv`", "- `D11_S5C_component_summary.csv`", "- `D11_S5C_scorecard_summary.json`"]
    (out_dir / "D11_S5C_RECOMMENDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
