#!/usr/bin/env python
"""D11-S5B low-target gate/sign diagnostic analysis.

This script does NOT launch training. It reads existing D11-S5A prediction.npz
files and scorecard CSVs, then diagnoses why the S5A low-target candidates
improved global MAE but worsened low_target / low_target_le_2p75 segments.

Outputs are written under:
  E:\\XJTU battery dataset\\_gv1_cache\\xjtu_batch134_d11_s5b_lowtarget_gate_sign_analysis
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
    "current_event_gate": ["voltage_current_event_gate", "current_event_gate"],
    "profile_event_gate": ["voltage_profile_event_gate", "profile_event_gate"],
    "low_tail_correction": ["voltage_low_tail_correction", "low_tail_correction", "v_low_tail_scaled"],
    "event_correction": ["voltage_event_correction", "event_correction", "v_event_scaled"],
    "temperature_correction": ["voltage_temperature_correction", "temperature_correction", "v_temperature_scaled"],
    "softsign_correction": ["voltage_softsign_correction", "voltage_correction", "v_correction"],
    "ocv_baseline": ["voltage_ocv_baseline", "v_ocv"],
    "direct_head": ["voltage_direct_head", "v_direct"],
    "base_branch": ["voltage_base_branch", "v_base"],
    "event_branch_delta": ["voltage_event_branch_delta", "v_event_total"],
}

FOCUS_SEGMENTS = ["all", "low_target", "low_target_le_2p75", "rest_I_zero", "high_target_ge_4p10", "charge_I_positive", "discharge_I_negative"]


def _to_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return np.asarray(arr, dtype=float).reshape(-1)


def pick_key(npz, keys: Iterable[str]) -> Optional[str]:
    files = list(npz.files)
    lower_map = {k.lower(): k for k in files}
    for k in keys:
        if k in files:
            return k
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None


def safe_float(x, default=math.nan) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    n = min([len(a) for a in arrays] or [0])
    if n == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a[:n])
    return mask


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    n = min(len(y_true), len(y_pred))
    yt = y_true[:n]
    yp = y_pred[:n]
    base = finite_mask(yt, yp)
    if mask is not None:
        base &= mask[:n]
    yt = yt[base]
    yp = yp[base]
    count = int(len(yt))
    if count == 0:
        return {"n": 0, "MAE_V": math.nan, "RMSE_V": math.nan, "corr": math.nan, "bias_V": math.nan,
                "pred_over_frac": math.nan, "pred_under_frac": math.nan, "target_min_V": math.nan, "target_max_V": math.nan,
                "pred_min_V": math.nan, "pred_max_V": math.nan}
    err = yp - yt
    corr = math.nan
    if count >= 2 and float(np.std(yt)) > 1e-12 and float(np.std(yp)) > 1e-12:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    return {
        "n": count,
        "MAE_V": float(np.mean(np.abs(err))),
        "RMSE_V": float(np.sqrt(np.mean(err ** 2))),
        "corr": corr,
        "bias_V": float(np.mean(err)),
        "pred_over_frac": float(np.mean(err > 0)),
        "pred_under_frac": float(np.mean(err < 0)),
        "target_min_V": float(np.min(yt)),
        "target_max_V": float(np.max(yt)),
        "pred_min_V": float(np.min(yp)),
        "pred_max_V": float(np.max(yp)),
    }


def infer_protocol(profile: str) -> str:
    if "R2.5" in profile or "R25" in profile or "R2_5" in profile:
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
    pk = pick_key(npz, PRED_KEYS)
    tk = pick_key(npz, TRUE_KEYS)
    if pk is None or tk is None:
        return None
    out: Dict[str, np.ndarray] = {
        "pred": _to_1d(npz[pk]),
        "true": _to_1d(npz[tk]),
    }
    time_k = pick_key(npz, TIME_KEYS)
    current_k = pick_key(npz, CURRENT_KEYS)
    out["time"] = _to_1d(npz[time_k]) if time_k else np.arange(len(out["true"]), dtype=float)
    out["current"] = _to_1d(npz[current_k]) if current_k else np.full(len(out["true"]), np.nan)
    for cname, keys in COMPONENT_KEYS.items():
        k = pick_key(npz, keys)
        if k is not None:
            out[cname] = _to_1d(npz[k])
    return out


def segment_masks(y_true: np.ndarray, y_pred: np.ndarray, time: np.ndarray, current: np.ndarray) -> Dict[str, np.ndarray]:
    n = min(len(y_true), len(y_pred), len(time), len(current))
    yt = y_true[:n]
    yp = y_pred[:n]
    tt = time[:n]
    ii = current[:n]
    finite = finite_mask(yt, yp)
    if np.isfinite(ii).any():
        finite &= np.isfinite(ii)
    else:
        ii = np.zeros(n)
    if np.isfinite(tt).any():
        q1, q2 = np.nanquantile(tt[np.isfinite(tt)], [1/3, 2/3])
    else:
        q1, q2 = n / 3, 2 * n / 3
    return {
        "all": finite,
        "charge_I_positive": finite & (ii > 1e-8),
        "discharge_I_negative": finite & (ii < -1e-8),
        "rest_I_zero": finite & (np.abs(ii) <= 1e-8),
        "low_target": finite & (yt <= 2.90),
        "low_target_le_2p75": finite & (yt <= 2.75),
        "high_target_ge_4p10": finite & (yt >= 4.10),
        "pred_high_overshoot_gt_4p35": finite & (yp > 4.35),
        "early_time_third": finite & (tt <= q1),
        "middle_time_third": finite & (tt > q1) & (tt <= q2),
        "late_time_third": finite & (tt > q2),
    }


def safe_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    n = min(len(arr), len(mask))
    if n == 0:
        return math.nan
    a = arr[:n]
    m = mask[:n] & np.isfinite(a)
    if not np.any(m):
        return math.nan
    return float(np.mean(a[m]))


def safe_frac_condition(arr: np.ndarray, mask: np.ndarray, op: str) -> float:
    n = min(len(arr), len(mask))
    if n == 0:
        return math.nan
    a = arr[:n]
    m = mask[:n] & np.isfinite(a)
    if not np.any(m):
        return math.nan
    if op == "lt0":
        return float(np.mean(a[m] < 0))
    if op == "gt0":
        return float(np.mean(a[m] > 0))
    if op == "abs_gt_1e-6":
        return float(np.mean(np.abs(a[m]) > 1e-6))
    return math.nan


def extract_mode_profile(path: Path) -> Tuple[str, str]:
    # Expected: prediction_root / mode / profile / prediction.npz
    return path.parent.parent.name, path.parent.name


def group_mean(rows: List[Dict[str, object]], keys: List[str], metrics: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        k = tuple(r.get(x, "") for x in keys)
        groups.setdefault(k, []).append(r)
    out: List[Dict[str, object]] = []
    for k, vals in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec = {name: value for name, value in zip(keys, k)}
        rec["n_rows"] = len(vals)
        for m in metrics:
            arr = np.array([safe_float(v.get(m)) for v in vals], dtype=float)
            rec[f"mean_{m}"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan
        out.append(rec)
    return out


def compare_duplicates(predictions: Dict[Tuple[str, str], np.ndarray]) -> List[Dict[str, object]]:
    profiles = sorted({p for _, p in predictions.keys()})
    modes = sorted({m for m, _ in predictions.keys()})
    rows: List[Dict[str, object]] = []
    for i, m1 in enumerate(modes):
        for m2 in modes[i + 1:]:
            diffs = []
            same_count = 0
            profile_count = 0
            for prof in profiles:
                a = predictions.get((m1, prof))
                b = predictions.get((m2, prof))
                if a is None or b is None:
                    continue
                n = min(len(a), len(b))
                if n == 0:
                    continue
                profile_count += 1
                d = float(np.nanmax(np.abs(a[:n] - b[:n])))
                diffs.append(d)
                if d < 1e-10:
                    same_count += 1
            rows.append({
                "mode_a": m1,
                "mode_b": m2,
                "profile_count_compared": profile_count,
                "identical_profile_count": same_count,
                "max_abs_diff_max": float(np.nanmax(diffs)) if diffs else math.nan,
                "max_abs_diff_mean": float(np.nanmean(diffs)) if diffs else math.nan,
                "duplicate_suspected": bool(profile_count > 0 and same_count == profile_count),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--prediction_root", default=None)
    ap.add_argument("--s5a_scorecard_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    prediction_root = Path(args.prediction_root) if args.prediction_root else cache_root / "xjtu_batch134_d11_s5a_lowtarget_sign_gate_diagnosis"
    s5a_scorecard_dir = Path(args.s5a_scorecard_dir) if args.s5a_scorecard_dir else cache_root / "xjtu_batch134_d11_s5a_lowtarget_sign_gate_scorecard"
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d11_s5b_lowtarget_gate_sign_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(prediction_root.rglob("prediction.npz"))
    if not pred_paths:
        raise FileNotFoundError(f"No prediction.npz files found under {prediction_root}")

    by_profile_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    predictions: Dict[Tuple[str, str], np.ndarray] = {}

    for path in pred_paths:
        mode, profile = extract_mode_profile(path)
        protocol = infer_protocol(profile)
        data = load_prediction(path)
        if data is None:
            by_profile_rows.append({"mode": mode, "profile": profile, "protocol": protocol, "status": "prediction_load_failed", "prediction_path": str(path)})
            continue
        y_true = data["true"]
        y_pred = data["pred"]
        time = data["time"]
        current = data["current"]
        n = min(len(y_true), len(y_pred), len(time), len(current))
        y_true = y_true[:n]
        y_pred = y_pred[:n]
        time = time[:n]
        current = current[:n]
        predictions[(mode, profile)] = y_pred.copy()
        masks = segment_masks(y_true, y_pred, time, current)

        row: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "status": "loaded", "prediction_path": str(path)}
        row.update({f"all_{k}": v for k, v in metric_dict(y_true, y_pred, masks["all"]).items()})
        row.update({f"low_target_{k}": v for k, v in metric_dict(y_true, y_pred, masks["low_target"]).items()})
        row.update({f"low_target_le_2p75_{k}": v for k, v in metric_dict(y_true, y_pred, masks["low_target_le_2p75"]).items()})
        by_profile_rows.append(row)

        for seg in FOCUS_SEGMENTS:
            m = masks[seg]
            rec: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg}
            rec.update(metric_dict(y_true, y_pred, m))
            seg_rows.append(rec)

        for seg in ["all", "low_target", "low_target_le_2p75", "rest_I_zero"]:
            m = masks[seg]
            rec: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg, "n": int(np.sum(m))}
            err = y_pred[:len(m)] - y_true[:len(m)]
            rec["err_mean_V"] = safe_mean(err, m)
            rec["err_positive_frac"] = safe_frac_condition(err, m, "gt0")
            rec["err_negative_frac"] = safe_frac_condition(err, m, "lt0")
            for cname in COMPONENT_KEYS.keys():
                if cname in data:
                    arr = data[cname]
                    rec[f"{cname}_mean"] = safe_mean(arr, m)
                    rec[f"{cname}_negative_frac"] = safe_frac_condition(arr, m, "lt0")
                    rec[f"{cname}_positive_frac"] = safe_frac_condition(arr, m, "gt0")
                    rec[f"{cname}_active_frac"] = safe_frac_condition(arr, m, "abs_gt_1e-6")
            component_rows.append(rec)

    dup_rows = compare_duplicates(predictions)

    write_csv(out_dir / "D11_S5B_by_profile_gate_sign.csv", by_profile_rows)
    write_csv(out_dir / "D11_S5B_segment_focus_metrics.csv", seg_rows)
    write_csv(out_dir / "D11_S5B_component_gate_sign_metrics.csv", component_rows)
    write_csv(out_dir / "D11_S5B_duplicate_mode_check.csv", dup_rows)

    seg_summary = group_mean(seg_rows, ["mode", "segment"], ["MAE_V", "RMSE_V", "corr", "bias_V", "n", "pred_over_frac"])
    comp_summary = group_mean(component_rows, ["mode", "segment"], [
        "err_mean_V", "err_positive_frac", "err_negative_frac",
        "low_gate_mean", "low_gate_active_frac",
        "low_tail_correction_mean", "low_tail_correction_negative_frac", "low_tail_correction_positive_frac", "low_tail_correction_active_frac",
        "softsign_correction_mean", "event_correction_mean", "temperature_correction_mean",
    ])
    write_csv(out_dir / "D11_S5B_mode_segment_summary.csv", seg_summary)
    write_csv(out_dir / "D11_S5B_mode_component_summary.csv", comp_summary)

    # Load S5A summary/tradeoff if present, to include in recommendation.
    s5a_recommendation = s5a_scorecard_dir / "D11_S5A_RECOMMENDATION.md"
    s5a_summary_json = s5a_scorecard_dir / "D11_S5A_scorecard_summary.json"
    s5a_mode_segment_csv = s5a_scorecard_dir / "D11_S5A_mode_segment_summary.csv"
    s5a_global_tradeoff_csv = s5a_scorecard_dir / "D11_S5A_global_vs_lowtarget_tradeoff.csv"

    # Diagnosis rules.
    duplicates = [r for r in dup_rows if r.get("duplicate_suspected")]
    lowtarget_summary = [r for r in seg_summary if r.get("segment") in {"low_target", "low_target_le_2p75"}]
    comp_low = [r for r in comp_summary if r.get("segment") in {"low_target", "low_target_le_2p75"}]

    baseline_low = {r["segment"]: r for r in lowtarget_summary if r.get("mode") == "baseline_d951"}
    candidate_findings: List[str] = []
    for r in lowtarget_summary:
        mode = str(r.get("mode"))
        seg = str(r.get("segment"))
        if mode == "baseline_d951":
            continue
        b = baseline_low.get(seg)
        if b:
            delta = safe_float(r.get("mean_MAE_V")) - safe_float(b.get("mean_MAE_V"))
            candidate_findings.append(f"{mode} / {seg}: delta_MAE_vs_baseline = {delta:+.6f} V")

    gate_findings: List[str] = []
    for r in comp_low:
        mode = str(r.get("mode"))
        seg = str(r.get("segment"))
        lg = safe_float(r.get("mean_low_gate_mean"))
        lc = safe_float(r.get("mean_low_tail_correction_mean"))
        neg = safe_float(r.get("mean_low_tail_correction_negative_frac"))
        err_pos = safe_float(r.get("mean_err_positive_frac"))
        gate_findings.append(
            f"{mode} / {seg}: low_gate_mean={lg:.6g}, low_tail_correction_mean={lc:.6g}, "
            f"correction_negative_frac={neg:.6g}, err_positive_frac={err_pos:.6g}"
        )

    recommendation_lines = []
    recommendation_lines += ["# D11-S5B low-target gate/sign diagnostic recommendation", ""]
    recommendation_lines += ["## Status", "", f"- Prediction root: `{prediction_root}`", f"- Output directory: `{out_dir}`", f"- Prediction files inspected: `{len(pred_paths)}`", ""]
    recommendation_lines += ["## Key conclusion", ""]
    recommendation_lines += [
        "D11-S5B is a diagnostic-only analysis. It does not promote any S5A candidate and does not launch training.",
        "The previous S5A result remains blocked from 200ks confirmation until low_target and low_target_le_2p75 improve directly.",
        "",
    ]
    recommendation_lines += ["## Duplicate mode check", ""]
    if duplicates:
        recommendation_lines.append("Potential duplicate modes detected:")
        for r in duplicates:
            recommendation_lines.append(f"- `{r['mode_a']}` vs `{r['mode_b']}`: identical profiles = {r['identical_profile_count']}/{r['profile_count_compared']}")
    else:
        recommendation_lines.append("No fully duplicate prediction modes detected by exact prediction comparison.")
    recommendation_lines += ["", "## Low-target MAE findings", ""]
    for line in candidate_findings:
        recommendation_lines.append(f"- {line}")
    recommendation_lines += ["", "## Gate/sign diagnostics", ""]
    if gate_findings:
        for line in gate_findings:
            recommendation_lines.append(f"- {line}")
    else:
        recommendation_lines.append("No component/gate arrays were found in prediction.npz. The model must save voltage_low_gate and voltage_low_tail_correction for a full sign diagnosis.")
    recommendation_lines += ["", "## Next action", ""]
    recommendation_lines += [
        "1. If low_gate_mean is missing or low on low_target, fix the output transform to save/activate a target-independent low-voltage/SOC gate.",
        "2. If err_positive_frac is high on low_target but low_tail_correction_mean is positive, the correction sign is wrong; redesign as a downward-only negative correction for low predicted-SOC/low-OCV regimes.",
        "3. If lowtarget_gate_probe and lowtarget_downward_mild are duplicate, fix the mode-to-args mapping before any new training.",
        "4. Do not run 200ks confirmation until low_target and low_target_le_2p75 MAE decrease versus baseline in a 40ks diagnostic.",
    ]
    (out_dir / "D11_S5B_RECOMMENDATION.md").write_text("\n".join(recommendation_lines) + "\n", encoding="utf-8")

    summary = {
        "ok": True,
        "stage": "D11-S5B low-target gate/sign diagnostic analysis",
        "cache_root": str(cache_root),
        "prediction_root": str(prediction_root),
        "s5a_scorecard_dir": str(s5a_scorecard_dir),
        "out_dir": str(out_dir),
        "prediction_count": len(pred_paths),
        "loaded_run_rows": len(by_profile_rows),
        "segment_rows": len(seg_rows),
        "component_rows": len(component_rows),
        "duplicate_mode_pairs": duplicates,
        "s5a_files_seen": {
            "recommendation": s5a_recommendation.exists(),
            "summary_json": s5a_summary_json.exists(),
            "mode_segment_csv": s5a_mode_segment_csv.exists(),
            "global_tradeoff_csv": s5a_global_tradeoff_csv.exists(),
        },
        "verdict": "d11_s5b_gate_sign_analysis_completed",
        "next_action": "redesign_mode_mapping_and_downward_gate_then_repeat_40ks; do_not_expand_to_200ks_yet",
    }
    (out_dir / "D11_S5B_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
