#!/usr/bin/env python
"""Collect D11-S5A low-target sign/gate diagnostic scorecard from predictions.

No training is launched.  The script reads prediction.npz files produced by
D11-S5A generated commands and summarizes global metrics, low-target segments,
and optional output-transform component/gate diagnostics.
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

# Optional component outputs saved by the current GV1 output transform.  The
# script tolerates missing keys and records NaN so it can run across versions.
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

SEGMENT_ORDER = [
    "all",
    "charge_I_positive",
    "discharge_I_negative",
    "rest_I_zero",
    "low_target",
    "low_target_le_2p75",
    "high_target_ge_4p10",
    "pred_high_overshoot_gt_4p35",
    "early_time_third",
    "middle_time_third",
    "late_time_third",
]


def _to_1d(a) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)
        arr = arr[:, 0]
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


def finite_pair(y_true: np.ndarray, y_pred: np.ndarray, *extra: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    lengths = [len(y_true), len(y_pred)] + [len(e) for e in extra]
    n = min(lengths) if lengths else 0
    y_true = y_true[:n]; y_pred = y_pred[:n]
    extras = [e[:n] for e in extra]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    for e in extras:
        mask &= np.isfinite(e)
    return y_true[mask], y_pred[mask], [e[mask] for e in extras]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred, _ = finite_pair(y_true, y_pred)
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "MAE_V": math.nan, "RMSE_V": math.nan, "corr": math.nan, "bias_V": math.nan,
                "target_min_V": math.nan, "target_max_V": math.nan, "pred_min_V": math.nan, "pred_max_V": math.nan}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if n >= 2 and float(np.std(y_true)) > 1e-12 and float(np.std(y_pred)) > 1e-12:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = math.nan
    return {"n": n, "MAE_V": mae, "RMSE_V": rmse, "corr": corr, "bias_V": bias,
            "target_min_V": float(np.min(y_true)), "target_max_V": float(np.max(y_true)),
            "pred_min_V": float(np.min(y_pred)), "pred_max_V": float(np.max(y_pred))}


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
    if not np.any(m):
        return math.nan
    return float(np.mean(a[m]))


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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_mean(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in group_keys)
        groups.setdefault(key, []).append(r)
    out: List[Dict[str, object]] = []
    for key, vals in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n_rows"] = len(vals)
        for m in ["MAE_V", "RMSE_V", "corr", "bias_V", "n"]:
            arr = np.array([float(v.get(m, np.nan)) for v in vals], dtype=float)
            rec[f"mean_{m}"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan
        # Component diagnostics.
        for c in COMPONENT_KEYS.keys():
            for suffix in ["mean", "lowtarget_mean"]:
                col = f"{c}_{suffix}"
                arr = np.array([float(v.get(col, np.nan)) for v in vals], dtype=float)
                if np.isfinite(arr).any():
                    rec[f"mean_{col}"] = float(np.nanmean(arr))
        out.append(rec)
    return out


def extract_mode_profile(path: Path) -> Tuple[str, str]:
    # Expected path: root / mode / profile / prediction.npz
    profile = path.parent.name
    mode = path.parent.parent.name if path.parent.parent.name else "unknown"
    return mode, profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    pred_root = cache_root / "xjtu_batch134_d11_s5a_lowtarget_sign_gate_diagnosis"
    out_dir = cache_root / "xjtu_batch134_d11_s5a_lowtarget_sign_gate_scorecard"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_root.rglob("prediction.npz"))
    run_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []
    gate_rows: List[Dict[str, object]] = []
    notes: List[str] = []

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
        run_row: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "status": "strict_completed_metrics_ok", "prediction_path": str(p)}
        run_row.update(run_m)
        run_rows.append(run_row)
        masks = segment_masks(y_true, y_pred, time, current)
        # Per-segment metrics and component means.
        for seg in SEGMENT_ORDER:
            mask = masks.get(seg)
            if mask is None:
                continue
            y_t = y_true[:len(mask)][mask]
            y_p = y_pred[:len(mask)][mask]
            rec: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg}
            rec.update(metrics(y_t, y_p))
            for cname in COMPONENT_KEYS.keys():
                if cname in data:
                    rec[f"{cname}_mean"] = safe_mean(data[cname], mask)
            seg_rows.append(rec)
        # Dedicated low-target gate/sign diagnostics.
        for seg in ["low_target", "low_target_le_2p75", "all"]:
            mask = masks.get(seg, np.zeros(n, dtype=bool))
            rec: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg, "n": int(np.sum(mask))}
            err = y_pred[:len(mask)] - y_true[:len(mask)]
            if np.any(mask):
                rec["bias_V"] = float(np.mean(err[mask]))
                rec["MAE_V"] = float(np.mean(np.abs(err[mask])))
                rec["pred_minus_true_positive_frac"] = float(np.mean(err[mask] > 0))
            else:
                rec["bias_V"] = rec["MAE_V"] = rec["pred_minus_true_positive_frac"] = math.nan
            for cname in COMPONENT_KEYS.keys():
                if cname in data:
                    rec[f"{cname}_mean"] = safe_mean(data[cname], mask)
                else:
                    rec[f"{cname}_mean"] = math.nan
            gate_rows.append(rec)

    mode_summary = group_mean([r for r in run_rows if r.get("status") == "strict_completed_metrics_ok"], ["mode"])
    mode_segment_summary = group_mean(seg_rows, ["mode", "segment"])
    mode_protocol_segment_summary = group_mean(seg_rows, ["mode", "protocol", "segment"])
    gate_summary = group_mean(gate_rows, ["mode", "segment"])

    write_csv(out_dir / "D11_S5A_run_metrics.csv", run_rows)
    write_csv(out_dir / "D11_S5A_segment_metrics.csv", seg_rows)
    write_csv(out_dir / "D11_S5A_mode_summary.csv", mode_summary)
    write_csv(out_dir / "D11_S5A_mode_segment_summary.csv", mode_segment_summary)
    write_csv(out_dir / "D11_S5A_mode_protocol_segment_summary.csv", mode_protocol_segment_summary)
    write_csv(out_dir / "D11_S5A_lowtarget_gate_diagnostic.csv", gate_rows)
    write_csv(out_dir / "D11_S5A_lowtarget_gate_summary.csv", gate_summary)

    # Build tradeoff table versus baseline.
    baseline = {r.get("segment"): r for r in mode_segment_summary if r.get("mode") == "baseline_d951"}
    tradeoff: List[Dict[str, object]] = []
    for r in mode_segment_summary:
        mode = str(r.get("mode"))
        seg = str(r.get("segment"))
        if mode == "baseline_d951" or seg not in baseline:
            continue
        b = baseline[seg]
        rec = {"mode": mode, "segment": seg}
        for key in ["mean_MAE_V", "mean_corr", "mean_bias_V"]:
            rv = float(r.get(key, math.nan)); bv = float(b.get(key, math.nan))
            rec[f"candidate_{key}"] = rv; rec[f"baseline_{key}"] = bv
            rec[f"delta_{key}"] = rv - bv if math.isfinite(rv) and math.isfinite(bv) else math.nan
        tradeoff.append(rec)
    write_csv(out_dir / "D11_S5A_global_vs_lowtarget_tradeoff.csv", tradeoff)

    # Recommendation.
    summary = {
        "ok": True,
        "stage": "D11-S5A low-target sign/gate diagnosis scorecard from predictions",
        "prediction_root": str(pred_root),
        "out_dir": str(out_dir),
        "run_count": len(run_rows),
        "expected_run_count": 24,
        "counts": {},
        "mode_summary": mode_summary,
        "verdict": "d11_s5a_diagnosis_completed" if run_rows else "d11_s5a_no_predictions_found",
        "notes": notes,
    }
    counts: Dict[str, int] = {}
    for r in run_rows:
        status = str(r.get("status", "unknown")); counts[status] = counts.get(status, 0) + 1
    summary["counts"] = counts

    # Decision logic: pass only if low_target and low_target_le_2p75 both improve.
    rec_lines = [
        "# D11-S5A low-target sign/gate diagnosis recommendation",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Run count: `{len(run_rows)}`",
        f"- Completed metrics OK: `{counts.get('strict_completed_metrics_ok', 0)}`",
        "",
        "## Mode summary",
        "",
        "| mode | n_rows | mean_MAE_V | mean_corr | mean_bias_V |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in mode_summary:
        rec_lines.append(f"| {r.get('mode')} | {r.get('n_rows')} | {r.get('mean_MAE_V')} | {r.get('mean_corr')} | {r.get('mean_bias_V')} |")
    rec_lines += ["", "## Low-target tradeoff versus baseline", "", "| mode | segment | ΔMAE vs baseline | Δcorr vs baseline | Δbias vs baseline |", "|---|---|---:|---:|---:|"]
    for r in tradeoff:
        if str(r.get("segment")) in {"low_target", "low_target_le_2p75", "all", "rest_I_zero", "pred_high_overshoot_gt_4p35"}:
            rec_lines.append(f"| {r.get('mode')} | {r.get('segment')} | {r.get('delta_mean_MAE_V')} | {r.get('delta_mean_corr')} | {r.get('delta_mean_bias_V')} |")

    # Pick candidate by required criterion.
    candidate_names = sorted({str(r.get("mode")) for r in tradeoff if str(r.get("mode")) != "baseline_d951"})
    viable = []
    for cand in candidate_names:
        t_low = next((r for r in tradeoff if r.get("mode") == cand and r.get("segment") == "low_target"), None)
        t_275 = next((r for r in tradeoff if r.get("mode") == cand and r.get("segment") == "low_target_le_2p75"), None)
        t_all = next((r for r in tradeoff if r.get("mode") == cand and r.get("segment") == "all"), None)
        if not (t_low and t_275 and t_all):
            continue
        d_low = float(t_low.get("delta_mean_MAE_V", math.nan))
        d_275 = float(t_275.get("delta_mean_MAE_V", math.nan))
        d_all = float(t_all.get("delta_mean_MAE_V", math.nan))
        d_corr_all = float(t_all.get("delta_mean_corr", math.nan))
        if math.isfinite(d_low) and math.isfinite(d_275) and d_low < 0 and d_275 < 0 and d_all <= 0.01 and d_corr_all >= -0.01:
            viable.append(cand)
    rec_lines += ["", "## Decision", ""]
    if viable:
        rec_lines.append("At least one candidate improves both low_target and low_target_le_2p75 without material global damage. Run a 6-profile 200ks confirmation for the best viable candidate only.")
        rec_lines.append(f"Viable candidates: `{viable}`")
        summary["next_action"] = "run_6profile_200ks_confirmation_for_viable_lowtarget_candidate"
    else:
        rec_lines.append("No candidate satisfies the low-target improvement criterion. Do not expand to 200ks; inspect gate/sign diagnostics and redesign the low-target correction.")
        summary["next_action"] = "do_not_expand_to_200ks_redesign_lowtarget_gate_or_sign"
    rec_lines += ["", "## Files to inspect", "", "- `D11_S5A_lowtarget_gate_diagnostic.csv`", "- `D11_S5A_lowtarget_gate_summary.csv`", "- `D11_S5A_global_vs_lowtarget_tradeoff.csv`", "- `D11_S5A_mode_segment_summary.csv`"]

    (out_dir / "D11_S5A_RECOMMENDATION.md").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")
    (out_dir / "D11_S5A_scorecard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
