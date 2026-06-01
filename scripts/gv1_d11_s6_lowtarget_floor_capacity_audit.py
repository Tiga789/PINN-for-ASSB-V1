#!/usr/bin/env python
"""D11-S6 low-target floor / model-capacity audit.

This script does NOT launch training.  It audits existing D11-S5C prediction.npz
files after D11-S5C showed global MAE improvements but low_target and
low_target_le_2p75 remained worse than the D9.5.1 baseline.

The goal is to answer:
1) Are low-target points still systematically over-predicted?
2) Is there an apparent voltage-floor / output-transform barrier?
3) Which voltage components dominate low-target predictions?
4) Is amplitude tuning insufficient, requiring output-transform/model-capacity redesign?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
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
    "temperature_correction": ["voltage_temperature_correction", "temperature_correction", "v_temperature_scaled"],
    "ocv_baseline": ["voltage_ocv_baseline", "v_ocv"],
    "direct_head": ["voltage_direct_head", "v_direct"],
    "base_branch": ["voltage_base_branch", "v_base"],
    "event_branch_delta": ["voltage_event_branch_delta", "v_event_total"],
}
SEGMENTS = {
    "all": lambda y, yp, t, i: np.isfinite(y) & np.isfinite(yp),
    "low_target": lambda y, yp, t, i: np.isfinite(y) & np.isfinite(yp) & (y <= 2.90),
    "low_target_le_2p75": lambda y, yp, t, i: np.isfinite(y) & np.isfinite(yp) & (y <= 2.75),
    "rest_I_zero": lambda y, yp, t, i: np.isfinite(y) & np.isfinite(yp) & np.isfinite(i) & (np.abs(i) <= 1e-8),
    "high_target_ge_4p10": lambda y, yp, t, i: np.isfinite(y) & np.isfinite(yp) & (y >= 4.10),
    "pred_high_overshoot_gt_4p35": lambda y, yp, t, i: np.isfinite(y) & np.isfinite(yp) & (yp > 4.35),
}
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


def safe_mean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return math.nan
    m = np.isfinite(a)
    return float(np.mean(a[m])) if np.any(m) else math.nan


def safe_min(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return math.nan
    m = np.isfinite(a)
    return float(np.min(a[m])) if np.any(m) else math.nan


def safe_max(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return math.nan
    m = np.isfinite(a)
    return float(np.max(a[m])) if np.any(m) else math.nan


def corr(y: np.ndarray, yp: np.ndarray) -> float:
    if len(y) < 2 or np.std(y) <= 1e-12 or np.std(yp) <= 1e-12:
        return math.nan
    return float(np.corrcoef(y, yp)[0, 1])


def metrics(y: np.ndarray, yp: np.ndarray) -> Dict[str, float]:
    n = min(len(y), len(yp))
    y = y[:n]
    yp = yp[:n]
    mask = np.isfinite(y) & np.isfinite(yp)
    y = y[mask]
    yp = yp[mask]
    if len(y) == 0:
        return {"n": 0, "MAE_V": math.nan, "RMSE_V": math.nan, "corr": math.nan, "bias_V": math.nan,
                "pred_over_frac": math.nan, "pred_under_frac": math.nan,
                "target_min_V": math.nan, "target_max_V": math.nan, "pred_min_V": math.nan, "pred_max_V": math.nan,
                "pred_min_minus_target_max_V": math.nan, "pred_min_minus_target_min_V": math.nan}
    err = yp - y
    return {
        "n": int(len(y)),
        "MAE_V": float(np.mean(np.abs(err))),
        "RMSE_V": float(np.sqrt(np.mean(err ** 2))),
        "corr": corr(y, yp),
        "bias_V": float(np.mean(err)),
        "pred_over_frac": float(np.mean(err > 0)),
        "pred_under_frac": float(np.mean(err < 0)),
        "target_min_V": safe_min(y),
        "target_max_V": safe_max(y),
        "pred_min_V": safe_min(yp),
        "pred_max_V": safe_max(yp),
        "pred_min_minus_target_max_V": safe_min(yp) - safe_max(y),
        "pred_min_minus_target_min_V": safe_min(yp) - safe_min(y),
    }


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
    n = min(len(out["pred"]), len(out["true"]))
    out["pred"] = out["pred"][:n]
    out["true"] = out["true"][:n]
    t_key = pick_key(npz, TIME_KEYS)
    i_key = pick_key(npz, CURRENT_KEYS)
    out["time"] = _to_1d(npz[t_key])[:n] if t_key else np.arange(n, dtype=float)
    out["current"] = _to_1d(npz[i_key])[:n] if i_key else np.full(n, np.nan)
    for cname, keys in COMPONENT_KEYS.items():
        k = pick_key(npz, keys)
        if k is not None:
            out[cname] = _to_1d(npz[k])[:n]
    out["keys"] = np.array(npz.files, dtype=object)
    return out


def infer_protocol(profile: str) -> str:
    if "R2.5" in profile or "R25" in profile:
        return "R2.5"
    if "R3" in profile:
        return "R3"
    if "2C" in profile:
        return "2C"
    return "unknown"


def extract_mode_profile(path: Path) -> Tuple[str, str]:
    profile = path.parent.name
    mode = path.parent.parent.name if path.parent.parent.name else "unknown"
    return mode, profile


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
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
    out: List[Dict[str, object]] = []
    numeric_cols = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, np.floating)) and k not in group_keys})
    for key, vals in sorted(groups.items(), key=lambda x: tuple(str(v) for v in x[0])):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n_rows"] = len(vals)
        for col in numeric_cols:
            arr = np.array([float(v.get(col, np.nan)) for v in vals], dtype=float)
            rec[f"mean_{col}"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan
        out.append(rec)
    return out


def audit_output_transform(project_root: Path) -> Dict[str, object]:
    path = project_root / "gv1" / "output_transform.py"
    out: Dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = [
        "voltage_min_V", "voltage_max_V", "voltage_floor_V", "voltage_ceil_V",
        "enable_voltage_hard_clamp", "direct_voltage_mix", "low_voltage_gate_center_V",
        "low_voltage_gate_width_V", "low_tail", "voltage_low_tail_correction",
        "voltage_direct_head", "voltage_ocv_baseline", "voltage_base_branch",
    ]
    out["pattern_hits"] = {p: (p in text) for p in patterns}
    lines = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(p in line for p in patterns):
            lines.append({"line": i, "text": line.strip()})
    out["relevant_lines"] = lines[:200]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1")
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--prediction_root", default="")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    cache_root = Path(args.cache_root)
    pred_root = Path(args.prediction_root) if args.prediction_root else cache_root / "xjtu_batch134_d11_s5c_lowtarget_amplitude_repair"
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d11_s6_lowtarget_floor_capacity_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_root.rglob("prediction.npz"))
    run_rows: List[Dict[str, object]] = []
    segment_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    key_rows: List[Dict[str, object]] = []

    for p in pred_paths:
        mode, profile = extract_mode_profile(p)
        protocol = infer_protocol(profile)
        data = load_prediction(p)
        if data is None:
            run_rows.append({"mode": mode, "profile": profile, "protocol": protocol, "status": "prediction_load_failed", "path": str(p)})
            continue
        y = data["true"]
        yp = data["pred"]
        t = data["time"]
        cur = data["current"]
        run_row = {"mode": mode, "profile": profile, "protocol": protocol, "status": "loaded", "path": str(p)}
        run_row.update({f"all_{k}": v for k, v in metrics(y, yp).items()})
        run_rows.append(run_row)
        key_rows.append({"mode": mode, "profile": profile, "protocol": protocol, "npz_keys": ";".join(str(k) for k in data.get("keys", []))})

        for seg_name, fn in SEGMENTS.items():
            n = min(len(y), len(yp), len(t), len(cur))
            yy, pp, tt, ii = y[:n], yp[:n], t[:n], cur[:n]
            mask = fn(yy, pp, tt, ii)
            m = metrics(yy[mask], pp[mask])
            seg_row: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg_name}
            seg_row.update(m)
            segment_rows.append(seg_row)
            comp_row: Dict[str, object] = {"mode": mode, "profile": profile, "protocol": protocol, "segment": seg_name, "n": int(np.sum(mask))}
            err = pp[mask] - yy[mask]
            comp_row["err_mean_V"] = safe_mean(err)
            comp_row["err_positive_frac"] = float(np.mean(err > 0)) if err.size else math.nan
            comp_row["err_negative_frac"] = float(np.mean(err < 0)) if err.size else math.nan
            for cname in COMPONENT_KEYS.keys():
                if cname in data:
                    arr = data[cname][:n][mask]
                    comp_row[f"{cname}_mean"] = safe_mean(arr)
                    comp_row[f"{cname}_min"] = safe_min(arr)
                    comp_row[f"{cname}_max"] = safe_max(arr)
                    comp_row[f"{cname}_negative_frac"] = float(np.mean(arr < 0)) if arr.size else math.nan
                    comp_row[f"{cname}_positive_frac"] = float(np.mean(arr > 0)) if arr.size else math.nan
            component_rows.append(comp_row)

    mode_summary = group_mean(run_rows, ["mode"])
    mode_segment_summary = group_mean(segment_rows, ["mode", "segment"])
    mode_component_summary = group_mean(component_rows, ["mode", "segment"])

    # Baseline comparisons on focus segments.
    by_mode_seg: Dict[Tuple[str, str], Dict[str, object]] = {(r["mode"], r["segment"]): r for r in mode_segment_summary}
    comparison_rows: List[Dict[str, object]] = []
    for r in mode_segment_summary:
        mode = str(r.get("mode"))
        seg = str(r.get("segment"))
        if mode == BASELINE_MODE:
            continue
        base = by_mode_seg.get((BASELINE_MODE, seg))
        if base:
            comparison_rows.append({
                "mode": mode,
                "segment": seg,
                "candidate_mean_MAE_V": r.get("mean_MAE_V", math.nan),
                "baseline_mean_MAE_V": base.get("mean_MAE_V", math.nan),
                "candidate_minus_baseline_MAE_V": float(r.get("mean_MAE_V", math.nan)) - float(base.get("mean_MAE_V", math.nan)),
                "candidate_mean_corr": r.get("mean_corr", math.nan),
                "baseline_mean_corr": base.get("mean_corr", math.nan),
                "candidate_minus_baseline_corr": float(r.get("mean_corr", math.nan)) - float(base.get("mean_corr", math.nan)),
                "candidate_mean_bias_V": r.get("mean_bias_V", math.nan),
                "baseline_mean_bias_V": base.get("mean_bias_V", math.nan),
                "candidate_minus_baseline_bias_V": float(r.get("mean_bias_V", math.nan)) - float(base.get("mean_bias_V", math.nan)),
                "candidate_pred_min_minus_target_max_V": r.get("mean_pred_min_minus_target_max_V", math.nan),
                "baseline_pred_min_minus_target_max_V": base.get("mean_pred_min_minus_target_max_V", math.nan),
            })

    # Find whether each candidate worsens low-target while improving all.
    candidate_decisions = []
    modes = [str(r.get("mode")) for r in mode_summary if str(r.get("mode")) != BASELINE_MODE]
    for mode in modes:
        def d(seg: str, col: str = "candidate_minus_baseline_MAE_V") -> float:
            for x in comparison_rows:
                if x["mode"] == mode and x["segment"] == seg:
                    return float(x.get(col, math.nan))
            return math.nan
        all_delta = d("all")
        low_delta = d("low_target")
        low275_delta = d("low_target_le_2p75")
        rest_delta = d("rest_I_zero")
        high_delta = d("high_target_ge_4p10")
        floor_gap = d("low_target", "candidate_pred_min_minus_target_max_V")
        candidate_decisions.append({
            "mode": mode,
            "all_mae_delta_V": all_delta,
            "low_target_mae_delta_V": low_delta,
            "low_target_le_2p75_mae_delta_V": low275_delta,
            "rest_I_zero_mae_delta_V": rest_delta,
            "high_target_mae_delta_V": high_delta,
            "candidate_low_target_pred_min_minus_target_max_V": floor_gap,
            "global_improves": bool(np.isfinite(all_delta) and all_delta < 0),
            "low_target_improves": bool(np.isfinite(low_delta) and low_delta < -0.02 and np.isfinite(low275_delta) and low275_delta < -0.02),
            "floor_barrier_suspected": bool(np.isfinite(floor_gap) and floor_gap > 0.30),
        })

    output_transform_audit = audit_output_transform(project_root)

    write_csv(out_dir / "D11_S6_run_level_floor_audit.csv", run_rows)
    write_csv(out_dir / "D11_S6_segment_floor_audit.csv", segment_rows)
    write_csv(out_dir / "D11_S6_component_lowtarget_audit.csv", component_rows)
    write_csv(out_dir / "D11_S6_npz_key_audit.csv", key_rows)
    write_csv(out_dir / "D11_S6_mode_summary.csv", mode_summary)
    write_csv(out_dir / "D11_S6_mode_segment_summary.csv", mode_segment_summary)
    write_csv(out_dir / "D11_S6_mode_component_summary.csv", mode_component_summary)
    write_csv(out_dir / "D11_S6_candidate_vs_baseline.csv", comparison_rows)
    write_csv(out_dir / "D11_S6_candidate_decisions.csv", candidate_decisions)
    (out_dir / "D11_S6_output_transform_static_audit.json").write_text(json.dumps(output_transform_audit, indent=2), encoding="utf-8")

    any_low_improve = any(c["low_target_improves"] for c in candidate_decisions)
    all_candidates_global_improve = all(c["global_improves"] for c in candidate_decisions) if candidate_decisions else False
    floor_suspected = any(c["floor_barrier_suspected"] for c in candidate_decisions)
    summary = {
        "ok": True,
        "stage": "D11-S6 low-target floor / model-capacity audit",
        "prediction_root": str(pred_root),
        "out_dir": str(out_dir),
        "prediction_count": len(pred_paths),
        "loaded_run_count": len([r for r in run_rows if r.get("status") == "loaded"]),
        "mode_summary": mode_summary,
        "candidate_decisions": candidate_decisions,
        "any_low_target_improvement": any_low_improve,
        "all_candidates_global_improve": all_candidates_global_improve,
        "floor_barrier_suspected": floor_suspected,
        "verdict": "d11_s6_audit_completed",
        "next_action": "do_not_run_200ks_confirmation; redesign output transform or low-voltage anchor if low-target remains worse",
    }
    (out_dir / "D11_S6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown recommendation.
    lines = [
        "# D11-S6 low-target floor / model-capacity audit recommendation",
        "",
        f"- Prediction root: `{pred_root}`",
        f"- Loaded runs: `{summary['loaded_run_count']}` / `{summary['prediction_count']}`",
        f"- Verdict: `{summary['verdict']}`",
        "",
        "## Main finding",
        "",
    ]
    if not any_low_improve and all_candidates_global_improve:
        lines.append("All amplitude candidates improve global MAE but none improves the low_target and low_target_le_2p75 segments.  This blocks any 200 ks confirmation.")
    elif any_low_improve:
        lines.append("At least one candidate appears to improve low-target segments.  Inspect candidate decisions before deciding whether to prepare a separate 200 ks confirmation.")
    else:
        lines.append("No reliable candidate improvement was detected.  Treat D11-S6 as a diagnostic audit only.")
    if floor_suspected:
        lines.append("A low-voltage floor/output-transform barrier is suspected because candidate low-target predictions remain far above the low-target range.")
    lines += [
        "",
        "## Candidate decisions",
        "",
        "| mode | global ΔMAE | low_target ΔMAE | low<=2.75 ΔMAE | floor suspected | low improves |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in candidate_decisions:
        lines.append(
            f"| {c['mode']} | {c['all_mae_delta_V']} | {c['low_target_mae_delta_V']} | {c['low_target_le_2p75_mae_delta_V']} | {c['floor_barrier_suspected']} | {c['low_target_improves']} |"
        )
    lines += [
        "",
        "## Recommended next step",
        "",
        "Do not run D11-S6 as a confirmation experiment.  The next coding step should target one of these mechanisms:",
        "",
        "1. output-transform redesign that permits lower predicted voltage in true low-target regions without hard clamping;",
        "2. low-voltage anchor / supervised tail anchor with explicit low_target selection criterion;",
        "3. protocol-specific or P2D-like high-rate correction if low-target failures are tied to electrolyte/transport limits;",
        "4. add prediction component logging if any required component fields are missing.",
        "",
        "Relevant files: `D11_S6_candidate_vs_baseline.csv`, `D11_S6_mode_component_summary.csv`, `D11_S6_output_transform_static_audit.json`.",
    ]
    (out_dir / "D11_S6_RECOMMENDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
