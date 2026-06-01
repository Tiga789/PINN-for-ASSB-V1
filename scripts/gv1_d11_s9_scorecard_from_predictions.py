#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D11-S9 scorecard for trainable localized P2D-like correction outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen = set(); keys=[]
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k); keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def first_array(data: np.lib.npyio.NpzFile, names: Sequence[str]) -> Optional[np.ndarray]:
    keys = set(data.files)
    for n in names:
        if n in keys:
            return np.asarray(data[n])
    lower = {k.lower(): k for k in data.files}
    for n in names:
        if n.lower() in lower:
            return np.asarray(data[lower[n.lower()]])
    return None


def force_1d(x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            arr = arr.reshape(arr.shape[0], -1)[:, 0]
    arr = arr.astype(float, copy=False).reshape(-1)
    if n is not None:
        arr = arr[:n] if arr.size >= n else np.concatenate([arr, np.full(n-arr.size, np.nan)])
    return arr


def scalar_str(data: np.lib.npyio.NpzFile, name: str, default: str = "") -> str:
    if name not in data.files:
        return default
    try:
        arr = np.asarray(data[name])
        if arr.shape == ():
            return str(arr.item())
        if arr.size:
            return str(arr.reshape(-1)[0])
    except Exception:
        return default
    return default


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=float).reshape(-1); p = np.asarray(p, dtype=float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(p)
    if not np.any(m):
        return {"n": 0, "MAE_V": float("nan"), "RMSE_V": float("nan"), "corr": float("nan"), "bias_V": float("nan"), "pred_over_frac": float("nan"), "pred_under_frac": float("nan")}
    yt = y[m]; yp = p[m]; e = yp - yt
    corr = float(np.corrcoef(yt, yp)[0,1]) if yt.size >= 2 and np.nanstd(yt) > 1e-12 and np.nanstd(yp) > 1e-12 else float("nan")
    return {
        "n": int(yt.size), "MAE_V": float(np.mean(np.abs(e))), "RMSE_V": float(np.sqrt(np.mean(e*e))),
        "corr": corr, "bias_V": float(np.mean(e)), "pred_over_frac": float(np.mean(e > 0)), "pred_under_frac": float(np.mean(e < 0)),
        "target_min_V": float(np.nanmin(yt)), "target_max_V": float(np.nanmax(yt)), "pred_min_V": float(np.nanmin(yp)), "pred_max_V": float(np.nanmax(yp)),
    }


def infer_protocol(profile: str, path: Path) -> str:
    text = f"{profile} {path}".lower()
    if "r2.5" in text or "r25" in text or "batch-3" in text or "batch3" in text:
        return "R2.5"
    if "r3" in text or "batch-4" in text or "batch4" in text:
        return "R3"
    if "2c" in text or "batch-1" in text or "batch1" in text:
        return "2C"
    return "unknown"


def load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        y = first_array(data, ["voltage_exp", "voltage_true", "target_voltage", "y_true", "V_true"])
        p = first_array(data, ["voltage_exp_pred", "voltage_pred", "pred_voltage", "y_pred", "V_pred"])
        if y is None or p is None:
            raise ValueError(f"Missing voltage arrays in {path}. keys={data.files}")
        n = min(np.asarray(y).size, np.asarray(p).size)
        y = force_1d(y, n); p = force_1d(p, n)
        base = first_array(data, ["voltage_exp_base_pred", "voltage_base_pred", "base_voltage_pred"])
        if base is None:
            base = p.copy()
        base = force_1d(base, n)
        I = first_array(data, ["I_profile", "current_A", "I_A", "current", "I"])
        if I is None:
            I = np.zeros(n)
        I = force_1d(I, n)
        t = first_array(data, ["t_global_s", "time_s", "t_s", "t", "time"])
        if t is None:
            t = np.arange(n, dtype=float)
        t = force_1d(t, n)
        mode = scalar_str(data, "mode", path.parent.parent.name)
        profile = scalar_str(data, "profile", path.parent.name)
        protocol = scalar_str(data, "protocol", infer_protocol(profile, path))
        split = scalar_str(data, "split", "unknown")
        deficit = first_array(data, ["p2dlike_trainable_deficit_V", "p2dlike_deficit_V"])
        if deficit is None:
            deficit = p*0.0
        deficit = force_1d(deficit, n)
        gate = first_array(data, ["transport_gate_s9", "low_gate_s9", "voltage_low_gate", "low_gate"])
        if gate is None:
            gate = np.full(n, np.nan)
        gate = force_1d(gate, n)
    return {"path": path, "y": y, "p": p, "base": base, "I": I, "t": t, "mode": mode, "profile": profile, "protocol": protocol, "split": split, "deficit": deficit, "gate": gate}


def masks(y: np.ndarray, p: np.ndarray, I: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    m = np.isfinite(y) & np.isfinite(p)
    absI = np.abs(I)
    finite = absI[np.isfinite(absI)]
    thr = max(float(np.nanpercentile(finite, 10)) if finite.size else 0.0, 1e-12)
    n = y.size
    order = np.argsort(np.nan_to_num(t, nan=0.0))
    thirds = np.zeros(n, dtype=int); thirds[order[:max(1,n//3)]] = 0; thirds[order[max(1,n//3):max(2,n*2//3)]] = 1; thirds[order[max(2,n*2//3):]] = 2
    return {
        "all": m,
        "charge_I_positive": m & (I > thr),
        "discharge_I_negative": m & (I < -thr),
        "rest_I_zero": m & (absI <= thr),
        "low_target": m & (y <= 3.0),
        "low_target_le_2p75": m & (y <= 2.75),
        "high_target_ge_4p10": m & (y >= 4.10),
        "pred_high_overshoot_gt_4p35": m & (p > 4.35),
        "early_time_third": m & (thirds == 0),
        "middle_time_third": m & (thirds == 1),
        "late_time_third": m & (thirds == 2),
    }


def group_mean(rows: List[Dict[str, Any]], keys: List[str], metric_keys: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any,...], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    out = []
    for kval, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = {k:v for k,v in zip(keys, kval)}
        row["n_rows"] = len(rs)
        for mk in metric_keys:
            vals = [float(r.get(mk, float("nan"))) for r in rs]
            vals = [v for v in vals if math.isfinite(v)]
            row[f"mean_{mk}"] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s9_trainable_p2dlike_correction")
    ap.add_argument("--out_dir", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s9_trainable_p2dlike_correction_scorecard")
    args = ap.parse_args()
    root = Path(args.prediction_root); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.rglob("prediction.npz"))
    recs = []
    for p in paths:
        try:
            recs.append(load_npz(p))
        except Exception as e:
            print(f"WARNING failed to load {p}: {e}")
    if not recs:
        raise RuntimeError(f"No prediction.npz found under {root}")

    run_rows: List[Dict[str, Any]] = []
    seg_rows: List[Dict[str, Any]] = []
    comp_rows: List[Dict[str, Any]] = []
    for r in recs:
        m_all = metrics(r["y"], r["p"])
        row = {"mode": r["mode"], "profile": r["profile"], "protocol": r["protocol"], "split": r["split"], "status": "strict_completed_metrics_ok", "prediction_path": str(r["path"])}
        row.update(m_all)
        run_rows.append(row)
        for sname, mask in masks(r["y"], r["p"], r["I"], r["t"]).items():
            if np.any(mask):
                sm = metrics(r["y"][mask], r["p"][mask])
            else:
                sm = {"n": 0, "MAE_V": float("nan"), "RMSE_V": float("nan"), "corr": float("nan"), "bias_V": float("nan"), "pred_over_frac": float("nan"), "pred_under_frac": float("nan")}
            srow = {"mode": r["mode"], "profile": r["profile"], "protocol": r["protocol"], "split": r["split"], "segment": sname}
            srow.update(sm)
            seg_rows.append(srow)
            mask2 = mask & np.isfinite(r["deficit"])
            comp_rows.append({
                "mode": r["mode"], "profile": r["profile"], "protocol": r["protocol"], "split": r["split"], "segment": sname,
                "n": int(np.sum(mask)),
                "deficit_mean_V": float(np.nanmean(r["deficit"][mask2])) if np.any(mask2) else float("nan"),
                "deficit_max_V": float(np.nanmax(r["deficit"][mask2])) if np.any(mask2) else float("nan"),
                "gate_mean": float(np.nanmean(r["gate"][mask])) if np.any(mask) else float("nan"),
            })

    metric_keys = ["MAE_V", "RMSE_V", "corr", "bias_V", "pred_over_frac", "pred_under_frac", "n"]
    mode_summary = group_mean(run_rows, ["mode"], metric_keys)
    mode_split_summary = group_mean(run_rows, ["mode", "split"], metric_keys)
    mode_segment_summary = group_mean(seg_rows, ["mode", "segment"], metric_keys)
    mode_split_segment_summary = group_mean(seg_rows, ["mode", "split", "segment"], metric_keys)
    comp_summary = group_mean(comp_rows, ["mode", "segment"], ["deficit_mean_V", "deficit_max_V", "gate_mean", "n"])

    # Baseline comparisons by all/eval segment means.  Prefer eval split for decision if present.
    def get_mean(rows, mode, segment, split=None, key="mean_MAE_V"):
        for r in rows:
            if r.get("mode") == mode and r.get("segment") == segment and (split is None or r.get("split") == split):
                return r.get(key, float("nan"))
        return float("nan")

    modes = [r["mode"] for r in mode_summary if r.get("mode") != "baseline_copy"]
    trade_rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    split_for_decision = "eval" if any(r.get("split") == "eval" for r in mode_split_segment_summary) else None
    for mode in modes:
        for seg in ["all", "low_target", "low_target_le_2p75", "rest_I_zero", "high_target_ge_4p10", "discharge_I_negative"]:
            # Segment summary by split is not created with segment all? yes all exists.
            rows_src = mode_split_segment_summary if split_for_decision else mode_segment_summary
            b_mae = get_mean(rows_src, "baseline_copy", seg, split_for_decision, "mean_MAE_V")
            c_mae = get_mean(rows_src, mode, seg, split_for_decision, "mean_MAE_V")
            b_corr = get_mean(rows_src, "baseline_copy", seg, split_for_decision, "mean_corr")
            c_corr = get_mean(rows_src, mode, seg, split_for_decision, "mean_corr")
            trade_rows.append({
                "candidate": mode, "decision_split": split_for_decision or "all", "segment": seg,
                "candidate_mean_MAE_V": c_mae, "baseline_mean_MAE_V": b_mae, "candidate_minus_baseline_MAE_V": c_mae - b_mae if math.isfinite(c_mae) and math.isfinite(b_mae) else float("nan"),
                "candidate_mean_corr": c_corr, "baseline_mean_corr": b_corr, "candidate_minus_baseline_corr": c_corr - b_corr if math.isfinite(c_corr) and math.isfinite(b_corr) else float("nan"),
            })
        def delta(seg, key="mean_MAE_V"):
            rows_src = mode_split_segment_summary if split_for_decision else mode_segment_summary
            b = get_mean(rows_src, "baseline_copy", seg, split_for_decision, key)
            c = get_mean(rows_src, mode, seg, split_for_decision, key)
            return c - b if math.isfinite(c) and math.isfinite(b) else float("nan")
        d_all = delta("all")
        d_low = delta("low_target")
        d_deep = delta("low_target_le_2p75")
        d_rest = delta("rest_I_zero")
        d_high = delta("high_target_ge_4p10")
        d_corr_all = delta("all", "mean_corr")
        low_ok = math.isfinite(d_low) and math.isfinite(d_deep) and d_low <= -0.020 and d_deep <= -0.020
        global_ok = math.isfinite(d_all) and d_all <= 0.005
        corr_ok = (not math.isfinite(d_corr_all)) or d_corr_all >= -0.005
        rest_ok = (not math.isfinite(d_rest)) or d_rest <= 0.015
        high_ok = (not math.isfinite(d_high)) or d_high <= 0.015
        promote = low_ok and global_ok and corr_ok and rest_ok and high_ok
        decisions.append({"mode": mode, "decision_split": split_for_decision or "all", "low_ok": low_ok, "global_ok": global_ok, "corr_ok": corr_ok, "rest_ok": rest_ok, "high_ok": high_ok, "promote_candidate": promote, "delta_all_MAE_V": d_all, "delta_low_target_MAE_V": d_low, "delta_low_target_le_2p75_MAE_V": d_deep, "delta_rest_MAE_V": d_rest, "delta_high_MAE_V": d_high, "delta_all_corr": d_corr_all})

    promoted = [d["mode"] for d in decisions if d["promote_candidate"]]
    write_csv(out / "D11_S9_run_metrics.csv", run_rows)
    write_csv(out / "D11_S9_segment_metrics.csv", seg_rows)
    write_csv(out / "D11_S9_mode_summary.csv", mode_summary)
    write_csv(out / "D11_S9_mode_split_summary.csv", mode_split_summary)
    write_csv(out / "D11_S9_mode_segment_summary.csv", mode_segment_summary)
    write_csv(out / "D11_S9_mode_split_segment_summary.csv", mode_split_segment_summary)
    write_csv(out / "D11_S9_component_summary.csv", comp_summary)
    write_csv(out / "D11_S9_global_vs_lowtarget_tradeoff.csv", trade_rows)
    write_csv(out / "D11_S9_candidate_decisions.csv", decisions)

    summary = {
        "ok": True, "stage": "D11-S9 trainable localized P2D-like correction scorecard",
        "prediction_root": str(root), "out_dir": str(out), "run_count": len(run_rows),
        "expected_min_run_count": 6 * 5, "counts": {"strict_completed_metrics_ok": len(run_rows)},
        "mode_summary": mode_summary, "decision_split": split_for_decision or "all", "candidate_decisions": decisions,
        "promoted_candidates": promoted, "verdict": "d11_s9_scorecard_completed",
        "next_action": "promote_to_200ks_only_if_candidate_exists_else_redesign_with_true_trainable_model_or_protocol_specific_adapter",
    }
    json_dump(out / "D11_S9_scorecard_summary.json", summary)
    with (out / "D11_S9_RECOMMENDATION.md").open("w", encoding="utf-8") as f:
        f.write("# D11-S9 trainable localized P2D-like correction recommendation\n\n")
        f.write(f"- Run count: `{len(run_rows)}`\n")
        f.write(f"- Decision split: `{split_for_decision or 'all'}`\n")
        f.write(f"- Promoted candidates: `{promoted}`\n\n")
        f.write("## Candidate decisions\n\n")
        f.write("| mode | low_ok | global_ok | corr_ok | rest_ok | high_ok | promote | d_all | d_low | d_deep |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for d in decisions:
            f.write(f"| {d['mode']} | {d['low_ok']} | {d['global_ok']} | {d['corr_ok']} | {d['rest_ok']} | {d['high_ok']} | {d['promote_candidate']} | {d['delta_all_MAE_V']} | {d['delta_low_target_MAE_V']} | {d['delta_low_target_le_2p75_MAE_V']} |\n")
        f.write("\n## Decision rule\n\n")
        f.write("Promote only if low_target and low_target_le_2p75 MAE both drop by at least 20 mV while all/rest/high-target metrics remain stable.\n\n")
        if promoted:
            f.write("## Next action\n\nRun 6-profile 200ks confirmation for the promoted candidate(s), still excluding battery-8 and keeping metadata_on disabled.\n")
        else:
            f.write("## Next action\n\nNo candidate qualifies.  Do not expand to 200ks; redesign the localized correction with protocol-specific adapter or train inside the voltage model rather than post-hoc ridge.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
