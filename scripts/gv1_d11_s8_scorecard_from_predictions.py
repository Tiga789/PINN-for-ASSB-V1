#!/usr/bin/env python
"""Collect D11-S8 P2D-like transport correction scorecard from prediction.npz files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _as_1d(x):
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim > 1:
        return x.reshape(-1)
    return x


def _pick(data: Dict[str, np.ndarray], candidates) -> Optional[str]:
    for k in candidates:
        if k in data:
            return k
    lower = {k.lower(): k for k in data}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _metrics(y, yp):
    y = _as_1d(y).astype(float); yp = _as_1d(yp).astype(float)
    n = min(y.size, yp.size)
    y = y[:n]; yp = yp[:n]
    m = np.isfinite(y) & np.isfinite(yp)
    if not np.any(m):
        return dict(n=0, MAE_V=np.nan, RMSE_V=np.nan, corr=np.nan, bias_V=np.nan, pred_over_frac=np.nan, pred_under_frac=np.nan, target_min_V=np.nan, target_max_V=np.nan, pred_min_V=np.nan, pred_max_V=np.nan)
    y = y[m]; yp = yp[m]
    err = yp - y
    corr = np.nan
    if y.size >= 2 and np.nanstd(y) > 0 and np.nanstd(yp) > 0:
        corr = float(np.corrcoef(y, yp)[0,1])
    return dict(
        n=int(y.size), MAE_V=float(np.mean(np.abs(err))), RMSE_V=float(np.sqrt(np.mean(err**2))), corr=corr,
        bias_V=float(np.mean(err)), pred_over_frac=float(np.mean(err > 0)), pred_under_frac=float(np.mean(err < 0)),
        target_min_V=float(np.min(y)), target_max_V=float(np.max(y)), pred_min_V=float(np.min(yp)), pred_max_V=float(np.max(yp))
    )


def _segment_masks(y, i):
    n = y.size
    masks = {"all": np.ones(n, dtype=bool)}
    if i is not None and i.size == n:
        thr = max(1e-12, np.nanpercentile(np.abs(i), 5) * 0.1)
        masks["charge_I_positive"] = i > thr
        masks["discharge_I_negative"] = i < -thr
        masks["rest_I_zero"] = np.abs(i) <= thr
    idx = np.arange(n)
    masks["early_time_third"] = idx < n/3
    masks["middle_time_third"] = (idx >= n/3) & (idx < 2*n/3)
    masks["late_time_third"] = idx >= 2*n/3
    masks["low_target"] = y <= 2.90
    masks["low_target_le_2p75"] = y <= 2.75
    masks["high_target_ge_4p10"] = y >= 4.10
    return masks


def _infer_profile(path: Path, data: Dict[str, np.ndarray]) -> str:
    if "d11_s8_profile" in data:
        try: return str(np.asarray(data["d11_s8_profile"]).item())
        except Exception: pass
    return path.parent.name


def _infer_protocol(path: Path, data: Dict[str, np.ndarray]) -> str:
    if "d11_s8_protocol" in data:
        try: return str(np.asarray(data["d11_s8_protocol"]).item())
        except Exception: pass
    s = str(path).lower()
    if "r2.5" in s or "r25" in s or "batch-3" in s: return "R2.5"
    if "r3" in s or "batch-4" in s: return "R3"
    if "2c" in s or "batch-1" in s: return "2C"
    return "unknown"


def _infer_mode(path: Path, data: Dict[str, np.ndarray]) -> str:
    if "d11_s8_mode" in data:
        try: return str(np.asarray(data["d11_s8_mode"]).item())
        except Exception: pass
    return path.parent.parent.name


def _mean_group(rows, group_keys, value_keys):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_rows"] = len(rs)
        for v in value_keys:
            arr = np.array([float(x.get(v, np.nan)) for x in rs], dtype=float)
            row[f"mean_{v}"] = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_root", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s8_p2dlike_transport_correction")
    ap.add_argument("--out_dir", default=r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s8_p2dlike_transport_correction_scorecard")
    args = ap.parse_args()
    pred_root = Path(args.prediction_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    seg_rows = []
    comp_rows = []
    preds = sorted(pred_root.rglob("prediction.npz"))
    for p in preds:
        z = np.load(p, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        pred_key = _pick(data, ["voltage_exp_pred", "voltage_pred", "phis_c_pred", "pred_voltage", "y_pred", "pred"])
        true_key = _pick(data, ["voltage_exp_true", "voltage_true", "target_voltage", "voltage_exp", "y_true", "true"])
        curr_key = _pick(data, ["I_profile", "current_A", "current", "I", "current_profile"])
        if pred_key is None or true_key is None:
            continue
        y = _as_1d(data[true_key]).astype(float)
        yp = _as_1d(data[pred_key]).astype(float)
        n = min(y.size, yp.size); y = y[:n]; yp = yp[:n]
        i = _as_1d(data[curr_key]).astype(float)[:n] if curr_key is not None else None
        mode = _infer_mode(p, data); profile = _infer_profile(p, data); protocol = _infer_protocol(p, data)
        base = dict(mode=mode, profile=profile, protocol=protocol, status="strict_completed_metrics_ok", prediction_path=str(p))
        mr = {**base, **_metrics(y, yp)}; run_rows.append(mr)
        for seg, mask in _segment_masks(y, i).items():
            mask = mask[:n]
            if not np.any(mask):
                continue
            seg_rows.append({**base, "segment": seg, **_metrics(y[mask], yp[mask])})
        # component diagnostics if available
        comp = {**base}
        for k in ["d11_s8_transport_deficit_V", "d11_s8_low_gate", "d11_s8_discharge_gate", "d11_s8_current_gate", "d11_s8_capacity_gate", "d11_s8_high_voltage_guard"]:
            if k in data:
                arr = _as_1d(data[k]).astype(float)[:n]
                comp[f"{k}_mean"] = float(np.nanmean(arr))
                comp[f"{k}_max"] = float(np.nanmax(arr))
        comp_rows.append(comp)

    value_keys = ["MAE_V", "RMSE_V", "corr", "bias_V", "pred_over_frac", "pred_under_frac", "n"]
    mode_summary = _mean_group(run_rows, ["mode"], value_keys)
    mode_segment_summary = _mean_group(seg_rows, ["mode", "segment"], value_keys)
    mode_protocol_summary = _mean_group(run_rows, ["mode", "protocol"], value_keys)

    # tradeoff relative to baseline_copy if present, else baseline_d951
    baseline_mode = "baseline_copy" if any(r["mode"] == "baseline_copy" for r in mode_segment_summary) else "baseline_d951"
    baseline_by_seg = {r["segment"]: r for r in mode_segment_summary if r["mode"] == baseline_mode}
    trade = []
    for r in mode_segment_summary:
        if r["mode"] == baseline_mode: continue
        b = baseline_by_seg.get(r["segment"])
        if not b: continue
        trade.append({
            "candidate": r["mode"], "segment": r["segment"],
            "candidate_mean_MAE_V": r.get("mean_MAE_V"), "baseline_mean_MAE_V": b.get("mean_MAE_V"),
            "candidate_minus_baseline_MAE_V": (r.get("mean_MAE_V") - b.get("mean_MAE_V")) if np.isfinite(r.get("mean_MAE_V", np.nan)) and np.isfinite(b.get("mean_MAE_V", np.nan)) else np.nan,
            "candidate_mean_corr": r.get("mean_corr"), "baseline_mean_corr": b.get("mean_corr"),
            "candidate_minus_baseline_corr": (r.get("mean_corr") - b.get("mean_corr")) if np.isfinite(r.get("mean_corr", np.nan)) and np.isfinite(b.get("mean_corr", np.nan)) else np.nan,
            "candidate_mean_bias_V": r.get("mean_bias_V"), "baseline_mean_bias_V": b.get("mean_bias_V"),
            "candidate_minus_baseline_bias_V": (r.get("mean_bias_V") - b.get("mean_bias_V")) if np.isfinite(r.get("mean_bias_V", np.nan)) and np.isfinite(b.get("mean_bias_V", np.nan)) else np.nan,
        })

    decisions = []
    for mode in sorted({r["mode"] for r in run_rows if r["mode"] != baseline_mode}):
        d = {"mode": mode}
        segd = {x["segment"]: x for x in trade if x["candidate"] == mode}
        def delta(seg, key="candidate_minus_baseline_MAE_V"):
            return segd.get(seg, {}).get(key, np.nan)
        d["low_ok"] = (delta("low_target") <= -0.02) and (delta("low_target_le_2p75") <= -0.02)
        d["global_ok"] = delta("all") <= 0.01
        d["corr_ok"] = delta("all", "candidate_minus_baseline_corr") >= -0.01
        d["rest_ok"] = delta("rest_I_zero") <= 0.03 or not np.isfinite(delta("rest_I_zero"))
        d["high_ok"] = delta("high_target_ge_4p10") <= 0.03 or not np.isfinite(delta("high_target_ge_4p10"))
        d["promote_candidate"] = bool(d["low_ok"] and d["global_ok"] and d["corr_ok"] and d["rest_ok"] and d["high_ok"])
        decisions.append(d)

    def write_csv(name, rows):
        path = out_dir / name
        if rows:
            keys = sorted(set().union(*[r.keys() for r in rows]))
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(rows)
        else:
            path.write_text("", encoding="utf-8")

    write_csv("D11_S8_run_metrics.csv", run_rows)
    write_csv("D11_S8_segment_metrics.csv", seg_rows)
    write_csv("D11_S8_mode_summary.csv", mode_summary)
    write_csv("D11_S8_mode_segment_summary.csv", mode_segment_summary)
    write_csv("D11_S8_mode_protocol_summary.csv", mode_protocol_summary)
    write_csv("D11_S8_global_vs_lowtarget_tradeoff.csv", trade)
    write_csv("D11_S8_component_summary.csv", comp_rows)
    write_csv("D11_S8_candidate_decisions.csv", decisions)

    summary = {
        "ok": True,
        "stage": "D11-S8 P2D-like low-voltage transport deficit correction scorecard",
        "prediction_root": str(pred_root),
        "out_dir": str(out_dir),
        "run_count": len(run_rows),
        "expected_run_count": 30,  # 6 profiles x 5 modes
        "counts": {"strict_completed_metrics_ok": len(run_rows)},
        "mode_summary": mode_summary,
        "candidate_decisions": decisions,
        "promoted_candidates": [d["mode"] for d in decisions if d.get("promote_candidate")],
        "verdict": "d11_s8_scorecard_completed",
        "next_action": "promote_only_if_low_target_improves_else_redesign_protocol_specific_or_p2d_like_trainable_head",
    }
    (out_dir / "D11_S8_scorecard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# D11-S8 P2D-like transport deficit correction recommendation", "", f"- Run count: `{len(run_rows)}`", f"- Promoted candidates: `{summary['promoted_candidates']}`", "", "## Candidate decisions", "", "| mode | low_ok | global_ok | corr_ok | rest_ok | high_ok | promote |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in decisions:
        md.append(f"| {d['mode']} | {d['low_ok']} | {d['global_ok']} | {d['corr_ok']} | {d['rest_ok']} | {d['high_ok']} | {d['promote_candidate']} |")
    md += ["", "## Decision rule", "", "Promote only if low_target and low_target_le_2p75 MAE both drop by at least 20 mV while all/rest/high-target metrics remain stable.", "", "## Files", "", "- D11_S8_mode_segment_summary.csv", "- D11_S8_global_vs_lowtarget_tradeoff.csv", "- D11_S8_component_summary.csv", "- D11_S8_candidate_decisions.csv"]
    (out_dir / "D11_S8_RECOMMENDATION.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
