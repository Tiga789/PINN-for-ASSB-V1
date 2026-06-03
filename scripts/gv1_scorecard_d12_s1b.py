#!/usr/bin/env python
"""Build D12-S1B preservation-tightened 6-profile scorecard from prediction.npz files.

The script intentionally computes metrics directly from prediction.npz to avoid
older collector read-error issues.  It produces:

- D12_S1B_run_metrics.csv
- D12_S1B_segment_metrics.csv
- D12_S1B_mode_summary.csv
- D12_S1B_candidate_decisions.csv
- D12_S1B_scorecard_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _as1(x: Any) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _safe_corr(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3:
        return float("nan")
    yy = y - float(np.nanmean(y))
    pp = p - float(np.nanmean(p))
    den = math.sqrt(float(np.nanmean(yy * yy)) * float(np.nanmean(pp * pp)))
    if not np.isfinite(den) or den <= 1e-12:
        return float("nan")
    return float(np.nanmean(yy * pp) / den)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
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


def _parse_run(path: Path) -> tuple[str, str]:
    name = path.parent.name
    # Expected: <mode>__<profile>; fallback robustly.
    if "__" in name:
        mode, profile = name.split("__", 1)
    else:
        parts = name.split("_Batch-")
        if len(parts) == 2:
            mode, profile = parts[0], "Batch-" + parts[1]
        else:
            mode, profile = name, name
    return mode, profile


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _segment_masks(v: np.ndarray, pred: np.ndarray, current: np.ndarray) -> dict[str, np.ndarray]:
    n = len(v)
    finite = np.isfinite(v) & np.isfinite(pred)
    return {
        "all": finite,
        "low_target": finite & (v <= 3.00),
        "low_target_le_2p75": finite & (v <= 2.75),
        "normal_target_gt_3p20": finite & (v > 3.20),
        "high_target_ge_4p10": finite & (v >= 4.10),
        "rest_I_zero": finite & (np.abs(current) <= 1e-10),
        "charge_I_positive": finite & (current > 1e-10),
        "discharge_I_negative": finite & (current < -1e-10),
        "pred_high_overshoot_gt_4p35": finite & (pred > 4.35),
        "target_mid_3p0_4p1": finite & (v > 3.00) & (v < 4.10),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--baseline_mode", default="baseline_d951")
    ap.add_argument("--min_low_improve_V", type=float, default=0.020)
    ap.add_argument("--max_global_regress_V", type=float, default=0.005)
    ap.add_argument("--max_corr_drop", type=float, default=0.005)
    ap.add_argument("--max_rest_regress_V", type=float, default=0.020)
    ap.add_argument("--max_high_regress_V", type=float, default=0.020)
    ap.add_argument("--max_normal_regress_V", type=float, default=0.005)
    args = ap.parse_args()

    root = Path(args.runs_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "D12_S1B_scorecard"
    pred_files = sorted(root.rglob("prediction.npz"))
    run_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []

    for pred_path in pred_files:
        try:
            arrays = _load_prediction(pred_path)
            y = _as1(arrays.get("voltage_exp"))
            p = _as1(arrays.get("voltage_exp_pred", arrays.get("phis_c_pred")))
            cur = _as1(arrays.get("I_profile", np.zeros_like(y)))
            if len(y) != len(p):
                n = min(len(y), len(p))
                y = y[:n]
                p = p[:n]
                cur = cur[:n]
            mode, profile = _parse_run(pred_path)
            m_all = _metrics(y, p)
            p2d = arrays.get("voltage_p2d_transport_deficit")
            p2d_mean = float(np.nanmean(_as1(p2d))) if p2d is not None and len(_as1(p2d)) else float("nan")
            run_rows.append(
                {
                    "mode": mode,
                    "profile": profile,
                    "prediction_npz": str(pred_path),
                    "p2d_deficit_mean_V": p2d_mean,
                    **m_all,
                    "status": "metrics_ok",
                }
            )
            for seg, mask in _segment_masks(y, p, cur).items():
                mm = _metrics(y[mask], p[mask]) if mask.any() else _metrics(np.asarray([]), np.asarray([]))
                seg_rows.append({"mode": mode, "profile": profile, "segment": seg, **mm})
        except Exception as exc:  # noqa: BLE001
            mode, profile = _parse_run(pred_path)
            run_rows.append(
                {
                    "mode": mode,
                    "profile": profile,
                    "prediction_npz": str(pred_path),
                    "status": "read_error",
                    "error": repr(exc),
                }
            )

    # Mode summaries.
    mode_rows: list[dict[str, Any]] = []
    modes = sorted({r["mode"] for r in run_rows})
    for mode in modes:
        rows = [r for r in run_rows if r.get("mode") == mode and r.get("status") == "metrics_ok"]
        if not rows:
            mode_rows.append({"mode": mode, "n": 0, "ok": 0})
            continue
        for key in ["MAE_V", "RMSE_V", "corr", "bias_V", "p2d_deficit_mean_V"]:
            vals = np.asarray([float(r.get(key, np.nan)) for r in rows], dtype=float)
            mode_rows.append(
                {
                    "mode": mode,
                    "metric": key,
                    "n": int(len(vals)),
                    "mean": float(np.nanmean(vals)),
                    "median": float(np.nanmedian(vals)),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                }
            )

    # Candidate decisions compared with baseline per profile.
    base_mode = str(args.baseline_mode)
    by_profile_segment: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in seg_rows:
        by_profile_segment[(r["mode"], r["profile"], r["segment"])] = r
    decisions: list[dict[str, Any]] = []
    for mode in modes:
        if mode == base_mode:
            continue
        profiles = sorted({r["profile"] for r in run_rows if r.get("mode") == mode})
        per_profile = []
        for prof in profiles:
            b_all = by_profile_segment.get((base_mode, prof, "all"))
            c_all = by_profile_segment.get((mode, prof, "all"))
            b_low = by_profile_segment.get((base_mode, prof, "low_target"))
            c_low = by_profile_segment.get((mode, prof, "low_target"))
            b_deep = by_profile_segment.get((base_mode, prof, "low_target_le_2p75"))
            c_deep = by_profile_segment.get((mode, prof, "low_target_le_2p75"))
            b_rest = by_profile_segment.get((base_mode, prof, "rest_I_zero"))
            c_rest = by_profile_segment.get((mode, prof, "rest_I_zero"))
            b_high = by_profile_segment.get((base_mode, prof, "high_target_ge_4p10"))
            c_high = by_profile_segment.get((mode, prof, "high_target_ge_4p10"))
            b_normal = by_profile_segment.get((base_mode, prof, "normal_target_gt_3p20"))
            c_normal = by_profile_segment.get((mode, prof, "normal_target_gt_3p20"))
            if not all([b_all, c_all, b_low, c_low]):
                continue
            delta_all = float(c_all["MAE_V"]) - float(b_all["MAE_V"])
            delta_low = float(c_low["MAE_V"]) - float(b_low["MAE_V"])
            delta_deep = float(c_deep["MAE_V"]) - float(b_deep["MAE_V"]) if b_deep and c_deep and int(b_deep["n"]) > 0 else float("nan")
            delta_rest = float(c_rest["MAE_V"]) - float(b_rest["MAE_V"]) if b_rest and c_rest and int(b_rest["n"]) > 0 else 0.0
            delta_high = float(c_high["MAE_V"]) - float(b_high["MAE_V"]) if b_high and c_high and int(b_high["n"]) > 0 else 0.0
            delta_normal = float(c_normal["MAE_V"]) - float(b_normal["MAE_V"]) if b_normal and c_normal and int(b_normal["n"]) > 0 else 0.0
            delta_corr = float(c_all["corr"]) - float(b_all["corr"])
            per_profile.append(
                {
                    "delta_all": delta_all,
                    "delta_low": delta_low,
                    "delta_deep": delta_deep,
                    "delta_rest": delta_rest,
                    "delta_high": delta_high,
                    "delta_normal": delta_normal,
                    "delta_corr": delta_corr,
                    "deep_n": int(c_deep["n"]) if c_deep else 0,
                }
            )
        if not per_profile:
            continue
        mean = lambda k: float(np.nanmean([x[k] for x in per_profile]))  # noqa: E731
        deep_available = any(x["deep_n"] > 0 for x in per_profile)
        low_ok = mean("delta_low") <= -float(args.min_low_improve_V)
        deep_ok = (not deep_available) or (mean("delta_deep") <= -float(args.min_low_improve_V))
        global_ok = mean("delta_all") <= float(args.max_global_regress_V)
        corr_ok = mean("delta_corr") >= -float(args.max_corr_drop)
        rest_ok = mean("delta_rest") <= float(args.max_rest_regress_V)
        high_ok = mean("delta_high") <= float(args.max_high_regress_V)
        normal_ok = mean("delta_normal") <= float(args.max_normal_regress_V)
        promote = bool(low_ok and deep_ok and global_ok and corr_ok and rest_ok and high_ok and normal_ok)
        decisions.append(
            {
                "mode": mode,
                "profile_count": len(per_profile),
                "delta_all_MAE_V": mean("delta_all"),
                "delta_low_target_MAE_V": mean("delta_low"),
                "delta_low_le_2p75_MAE_V": mean("delta_deep"),
                "delta_rest_MAE_V": mean("delta_rest"),
                "delta_high_MAE_V": mean("delta_high"),
                "delta_normal_MAE_V": mean("delta_normal"),
                "delta_corr": mean("delta_corr"),
                "low_ok": low_ok,
                "deep_ok": deep_ok,
                "global_ok": global_ok,
                "corr_ok": corr_ok,
                "rest_ok": rest_ok,
                "high_ok": high_ok,
                "normal_ok": normal_ok,
                "promote_to_200ks": promote,
            }
        )

    _write_csv(out_dir / "D12_S1B_run_metrics.csv", run_rows)
    _write_csv(out_dir / "D12_S1B_segment_metrics.csv", seg_rows)
    _write_csv(out_dir / "D12_S1B_mode_summary.csv", mode_rows)
    _write_csv(out_dir / "D12_S1B_candidate_decisions.csv", decisions)
    summary = {
        "ok": True,
        "runs_root": str(root),
        "output_dir": str(out_dir),
        "prediction_count": len(pred_files),
        "metrics_ok_count": sum(1 for r in run_rows if r.get("status") == "metrics_ok"),
        "read_error_count": sum(1 for r in run_rows if r.get("status") == "read_error"),
        "baseline_mode": base_mode,
        "promoted_candidates": [d["mode"] for d in decisions if d.get("promote_to_200ks")],
        "decision_rule": {
            "low_and_deep_MAE_improve_at_least_V": float(args.min_low_improve_V),
            "global_MAE_regress_no_more_than_V": float(args.max_global_regress_V),
            "corr_drop_no_more_than": float(args.max_corr_drop),
            "rest_high_regress_no_more_than_V": float(args.max_rest_regress_V),
            "normal_regress_no_more_than_V": float(args.max_normal_regress_V),
        },
    }
    (out_dir / "D12_S1B_scorecard_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
