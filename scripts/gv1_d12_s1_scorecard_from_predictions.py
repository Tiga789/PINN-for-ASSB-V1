#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build D12-S1 metadata ablation scorecard directly from prediction.npz.

This collector intentionally does not depend on d10_voltage_metrics.json. It
computes voltage metrics directly from each completed run's prediction.npz and
then writes per-run, per-segment, and per-mode summaries.

It does not launch training and does not modify D9.6/D9.5.1 source files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def to_float(x: Any, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def corrcoef(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 3:
        return math.nan
    yy = y[mask].astype(float)
    pp = p[mask].astype(float)
    if float(np.nanstd(yy)) <= 1e-12 or float(np.nanstd(pp)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(yy, pp)[0, 1])


def metrics_for(label: str, y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = mask & np.isfinite(y) & np.isfinite(p)
    n = int(mask.sum())
    if n <= 0:
        return {"label": label, "n": 0}
    err = p[mask] - y[mask]
    return {
        "label": label,
        "n": n,
        "mae_V": float(np.nanmean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.nanmean(err ** 2))),
        "bias_V": float(np.nanmean(err)),
        "corr": corrcoef(y[mask], p[mask]),
        "target_min_V": float(np.nanmin(y[mask])),
        "target_max_V": float(np.nanmax(y[mask])),
        "pred_min_V": float(np.nanmin(p[mask])),
        "pred_max_V": float(np.nanmax(p[mask])),
        "pred_upper_frac_ge_4p269": float(np.nanmean(p[mask] >= 4.269)),
        "pred_overshoot_frac_gt_4p35": float(np.nanmean(p[mask] > 4.35)),
        "pred_low_frac_le_2p75": float(np.nanmean(p[mask] <= 2.75)),
        "target_low_frac_le_2p75": float(np.nanmean(y[mask] <= 2.75)),
    }


def find_key(keys: set[str], candidates: list[str]) -> str | None:
    for key in candidates:
        if key in keys:
            return key
    return None


def compute_prediction_metrics(prediction_npz: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with np.load(prediction_npz, allow_pickle=True) as z:
        keys = set(z.files)
        y_key = find_key(keys, ["voltage_exp", "target_voltage", "voltage_true", "y_voltage"])
        p_key = find_key(keys, ["voltage_exp_pred", "phis_c_pred", "voltage_pred", "pred_voltage"])
        if y_key is None:
            raise KeyError(f"No target voltage key found in {prediction_npz}; keys={sorted(keys)}")
        if p_key is None:
            raise KeyError(f"No prediction voltage key found in {prediction_npz}; keys={sorted(keys)}")
        y = np.asarray(z[y_key], dtype=float).reshape(-1)
        p = np.asarray(z[p_key], dtype=float).reshape(-1)
        if "I_profile" in keys:
            I = np.asarray(z["I_profile"], dtype=float).reshape(-1)
        elif "current_A" in keys:
            I = np.asarray(z["current_A"], dtype=float).reshape(-1)
        else:
            I = np.zeros_like(y)
        if "t_global_s" in keys:
            t = np.asarray(z["t_global_s"], dtype=float).reshape(-1)
        elif "t_s" in keys:
            t = np.asarray(z["t_s"], dtype=float).reshape(-1)
        else:
            t = np.arange(len(y), dtype=float)

    n = min(len(y), len(p), len(I), len(t))
    y, p, I, t = y[:n], p[:n], I[:n], t[:n]
    finite = np.isfinite(y) & np.isfinite(p)
    abs_i = np.abs(I[np.isfinite(I)])
    high_i = float(np.nanquantile(abs_i, 0.90)) if len(abs_i) else 0.0
    eps = 1e-8
    masks = {
        "all": np.ones(n, dtype=bool),
        "charge_I_pos": I > eps,
        "discharge_I_neg": I < -eps,
        "rest_I_zero": np.abs(I) <= eps,
        "low_target": y <= 2.75,
        "high_target": y >= 4.10,
        "mid_target": (y > 2.75) & (y < 4.10),
        "high_current_abs": np.abs(I) >= max(high_i, eps),
    }
    rows = [metrics_for(name, y, p, mask & finite) for name, mask in masks.items()]
    payload = {
        "ok": True,
        "stage": "D12-S1 voltage metrics computed from prediction.npz",
        "prediction_npz": str(prediction_npz),
        "n": int(n),
        "target_key": y_key,
        "prediction_key": p_key,
        "current_high_quantile_threshold_A": high_i,
        "metrics": {row["label"]: row for row in rows},
    }
    return payload, rows


def mode_from_name(name: str) -> str:
    match = re.search(r"d12_s1_metadata_(off|zero|on)_", name)
    if match:
        return match.group(1)
    match = re.search(r"d12_runtime_metadata_(off|zero|on)_", name)
    return match.group(1) if match else "unknown"


def profile_from_name(name: str) -> str:
    # New name: xjtu_batch134_d12_s1_metadata_on_Batch-1_2C_battery-1_TRUE_SMOKE_40ks_e100
    match = re.search(r"metadata_(?:off|zero|on)_(.*?)_TRUE_SMOKE", name)
    return match.group(1) if match else ""


def find_prediction(run_dir: Path) -> Path | None:
    direct = run_dir / "prediction.npz"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.rglob("prediction.npz"))
    return candidates[0] if candidates else None


def small_listing(run_dir: Path) -> str:
    try:
        items = sorted(run_dir.iterdir(), key=lambda p: p.name)[:50]
        return "; ".join(f"{p.name}{'/' if p.is_dir() else ''}" for p in items)
    except Exception as exc:
        return f"listing_failed: {exc}"


def infer_metadata_dim(mode: str, run_dir: Path) -> int | str:
    if mode == "off":
        return 0
    # Try metadata runtime file first.
    for name in ["d12_metadata_runtime_summary.json", "metadata_runtime_summary.json"]:
        path = run_dir / name
        if path.exists():
            try:
                data = read_json(path)
                md = data.get("metadata", data.get("d12_metadata_runtime", data))
                if "metadata_dim" in md:
                    return int(md["metadata_dim"])
            except Exception:
                pass
    # Then infer from training summary condition_dim or condition_vector length.
    for path in list(run_dir.glob("*.json")):
        try:
            data = read_json(path)
        except Exception:
            continue
        mc = data.get("model_config", {}) if isinstance(data, dict) else {}
        if "condition_dim" in mc:
            return max(0, int(mc["condition_dim"]) - 8)
        ds = data.get("dataset", {}) if isinstance(data, dict) else {}
        cv = ds.get("condition_vector")
        if isinstance(cv, list):
            return max(0, len(cv) - 8)
    return ""


def status_from_row(row: dict[str, Any]) -> str:
    if row.get("metrics_source") == "computed_from_prediction":
        mae = to_float(row.get("mae_V"))
        corr = to_float(row.get("corr"))
        if np.isfinite(mae) and np.isfinite(corr):
            return "smoke_completed_metrics_ok"
        if np.isfinite(mae):
            return "smoke_completed_metrics_review"
        return "metrics_nan_review"
    return str(row.get("status", "read_error"))


def mean(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.nanmean(finite)) if finite else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--include_legacy_true_smoke", action="store_true", help="Also scan older xjtu_batch134_d12_runtime_metadata_*TRUE_SMOKE dirs.")
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d12_s1_metadata_ablation_scorecard"

    run_dirs = sorted([
        p for p in cache_root.iterdir()
        if p.is_dir() and p.name.startswith("xjtu_batch134_d12_s1_metadata_") and "TRUE_SMOKE" in p.name
    ])
    if args.include_legacy_true_smoke:
        run_dirs += sorted([
            p for p in cache_root.iterdir()
            if p.is_dir() and p.name.startswith("xjtu_batch134_d12_runtime_metadata_") and "TRUE_SMOKE" in p.name
        ])

    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        mode = mode_from_name(run_dir.name)
        row: dict[str, Any] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "mode": mode,
            "metadata_profile_id": profile_from_name(run_dir.name),
            "metadata_dim": infer_metadata_dim(mode, run_dir),
        }
        prediction = find_prediction(run_dir)
        if prediction is None:
            row.update({
                "status": "missing_prediction",
                "error": "No prediction.npz found in run directory.",
                "file_listing": small_listing(run_dir),
            })
        else:
            try:
                metrics_payload, segs = compute_prediction_metrics(prediction)
                write_json(run_dir / "d12_s1_voltage_metrics.json", metrics_payload)
                write_csv(run_dir / "d12_s1_voltage_metrics_by_segment.csv", segs)
                all_metrics = metrics_payload["metrics"].get("all", {})
                row.update(all_metrics)
                row["prediction_npz"] = str(prediction)
                row["metrics_source"] = "computed_from_prediction"
                for seg in segs:
                    rr = dict(seg)
                    rr.update({"run_name": run_dir.name, "mode": mode, "metadata_profile_id": row["metadata_profile_id"]})
                    segment_rows.append(rr)
            except Exception as exc:
                row.update({
                    "status": "metrics_compute_error",
                    "error": str(exc),
                    "file_listing": small_listing(run_dir),
                })
        row["status"] = status_from_row(row)
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1

    mode_rows: list[dict[str, Any]] = []
    for mode in sorted(set(str(row.get("mode")) for row in rows)):
        group = [row for row in rows if str(row.get("mode")) == mode]
        ok_group = [row for row in group if row.get("status") == "smoke_completed_metrics_ok"]
        mode_rows.append({
            "mode": mode,
            "n": len(group),
            "ok": len(ok_group),
            "mean_mae_V": mean([to_float(row.get("mae_V")) for row in ok_group]),
            "mean_rmse_V": mean([to_float(row.get("rmse_V")) for row in ok_group]),
            "mean_corr": mean([to_float(row.get("corr")) for row in ok_group]),
            "mean_bias_V": mean([to_float(row.get("bias_V")) for row in ok_group]),
        })

    completed = counts.get("smoke_completed_metrics_ok", 0) + counts.get("smoke_completed_metrics_review", 0)
    if not rows:
        verdict = "d12_s1_no_run_dirs_found"
    elif completed == len(rows):
        verdict = "d12_s1_all_runs_completed_metrics_ok"
    elif completed > 0:
        verdict = "d12_s1_partial_runs_completed_review_missing"
    else:
        verdict = "d12_s1_no_completed_predictions"

    summary = {
        "ok": True,
        "stage": "D12-S1 metadata ablation scorecard from predictions",
        "run_count": len(rows),
        "counts": counts,
        "mode_summary": mode_rows,
        "mean_mae_V": mean([to_float(row.get("mae_V")) for row in rows]),
        "mean_corr": mean([to_float(row.get("corr")) for row in rows]),
        "out_dir": str(out_dir),
        "verdict": verdict,
        "note": "No training launched. Metrics computed directly from existing prediction.npz files.",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "d12_s1_scorecard.csv", rows)
    write_csv(out_dir / "d12_s1_segment_metrics.csv", segment_rows)
    write_csv(out_dir / "d12_s1_mode_summary.csv", mode_rows)
    write_json(out_dir / "d12_s1_scorecard_summary.json", summary)
    md_lines = [
        "# D12-S1 metadata ablation scorecard",
        "",
        "## Verdict",
        "",
        f"```text\n{verdict}\n```",
        "",
        "## Counts",
        "",
        "```json",
        json.dumps(counts, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Mode summary",
        "",
        "| mode | n | ok | mean_MAE_V | mean_corr | mean_bias_V |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in mode_rows:
        md_lines.append(
            f"| {row['mode']} | {row['n']} | {row['ok']} | {row['mean_mae_V']} | {row['mean_corr']} | {row['mean_bias_V']} |"
        )
    (out_dir / "D12_S1_SCORECARD_RECOMMENDATION.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
