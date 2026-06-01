#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build D12-S3 metadata ablation scorecard directly from prediction.npz.

The collector does not depend on existing metrics JSON files. It scans D12-S3
run directories, computes voltage metrics from prediction.npz, and writes
per-run, per-segment, per-mode and per-protocol summaries.
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


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
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
            current = np.asarray(z["I_profile"], dtype=float).reshape(-1)
        elif "current_A" in keys:
            current = np.asarray(z["current_A"], dtype=float).reshape(-1)
        else:
            current = np.zeros_like(y)
        if "t_global_s" in keys:
            time = np.asarray(z["t_global_s"], dtype=float).reshape(-1)
        elif "t_s" in keys:
            time = np.asarray(z["t_s"], dtype=float).reshape(-1)
        else:
            time = np.arange(len(y), dtype=float)
    n = min(len(y), len(p), len(current), len(time))
    y, p, current, time = y[:n], p[:n], current[:n], time[:n]
    finite = np.isfinite(y) & np.isfinite(p)
    abs_i = np.abs(current[np.isfinite(current)])
    high_i = float(np.nanquantile(abs_i, 0.90)) if len(abs_i) else 0.0
    eps = 1e-8
    masks = {
        "all": np.ones(n, dtype=bool),
        "charge_I_pos": current > eps,
        "discharge_I_neg": current < -eps,
        "rest_I_zero": np.abs(current) <= eps,
        "low_target": y <= 2.75,
        "high_target": y >= 4.10,
        "mid_target": (y > 2.75) & (y < 4.10),
        "high_current_abs": np.abs(current) >= max(high_i, eps),
    }
    rows = [metrics_for(name, y, p, mask & finite) for name, mask in masks.items()]
    payload = {
        "ok": True,
        "stage": "D12-S3 voltage metrics computed from prediction.npz",
        "prediction_npz": str(prediction_npz),
        "n": int(n),
        "target_key": y_key,
        "prediction_key": p_key,
        "current_high_quantile_threshold_A": high_i,
        "metrics": {row["label"]: row for row in rows},
    }
    return payload, rows


def mode_from_name(name: str) -> str:
    match = re.search(r"d12_s3_metadata_(off|zero|on)_", name)
    if match:
        return match.group(1)
    match = re.search(r"d12_runtime_metadata_(off|zero|on)_", name)
    return match.group(1) if match else "unknown"


def profile_from_name(name: str) -> str:
    match = re.search(r"metadata_(?:off|zero|on)_(.*?)_STRICT_40ks", name)
    if match:
        return match.group(1)
    match = re.search(r"metadata_(?:off|zero|on)_(.*?)_TRUE_SMOKE", name)
    return match.group(1) if match else ""


def protocol_from_profile(profile: str) -> str:
    if "R2.5" in profile or "R25" in profile:
        return "R2.5"
    if "R3" in profile:
        return "R3"
    if "2C" in profile:
        return "2C"
    return "unknown"


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
    for path in list(run_dir.glob("*.json")):
        try:
            data = read_json(path)
        except Exception:
            continue
        model_config = data.get("model_config", {}) if isinstance(data, dict) else {}
        if "condition_dim" in model_config:
            return max(0, int(model_config["condition_dim"]) - 8)
        dataset = data.get("dataset", {}) if isinstance(data, dict) else {}
        condition_vector = dataset.get("condition_vector")
        if isinstance(condition_vector, list):
            return max(0, len(condition_vector) - 8)
    return ""


def status_from_row(row: dict[str, Any]) -> str:
    if row.get("metrics_source") == "computed_from_prediction":
        mae = to_float(row.get("mae_V"))
        corr = to_float(row.get("corr"))
        if np.isfinite(mae) and np.isfinite(corr):
            return "strict_completed_metrics_ok"
        if np.isfinite(mae):
            return "strict_completed_metrics_review"
        return "metrics_nan_review"
    return str(row.get("status", "read_error"))


def mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.nanmean(finite)) if finite else math.nan


def summarise(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted(set(str(row.get(group_key, "unknown")) for row in rows)):
        group = [row for row in rows if str(row.get(group_key, "unknown")) == key]
        ok_group = [row for row in group if row.get("status") == "strict_completed_metrics_ok"]
        out.append({
            group_key: key,
            "n": len(group),
            "ok": len(ok_group),
            "mean_mae_V": mean([to_float(row.get("mae_V")) for row in ok_group]),
            "mean_rmse_V": mean([to_float(row.get("rmse_V")) for row in ok_group]),
            "mean_corr": mean([to_float(row.get("corr")) for row in ok_group]),
            "mean_bias_V": mean([to_float(row.get("bias_V")) for row in ok_group]),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--expected_run_count", type=int, default=69)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d12_s3_metadata_ablation_scorecard"
    run_dirs = sorted([
        path for path in cache_root.iterdir()
        if path.is_dir()
        and path.name.startswith("xjtu_batch134_d12_s3_metadata_")
        and "STRICT_40ks" in path.name
    ])

    rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        mode = mode_from_name(run_dir.name)
        profile = profile_from_name(run_dir.name)
        row: dict[str, Any] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "mode": mode,
            "metadata_profile_id": profile,
            "protocol": protocol_from_profile(profile),
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
                metrics_payload, segment_metrics = compute_prediction_metrics(prediction)
                write_json(run_dir / "d12_s3_voltage_metrics.json", metrics_payload)
                write_csv(run_dir / "d12_s3_voltage_metrics_by_segment.csv", segment_metrics)
                all_metrics = metrics_payload["metrics"].get("all", {})
                row.update(all_metrics)
                row["prediction_npz"] = str(prediction)
                row["metrics_source"] = "computed_from_prediction"
                for segment in segment_metrics:
                    segment_row = dict(segment)
                    segment_row.update({
                        "run_name": run_dir.name,
                        "mode": mode,
                        "metadata_profile_id": profile,
                        "protocol": row["protocol"],
                    })
                    segment_rows.append(segment_row)
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
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    mode_summary = summarise(rows, "mode")
    protocol_summary = summarise(rows, "protocol")
    completed = counts.get("strict_completed_metrics_ok", 0) + counts.get("strict_completed_metrics_review", 0)
    if not rows:
        verdict = "d12_s3_no_run_dirs_found"
    elif len(rows) != args.expected_run_count:
        verdict = "d12_s3_run_count_mismatch_review"
    elif completed == len(rows):
        verdict = "d12_s3_all_runs_completed_metrics_ok"
    elif completed > 0:
        verdict = "d12_s3_partial_runs_completed_review_missing"
    else:
        verdict = "d12_s3_no_completed_predictions"

    summary = {
        "ok": True,
        "stage": "D12-S3 clean 23-profile metadata ablation scorecard from predictions",
        "run_count": len(rows),
        "expected_run_count": args.expected_run_count,
        "counts": counts,
        "mode_summary": mode_summary,
        "protocol_summary": protocol_summary,
        "mean_mae_V": mean([to_float(row.get("mae_V")) for row in rows]),
        "mean_corr": mean([to_float(row.get("corr")) for row in rows]),
        "out_dir": str(out_dir),
        "verdict": verdict,
        "note": "No training launched. Metrics computed directly from existing prediction.npz files.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "d12_s3_scorecard.csv", rows)
    write_csv(out_dir / "d12_s3_segment_metrics.csv", segment_rows)
    write_csv(out_dir / "d12_s3_mode_summary.csv", mode_summary)
    write_csv(out_dir / "d12_s3_protocol_summary.csv", protocol_summary)
    write_json(out_dir / "d12_s3_scorecard_summary.json", summary)

    md_lines = [
        "# D12-S3 clean 23-profile metadata ablation scorecard",
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
    for row in mode_summary:
        md_lines.append(
            f"| {row['mode']} | {row['n']} | {row['ok']} | {row['mean_mae_V']} | {row['mean_corr']} | {row['mean_bias_V']} |"
        )
    md_lines += [
        "",
        "## Protocol summary",
        "",
        "| protocol | n | ok | mean_MAE_V | mean_corr | mean_bias_V |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in protocol_summary:
        md_lines.append(
            f"| {row['protocol']} | {row['n']} | {row['ok']} | {row['mean_mae_V']} | {row['mean_corr']} | {row['mean_bias_V']} |"
        )
    (out_dir / "D12_S3_SCORECARD_RECOMMENDATION.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
