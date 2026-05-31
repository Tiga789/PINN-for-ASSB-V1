#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rescue D12 TRUE SMOKE scorecard by computing metrics from prediction.npz.

The previous D12 TRUE SMOKE collector expected d10_voltage_metrics.json, but
TRUE_SMOKE triplet did not call the D10 metrics step.  This script is
idempotent: for each TRUE_SMOKE run directory, it will
  1) keep existing d10_voltage_metrics.json if present;
  2) otherwise compute it from prediction.npz if prediction exists;
  3) otherwise report missing_prediction with a compact file listing.

No training is launched by this script.
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _to_float(x: Any, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _corr(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if int(mask.sum()) < 3:
        return math.nan
    yy = y[mask].astype(float)
    pp = p[mask].astype(float)
    if float(np.nanstd(yy)) <= 1e-12 or float(np.nanstd(pp)) <= 1e-12:
        return math.nan
    return float(np.corrcoef(yy, pp)[0, 1])


def _metrics(label: str, y: np.ndarray, p: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = mask & np.isfinite(y) & np.isfinite(p)
    n = int(mask.sum())
    if n == 0:
        return {"label": label, "n": 0}
    err = p[mask] - y[mask]
    return {
        "label": label,
        "n": n,
        "mae_V": float(np.nanmean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.nanmean(err ** 2))),
        "bias_V": float(np.nanmean(err)),
        "corr": _corr(y[mask], p[mask]),
        "target_min_V": float(np.nanmin(y[mask])),
        "target_max_V": float(np.nanmax(y[mask])),
        "pred_min_V": float(np.nanmin(p[mask])),
        "pred_max_V": float(np.nanmax(p[mask])),
        "pred_upper_frac_ge_4p269": float(np.nanmean(p[mask] >= 4.269)),
        "pred_overshoot_frac_gt_4p35": float(np.nanmean(p[mask] > 4.35)),
        "pred_low_frac_le_2p75": float(np.nanmean(p[mask] <= 2.75)),
        "target_low_frac_le_2p75": float(np.nanmean(y[mask] <= 2.75)),
    }


def _compute_metrics_from_prediction(pred_path: Path) -> dict[str, Any]:
    with np.load(pred_path, allow_pickle=True) as z:
        keys = set(z.files)
        if "voltage_exp" in keys:
            y = np.asarray(z["voltage_exp"], dtype=float).reshape(-1)
        elif "target_voltage" in keys:
            y = np.asarray(z["target_voltage"], dtype=float).reshape(-1)
        else:
            raise KeyError(f"{pred_path} has no voltage_exp/target_voltage; keys={sorted(keys)}")
        if "voltage_exp_pred" in keys:
            p = np.asarray(z["voltage_exp_pred"], dtype=float).reshape(-1)
        elif "phis_c_pred" in keys:
            p = np.asarray(z["phis_c_pred"], dtype=float).reshape(-1)
        elif "voltage_pred" in keys:
            p = np.asarray(z["voltage_pred"], dtype=float).reshape(-1)
        else:
            raise KeyError(f"{pred_path} has no voltage_exp_pred/phis_c_pred/voltage_pred; keys={sorted(keys)}")
        I = np.asarray(z["I_profile"], dtype=float).reshape(-1) if "I_profile" in keys else np.zeros_like(y)
        t = np.asarray(z["t_global_s"], dtype=float).reshape(-1) if "t_global_s" in keys else np.arange(len(y), dtype=float)
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
    rows = [_metrics(name, y, p, mask & finite) for name, mask in masks.items()]
    return {
        "ok": True,
        "stage": "D12 TRUE SMOKE rescued voltage metrics from prediction.npz",
        "prediction_npz": str(pred_path),
        "n": int(n),
        "current_high_quantile_threshold_A": high_i,
        "metrics": {r["label"]: r for r in rows},
        "segments": rows,
    }


def _mode_from_name(name: str) -> str:
    m = re.search(r"d12_runtime_metadata_(off|zero|on)_", name)
    return m.group(1) if m else "unknown"


def _find_prediction(run_dir: Path) -> Path | None:
    direct = run_dir / "prediction.npz"
    if direct.exists():
        return direct
    cands = sorted(run_dir.rglob("prediction.npz"))
    return cands[0] if cands else None


def _small_listing(run_dir: Path) -> str:
    try:
        items = sorted([p for p in run_dir.iterdir()], key=lambda p: p.name)[:30]
        return "; ".join(f"{p.name}{'/' if p.is_dir() else ''}" for p in items)
    except Exception as exc:
        return f"listing_failed: {exc}"


def _status(row: dict[str, Any]) -> str:
    if row.get("metrics_source") in {"existing_json", "computed_from_prediction"}:
        mae = _to_float(row.get("mae_V"))
        corr = _to_float(row.get("corr"))
        if np.isfinite(mae) and np.isfinite(corr):
            return "smoke_completed_metrics_ok"
        if np.isfinite(mae):
            return "smoke_completed_metrics_review"
        return "metrics_json_read_error"
    return str(row.get("status", "read_error"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d12_runtime_metadata_true_smoke_scorecard_rescued"
    run_dirs = sorted([
        p for p in cache_root.iterdir()
        if p.is_dir() and p.name.startswith("xjtu_batch134_d12_runtime_metadata_") and "TRUE_SMOKE" in p.name
    ])

    rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    for d in run_dirs:
        row: dict[str, Any] = {"run_dir": str(d), "run_name": d.name, "mode": _mode_from_name(d.name)}
        meta_path = d / "d12_metadata_runtime_summary.json"
        if meta_path.exists():
            try:
                md = _read_json(meta_path).get("metadata", {})
                row.update({
                    "metadata_dim": md.get("metadata_dim"),
                    "metadata_profile_id": md.get("profile_id"),
                    "metadata_mode": md.get("mode"),
                })
            except Exception as exc:
                row["metadata_read_error"] = str(exc)
        metrics_path = d / "d10_voltage_metrics.json"
        if not metrics_path.exists():
            pred_path = _find_prediction(d)
            if pred_path is not None:
                try:
                    metrics = _compute_metrics_from_prediction(pred_path)
                    _write_json(metrics_path, {k: v for k, v in metrics.items() if k != "segments"})
                    _write_csv(d / "d10_voltage_metrics_by_segment.csv", metrics["segments"])
                    row["metrics_source"] = "computed_from_prediction"
                    row["prediction_npz"] = str(pred_path)
                except Exception as exc:
                    row["status"] = "metrics_compute_error"
                    row["error"] = str(exc)
                    row["file_listing"] = _small_listing(d)
            else:
                row["status"] = "missing_prediction"
                row["error"] = "No prediction.npz found; training likely did not complete or saved under an unexpected name."
                row["file_listing"] = _small_listing(d)
        else:
            row["metrics_source"] = "existing_json"

        if metrics_path.exists():
            try:
                data = _read_json(metrics_path)
                allm = data.get("metrics", {}).get("all", data.get("all", {}))
                row.update(allm)
                for label, m in data.get("metrics", {}).items():
                    rr = dict(m)
                    rr.update({"run_name": d.name, "mode": row.get("mode")})
                    seg_rows.append(rr)
            except Exception as exc:
                row["status"] = "metrics_json_read_error"
                row["error"] = str(exc)
        row["status"] = _status(row)
        rows.append(row)

    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r.get("status", "read_error"))] = counts.get(str(r.get("status", "read_error")), 0) + 1
    maes = [_to_float(r.get("mae_V")) for r in rows if np.isfinite(_to_float(r.get("mae_V")))]
    corrs = [_to_float(r.get("corr")) for r in rows if np.isfinite(_to_float(r.get("corr")))]
    completed = counts.get("smoke_completed_metrics_ok", 0) + counts.get("smoke_completed_metrics_review", 0)
    if len(rows) == 0:
        verdict = "d12_true_smoke_no_run_dirs_found"
    elif completed == len(rows):
        verdict = "d12_true_smoke_metrics_rescued_all_completed"
    elif completed > 0:
        verdict = "d12_true_smoke_partial_metrics_rescued_review_missing_runs"
    else:
        verdict = "d12_true_smoke_no_completed_predictions_rerun_required"
    summary = {
        "ok": True,
        "stage": "D12 TRUE SMOKE rescued scorecard",
        "profile_count": len(rows),
        "counts": counts,
        "mean_mae_V": float(np.nanmean(maes)) if maes else math.nan,
        "mean_corr": float(np.nanmean(corrs)) if corrs else math.nan,
        "out_dir": str(out_dir),
        "verdict": verdict,
        "note": "No training was launched. Metrics were computed only from existing prediction.npz files when present.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "d12_true_smoke_scorecard_rescued.csv", rows)
    _write_csv(out_dir / "d12_true_smoke_segment_metrics_rescued.csv", seg_rows)
    _write_json(out_dir / "d12_true_smoke_scorecard_rescued_summary.json", summary)
    md = [
        "# D12 TRUE SMOKE rescued scorecard",
        "",
        "## Verdict",
        "",
        f"```text\n{verdict}\n```",
        "",
        "## What this rescue did",
        "",
        "- It did not launch training.",
        "- It searched existing TRUE_SMOKE run directories.",
        "- It computed `d10_voltage_metrics.json` from `prediction.npz` when metrics were missing.",
        "- If a run has `missing_prediction`, that specific run must be rerun with the corrected TRUE_SMOKE command.",
        "",
        "## Counts",
        "",
        "```json\n" + json.dumps(counts, ensure_ascii=False, indent=2) + "\n```",
    ]
    (out_dir / "D12_TRUE_SMOKE_RESCUE_RECOMMENDATION.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
