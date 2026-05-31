#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D10-P0 battery-8 outlier/regime judgement for GV1.

This script is intentionally diagnostic-only.  It reads existing D9.6/D9.7
outputs and produces an auditable recommendation before any new 24-profile
200ks/500ks training is launched.

Expected existing outputs, with default paths under E:/XJTU battery dataset/_gv1_cache:
  - xjtu_batch134_d97_battery8_outlier_diagnosis/d97_battery8_diagnosis_summary.json
  - xjtu_batch134_d97_battery8_outlier_diagnosis/diagnosis_plots/d97_candidate_metrics_table.csv
  - xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96/metrics_borderline_200ks.json
  - xjtu_batch134_train_conditioned_pinn_multicell_24x40ks_d96/scorecard_d96_40ks.json
  - xjtu_batch134_train_conditioned_pinn_multicell_6x200ks_d96/scorecard_d96_200ks.json

Outputs:
  - d10_p0_battery8_judgement_summary.json
  - d10_p0_candidate_metrics_normalized.csv
  - d10_p0_segment_metrics_table.csv
  - d10_p0_peer_comparison_table.csv
  - D10_P0_RECOMMENDATION.md
  - optional plots under plots/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_CACHE_ROOT = Path(r"E:/XJTU battery dataset/_gv1_cache")


def _as_path(x: str | Path | None) -> Path | None:
    if x is None:
        return None
    s = str(x).strip().strip('"')
    if not s:
        return None
    return Path(s)


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, data: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


MappingLike = dict[str, Any]


def _to_float(x: Any, default: float = math.nan) -> float:
    if x is None:
        return default
    if isinstance(x, (int, float, np.floating, np.integer)):
        try:
            return float(x)
        except Exception:
            return default
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return default
    s = s.replace("≈", "").replace("V", "").replace("v", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return default


def _pick(row: dict[str, Any], aliases: Iterable[str], default: Any = None) -> Any:
    lower = {str(k).lower(): k for k in row.keys()}
    for name in aliases:
        key = lower.get(name.lower())
        if key is not None:
            return row.get(key)
    return default


def _label_of(row: dict[str, Any], fallback: str = "") -> str:
    return str(_pick(row, ["label", "run", "profile", "profile_id", "name", "path", "output_dir"], fallback) or fallback)


def _norm_metric_row(row: dict[str, Any], label: str | None = None) -> dict[str, Any]:
    lab = str(label if label is not None else _label_of(row))
    return {
        "label": lab,
        "mae_V": _to_float(_pick(row, ["mae_V", "mae", "MAE", "voltage_mae", "voltage_mae_V"])),
        "rmse_V": _to_float(_pick(row, ["rmse_V", "rmse", "RMSE", "voltage_rmse", "voltage_rmse_V"])),
        "bias_V": _to_float(_pick(row, ["bias_V", "bias", "BIAS", "voltage_bias", "mean_error_V"])),
        "corr": _to_float(_pick(row, ["corr", "correlation", "pearson_r", "r"])),
        "r2": _to_float(_pick(row, ["r2", "R2", "R_squared"])),
        "pred_min_V": _to_float(_pick(row, ["pred_min_V", "voltage_pred_min", "prediction_min", "v_pred_min"])),
        "pred_max_V": _to_float(_pick(row, ["pred_max_V", "voltage_pred_max", "prediction_max", "v_pred_max"])),
        "target_min_V": _to_float(_pick(row, ["target_min_V", "voltage_target_min", "voltage_exp_min", "v_true_min"])),
        "target_max_V": _to_float(_pick(row, ["target_max_V", "voltage_target_max", "voltage_exp_max", "v_true_max"])),
        "pred_upper_frac_ge_4p269": _to_float(_pick(row, ["pred_upper_frac_ge_4p269", "upper_frac", "upper_frac_ge_4p269"])),
        "pred_overshoot_frac_gt_4p35": _to_float(_pick(row, ["pred_overshoot_frac_gt_4p35", "overshoot_frac", "overshoot_frac_gt_4p35"])),
        "status": str(_pick(row, ["status", "verdict", "pass_status"], "") or ""),
    }


def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, float]:
    label = str(row.get("label", "")).lower()
    # Prefer D9.6/A reproduce/original as the baseline if present.
    if any(k in label for k in ["d9.6 original", "d96 original", "original", "a reproduce", "reproduce", "baseline"]):
        pri = 0
    elif "d9.6" in label or "d96" in label:
        pri = 1
    else:
        pri = 2
    mae = _to_float(row.get("mae_V"), default=math.inf)
    return (pri, mae if np.isfinite(mae) else math.inf)


def _is_battery8_label(label: str) -> bool:
    s = label.lower().replace("_", "-").replace(" ", "-")
    return "battery-8" in s or "battery8" in s or "0008" in s


def _is_b1_2c_label(label: str) -> bool:
    s = label.lower().replace("_", "-")
    return ("b1" in s or "batch-1" in s or "batch1" in s) and "2c" in s


def _iter_dicts(obj: Any, path: str = "root") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from _iter_dicts(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_dicts(v, f"{path}[{i}]")


def _collect_metric_like_dicts(obj: Any, default_label_prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if obj is None:
        return rows
    for path, d in _iter_dicts(obj):
        keys = {str(k).lower() for k in d.keys()}
        has_error = any(k in keys for k in ["mae", "mae_v", "rmse", "rmse_v", "corr", "correlation"])
        if not has_error:
            continue
        label = _label_of(d, fallback=path)
        if label == path and default_label_prefix:
            label = f"{default_label_prefix}:{path}"
        row = _norm_metric_row(d, label=label)
        row["json_path"] = path
        rows.append(row)
    return rows


def _discover_first(root: Path, names: list[str]) -> Path | None:
    if not root.exists():
        return None
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    for name in names:
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None


def _median_mad(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(arr) == 0:
        return math.nan, math.nan
    med = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - med)))
    if mad <= 1e-12:
        mad = float(np.nanstd(arr))
    if mad <= 1e-12:
        mad = math.nan
    return med, mad


def _classify_candidate(label: str) -> str:
    s = label.lower().replace("_", ".")
    if "d9.6.1" in s or "d961" in s:
        return "rejected_d9_6_1_high_voltage_saturation"
    if "d9.6.2" in s or "d962" in s:
        return "rejected_d9_6_2_voltage_range_collapse"
    if "d9.6.3" in s or "d963" in s or "lower lr" in s or "seed7" in s:
        return "rejected_d9_6_3_training_strategy_no_improvement"
    if "d9.6" in s or "d96" in s or "original" in s or "reproduce" in s or "baseline" in s:
        return "mainline_d9_6_reference"
    return "other_candidate"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_plots(out_dir: Path, candidates: list[dict[str, Any]], segments: list[dict[str, Any]], peers: list[dict[str, Any]]) -> list[str]:
    made: list[str] = []
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return made
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def finite_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        return [r for r in rows if np.isfinite(_to_float(r.get(key)))]

    rows = finite_rows(candidates, "mae_V")
    if rows:
        labels = [str(r.get("label", ""))[:36] for r in rows]
        vals = [_to_float(r.get("mae_V")) for r in rows]
        plt.figure(figsize=(max(8, 0.45 * len(rows)), 4.5))
        plt.bar(range(len(rows)), vals)
        plt.xticks(range(len(rows)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("MAE (V)")
        plt.title("D10-P0 battery-8 candidate comparison")
        plt.tight_layout()
        p = plot_dir / "candidate_mae_comparison.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append(str(p))

    rows = finite_rows(segments, "mae_V")
    if rows:
        labels = [str(r.get("label", ""))[:36] for r in rows]
        vals = [_to_float(r.get("mae_V")) for r in rows]
        plt.figure(figsize=(max(8, 0.45 * len(rows)), 4.5))
        plt.bar(range(len(rows)), vals)
        plt.xticks(range(len(rows)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel("MAE (V)")
        plt.title("D10-P0 battery-8 segment/regime MAE")
        plt.tight_layout()
        p = plot_dir / "segment_mae_comparison.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append(str(p))

    peer_rows = finite_rows(peers, "mae_V")
    if peer_rows:
        labels = [str(r.get("label", ""))[:28] for r in peer_rows]
        vals = [_to_float(r.get("mae_V")) for r in peer_rows]
        colors = ["tab:red" if bool(r.get("is_battery8")) else "tab:blue" for r in peer_rows]
        plt.figure(figsize=(max(8, 0.33 * len(peer_rows)), 4.5))
        plt.bar(range(len(peer_rows)), vals, color=colors)
        plt.xticks(range(len(peer_rows)), labels, rotation=70, ha="right", fontsize=7)
        plt.ylabel("MAE (V)")
        plt.title("24-profile 40ks peer comparison; red = battery-8")
        plt.tight_layout()
        p = plot_dir / "peer_mae_24x40ks.png"
        plt.savefig(p, dpi=180)
        plt.close()
        made.append(str(p))
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="D10-P0 battery-8 outlier/regime judgement from D9.6/D9.7 artifacts.")
    ap.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--d97_dir", default=None)
    ap.add_argument("--d97_summary_json", default=None)
    ap.add_argument("--d97_candidate_csv", default=None)
    ap.add_argument("--d96_b8_200ks_metrics_json", default=None)
    ap.add_argument("--scorecard_24x40_json", default=None)
    ap.add_argument("--scorecard_6x200_json", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    d97_dir = _as_path(args.d97_dir) or (cache_root / "xjtu_batch134_d97_battery8_outlier_diagnosis")
    out_dir = _as_path(args.out_dir) or (cache_root / "xjtu_batch134_d10_p0_battery8_regime_judgement")
    out_dir.mkdir(parents=True, exist_ok=True)

    d97_summary_path = _as_path(args.d97_summary_json) or _discover_first(d97_dir, ["d97_battery8_diagnosis_summary.json"])
    d97_candidate_csv = _as_path(args.d97_candidate_csv) or _discover_first(d97_dir, ["d97_candidate_metrics_table.csv"])
    d96_b8_metrics_path = _as_path(args.d96_b8_200ks_metrics_json) or (
        cache_root / "xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96" / "metrics_borderline_200ks.json"
    )
    scorecard_24x40_path = _as_path(args.scorecard_24x40_json) or (
        cache_root / "xjtu_batch134_train_conditioned_pinn_multicell_24x40ks_d96" / "scorecard_d96_40ks.json"
    )
    scorecard_6x200_path = _as_path(args.scorecard_6x200_json) or (
        cache_root / "xjtu_batch134_train_conditioned_pinn_multicell_6x200ks_d96" / "scorecard_d96_200ks.json"
    )

    d97_summary = _read_json(d97_summary_path)
    d96_b8_metrics = _read_json(d96_b8_metrics_path)
    score_24 = _read_json(scorecard_24x40_path)
    score_6x200 = _read_json(scorecard_6x200_path)

    # Candidate table: D9.6 original vs failed variants.
    candidate_rows: list[dict[str, Any]] = []
    for row in _read_csv(d97_candidate_csv):
        nr = _norm_metric_row(row)
        nr["candidate_class"] = _classify_candidate(str(nr["label"]))
        candidate_rows.append(nr)
    if d96_b8_metrics:
        # Include explicit original D9.6 200ks metrics even if D9.7 CSV is missing.
        for row in _collect_metric_like_dicts(d96_b8_metrics, default_label_prefix="d96_b8_metrics"):
            row["label"] = row.get("label") or "D9.6_original_battery8_200ks"
            if "root" in str(row.get("json_path", "")):
                row["label"] = "D9.6_original_battery8_200ks"
            row["candidate_class"] = _classify_candidate(str(row["label"]))
            candidate_rows.append(row)
    # Deduplicate by label + approximate mae.
    seen = set()
    dedup_candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = (str(row.get("label", "")), round(_to_float(row.get("mae_V"), -999), 6))
        if key in seen:
            continue
        seen.add(key)
        dedup_candidates.append(row)
    candidate_rows = sorted(dedup_candidates, key=_candidate_sort_key)

    baseline_candidate = candidate_rows[0] if candidate_rows else {}
    # If the first row is not obviously mainline, pick a mainline row by class.
    for row in candidate_rows:
        if row.get("candidate_class") == "mainline_d9_6_reference":
            baseline_candidate = row
            break

    # Segment/regime table from D9.7 summary.
    segment_rows = _collect_metric_like_dicts(d97_summary, default_label_prefix="d97")
    # Keep rows that look like segment/regime names or are not candidate classes.
    keep_segments: list[dict[str, Any]] = []
    for row in segment_rows:
        label = str(row.get("label", ""))
        low = label.lower()
        if any(k in low for k in ["charge", "discharge", "target", "voltage", "current", "mid", "high", "low", "segment", "regime"]):
            keep_segments.append(row)
    if not keep_segments:
        keep_segments = segment_rows
    segment_rows = keep_segments

    # Peer comparison from D9.6 24x40 scorecard.
    peer_rows = _collect_metric_like_dicts(score_24, default_label_prefix="score24")
    # Retain profile-level rows; remove aggregate rows if they have labels like global/mean.
    profile_peer_rows: list[dict[str, Any]] = []
    for row in peer_rows:
        label = str(row.get("label", ""))
        low = label.lower()
        if any(k in low for k in ["global", "summary", "mean", "aggregate"]):
            continue
        if not np.isfinite(_to_float(row.get("mae_V"))) and not np.isfinite(_to_float(row.get("corr"))):
            continue
        row["is_battery8"] = _is_battery8_label(label)
        row["is_b1_2c"] = _is_b1_2c_label(label)
        profile_peer_rows.append(row)
    peer_rows = profile_peer_rows

    non_b8_maes = [_to_float(r.get("mae_V")) for r in peer_rows if not r.get("is_battery8")]
    non_b8_corrs = [_to_float(r.get("corr")) for r in peer_rows if not r.get("is_battery8")]
    median_mae, mad_mae = _median_mad(non_b8_maes)
    median_corr, mad_corr = _median_mad(non_b8_corrs)
    for row in peer_rows:
        mae = _to_float(row.get("mae_V"))
        corr = _to_float(row.get("corr"))
        row["peer_median_mae_V_excluding_b8"] = median_mae
        row["peer_mad_mae_V_excluding_b8"] = mad_mae
        row["mae_mad_z_vs_peers"] = (mae - median_mae) / mad_mae if np.isfinite(mae) and np.isfinite(median_mae) and np.isfinite(mad_mae) and mad_mae > 0 else math.nan
        row["peer_median_corr_excluding_b8"] = median_corr
        row["corr_gap_vs_peer_median"] = median_corr - corr if np.isfinite(corr) and np.isfinite(median_corr) else math.nan

    b8_peer_rows = [r for r in peer_rows if r.get("is_battery8")]
    b8_peer = b8_peer_rows[0] if b8_peer_rows else {}

    # Extract important segments.
    def find_segment(*patterns: str) -> dict[str, Any]:
        for row in segment_rows:
            lab = str(row.get("label", "")).lower()
            if all(p.lower() in lab for p in patterns):
                return row
        return {}

    charge_seg = find_segment("charge")
    discharge_seg = find_segment("discharge")
    if not charge_seg:
        charge_seg = find_segment("I_pos") or find_segment("pos")
    if not discharge_seg:
        discharge_seg = find_segment("I_neg") or find_segment("neg")

    charge_mae = _to_float(charge_seg.get("mae_V")) if charge_seg else math.nan
    discharge_mae = _to_float(discharge_seg.get("mae_V")) if discharge_seg else math.nan
    charge_corr = _to_float(charge_seg.get("corr")) if charge_seg else math.nan
    discharge_corr = _to_float(discharge_seg.get("corr")) if discharge_seg else math.nan

    segment_asymmetry = bool(
        np.isfinite(charge_mae)
        and np.isfinite(discharge_mae)
        and discharge_mae >= max(0.08, 2.0 * max(charge_mae, 1e-9))
        and (not np.isfinite(charge_corr) or charge_corr >= 0.90)
        and (not np.isfinite(discharge_corr) or discharge_corr <= 0.85)
    )
    peer_outlier = bool(
        b8_peer
        and (
            str(b8_peer.get("status", "")).lower().find("border") >= 0
            or (_to_float(b8_peer.get("mae_mad_z_vs_peers")) >= 2.0)
            or (_to_float(b8_peer.get("corr_gap_vs_peer_median")) >= 0.08)
        )
    )
    overshoot_risk = bool(
        _to_float(baseline_candidate.get("pred_max_V")) > 4.35
        or _to_float(baseline_candidate.get("pred_upper_frac_ge_4p269")) >= 0.005
        or _to_float(baseline_candidate.get("pred_overshoot_frac_gt_4p35")) >= 0.001
    )

    # Compare failed candidates to baseline.
    baseline_mae = _to_float(baseline_candidate.get("mae_V"))
    worse_candidates = []
    for row in candidate_rows:
        cls = str(row.get("candidate_class", ""))
        if not cls.startswith("rejected"):
            continue
        mae = _to_float(row.get("mae_V"))
        if np.isfinite(mae) and np.isfinite(baseline_mae) and mae > baseline_mae:
            worse_candidates.append(str(row.get("label", "")))
    failed_repairs_confirmed = len(worse_candidates) >= 2 or any(str(r.get("candidate_class", "")).startswith("rejected") for r in candidate_rows)

    if segment_asymmetry and (peer_outlier or overshoot_risk):
        verdict = "battery8_flagged_late_2C_discharge_regime_outlier_keep_D9_6_mainline"
        next_action = [
            "Do not overwrite D9.6/D9.5.1 mainline with D9.6.1/D9.6.2/D9.6.3-style guards.",
            "Treat B1_2C battery-8 as a flagged outlier/regime case for now.",
            "Run 23-profile 200ks excluding or explicitly flagging battery-8 to validate non-outlier medium-window generalization.",
            "Only after 23-profile 200ks passes, consider a lightweight battery-8-specific profile-level affine/discharge calibration; avoid strong voltage clamps.",
        ]
    elif segment_asymmetry:
        verdict = "battery8_regime_specific_discharge_issue_peer_evidence_incomplete"
        next_action = [
            "Keep D9.6 mainline.",
            "Compare battery-8 against B1_2C peer cells before any new repair.",
            "If peer comparison confirms outlier behavior, run 23-profile 200ks excluding/flagging battery-8.",
        ]
    else:
        verdict = "inconclusive_need_more_battery8_peer_and_segment_evidence"
        next_action = [
            "Keep D9.6 mainline temporarily.",
            "Inspect D9.7 plots and segment CSV manually.",
            "Do not run 24-profile 200ks until battery-8 classification is resolved.",
        ]

    evidence = {
        "paths": {
            "cache_root": str(cache_root),
            "d97_dir": str(d97_dir),
            "d97_summary_json": str(d97_summary_path) if d97_summary_path else None,
            "d97_candidate_csv": str(d97_candidate_csv) if d97_candidate_csv else None,
            "d96_b8_200ks_metrics_json": str(d96_b8_metrics_path),
            "scorecard_24x40_json": str(scorecard_24x40_path),
            "scorecard_6x200_json": str(scorecard_6x200_path),
            "out_dir": str(out_dir),
        },
        "file_exists": {
            "d97_summary_json": bool(d97_summary_path and d97_summary_path.exists()),
            "d97_candidate_csv": bool(d97_candidate_csv and d97_candidate_csv.exists()),
            "d96_b8_200ks_metrics_json": bool(d96_b8_metrics_path and d96_b8_metrics_path.exists()),
            "scorecard_24x40_json": bool(scorecard_24x40_path and scorecard_24x40_path.exists()),
            "scorecard_6x200_json": bool(scorecard_6x200_path and scorecard_6x200_path.exists()),
        },
        "baseline_candidate": baseline_candidate,
        "charge_segment": charge_seg,
        "discharge_segment": discharge_seg,
        "segment_asymmetry": segment_asymmetry,
        "peer_outlier": peer_outlier,
        "overshoot_risk": overshoot_risk,
        "failed_repairs_confirmed": failed_repairs_confirmed,
        "worse_repair_candidates": worse_candidates,
        "peer_reference": {
            "non_b8_median_mae_V": median_mae,
            "non_b8_mad_mae_V": mad_mae,
            "non_b8_median_corr": median_corr,
            "non_b8_mad_corr": mad_corr,
            "battery8_peer_row": b8_peer,
        },
    }

    plots = _make_plots(out_dir, candidate_rows, segment_rows, peer_rows) if args.make_plots else []
    summary: dict[str, Any] = {
        "ok": True,
        "stage": "D10-P0 battery-8 outlier/regime judgement",
        "verdict": verdict,
        "recommended_next_action": next_action,
        "evidence": evidence,
        "counts": {
            "candidate_rows": len(candidate_rows),
            "segment_rows": len(segment_rows),
            "peer_rows": len(peer_rows),
        },
        "plots": plots,
    }

    _write_csv(out_dir / "d10_p0_candidate_metrics_normalized.csv", candidate_rows)
    _write_csv(out_dir / "d10_p0_segment_metrics_table.csv", segment_rows)
    _write_csv(out_dir / "d10_p0_peer_comparison_table.csv", peer_rows)
    _write_json(out_dir / "d10_p0_battery8_judgement_summary.json", summary)

    rec_lines = [
        "# D10-P0 battery-8 judgement recommendation",
        "",
        f"Verdict: `{verdict}`",
        "",
        "## Recommended next action",
    ]
    for item in next_action:
        rec_lines.append(f"- {item}")
    rec_lines += [
        "",
        "## Key evidence used by this script",
        f"- segment_asymmetry = {segment_asymmetry}",
        f"- peer_outlier = {peer_outlier}",
        f"- overshoot_risk = {overshoot_risk}",
        f"- failed_repairs_confirmed = {failed_repairs_confirmed}",
        f"- baseline_candidate = {baseline_candidate.get('label', '')}",
        "",
        "## Output files",
        "- d10_p0_battery8_judgement_summary.json",
        "- d10_p0_candidate_metrics_normalized.csv",
        "- d10_p0_segment_metrics_table.csv",
        "- d10_p0_peer_comparison_table.csv",
    ]
    if plots:
        rec_lines += ["", "## Plots"] + [f"- {p}" for p in plots]
    (out_dir / "D10_P0_RECOMMENDATION.md").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "verdict": verdict, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
