#!/usr/bin/env python
"""Build a D9.6 multi-profile scorecard from gv1_prediction_metrics.py output."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


def _as_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _protocol(run: str) -> str:
    if re.search(r"R25|R2\.5", run, flags=re.I):
        return "R2.5"
    if re.search(r"(?:^|_)R3(?:_|$)", run, flags=re.I):
        return "R3"
    if re.search(r"(?:^|_)2C(?:_|$)", run, flags=re.I):
        return "2C"
    return "unknown"


def _window_tag(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    tend = max(float(r.get("t_end_s", 0.0) or 0.0) for r in rows)
    if tend >= 999_000:
        return "1Ms"
    if tend >= 499_000:
        return "500ks"
    if tend >= 199_000:
        return "200ks"
    if tend >= 39_000:
        return "40ks"
    return f"{int(tend)}s"


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _row_status(row: dict[str, Any], window: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    mae = _finite(row.get("mae_V"))
    corr = _finite(row.get("corr"))
    upper = _finite(row.get("pred_upper_frac_ge_4p269")) or 0.0
    over = _finite(row.get("pred_overshoot_frac_gt_4p35")) or 0.0
    under = _finite(row.get("pred_undershoot_frac_lt_2p35")) or 0.0

    # D9.6 is a verification step. We keep thresholds realistic for cross-cell checks,
    # not as strict as single-profile calibration.
    if mae is None or corr is None:
        return "fail", ["missing mae/corr"]
    if corr < 0.85:
        reasons.append(f"corr<{0.85:.2f}")
    if mae > 0.18:
        reasons.append("mae>0.18V")
    if upper > 0.05:
        reasons.append("4.269V saturation>5%")
    if over > 0.005:
        reasons.append(">4.35V overshoot>0.5%")
    if under > 0.005:
        reasons.append("<2.35V undershoot>0.5%")
    if reasons:
        return "fail", reasons

    pass_corr = 0.90 if window in {"40ks", "200ks"} else 0.88
    pass_mae = 0.12 if window in {"40ks", "200ks"} else 0.14
    if corr >= pass_corr and mae <= pass_mae and upper <= 0.02 and over <= 0.001 and under <= 0.001:
        return "pass", []
    return "borderline", ["usable but below preferred pass thresholds"]


def _mean_finite(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [_finite(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return float(mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Create D9.6 multi-profile scorecard.")
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    metrics_path = Path(args.metrics_json)
    rows = _as_rows(json.loads(metrics_path.read_text(encoding="utf-8")))
    window = _window_tag(rows)
    per_run = []
    for r in rows:
        status, reasons = _row_status(r, window)
        per_run.append({
            "run": r.get("run"),
            "protocol": _protocol(str(r.get("run", ""))),
            "status": status,
            "reasons": reasons,
            "mae_V": r.get("mae_V"),
            "rmse_V": r.get("rmse_V"),
            "corr": r.get("corr"),
            "bias_V": r.get("bias_V"),
            "pred_low_voltage_frac_le_2p75": r.get("pred_low_voltage_frac_le_2p75"),
            "target_low_voltage_frac_le_2p75": r.get("target_low_voltage_frac_le_2p75"),
            "pred_upper_frac_ge_4p269": r.get("pred_upper_frac_ge_4p269"),
            "prediction_npz": r.get("prediction_npz"),
        })

    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_protocol.setdefault(_protocol(str(r.get("run", ""))), []).append(r)

    protocol_summary = {}
    for p, rs in by_protocol.items():
        protocol_summary[p] = {
            "n": len(rs),
            "mean_mae_V": _mean_finite(rs, "mae_V"),
            "mean_rmse_V": _mean_finite(rs, "rmse_V"),
            "mean_corr": _mean_finite(rs, "corr"),
            "mean_bias_V": _mean_finite(rs, "bias_V"),
            "mean_pred_upper_frac_ge_4p269": _mean_finite(rs, "pred_upper_frac_ge_4p269"),
            "mean_pred_low_voltage_frac_le_2p75": _mean_finite(rs, "pred_low_voltage_frac_le_2p75"),
        }

    statuses = [r["status"] for r in per_run]
    fail_count = statuses.count("fail")
    borderline_count = statuses.count("borderline")
    pass_count = statuses.count("pass")
    if fail_count > 0:
        overall = "fail"
        recommendation = "暂停扩窗；先检查 fail profile 的协议/低压段/高压饱和，再考虑 D9.6.1 修正。"
    elif borderline_count > 0:
        overall = "borderline_continue_carefully"
        recommendation = "可以进入下一窗口或少量扩窗，但不要直接 24-profile mid-window；优先看 borderline profile 图像。"
    else:
        overall = "pass"
        recommendation = "可以进入 D9.6 下一步：6-profile 200ks；若当前已是 200ks，则再考虑 24-profile 40ks。"

    out = {
        "ok": bool(rows),
        "stage": "GV1 D9.6 multi-profile verification scorecard",
        "metrics_json": str(metrics_path),
        "window_tag": window,
        "n_profiles": len(rows),
        "status_counts": {"pass": pass_count, "borderline": borderline_count, "fail": fail_count},
        "overall_status": overall,
        "recommendation": recommendation,
        "global_mean_mae_V": _mean_finite(rows, "mae_V"),
        "global_mean_rmse_V": _mean_finite(rows, "rmse_V"),
        "global_mean_corr": _mean_finite(rows, "corr"),
        "global_mean_bias_V": _mean_finite(rows, "bias_V"),
        "protocol_summary": protocol_summary,
        "per_run": per_run,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
