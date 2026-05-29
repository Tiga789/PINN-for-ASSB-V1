#!/usr/bin/env python
"""Compute voltage diagnostics for GV1 D9.5.1 prediction.npz files.

D9.5.1 keeps D9.3's regime diagnostics and adds hybrid-branch diagnostics for
profile-adaptive voltage transforms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) <= 2:
        return None
    if float(np.nanstd(a)) <= 0 or float(np.nanstd(b)) <= 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _stats_from_err(err: np.ndarray) -> dict[str, float | int | None]:
    if len(err) == 0:
        return {"n": 0, "mae_V": None, "rmse_V": None, "bias_V": None}
    return {
        "n": int(len(err)),
        "mae_V": float(np.mean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.mean(err ** 2))),
        "bias_V": float(np.mean(err)),
    }


def _one_metrics(pred_path: Path) -> dict[str, Any]:
    with np.load(pred_path, allow_pickle=True) as d:
        if "voltage_exp" not in d or "voltage_exp_pred" not in d:
            raise KeyError(f"{pred_path} must contain voltage_exp and voltage_exp_pred")
        y = d["voltage_exp"].astype(float).reshape(-1)
        yh = d["voltage_exp_pred"].astype(float).reshape(-1)
        t = d["t_global_s"].astype(float).reshape(-1) if "t_global_s" in d else np.arange(len(y), dtype=float)
        I = d["I_profile"].astype(float).reshape(-1) if "I_profile" in d else np.full_like(y, np.nan)
        T = d["temperature_C"].astype(float).reshape(-1) if "temperature_C" in d else np.full_like(y, np.nan)
        extra = {}
        for key in [
            "voltage_ocv_baseline",
            "voltage_direct_head",
            "voltage_ohmic_baseline",
            "voltage_softsign_correction",
            "voltage_low_tail_correction",
            "voltage_event_correction",
            "voltage_temperature_correction",
            "voltage_low_gate",
            "voltage_current_event_gate",
            "voltage_profile_event_gate",
            "voltage_base_branch",
            "voltage_event_branch_delta",
        ]:
            if key in d:
                extra[key] = d[key].astype(float).reshape(-1)

    n_candidates = [len(y), len(yh), len(t), len(I), len(T)] + [len(v) for v in extra.values()]
    n = min(n_candidates)
    y, yh, t, I, T = y[:n], yh[:n], t[:n], I[:n], T[:n]
    extra = {k: v[:n] for k, v in extra.items()}
    m = np.isfinite(y) & np.isfinite(yh)
    if int(m.sum()) < 2:
        return {"prediction_npz": str(pred_path), "ok": False, "reason": "fewer than two finite points"}

    y_m = y[m]
    yh_m = yh[m]
    t_m = t[m]
    I_m = I[m]
    T_m = T[m]
    err = yh_m - y_m
    bias = float(np.mean(err))
    yh_bc = yh_m - bias
    err_bc = yh_bc - y_m

    low_mask = y_m <= 2.75
    high_mask = y_m >= 4.10
    mid_mask = (~low_mask) & (~high_mask)
    q05 = float(np.nanquantile(y_m, 0.05))
    q10 = float(np.nanquantile(y_m, 0.10))
    q90 = float(np.nanquantile(y_m, 0.90))
    q95 = float(np.nanquantile(y_m, 0.95))
    low_q05_mask = y_m <= q05
    low_q10_mask = y_m <= q10
    high_q90_mask = y_m >= q90
    high_q95_mask = y_m >= q95

    absI = np.abs(I_m)
    high_current_mask = np.zeros_like(y_m, dtype=bool)
    if np.isfinite(absI).any():
        i90 = float(np.nanquantile(absI[np.isfinite(absI)], 0.90))
        high_current_mask = absI >= max(i90, 1e-12)
    else:
        i90 = float("nan")
    temp_event_mask = np.zeros_like(y_m, dtype=bool)
    if np.isfinite(T_m).any():
        tdev = np.abs(T_m - np.nanmedian(T_m))
        t90 = float(np.nanquantile(tdev[np.isfinite(tdev)], 0.90))
        temp_event_mask = tdev >= max(t90, 1e-12)
    else:
        t90 = float("nan")

    def subset_stats(mask: np.ndarray) -> dict[str, float | int | None]:
        return _stats_from_err(err[mask])

    pred_low_2p75 = yh_m <= 2.75
    pred_high_4p10 = yh_m >= 4.10
    out: dict[str, Any] = {
        "prediction_npz": str(pred_path),
        "run": pred_path.parent.name,
        "ok": True,
        "n": int(m.sum()),
        "t_start_s": float(t_m[0]),
        "t_end_s": float(t_m[-1]),
        "mae_V": float(np.mean(np.abs(err))),
        "rmse_V": float(np.sqrt(np.mean(err ** 2))),
        "bias_V": bias,
        "corr": _safe_corr(y_m, yh_m),
        "bias_corrected_mae_V": float(np.mean(np.abs(err_bc))),
        "bias_corrected_rmse_V": float(np.sqrt(np.mean(err_bc ** 2))),
        "bias_corrected_corr": _safe_corr(y_m, yh_bc),
        "voltage_exp_minmax": [float(np.min(y_m)), float(np.max(y_m))],
        "voltage_pred_minmax": [float(np.min(yh_m)), float(np.max(yh_m))],
        "voltage_exp_quantiles": {"q05": q05, "q10": q10, "q90": q90, "q95": q95},
        "voltage_pred_quantiles": {
            "q05": float(np.nanquantile(yh_m, 0.05)),
            "q10": float(np.nanquantile(yh_m, 0.10)),
            "q90": float(np.nanquantile(yh_m, 0.90)),
            "q95": float(np.nanquantile(yh_m, 0.95)),
        },
        "current_minmax_A": [float(np.nanmin(I_m)), float(np.nanmax(I_m))] if np.isfinite(I_m).any() else None,
        "temperature_minmax_C": [float(np.nanmin(T_m)), float(np.nanmax(T_m))] if np.isfinite(T_m).any() else None,
        "pred_upper_frac_ge_4p269": float(np.mean(yh_m >= 4.269)),
        "pred_upper_frac_ge_4p25": float(np.mean(yh_m >= 4.25)),
        "pred_overshoot_frac_gt_4p35": float(np.mean(yh_m > 4.35)),
        "pred_low_voltage_frac_le_2p75": float(np.mean(pred_low_2p75)),
        "pred_undershoot_frac_lt_2p35": float(np.mean(yh_m < 2.35)),
        "target_low_voltage_frac_le_2p75": float(np.mean(low_mask)),
        "target_high_voltage_frac_ge_4p10": float(np.mean(high_mask)),
        "low_coverage_gap_le_2p75_pred_minus_target": float(np.mean(pred_low_2p75) - np.mean(low_mask)),
        "high_coverage_gap_ge_4p10_pred_minus_target": float(np.mean(pred_high_4p10) - np.mean(high_mask)),
        "low_target_le_2p75": subset_stats(low_mask),
        "mid_target_2p75_to_4p10": subset_stats(mid_mask),
        "high_target_ge_4p10": subset_stats(high_mask),
        "low_target_le_q05": subset_stats(low_q05_mask),
        "low_target_le_q10": subset_stats(low_q10_mask),
        "high_target_ge_q90": subset_stats(high_q90_mask),
        "high_target_ge_q95": subset_stats(high_q95_mask),
        "high_current_absI_ge_q90": {"threshold_A": i90, **subset_stats(high_current_mask)},
        "temperature_event_absdev_ge_q90": {"threshold_C": t90, **subset_stats(temp_event_mask)},
    }
    for key, arr in extra.items():
        arr_m = arr[m]
        out[f"{key}_minmax"] = [float(np.nanmin(arr_m)), float(np.nanmax(arr_m))]
        out[f"{key}_mean"] = float(np.nanmean(arr_m))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute GV1 D9.5.1 prediction voltage diagnostics.")
    ap.add_argument("--prediction_npz", default=None, help="One prediction.npz file")
    ap.add_argument("--root", default=None, help="Root directory; recursively scans */prediction.npz")
    ap.add_argument("--output_json", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.prediction_npz:
        paths.append(Path(args.prediction_npz))
    if args.root:
        paths.extend(sorted(Path(args.root).glob("**/prediction.npz")))
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if not uniq:
        raise FileNotFoundError("No prediction.npz files found. Use --prediction_npz or --root.")

    rows = [_one_metrics(p) for p in uniq]
    payload: Any = rows[0] if len(rows) == 1 else rows
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
