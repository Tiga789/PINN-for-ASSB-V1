#!/usr/bin/env python
"""
D11-S8 P2D-like low-voltage transport-deficit correction from baseline predictions.

This script performs a *post-transform diagnostic correction* on an existing
baseline prediction.npz.  It does not train a model and does not modify the GV1
mainline.  The purpose is to test whether a P2D-like transport-deficit term can
pull low-voltage target regions downward without using target voltage in the
correction itself.

Correction form:
    V_corr = V_base - deficit_V
    deficit_V = scale * low_gate^p * discharge_gate * current_gate * capacity_gate

Only measured/replay signals and existing model components are used to build the
gates.  Target voltage is only preserved for scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _as_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        return x.reshape(1)
    if x.ndim > 1:
        return x.reshape(-1)
    return x


def _pick_key(data: Dict[str, np.ndarray], candidates) -> Optional[str]:
    keys = set(data.keys())
    for k in candidates:
        if k in keys:
            return k
    # permissive lowercase contains search
    lower = {k.lower(): k for k in data.keys()}
    for cand in candidates:
        c = cand.lower()
        if c in lower:
            return lower[c]
    return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _safe_norm_abs_current(i: Optional[np.ndarray], n: int) -> Tuple[np.ndarray, np.ndarray]:
    if i is None:
        return np.ones(n, dtype=np.float64), np.ones(n, dtype=np.float64)
    i = _as_1d(i).astype(np.float64)
    if i.size != n:
        i = np.resize(i, n)
    abs_i = np.abs(i)
    scale = np.nanpercentile(abs_i[abs_i > 0], 90) if np.any(abs_i > 0) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    current_gate = np.clip(abs_i / scale, 0.0, 2.0) / 2.0 + 0.5
    # XJTU convention in this workflow: discharge current is negative.
    # Make this soft to avoid overfitting exact zero/near-zero signs.
    s = max(scale, 1e-12)
    discharge_gate = _sigmoid((-i) / (0.15 * s))
    # Rest points should not receive large transport correction.
    rest_mask = abs_i < max(1e-12, 0.03 * scale)
    discharge_gate = np.where(rest_mask, 0.0, discharge_gate)
    return current_gate.astype(np.float64), discharge_gate.astype(np.float64)


def _capacity_gate(i: Optional[np.ndarray], n: int) -> np.ndarray:
    if i is None:
        return np.ones(n, dtype=np.float64)
    i = _as_1d(i).astype(np.float64)
    if i.size != n:
        i = np.resize(i, n)
    discharge = np.clip(-i, 0.0, None)
    cum = np.cumsum(discharge)
    mx = np.nanmax(cum) if cum.size else 0.0
    if not np.isfinite(mx) or mx <= 0:
        return np.ones(n, dtype=np.float64)
    q = cum / mx
    return np.clip(0.15 + 0.85 * np.power(q, 0.75), 0.0, 1.0).astype(np.float64)


def _mode_params(mode: str):
    # scale_V, low_power, high_target_guard_center, high_target_guard_width, max_deficit
    params = {
        "baseline_copy": dict(scale=0.0, power=1.0, current_power=0.0, cap_power=0.0, guard_center=3.35, guard_width=0.18, max_deficit=0.0),
        "p2dlike_transport_mild": dict(scale=0.22, power=1.15, current_power=0.7, cap_power=0.6, guard_center=3.35, guard_width=0.18, max_deficit=0.35),
        "p2dlike_transport_medium": dict(scale=0.38, power=1.10, current_power=0.8, cap_power=0.7, guard_center=3.40, guard_width=0.20, max_deficit=0.55),
        "p2dlike_transport_strong_guarded": dict(scale=0.60, power=1.05, current_power=0.8, cap_power=0.75, guard_center=3.45, guard_width=0.22, max_deficit=0.75),
        "p2dlike_transport_discharge_only": dict(scale=0.48, power=1.20, current_power=0.0, cap_power=0.6, guard_center=3.40, guard_width=0.20, max_deficit=0.60),
    }
    if mode not in params:
        raise ValueError(f"Unknown mode: {mode}. Available={sorted(params)}")
    return params[mode]


def _build_low_gate(data: Dict[str, np.ndarray], pred: np.ndarray, ocv_key: Optional[str], n: int, center: float, width: float) -> Tuple[np.ndarray, str]:
    if "voltage_low_gate" in data:
        g = _as_1d(data["voltage_low_gate"]).astype(np.float64)
        if g.size != n:
            g = np.resize(g, n)
        return np.clip(g, 0.0, 1.0), "voltage_low_gate"
    if ocv_key is not None:
        base = _as_1d(data[ocv_key]).astype(np.float64)
        if base.size != n:
            base = np.resize(base, n)
        return _sigmoid((center - base) / width), ocv_key
    # Last-resort proxy.  This will under-activate if pred is stuck high, which
    # is useful diagnostic evidence.
    return _sigmoid((center - pred) / width), "predicted_voltage_proxy"


def main() -> None:
    ap = argparse.ArgumentParser(description="D11-S8 post-transform P2D-like transport correction from baseline prediction.npz")
    ap.add_argument("--input_prediction", required=True)
    ap.add_argument("--output_prediction", required=True)
    ap.add_argument("--mode", required=True, choices=[
        "baseline_copy", "p2dlike_transport_mild", "p2dlike_transport_medium", "p2dlike_transport_strong_guarded", "p2dlike_transport_discharge_only"
    ])
    ap.add_argument("--profile", default="")
    ap.add_argument("--protocol", default="")
    args = ap.parse_args()

    in_path = Path(args.input_prediction)
    out_path = Path(args.output_prediction)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(in_path, allow_pickle=True)
    data = {k: z[k] for k in z.files}

    pred_key = _pick_key(data, ["voltage_exp_pred", "voltage_pred", "phis_c_pred", "pred_voltage", "y_pred", "pred"])
    true_key = _pick_key(data, ["voltage_exp_true", "voltage_true", "target_voltage", "voltage_exp", "y_true", "true"])
    current_key = _pick_key(data, ["I_profile", "current_A", "current", "I", "current_profile"])
    ocv_key = _pick_key(data, ["voltage_ocv_baseline", "voltage_ocv", "ocv_baseline"])

    if pred_key is None:
        raise KeyError(f"Could not identify predicted voltage key in {in_path}; keys={sorted(data.keys())[:80]}")
    pred = _as_1d(data[pred_key]).astype(np.float64)
    n = pred.size
    current = _as_1d(data[current_key]).astype(np.float64) if current_key is not None else None

    p = _mode_params(args.mode)
    low_gate, low_gate_source = _build_low_gate(data, pred, ocv_key, n, center=3.08, width=0.18)
    current_gate, discharge_gate = _safe_norm_abs_current(current, n)
    cap_gate = _capacity_gate(current, n)
    high_target_guard_base = _as_1d(data[ocv_key]).astype(np.float64) if ocv_key is not None else pred
    if high_target_guard_base.size != n:
        high_target_guard_base = np.resize(high_target_guard_base, n)
    # guard is near 1 in low-voltage regime and near 0 at high-voltage regime.
    high_guard = _sigmoid((p["guard_center"] - high_target_guard_base) / max(p["guard_width"], 1e-6))

    deficit = (
        p["scale"]
        * np.power(np.clip(low_gate, 0.0, 1.0), p["power"])
        * discharge_gate
        * np.power(np.clip(current_gate, 0.0, 1.0), p["current_power"])
        * np.power(np.clip(cap_gate, 0.0, 1.0), p["cap_power"])
        * np.clip(high_guard, 0.0, 1.0)
    )
    deficit = np.clip(deficit, 0.0, p["max_deficit"])
    corrected = pred - deficit

    # Preserve original arrays and append diagnostic arrays.
    out = dict(data)
    out[f"{pred_key}_D11S8_baseline"] = pred.astype(np.float32)
    out[pred_key] = corrected.astype(np.float32)
    if pred_key != "voltage_exp_pred" and "voltage_exp_pred" in out:
        out["voltage_exp_pred_D11S8_baseline"] = _as_1d(out["voltage_exp_pred"]).astype(np.float32)
        out["voltage_exp_pred"] = corrected.astype(np.float32)
    elif "voltage_exp_pred" not in out:
        out["voltage_exp_pred"] = corrected.astype(np.float32)
    out["d11_s8_transport_deficit_V"] = deficit.astype(np.float32)
    out["d11_s8_low_gate"] = low_gate.astype(np.float32)
    out["d11_s8_discharge_gate"] = discharge_gate.astype(np.float32)
    out["d11_s8_current_gate"] = current_gate.astype(np.float32)
    out["d11_s8_capacity_gate"] = cap_gate.astype(np.float32)
    out["d11_s8_high_voltage_guard"] = high_guard.astype(np.float32)
    out["d11_s8_mode"] = np.array(args.mode)
    out["d11_s8_profile"] = np.array(args.profile)
    out["d11_s8_protocol"] = np.array(args.protocol)

    np.savez_compressed(out_path, **out)

    summary = {
        "ok": True,
        "mode": args.mode,
        "profile": args.profile,
        "protocol": args.protocol,
        "input_prediction": str(in_path),
        "output_prediction": str(out_path),
        "pred_key": pred_key,
        "true_key": true_key,
        "current_key": current_key,
        "ocv_key": ocv_key,
        "low_gate_source": low_gate_source,
        "deficit_mean_V": float(np.nanmean(deficit)),
        "deficit_max_V": float(np.nanmax(deficit)),
        "low_gate_mean": float(np.nanmean(low_gate)),
        "discharge_gate_mean": float(np.nanmean(discharge_gate)),
    }
    out_path.with_suffix(".d11_s8_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
