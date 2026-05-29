#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def rows(payload: Any) -> list[dict[str, Any]]:
    return payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []

def f(x: Any, default: float = 0.0) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default

def width(row: dict[str, Any], key: str = "voltage_pred_minmax") -> float:
    mm = row.get(key)
    if not isinstance(mm, list) or len(mm) < 2:
        return 0.0
    return max(0.0, f(mm[1]) - f(mm[0]))

def score(row: dict[str, Any]) -> float:
    mae = f(row.get("mae_V"), 9.0)
    corr = f(row.get("corr"), 0.0)
    bias = abs(f(row.get("bias_V"), 9.0))
    upper = f(row.get("pred_upper_frac_ge_4p269"), 1.0)
    overshoot = f(row.get("pred_overshoot_frac_gt_4p35"), 1.0)
    pred_w = width(row)
    exp_w = width(row, "voltage_exp_minmax")
    collapse_penalty = max(0.0, min(1.10, 0.65 * exp_w) - pred_w)
    return mae + 0.25 * max(0.0, 0.90 - corr) + 0.05 * bias + 2.0 * upper + 6.0 * overshoot + 0.12 * collapse_penalty

def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    mae = f(row.get("mae_V"), 9.0)
    corr = f(row.get("corr"), 0.0)
    upper = f(row.get("pred_upper_frac_ge_4p269"), 1.0)
    overshoot = f(row.get("pred_overshoot_frac_gt_4p35"), 1.0)
    pred_w = width(row)
    exp_w = width(row, "voltage_exp_minmax")
    mm = row.get("voltage_pred_minmax") or [None, None]
    vmax = f(mm[1], 99.0) if isinstance(mm, list) and len(mm) >= 2 else 99.0
    if pred_w < 0.80 or (exp_w > 0 and pred_w < 0.55 * exp_w):
        reasons.append(f"voltage_range_collapsed pred_width={pred_w:.4f}, exp_width={exp_w:.4f}")
    if upper > 0.08:
        reasons.append(f"high_voltage_saturation pred_upper_frac_ge_4p269={upper:.4f}")
    if overshoot > 0.01 or vmax > 4.45:
        reasons.append(f"unsafe_overshoot overshoot_frac={overshoot:.4f}, vmax={vmax:.4f}")
    if mae > 0.14:
        reasons.append(f"mae_too_high={mae:.4f}")
    if corr < 0.88:
        reasons.append(f"corr_too_low={corr:.4f}")
    if reasons:
        return "fail", reasons
    if mae <= 0.10 and corr >= 0.90 and upper <= 0.03 and pred_w >= 1.0:
        return "pass", []
    return "usable", ["usable but not preferred-pass"]

def collect(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.rglob("metrics*.json")):
        if path.name.startswith(("scorecard", "selection")):
            continue
        for row in rows(load_json(path)):
            item = dict(row)
            item["candidate"] = path.parent.name
            item["metrics_json"] = str(path)
            out.append(item)
    return out

def slim(row: dict[str, Any]) -> dict[str, Any]:
    status, reasons = classify(row)
    return {
        "candidate": row.get("candidate") or row.get("run"),
        "status": status,
        "reasons": reasons,
        "score": score(row),
        "mae_V": row.get("mae_V"),
        "rmse_V": row.get("rmse_V"),
        "corr": row.get("corr"),
        "bias_V": row.get("bias_V"),
        "voltage_pred_minmax": row.get("voltage_pred_minmax"),
        "voltage_pred_width_V": width(row),
        "pred_upper_frac_ge_4p269": row.get("pred_upper_frac_ge_4p269"),
        "pred_upper_frac_ge_4p25": row.get("pred_upper_frac_ge_4p25"),
        "pred_overshoot_frac_gt_4p35": row.get("pred_overshoot_frac_gt_4p35"),
        "pred_low_voltage_frac_le_2p75": row.get("pred_low_voltage_frac_le_2p75"),
        "target_low_voltage_frac_le_2p75": row.get("target_low_voltage_frac_le_2p75"),
        "metrics_json": row.get("metrics_json"),
        "prediction_npz": row.get("prediction_npz"),
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Select D9.6.3 candidate for B1_2C battery-8 200ks.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline_json", default=None)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    candidates = [slim(x) for x in collect(Path(args.root))]
    if not candidates:
        raise FileNotFoundError(f"No metrics*.json found below {args.root}")
    candidates.sort(key=lambda x: (0 if x["status"] == "pass" else 1 if x["status"] == "usable" else 2, f(x["score"], 99.0)))
    best = candidates[0]
    baseline = None
    should_replace = False
    if args.baseline_json and Path(args.baseline_json).exists():
        base_rows = rows(load_json(Path(args.baseline_json)))
        if base_rows:
            b = dict(base_rows[0]); b["candidate"] = "original_d96_baseline"; b["metrics_json"] = str(Path(args.baseline_json))
            baseline = slim(b)
            should_replace = (
                best["status"] in {"pass", "usable"}
                and f(best["score"], 99.0) < 0.98 * f(baseline["score"], 99.0)
                and f(best["mae_V"], 9.0) <= f(baseline["mae_V"], 9.0) + 0.003
                and f(best["corr"], 0.0) >= f(baseline["corr"], 0.0) - 0.005
                and f(best["pred_upper_frac_ge_4p269"], 1.0) <= max(0.02, f(baseline["pred_upper_frac_ge_4p269"], 1.0) + 0.005)
            )
    payload = {
        "ok": True,
        "stage": "GV1 D9.6.3 borderline training-strategy selection",
        "root": args.root,
        "n_candidates": len(candidates),
        "status_counts": {k: sum(1 for x in candidates if x["status"] == k) for k in ["pass", "usable", "fail"]},
        "best_candidate": best,
        "baseline_d96": baseline,
        "should_replace_original_d96_for_borderline": bool(should_replace),
        "recommendation": "Candidate clears conservative replacement checks; review plots before promoting." if should_replace else "Keep original D9.6 mainline; D9.6.3 did not clear conservative replacement checks.",
        "candidates": candidates,
    }
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2); out.write_text(text, encoding="utf-8"); print(text)
if __name__ == "__main__":
    main()
