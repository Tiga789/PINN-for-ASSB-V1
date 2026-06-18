from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


INVENTORY_TARGETS = {"theta_a", "theta_c", "cs_a", "cs_c"}
GAUGE_TARGETS = {"phie"}
VOLTAGE_TARGETS = {"phis_c"}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except Exception:
            pass
    return []


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    if not keys:
        keys = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def row_get(row: Dict[str, Any], names: List[str], default: str = "") -> str:
    lower = {k.lower(): k for k in row.keys()}
    for n in names:
        k = lower.get(n.lower())
        if k is not None:
            return str(row.get(k, default))
    return default


def find_metric_csvs(s1_dir: Path) -> List[Path]:
    preferred = sorted(s1_dir.rglob("D17_G7S1_PER_TARGET_PROFILE_METRICS.csv"))
    if preferred:
        return preferred

    candidates = sorted(s1_dir.rglob("*PER_TARGET_PROFILE_METRICS*.csv"))
    # Avoid old duplicate D17_G2 file when possible.
    non_g2 = [p for p in candidates if "D17_G2_" not in p.name]
    return non_g2 or candidates


def dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in rows:
        key = (
            row_get(r, ["split"]),
            row_get(r, ["canonical_cell_uid", "cell_uid", "profile_id"]),
            row_get(r, ["protocol"]),
            row_get(r, ["semantic_branch"]),
            row_get(r, ["target"]),
            row_get(r, ["r2"]),
            row_get(r, ["bias"]),
            row_get(r, ["rmse"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def analyze_row(row: Dict[str, Any]) -> Dict[str, Any]:
    split = row_get(row, ["split"])
    cell = row_get(row, ["canonical_cell_uid", "cell_uid", "profile_id"])
    protocol = row_get(row, ["protocol"])
    branch = row_get(row, ["semantic_branch"])
    target = row_get(row, ["target"])

    n_points = safe_int(row_get(row, ["n_points", "n"]))
    r2 = safe_float(row_get(row, ["r2"]))
    mae = safe_float(row_get(row, ["mae"]))
    rmse = safe_float(row_get(row, ["rmse"]))
    bias = safe_float(row_get(row, ["bias"]))
    target_range = safe_float(row_get(row, ["target_range", "range"]))
    target_std = safe_float(row_get(row, ["target_std", "std"]))

    mse = rmse * rmse if math.isfinite(rmse) else float("nan")
    bias_mse = bias * bias if math.isfinite(bias) else float("nan")
    residual_mse_after_shift = (
        max(mse - bias_mse, 0.0)
        if math.isfinite(mse) and math.isfinite(bias_mse)
        else float("nan")
    )
    unbiased_rmse_proxy = (
        math.sqrt(residual_mse_after_shift)
        if math.isfinite(residual_mse_after_shift)
        else float("nan")
    )
    var_y = target_std * target_std if math.isfinite(target_std) else float("nan")

    shift_corrected_r2_proxy = (
        1.0 - residual_mse_after_shift / max(var_y, 1e-30)
        if math.isfinite(residual_mse_after_shift) and math.isfinite(var_y) and var_y > 0
        else float("nan")
    )
    shift_gain_r2_proxy = (
        shift_corrected_r2_proxy - r2
        if math.isfinite(shift_corrected_r2_proxy) and math.isfinite(r2)
        else float("nan")
    )
    bias_abs = abs(bias) if math.isfinite(bias) else float("nan")
    bias_abs_over_range = (
        bias_abs / target_range
        if math.isfinite(bias_abs) and math.isfinite(target_range) and target_range > 0
        else float("nan")
    )
    bias_mse_fraction = (
        bias_mse / mse
        if math.isfinite(bias_mse) and math.isfinite(mse) and mse > 0
        else float("nan")
    )
    rmse_over_std = (
        rmse / target_std
        if math.isfinite(rmse) and math.isfinite(target_std) and target_std > 0
        else float("nan")
    )

    # Optional rough range diagnostics if pred_min/max are present.
    target_min = safe_float(row_get(row, ["target_min"]))
    target_max = safe_float(row_get(row, ["target_max"]))
    pred_min = safe_float(row_get(row, ["pred_min"]))
    pred_max = safe_float(row_get(row, ["pred_max"]))
    pred_range = pred_max - pred_min if math.isfinite(pred_max) and math.isfinite(pred_min) else float("nan")
    range_ratio_pred_to_target = (
        pred_range / target_range
        if math.isfinite(pred_range) and math.isfinite(target_range) and target_range > 0
        else float("nan")
    )

    tags: List[str] = []
    if math.isfinite(r2) and r2 < 0.90:
        tags.append("low_r2")
    if math.isfinite(bias_mse_fraction) and bias_mse_fraction >= 0.50:
        tags.append("bias_dominated")
    if math.isfinite(bias_abs_over_range) and bias_abs_over_range >= 0.15:
        tags.append("large_shift")
    if target in INVENTORY_TARGETS and math.isfinite(shift_corrected_r2_proxy) and shift_corrected_r2_proxy >= 0.90 and r2 < 0.90:
        tags.append("inventory_shift_may_fix")
    if target in GAUGE_TARGETS and math.isfinite(shift_corrected_r2_proxy) and shift_corrected_r2_proxy >= 0.90 and r2 < 0.90:
        tags.append("phie_gauge_shift_may_fix")
    if math.isfinite(shift_corrected_r2_proxy) and shift_corrected_r2_proxy < 0.90 and r2 < 0.90:
        tags.append("not_fixed_by_constant_shift")
    if math.isfinite(range_ratio_pred_to_target) and (range_ratio_pred_to_target < 0.70 or range_ratio_pred_to_target > 1.30):
        tags.append("range_scale_mismatch_candidate")

    if target in INVENTORY_TARGETS:
        latent_family = "inventory_phase"
    elif target in GAUGE_TARGETS:
        latent_family = "phie_gauge"
    elif target in VOLTAGE_TARGETS:
        latent_family = "voltage_baseline"
    else:
        latent_family = "other"

    return {
        "split": split,
        "canonical_cell_uid": cell,
        "protocol": protocol,
        "semantic_branch": branch,
        "target": target,
        "latent_family": latent_family,
        "n_points": n_points,
        "r2_raw": r2,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "target_range": target_range,
        "target_std": target_std,
        "bias_abs": bias_abs,
        "bias_abs_over_range": bias_abs_over_range,
        "bias_mse_fraction": bias_mse_fraction,
        "rmse_over_target_std": rmse_over_std,
        "unbiased_rmse_proxy_after_constant_shift": unbiased_rmse_proxy,
        "shift_corrected_r2_proxy": shift_corrected_r2_proxy,
        "shift_gain_r2_proxy": shift_gain_r2_proxy,
        "pred_range": pred_range,
        "range_ratio_pred_to_target": range_ratio_pred_to_target,
        "tags": ";".join(tags),
    }


def mean(vals: List[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def min_finite(vals: List[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return min(vals) if vals else float("nan")


def max_finite(vals: List[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else float("nan")


def group_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            str(r["split"]),
            str(r["latent_family"]),
            str(r["target"]),
            str(r["protocol"]),
            str(r["semantic_branch"]),
        )
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for (split, fam, target, protocol, branch), rs in sorted(groups.items()):
        r2s = [safe_float(r["r2_raw"]) for r in rs]
        shift_r2s = [safe_float(r["shift_corrected_r2_proxy"]) for r in rs]
        gains = [safe_float(r["shift_gain_r2_proxy"]) for r in rs]
        bias_fracs = [safe_float(r["bias_mse_fraction"]) for r in rs]
        low_count = sum(1 for r in rs if safe_float(r["r2_raw"]) < 0.90)
        shift_fix_count = sum(1 for r in rs if "shift_may_fix" in str(r["tags"]))
        out.append({
            "split": split,
            "latent_family": fam,
            "target": target,
            "protocol": protocol,
            "semantic_branch": branch,
            "n_rows": len(rs),
            "low_r2_count": low_count,
            "shift_fix_candidate_count": shift_fix_count,
            "r2_raw_mean": mean(r2s),
            "r2_raw_min": min_finite(r2s),
            "shift_corrected_r2_proxy_mean": mean(shift_r2s),
            "shift_corrected_r2_proxy_min": min_finite(shift_r2s),
            "shift_gain_r2_proxy_mean": mean(gains),
            "bias_mse_fraction_mean": mean(bias_fracs),
            "bias_mse_fraction_max": max_finite(bias_fracs),
        })
    return out


def decide(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    low = [r for r in rows if safe_float(r["r2_raw"]) < 0.90]
    inv_low = [r for r in low if r["latent_family"] == "inventory_phase"]
    phie_low = [r for r in low if r["latent_family"] == "phie_gauge"]

    inv_fix = [r for r in inv_low if "inventory_shift_may_fix" in str(r["tags"])]
    phie_fix = [r for r in phie_low if "phie_gauge_shift_may_fix" in str(r["tags"])]
    not_shift = [r for r in low if "not_fixed_by_constant_shift" in str(r["tags"])]

    inv_fix_frac = len(inv_fix) / len(inv_low) if inv_low else float("nan")
    phie_fix_frac = len(phie_fix) / len(phie_low) if phie_low else float("nan")
    not_shift_frac = len(not_shift) / len(low) if low else float("nan")

    if low and not_shift_frac >= 0.50:
        recommendation = "DO_NOT_TRAIN_S1R_YET_FAILURE_NOT_EXPLAINED_BY_SIMPLE_SHIFT"
    elif inv_fix and phie_fix:
        recommendation = "TRY_S1R_PROFILE_LATENT_INVENTORY_AND_PHIE_GAUGE_SHIFT_ADAPTER"
    elif inv_fix:
        recommendation = "TRY_S1R_PROFILE_LATENT_INVENTORY_SHIFT_ADAPTER"
    elif phie_fix:
        recommendation = "TRY_S1R_PHIE_GAUGE_SHIFT_ADAPTER"
    else:
        recommendation = "NO_CLEAR_LATENT_SHIFT_DIRECTION_REVIEW_FEATURES_AND_MODEL_STRUCTURE"

    return {
        "low_r2_row_count": len(low),
        "inventory_low_r2_row_count": len(inv_low),
        "phie_low_r2_row_count": len(phie_low),
        "inventory_shift_fix_candidate_count": len(inv_fix),
        "phie_shift_fix_candidate_count": len(phie_fix),
        "not_fixed_by_constant_shift_count": len(not_shift),
        "inventory_shift_fix_fraction": inv_fix_frac,
        "phie_shift_fix_fraction": phie_fix_frac,
        "not_fixed_by_constant_shift_fraction": not_shift_frac,
        "recommendation": recommendation,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_dir", required=True)
    ap.add_argument("--s1_summary", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--r2_gate", type=float, default=0.90)
    args = ap.parse_args()

    s1_dir = Path(args.s1_dir)
    s1_summary = Path(args.s1_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_in = read_json(s1_summary)
    csvs = find_metric_csvs(s1_dir)
    raw_rows: List[Dict[str, Any]] = []
    for p in csvs:
        for r in read_csv_rows(p):
            if row_get(r, ["target"]) and row_get(r, ["r2"]):
                rr = dict(r)
                rr["_source_csv"] = str(p)
                raw_rows.append(rr)

    raw_rows = dedup_rows(raw_rows)
    rows = [analyze_row(r) for r in raw_rows]
    groups = group_rows(rows)
    decision = decide(rows)

    worst_rows = sorted(rows, key=lambda r: safe_float(r["r2_raw"]))[:30]
    shift_fix_rows = [
        r for r in rows
        if ("inventory_shift_may_fix" in r["tags"] or "phie_gauge_shift_may_fix" in r["tags"])
    ]
    shift_fix_rows = sorted(shift_fix_rows, key=lambda r: safe_float(r["shift_gain_r2_proxy"]), reverse=True)

    out_summary = {
        "protocol": "D17-G7-S1E_PROFILE_LATENT_EXPLAINABILITY_DIAGNOSTIC",
        "status": "PASS",
        "training_performed": False,
        "s1_dir": str(s1_dir),
        "s1_summary": str(s1_summary),
        "source_metric_csvs": [str(p) for p in csvs],
        "raw_metric_row_count_after_dedup": len(rows),
        "decision": decision,
        "s1_core": {
            "status": summary_in.get("status"),
            "selected_cycle_check_ready": summary_in.get("selected_cycle_check_ready"),
            "s2_ready": summary_in.get("s2_ready"),
            "recommendation": summary_in.get("recommendation"),
            "best_epoch": summary_in.get("best_epoch"),
            "fit_train_mean_r2": summary_in.get("fit_train_mean_r2"),
            "fit_train_min_r2": summary_in.get("fit_train_min_r2"),
            "internal_heldout_mean_r2": summary_in.get("internal_heldout_mean_r2"),
            "internal_heldout_min_r2": summary_in.get("internal_heldout_min_r2"),
            "validation_mean_r2": summary_in.get("validation_mean_r2"),
            "validation_min_r2": summary_in.get("validation_min_r2"),
        },
        "interpretation": {
            "constant_shift_proxy": "Uses rmse^2 - bias^2 to estimate the best constant shift correction. This is exact only for mean-shift removal under the metrics convention, and does not replace array-level affine diagnostics.",
            "do_not_enter_s2": True,
            "next_if_shift_fixable": "Implement small S1R adapter with profile-level inventory shift and/or phie gauge shift; do not run S2 directly.",
            "next_if_not_shift_fixable": "Do not train; inspect full profile features and model structure because the error is not reducible to low-dimensional shifts.",
        },
        "worst_rows_preview": worst_rows[:10],
        "shift_fix_candidates_preview": shift_fix_rows[:10],
    }

    summary_path = out_dir / "D17_G7S1E_PROFILE_LATENT_EXPLAINABILITY_SUMMARY.json"
    rows_csv = out_dir / "D17_G7S1E_SHIFT_LATENT_DIAGNOSTIC_ROWS.csv"
    groups_csv = out_dir / "D17_G7S1E_GROUP_SUMMARY.csv"
    worst_csv = out_dir / "D17_G7S1E_TOP_FAILURES.csv"
    fix_csv = out_dir / "D17_G7S1E_SHIFT_FIX_CANDIDATES.csv"

    summary_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows_csv, rows)
    write_csv(groups_csv, groups)
    write_csv(worst_csv, worst_rows)
    write_csv(fix_csv, shift_fix_rows)

    print(json.dumps({
        "status": "PASS",
        "recommendation": decision["recommendation"],
        "decision": decision,
        "raw_metric_row_count_after_dedup": len(rows),
        "summary_json": str(summary_path),
        "diagnostic_rows_csv": str(rows_csv),
        "group_summary_csv": str(groups_csv),
        "top_failures_csv": str(worst_csv),
        "shift_fix_candidates_csv": str(fix_csv),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
