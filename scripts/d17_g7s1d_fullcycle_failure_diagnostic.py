from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        for k in r.keys():
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


def discover_csv_files(s1_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(s1_dir.rglob("*.csv")):
        try:
            head_rows = read_csv_rows(p)[:3]
            cols = list(head_rows[0].keys()) if head_rows else []
            rows.append({
                "path": str(p),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "columns": "|".join(cols),
                "looks_like_metric": str(any(c.lower() in {"r2", "mae", "rmse", "bias", "target", "split", "canonical_cell_uid"} for c in cols)),
            })
        except Exception as e:
            rows.append({
                "path": str(p),
                "name": p.name,
                "size_bytes": "",
                "columns": "",
                "looks_like_metric": "False",
                "error": f"{type(e).__name__}: {e}",
            })
    return rows


def collect_metric_rows(csv_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for info in csv_files:
        if info.get("looks_like_metric") != "True":
            continue
        p = Path(info["path"])
        rows = read_csv_rows(p)
        for r in rows:
            lower_keys = {k.lower(): k for k in r.keys()}
            if "r2" not in lower_keys or "target" not in lower_keys:
                continue
            rr = dict(r)
            rr["_source_csv"] = str(p)
            all_rows.append(rr)
    return all_rows


def row_get(row: Dict[str, Any], names: List[str], default: str = "") -> str:
    lower = {k.lower(): k for k in row.keys()}
    for n in names:
        k = lower.get(n.lower())
        if k is not None:
            return str(row.get(k, default))
    return default


def enrich_metric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(row)
    target = row_get(row, ["target"], "")
    split = row_get(row, ["split"], "")
    cell = row_get(row, ["canonical_cell_uid", "cell_uid", "profile", "profile_id"], "")
    protocol = row_get(row, ["protocol"], "")
    branch = row_get(row, ["semantic_branch"], "")

    r2 = safe_float(row_get(row, ["r2"], ""))
    mae = safe_float(row_get(row, ["mae"], ""))
    rmse = safe_float(row_get(row, ["rmse"], ""))
    bias = safe_float(row_get(row, ["bias"], ""))
    target_range = safe_float(row_get(row, ["target_range", "range"], ""))
    target_std = safe_float(row_get(row, ["target_std", "std"], ""))

    bias_abs = abs(bias) if math.isfinite(bias) else float("nan")
    bias_over_range = bias_abs / target_range if math.isfinite(bias_abs) and math.isfinite(target_range) and target_range > 0 else float("nan")
    rmse_over_std = rmse / target_std if math.isfinite(rmse) and math.isfinite(target_std) and target_std > 0 else float("nan")

    failure_tags = []
    if math.isfinite(r2) and r2 < 0.90:
        failure_tags.append("low_r2")
    if math.isfinite(bias_over_range) and bias_over_range >= 0.15:
        failure_tags.append("dominant_bias")
    if target in {"theta_a", "theta_c", "cs_a", "cs_c"} and math.isfinite(bias_over_range) and bias_over_range >= 0.15:
        failure_tags.append("inventory_phase_offset_candidate")
    if target == "phie" and math.isfinite(bias_abs) and bias_abs >= 0.02:
        failure_tags.append("phie_gauge_offset_candidate")
    if target == "phis_c" and math.isfinite(bias_abs) and bias_abs >= 0.02:
        failure_tags.append("phis_c_voltage_bias_candidate")
    if math.isfinite(rmse_over_std) and rmse_over_std >= 0.5:
        failure_tags.append("shape_or_scale_error")

    r["_split"] = split
    r["_cell"] = cell
    r["_protocol"] = protocol
    r["_branch"] = branch
    r["_target"] = target
    r["_r2"] = r2
    r["_mae"] = mae
    r["_rmse"] = rmse
    r["_bias"] = bias
    r["_target_range"] = target_range
    r["_target_std"] = target_std
    r["_bias_abs"] = bias_abs
    r["_bias_abs_over_range"] = bias_over_range
    r["_rmse_over_target_std"] = rmse_over_std
    r["_failure_tags"] = ";".join(failure_tags)
    return r


def mean(vals: List[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def group_aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            str(r.get("_split", "")),
            str(r.get("_target", "")),
            str(r.get("_protocol", "")),
            str(r.get("_branch", "")),
        )
        groups.setdefault(key, []).append(r)

    out = []
    for (split, target, protocol, branch), rs in sorted(groups.items()):
        r2s = [safe_float(r.get("_r2")) for r in rs]
        biases = [safe_float(r.get("_bias")) for r in rs]
        bias_ratios = [safe_float(r.get("_bias_abs_over_range")) for r in rs]
        out.append({
            "split": split,
            "target": target,
            "protocol": protocol,
            "semantic_branch": branch,
            "n_rows": len(rs),
            "r2_mean": mean(r2s),
            "r2_min": min([v for v in r2s if math.isfinite(v)], default=float("nan")),
            "bias_mean": mean(biases),
            "bias_abs_over_range_mean": mean(bias_ratios),
            "bias_abs_over_range_max": max([v for v in bias_ratios if math.isfinite(v)], default=float("nan")),
        })
    return out


def compact_metric_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "split": r.get("_split", ""),
        "canonical_cell_uid": r.get("_cell", ""),
        "protocol": r.get("_protocol", ""),
        "semantic_branch": r.get("_branch", ""),
        "target": r.get("_target", ""),
        "r2": r.get("_r2", ""),
        "mae": r.get("_mae", ""),
        "rmse": r.get("_rmse", ""),
        "bias": r.get("_bias", ""),
        "target_range": r.get("_target_range", ""),
        "target_std": r.get("_target_std", ""),
        "bias_abs_over_range": r.get("_bias_abs_over_range", ""),
        "rmse_over_target_std": r.get("_rmse_over_target_std", ""),
        "failure_tags": r.get("_failure_tags", ""),
        "source_csv": r.get("_source_csv", ""),
    }


def summarize_from_json_only(s1: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fit_train_mean_r2": s1.get("fit_train_mean_r2"),
        "fit_train_min_r2": s1.get("fit_train_min_r2"),
        "internal_heldout_mean_r2": s1.get("internal_heldout_mean_r2"),
        "internal_heldout_min_r2": s1.get("internal_heldout_min_r2"),
        "validation_mean_r2": s1.get("validation_mean_r2"),
        "validation_min_r2": s1.get("validation_min_r2"),
        "worst_internal_target_profile": s1.get("worst_internal_target_profile", {}),
        "worst_validation_target_profile": s1.get("worst_validation_target_profile", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_dir", required=True)
    ap.add_argument("--s1_summary", required=True)
    ap.add_argument("--s0_summary", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    s1_dir = Path(args.s1_dir)
    s1_summary_path = Path(args.s1_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = read_json(s1_summary_path)
    s0 = read_json(Path(args.s0_summary)) if args.s0_summary and Path(args.s0_summary).exists() else {}

    csv_files = discover_csv_files(s1_dir)
    raw_metric_rows = collect_metric_rows(csv_files)
    metric_rows = [enrich_metric_row(r) for r in raw_metric_rows]

    worst_rows = sorted(
        metric_rows,
        key=lambda r: safe_float(r.get("_r2")),
    )[:30]
    worst_compact = [compact_metric_row(r) for r in worst_rows]
    groups = group_aggregate(metric_rows)

    fit = safe_float(s1.get("fit_train_mean_r2"))
    internal = safe_float(s1.get("internal_heldout_mean_r2"))
    validation = safe_float(s1.get("validation_mean_r2"))

    flags = []
    if math.isfinite(fit) and fit >= 0.98 and math.isfinite(internal) and internal < 0.90:
        flags.append("fit_train_overfit_internal_generalization_failure")
    if math.isfinite(fit) and fit >= 0.98 and math.isfinite(validation) and validation < 0.90:
        flags.append("fit_train_overfit_validation_generalization_failure")

    json_only = summarize_from_json_only(s1)

    if not metric_rows:
        recommendation = "LIMITED_DIAGNOSTIC_NO_DETAILED_METRICS_CSV_FOUND_USE_SUMMARY_ONLY"
    else:
        has_inventory = any("inventory_phase_offset_candidate" in str(r.get("_failure_tags", "")) for r in metric_rows)
        has_phie = any("phie_gauge_offset_candidate" in str(r.get("_failure_tags", "")) for r in metric_rows)
        if has_inventory and has_phie:
            recommendation = "DO_NOT_ENTER_S2_DIAGNOSE_INVENTORY_AND_PHIE_GAUGE_LATENTS"
        elif has_inventory:
            recommendation = "DO_NOT_ENTER_S2_DIAGNOSE_INVENTORY_PHASE_LATENT"
        elif has_phie:
            recommendation = "DO_NOT_ENTER_S2_DIAGNOSE_PHIE_GAUGE_LATENT"
        else:
            recommendation = "DO_NOT_ENTER_S2_REVIEW_GENERALIZATION_FAILURE"

    summary = {
        "protocol": "D17-G7-S1D_FULLCYCLE_FAILURE_DIAGNOSTIC",
        "status": "PASS",
        "training_performed": False,
        "recommendation": recommendation,
        "s1_summary": str(s1_summary_path),
        "s1_dir": str(s1_dir),
        "s0_summary": str(args.s0_summary or ""),
        "json_only_core": json_only,
        "overfit_flags": flags,
        "csv_file_count": len(csv_files),
        "metric_row_count": len(metric_rows),
        "worst_rows_preview": worst_compact[:10],
        "group_aggregate_preview": groups[:20],
        "interpretation": {
            "do_not_enter_s2": True,
            "reason": "S1 fit-train is high but internal/validation are low; diagnose profile-level inventory/gauge drift before any larger training.",
            "primary_questions": [
                "Is theta/cs failure dominated by constant inventory phase bias?",
                "Is phie failure dominated by gauge bias?",
                "Are failures concentrated in protocol/branch groups?",
                "Do detailed S1 metrics exist, or is only summary-level evidence available?"
            ],
        },
    }

    summary_path = out_dir / "D17_G7S1D_FULLCYCLE_FAILURE_DIAGNOSTIC_SUMMARY.json"
    file_csv = out_dir / "D17_G7S1D_FILE_DISCOVERY.csv"
    worst_csv = out_dir / "D17_G7S1D_WORST_TARGET_PROFILE_ROWS.csv"
    group_csv = out_dir / "D17_G7S1D_GROUP_AGGREGATES.csv"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(file_csv, csv_files)
    write_csv(worst_csv, worst_compact)
    write_csv(group_csv, groups)

    print(json.dumps({
        "status": "PASS",
        "recommendation": recommendation,
        "metric_row_count": len(metric_rows),
        "overfit_flags": flags,
        "summary_json": str(summary_path),
        "worst_rows_csv": str(worst_csv),
        "group_aggregates_csv": str(group_csv),
        "file_discovery_csv": str(file_csv),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
