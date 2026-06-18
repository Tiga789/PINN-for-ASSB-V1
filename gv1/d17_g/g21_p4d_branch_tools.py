from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def row_score(row: Mapping[str, Any]) -> float:
    return safe_float(row.get("r2"), 1e99)


def locate_g2_file(summary: Mapping[str, Any], g2_out_dir: str | Path, key: str, fallback_name: str) -> Path:
    files = summary.get("files", {}) if isinstance(summary.get("files"), Mapping) else {}
    candidate = files.get(key)
    if candidate:
        p = Path(str(candidate))
        if p.exists():
            return p
    return Path(g2_out_dir) / fallback_name


def add_row_diagnostics(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r0 in rows:
        r = dict(r0)
        r2 = safe_float(r.get("r2"))
        bias = safe_float(r.get("bias"), 0.0)
        tr = safe_float(r.get("target_range"), float("nan"))
        ts = safe_float(r.get("target_std"), float("nan"))
        mae = safe_float(r.get("mae"), float("nan"))
        r["abs_bias"] = abs(bias) if math.isfinite(bias) else float("nan")
        r["abs_bias_over_target_range"] = (abs(bias) / max(abs(tr), 1e-12)) if math.isfinite(bias) and math.isfinite(tr) else float("nan")
        r["abs_bias_over_target_std"] = (abs(bias) / max(abs(ts), 1e-12)) if math.isfinite(bias) and math.isfinite(ts) else float("nan")
        r["mae_over_target_range"] = (abs(mae) / max(abs(tr), 1e-12)) if math.isfinite(mae) and math.isfinite(tr) else float("nan")
        target = safe_str(r.get("target"))
        branch = safe_str(r.get("semantic_branch"))
        proto = safe_str(r.get("protocol"))
        tags: List[str] = []
        if branch == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH":
            tags.append("P4D_BRANCH")
        if proto in {"random_walk", "GEO"}:
            tags.append("P4D_PROTOCOL_FAMILY")
        if target in {"theta_a", "theta_c", "cs_a", "cs_c"} and safe_float(r.get("abs_bias_over_target_std"), 0.0) > 1.0:
            tags.append("INVENTORY_OR_PHASE_BIAS")
        if target == "phie" and r2 < 0.90:
            tags.append("PHIE_CONVENTION_OR_GAUGE_WEAKNESS")
        if math.isfinite(tr) and abs(tr) < 0.15 and r2 < 0.0:
            tags.append("SMALL_DYNAMIC_RANGE_R2_AMPLIFICATION")
        r["diagnostic_tags"] = ";".join(tags)
        out.append(r)
    return out


def aggregate_rows(rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = {}
    for r in rows:
        k = tuple(safe_str(r.get(f)) for f in group_fields)
        groups.setdefault(k, []).append(r)
    out: List[Dict[str, Any]] = []
    for k, gr in sorted(groups.items()):
        vals = [safe_float(r.get("r2")) for r in gr]
        vals = [v for v in vals if math.isfinite(v)]
        maes = [safe_float(r.get("mae")) for r in gr]
        maes = [v for v in maes if math.isfinite(v)]
        row = {f: k[i] for i, f in enumerate(group_fields)}
        row.update({
            "row_count": len(gr),
            "r2_mean": float(sum(vals) / len(vals)) if vals else float("nan"),
            "r2_min": float(min(vals)) if vals else float("nan"),
            "r2_max": float(max(vals)) if vals else float("nan"),
            "mae_mean": float(sum(maes) / len(maes)) if maes else float("nan"),
            "bad_r2_count_lt_0p90": int(sum(1 for v in vals if v < 0.90)),
            "bad_r2_count_lt_0": int(sum(1 for v in vals if v < 0.0)),
        })
        out.append(row)
    return out


def summarize_g2_failure(g2_summary: Mapping[str, Any], per_target_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = add_row_diagnostics(per_target_rows)
    internal = [r for r in rows if safe_str(r.get("split")) == "train_internal_heldout"]
    validation = [r for r in rows if safe_str(r.get("split")) == "validation_report_only"]
    fit = [r for r in rows if safe_str(r.get("split")) == "train_fit"]
    internal_sorted = sorted(internal, key=row_score)
    validation_sorted = sorted(validation, key=row_score)
    worst_internal = internal_sorted[0] if internal_sorted else None
    worst_validation = validation_sorted[0] if validation_sorted else None
    p4d_internal = [r for r in internal if safe_str(r.get("semantic_branch")) == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH"]
    p4d_fit = [r for r in fit if safe_str(r.get("semantic_branch")) == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH"]
    p4d_validation = [r for r in validation if safe_str(r.get("semantic_branch")) == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH"]

    failure_mode = "UNKNOWN"
    if worst_internal:
        branch = safe_str(worst_internal.get("semantic_branch"))
        proto = safe_str(worst_internal.get("protocol"))
        target = safe_str(worst_internal.get("target"))
        if branch == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH" and proto in {"random_walk", "GEO"} and target in {"theta_a", "theta_c", "cs_a", "cs_c"}:
            failure_mode = "P4D_RANDOM_WALK_OR_GEO_INVENTORY_PHASE_BIAS"
        elif branch == "D15-P4D_FULL_REPLAY_CURRENT_INTEGRAL_BRANCH":
            failure_mode = "P4D_BRANCH_SPECIFIC_FAILURE"
        elif target == "phie":
            failure_mode = "PHIE_GAUGE_FAILURE"
        else:
            failure_mode = "NON_P4D_GENERALIZATION_FAILURE"
    recommendation = "RUN_G21_P4D_BRANCH_REPAIR" if failure_mode.startswith("P4D") else "REVIEW_BEFORE_REPAIR"
    if not internal_sorted:
        recommendation = "MISSING_G2_PER_TARGET_INTERNAL_ROWS"
    return {
        "g2_status": g2_summary.get("status"),
        "g2_g3_ready": g2_summary.get("g3_ready"),
        "g2_recommendation": g2_summary.get("recommendation"),
        "g2_blockers": g2_summary.get("g3_blockers"),
        "failure_mode": failure_mode,
        "recommendation": recommendation,
        "worst_internal_target_profile": worst_internal,
        "worst_validation_target_profile": worst_validation,
        "internal_row_count": len(internal),
        "fit_row_count": len(fit),
        "validation_row_count": len(validation),
        "p4d_internal_row_count": len(p4d_internal),
        "p4d_fit_row_count": len(p4d_fit),
        "p4d_validation_row_count": len(p4d_validation),
        "internal_by_protocol_branch_target": aggregate_rows(internal, ["protocol", "semantic_branch", "target"]),
        "internal_by_profile_target": aggregate_rows(internal, ["canonical_cell_uid", "protocol", "semantic_branch", "target"]),
        "p4d_internal_worst_rows": sorted(p4d_internal, key=row_score)[:24],
        "validation_worst_rows": validation_sorted[:24],
    }


def build_repair_config(base_config: Mapping[str, Any], force_fit_profile_contains: Sequence[str] | None = None) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(dict(base_config), ensure_ascii=False))
    cfg["protocol"] = "D17-G2.1_P4D_BRANCH_REPAIR"
    force = list(cfg.get("force_fit_profile_contains", []))
    for x in (force_fit_profile_contains or ["Batch-4_R3_battery-4", "Batch-5_random_walk_battery-8"]):
        if x and x not in force:
            force.append(x)
    cfg["force_fit_profile_contains"] = force
    cfg["internal_heldout_profile_count"] = int(cfg.get("g21_internal_heldout_profile_count", 6))
    cfg["min_fit_per_group"] = int(cfg.get("g21_min_fit_per_group", 2))
    cfg["max_internal_per_group"] = int(cfg.get("g21_max_internal_per_group", 1))
    w = dict(cfg.get("target_group_weights", {}))
    w.update({"theta_a": 1.8, "theta_c": 1.8, "cs_a": 1.2, "cs_c": 1.2, "phie": 12.0, "phis_c": 3.0})
    cfg["target_group_weights"] = w
    notes = list(cfg.get("notes", []))
    notes.extend([
        "G2.1 pins known P4D random_walk failure profile into fit-train to test coverage vs. model-form failure.",
        "G2.1 uses protocol+branch stratification with min_fit_per_group=2 and max_internal_per_group=1.",
        "Validation remains report-only; frozen-test soft labels are not read."
    ])
    cfg["notes"] = notes
    return cfg
