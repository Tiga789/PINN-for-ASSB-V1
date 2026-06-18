from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Mapping[str, Any], path: str | Path) -> None:
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


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fields.append(k)
                seen.add(k)
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def finite(x: Any) -> bool:
    try:
        y = float(x)
        return math.isfinite(y)
    except Exception:
        return False


def win_path_resolve(path: Optional[str | Path], base_dir: Optional[str | Path] = None) -> Optional[Path]:
    if path is None:
        return None
    s = str(path)
    if not s:
        return None
    # On Windows this is a normal path. On POSIX test machines it will not exist;
    # still return a Path object without mutating the string too much.
    p = Path(s)
    if p.exists():
        return p
    if base_dir is not None:
        cand = Path(base_dir) / s
        if cand.exists():
            return cand
    # Try basename relative to base_dir as a robust fallback.
    if base_dir is not None:
        cand = Path(base_dir) / Path(s).name
        if cand.exists():
            return cand
    # Try slash-normalized path.
    p2 = Path(s.replace("\\", "/"))
    if p2.exists():
        return p2
    return p


def infer_g14_file(summary: Mapping[str, Any], g14_out_dir: str | Path, key: str, fallback_name: str) -> Path:
    files = summary.get("files", {}) if isinstance(summary.get("files"), Mapping) else {}
    p = win_path_resolve(files.get(key), g14_out_dir)
    if p is not None and p.exists():
        return p
    return Path(g14_out_dir) / fallback_name


def target_rows_by_split(rows: Sequence[Mapping[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows if str(r.get("split", "")) == split]


def profile_key(row: Mapping[str, Any]) -> str:
    return str(row.get("canonical_cell_uid") or row.get("cell_uid") or row.get("profile") or row.get("profile_id") or "UNKNOWN")


def group_rows_by_profile(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[profile_key(row)].append(dict(row))
    return dict(out)


def aggregate_profile_target_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_prof = group_rows_by_profile(rows)
    out: List[Dict[str, Any]] = []
    for prof, rs in by_prof.items():
        vals = [as_float(r.get("r2")) for r in rs]
        vals = [v for v in vals if math.isfinite(v)]
        worst = min(rs, key=lambda r: as_float(r.get("r2"), 1e99)) if rs else {}
        first = rs[0] if rs else {}
        out.append({
            "canonical_cell_uid": prof,
            "split": first.get("split", ""),
            "protocol": first.get("protocol", ""),
            "semantic_branch": first.get("semantic_branch", ""),
            "profile_target_r2_mean": float(sum(vals) / len(vals)) if vals else float("nan"),
            "profile_target_r2_min": float(min(vals)) if vals else float("nan"),
            "worst_target": worst.get("target", ""),
            "worst_target_r2": as_float(worst.get("r2")),
            "worst_target_mae": as_float(worst.get("mae")),
            "worst_target_range": as_float(worst.get("target_range")),
            "target_count": len(rs),
        })
    out.sort(key=lambda r: as_float(r.get("profile_target_r2_min"), 1e99))
    return out


def protocol_branch_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    c_proto = Counter(str(r.get("protocol", "")) for r in rows)
    c_branch = Counter(str(r.get("semantic_branch", "")) for r in rows)
    return {"protocol_counts": dict(c_proto), "semantic_branch_counts": dict(c_branch)}


def numeric_feature_columns(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    if not rows:
        return []
    exclude = {
        "split", "canonical_cell_uid", "cell_uid", "protocol", "semantic_branch", "branch", "source_stage",
        "replay_npz", "softlabel_npz", "softlabel_dir", "profile", "profile_index", "battery", "batch",
    }
    cols: List[str] = []
    for k in rows[0].keys():
        if k in exclude:
            continue
        vals = [as_float(r.get(k)) for r in rows[: min(len(rows), 20)]]
        good = [v for v in vals if math.isfinite(v)]
        if good:
            cols.append(k)
    return cols


def feature_coverage_audit(feature_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute rough distance of heldout/validation profiles to fit-train feature cloud.

    This is intentionally conservative: if feature audit CSV schema is unknown or
    missing, it reports missing instead of failing.
    """
    if not feature_rows:
        return [], {"status": "MISSING", "reason": "profile encoder feature audit CSV not found or empty"}
    split_values = sorted(set(str(r.get("split", "")) for r in feature_rows))
    fit_rows = [r for r in feature_rows if str(r.get("split", "")) in {"train_fit", "fit_train", "train"}]
    if not fit_rows:
        # Some older audit files may not carry split. Treat all rows as reference;
        # still produce per-profile stats if possible.
        fit_rows = list(feature_rows)
    cols = numeric_feature_columns(fit_rows)
    if not cols:
        return [], {"status": "MISSING", "reason": "no numeric feature columns detected", "split_values": split_values}
    ref_mean: Dict[str, float] = {}
    ref_std: Dict[str, float] = {}
    ref_min: Dict[str, float] = {}
    ref_max: Dict[str, float] = {}
    for c in cols:
        vals = [as_float(r.get(c)) for r in fit_rows]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        ref_mean[c] = float(sum(vals) / len(vals))
        var = sum((v - ref_mean[c]) ** 2 for v in vals) / max(1, len(vals) - 1)
        ref_std[c] = float(math.sqrt(max(var, 0.0)) or 1.0)
        ref_min[c] = float(min(vals))
        ref_max[c] = float(max(vals))
    rows_out: List[Dict[str, Any]] = []
    for r in feature_rows:
        prof = profile_key(r)
        z_abs: List[float] = []
        outside = 0
        max_feature = ""
        max_z = -1.0
        for c in cols:
            v = as_float(r.get(c))
            if not math.isfinite(v) or c not in ref_mean:
                continue
            z = abs((v - ref_mean[c]) / max(ref_std.get(c, 1.0), 1e-12))
            z_abs.append(z)
            if z > max_z:
                max_z = z
                max_feature = c
            if v < ref_min[c] or v > ref_max[c]:
                outside += 1
        rows_out.append({
            "canonical_cell_uid": prof,
            "split": r.get("split", ""),
            "protocol": r.get("protocol", ""),
            "semantic_branch": r.get("semantic_branch", ""),
            "feature_z_mean_abs": float(sum(z_abs) / len(z_abs)) if z_abs else float("nan"),
            "feature_z_max_abs": float(max(z_abs)) if z_abs else float("nan"),
            "feature_z_max_feature": max_feature,
            "features_outside_fit_minmax": outside,
            "numeric_feature_count": len(z_abs),
        })
    rows_out.sort(key=lambda r: (as_float(r.get("features_outside_fit_minmax"), -1), as_float(r.get("feature_z_max_abs"), -1)), reverse=True)
    return rows_out, {"status": "PASS", "numeric_feature_count": len(cols), "split_values": split_values}


def read_g0_semantics(path: Optional[str | Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    rows = read_csv_rows(p)
    out = {}
    for r in rows:
        for key in [r.get("canonical_cell_uid"), r.get("cell_uid")]:
            if key:
                out[str(key)] = dict(r)
    return out


def annotate_with_semantics(rows: Sequence[Mapping[str, Any]], sem: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        s = sem.get(str(r.get("canonical_cell_uid", "")), {})
        for k in ["source_stage", "semantic_branch", "cbar_source_a", "cbar_source_c", "J_source_a", "J_source_c", "phis_c_source", "phie_source"]:
            if k not in r or not r.get(k):
                if k in s:
                    r[k] = s.get(k)
        out.append(r)
    return out


def target_failure_summary(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    split_rows = target_rows_by_split(rows, split)
    by_target: Dict[str, List[float]] = defaultdict(list)
    for r in split_rows:
        t = str(r.get("target", ""))
        v = as_float(r.get("r2"))
        if t and math.isfinite(v):
            by_target[t].append(v)
    out: Dict[str, Any] = {"split": split, "target_count": len(by_target)}
    vals_all = []
    for t, vals in sorted(by_target.items()):
        out[f"{t}_r2_mean"] = float(sum(vals) / len(vals)) if vals else float("nan")
        out[f"{t}_r2_min"] = float(min(vals)) if vals else float("nan")
        vals_all.extend(vals)
    out["all_target_profile_r2_mean"] = float(sum(vals_all) / len(vals_all)) if vals_all else float("nan")
    out["all_target_profile_r2_min"] = float(min(vals_all)) if vals_all else float("nan")
    return out


def decide(summary: Mapping[str, Any], internal_rank: Sequence[Mapping[str, Any]], validation_rank: Sequence[Mapping[str, Any]], coverage_rows: Sequence[Mapping[str, Any]]) -> Tuple[str, List[str], List[str]]:
    blockers: List[str] = []
    actions: List[str] = []
    int_agg = summary.get("internal_heldout_per_target_aggregate", {}) if isinstance(summary.get("internal_heldout_per_target_aggregate"), Mapping) else {}
    val_agg = summary.get("validation_report_only_per_target_aggregate", {}) if isinstance(summary.get("validation_report_only_per_target_aggregate"), Mapping) else {}
    int_min = as_float(int_agg.get("all_target_profile_r2_min"))
    int_mean = as_float(int_agg.get("all_target_profile_r2_mean"))
    val_min = as_float(val_agg.get("all_target_profile_r2_min"))
    val_mean = as_float(val_agg.get("all_target_profile_r2_mean"))
    if math.isfinite(int_min) and int_min < 0.90:
        blockers.append(f"internal-heldout min R2 is below 0.90: {int_min:.6g}")
    if math.isfinite(int_mean) and int_mean < 0.95:
        blockers.append(f"internal-heldout mean R2 is below 0.95: {int_mean:.6g}")
    if internal_rank:
        w = internal_rank[0]
        blockers.append(f"worst internal-heldout profile={w.get('canonical_cell_uid')} target={w.get('worst_target')} r2={as_float(w.get('worst_target_r2')):.6g}")
        if as_float(w.get("profile_target_r2_min")) < 0.0:
            actions.append("Do not enter G2; isolate the worst internal-heldout profile and rerun G1.4/G1.5 with it moved into fit-train to test whether the failure is coverage or model form.")
    if math.isfinite(val_mean) and val_mean >= 0.95 and math.isfinite(val_min) and val_min >= 0.90:
        actions.append("Validation report-only is healthy; do not tune validation further. Focus only on train-internal heldout coverage/stability.")
    if coverage_rows:
        worst_cov = coverage_rows[0]
        if as_float(worst_cov.get("features_outside_fit_minmax"), 0.0) > 0:
            actions.append(f"Check feature coverage for {worst_cov.get('canonical_cell_uid')}: {worst_cov.get('features_outside_fit_minmax')} profile-summary features outside fit-train min/max.")
    if not actions:
        actions.append("Run targeted G1.5R with stratified internal-heldout and protocol-balanced fit-train; keep G1.4 validation report-only policy unchanged.")
    recommendation = "DO_NOT_ENTER_G2_RUN_G15R_STRATIFIED_HELDOUT_OR_COVERAGE_REPAIR" if blockers else "G1_5_TRIAGE_PASS_CONSIDER_G2_AFTER_CONFIRMING_FULL_TRAIN_INTERNAL_HELDOUT"
    return recommendation, blockers, actions


def make_candidate_config(base_config: Mapping[str, Any], summary: Mapping[str, Any], internal_rank: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = dict(base_config or {})
    # Conservative defaults: keep G1.4 architecture, remove the validation-driven focus,
    # and make train-internal heldout less brittle by stratifying in the next training script.
    cfg.setdefault("seed", 20260615)
    cfg["protocol"] = "D17-G1.5R_RECOMMENDED_STRATIFIED_INTERNAL_HELDOUT_REPAIR_CONFIG"
    cfg["note"] = "Generated by G1.5 triage. This is a recommendation file, not executed by the G1.5 diagnostic itself."
    cfg["recommended_changes"] = {
        "checkpoint_selection": "Use train-internal heldout all-target mean/min plus phie, not validation labels.",
        "internal_heldout_policy": "stratified_by_protocol_and_semantic_branch_or_rotate_kfold; avoid last-N contiguous profile holdout.",
        "phie_focus": "keep dedicated phie head but reduce phie-only early over-focus if internal heldout all-state metrics drop.",
        "coverage": "ensure every protocol present in internal-heldout is represented in fit-train by at least 2 profiles.",
        "do_not_use": "validation/frozen_test labels for training or checkpoint selection.",
    }
    cfg["target_group_weights_suggested"] = {
        "theta_a": 1.5,
        "theta_c": 1.5,
        "cs_a": 1.0,
        "cs_c": 1.0,
        "phie": 12.0,
        "phis_c": 3.0,
    }
    cfg["internal_heldout_profile_count"] = int(summary.get("dataset", {}).get("internal_heldout_profile_count", 4)) if isinstance(summary.get("dataset"), Mapping) else 4
    if internal_rank:
        cfg["known_worst_internal_heldout_profile_from_g14"] = internal_rank[0]
    return cfg


def run_g15_triage(
    g14_summary: str | Path,
    g14_out_dir: str | Path,
    out_dir: str | Path,
    config: Optional[Mapping[str, Any]] = None,
    g13_summary: Optional[str | Path] = None,
    g12_summary: Optional[str | Path] = None,
    g0_profile_semantics_csv: Optional[str | Path] = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(config or {})
    s14 = load_json(g14_summary)
    s13 = load_json(g13_summary) if g13_summary and Path(g13_summary).exists() else {}
    s12 = load_json(g12_summary) if g12_summary and Path(g12_summary).exists() else {}
    sem = read_g0_semantics(g0_profile_semantics_csv)

    per_target_csv = infer_g14_file(s14, g14_out_dir, "per_target_profile_metrics_csv", "D17_G14_PER_TARGET_PROFILE_METRICS.csv")
    profile_csv = infer_g14_file(s14, g14_out_dir, "profile_metrics_csv", "D17_G14_PROFILE_METRICS.csv")
    phie_csv = infer_g14_file(s14, g14_out_dir, "phie_robustness_audit_csv", "D17_G14_PHIE_ROBUSTNESS_AUDIT.csv")
    feature_csv = infer_g14_file(s14, g14_out_dir, "profile_encoder_feature_audit_csv", "D17_G14_PROFILE_ENCODER_FEATURE_AUDIT.csv")
    train_hist_csv = infer_g14_file(s14, g14_out_dir, "training_history_csv", "D17_G14_training_history.csv")

    per_target_rows = read_csv_rows(per_target_csv)
    profile_rows = read_csv_rows(profile_csv)
    phie_rows = read_csv_rows(phie_csv)
    feature_rows = read_csv_rows(feature_csv)
    train_hist = read_csv_rows(train_hist_csv)

    internal_trows = target_rows_by_split(per_target_rows, "train_internal_heldout")
    val_trows = target_rows_by_split(per_target_rows, "validation_report_only")
    fit_trows = target_rows_by_split(per_target_rows, "train_fit")
    internal_rank = annotate_with_semantics(aggregate_profile_target_rows(internal_trows), sem)
    val_rank = annotate_with_semantics(aggregate_profile_target_rows(val_trows), sem)
    fit_rank = aggregate_profile_target_rows(fit_trows)

    coverage_rows, coverage_summary = feature_coverage_audit(feature_rows)
    coverage_rows = annotate_with_semantics(coverage_rows, sem)
    # Keep top coverage rows in output but write full CSV separately.

    split_composition = {
        "train_fit_per_target_rows": protocol_branch_counts(fit_trows),
        "train_internal_heldout_per_target_rows": protocol_branch_counts(internal_trows),
        "validation_report_only_per_target_rows": protocol_branch_counts(val_trows),
    }

    target_aggregate_recomputed = {
        "train_fit": target_failure_summary(per_target_rows, "train_fit"),
        "train_internal_heldout": target_failure_summary(per_target_rows, "train_internal_heldout"),
        "validation_report_only": target_failure_summary(per_target_rows, "validation_report_only"),
    }

    recommendation, blockers, actions = decide(s14, internal_rank, val_rank, coverage_rows)
    status = "PASS" if per_target_rows else "REVIEW"
    if not per_target_rows:
        blockers.append("G14 per-target profile metrics CSV missing; triage only used summary JSON and is incomplete.")
        actions.append("Re-run G1.4 or provide D17_G14_PER_TARGET_PROFILE_METRICS.csv before deciding G1.5R.")

    # Compare G1.3 vs G1.4 headline results when available.
    comparison: Dict[str, Any] = {}
    if s13:
        comparison["G13"] = {
            "recommendation": s13.get("recommendation"),
            "g2_ready": s13.get("g2_ready"),
            "internal_heldout": s13.get("internal_heldout_per_target_aggregate", {}),
            "validation_report_only": s13.get("validation_report_only_per_target_aggregate", {}),
        }
    if s12:
        comparison["G12"] = {
            "recommendation": s12.get("recommendation"),
            "g2_ready": s12.get("g2_ready"),
            "train_closedset": s12.get("train_closedset_profile_aggregate", s12.get("train_closedset_per_target_aggregate", {})),
        }
    comparison["G14"] = {
        "recommendation": s14.get("recommendation"),
        "g2_ready": s14.get("g2_ready"),
        "g2_blockers": s14.get("g2_blockers"),
        "internal_heldout": s14.get("internal_heldout_per_target_aggregate", {}),
        "validation_report_only": s14.get("validation_report_only_per_target_aggregate", {}),
    }

    candidate_config = make_candidate_config(cfg, s14, internal_rank)
    cand_path = out / "D17_G15_RECOMMENDED_G15R_CONFIG.json"
    dump_json(candidate_config, cand_path)

    write_csv(internal_rank, out / "D17_G15_INTERNAL_HELDOUT_FAILURE_RANKING.csv")
    write_csv(val_rank, out / "D17_G15_VALIDATION_RANKING.csv")
    write_csv(fit_rank, out / "D17_G15_FIT_TRAIN_RANKING.csv")
    write_csv(coverage_rows, out / "D17_G15_PROFILE_COVERAGE_AUDIT.csv")
    if phie_rows:
        write_csv(phie_rows, out / "D17_G15_PHIE_PROFILE_AUDIT_COPY.csv")

    # Decision report markdown.
    md_lines = [
        "# D17-G1.5 internal-heldout failure triage",
        "",
        f"Created: {utc_now()}",
        "",
        "## Decision",
        f"- status: `{status}`",
        f"- recommendation: `{recommendation}`",
        "",
        "## Blockers",
    ]
    md_lines.extend([f"- {b}" for b in blockers] or ["- none"])
    md_lines.append("")
    md_lines.append("## Recommended actions")
    md_lines.extend([f"- {a}" for a in actions] or ["- none"])
    if internal_rank:
        md_lines.extend([
            "",
            "## Worst internal-heldout profiles",
        ])
        for r in internal_rank[:5]:
            md_lines.append(f"- {r.get('canonical_cell_uid')} | protocol={r.get('protocol')} | worst={r.get('worst_target')} R2={as_float(r.get('worst_target_r2')):.6g} | profile_min={as_float(r.get('profile_target_r2_min')):.6g}")
    (out / "D17_G15_DECISION_REPORT.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary: Dict[str, Any] = {
        "protocol": "D17-G1.5_INTERNAL_HELDOUT_FAILURE_TRIAGE",
        "created_at_utc": utc_now(),
        "status": status,
        "recommendation": recommendation,
        "g2_ready": False,
        "g2_blockers": blockers,
        "purpose": "Diagnose why G1.4 fixed validation phie but failed train-internal heldout before any G2 expansion.",
        "policy": {
            "training_performed": False,
            "checkpoint_selection_performed": False,
            "validation_softlabels_used_for_training": False,
            "frozen_test_softlabels_used": False,
            "not_a_G2_run": True,
        },
        "inputs": {
            "g14_summary": str(g14_summary),
            "g14_out_dir": str(g14_out_dir),
            "g13_summary": str(g13_summary) if g13_summary else "",
            "g12_summary": str(g12_summary) if g12_summary else "",
            "g0_profile_semantics_csv": str(g0_profile_semantics_csv) if g0_profile_semantics_csv else "",
            "per_target_profile_metrics_csv": str(per_target_csv),
            "profile_metrics_csv": str(profile_csv),
            "phie_robustness_audit_csv": str(phie_csv),
            "profile_encoder_feature_audit_csv": str(feature_csv),
            "training_history_csv": str(train_hist_csv),
        },
        "source_g14_status": {
            "status": s14.get("status"),
            "recommendation": s14.get("recommendation"),
            "g2_ready": s14.get("g2_ready"),
            "g2_blockers": s14.get("g2_blockers"),
            "best_epoch": s14.get("best_epoch"),
            "dataset": s14.get("dataset"),
        },
        "target_aggregate_recomputed_from_csv": target_aggregate_recomputed,
        "split_composition": split_composition,
        "worst_internal_heldout_profiles": internal_rank[:10],
        "worst_validation_profiles": val_rank[:10],
        "coverage_summary": coverage_summary,
        "worst_feature_coverage_profiles": coverage_rows[:10],
        "history_available": bool(train_hist),
        "comparison": comparison,
        "recommended_actions": actions,
        "files": {
            "summary_json": str(out / "D17_G15_INTERNAL_HELDOUT_TRIAGE_SUMMARY.json"),
            "decision_report_md": str(out / "D17_G15_DECISION_REPORT.md"),
            "internal_heldout_failure_ranking_csv": str(out / "D17_G15_INTERNAL_HELDOUT_FAILURE_RANKING.csv"),
            "validation_ranking_csv": str(out / "D17_G15_VALIDATION_RANKING.csv"),
            "fit_train_ranking_csv": str(out / "D17_G15_FIT_TRAIN_RANKING.csv"),
            "profile_coverage_audit_csv": str(out / "D17_G15_PROFILE_COVERAGE_AUDIT.csv"),
            "recommended_g15r_config_json": str(cand_path),
        },
    }
    dump_json(summary, out / "D17_G15_INTERNAL_HELDOUT_TRIAGE_SUMMARY.json")
    return summary
