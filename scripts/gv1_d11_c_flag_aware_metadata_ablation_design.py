#!/usr/bin/env python3
"""D11-C flag-aware metadata ablation design package.

This script is intentionally non-destructive.  It reads existing D10/D11 outputs
and writes design/audit artifacts for a future optional metadata-input ablation.
It does not start training and it does not modify the D9.6/D9.5.1 mainline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TARGET_BATCH_ID = "Batch-1"
TARGET_BATTERY_ID = "battery-8"
TARGET_PROTOCOL = "2C"
TARGET_PROFILE_ID_CANONICAL = "Batch-1_2C_battery-8"

D10P5_EXPECTED_VERDICT = "d10_p5_mainline_freeze_and_regime_policy_ready_for_d11"
D11B_ACCEPTABLE_VERDICTS = {
    "d11_b_battery8_feature_distance_boundary_supported_keep_flagged",
    "d11_b_battery8_feature_distance_weakly_supported_keep_flagged",
}


def _as_posixish(path: Path) -> str:
    return str(path).replace("\\", "/")


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fieldnames})


def _csv_value(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.12g}"
    return v


def to_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y", "ok", "pass"}
    return False


def first_nonempty(row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


def infer_profile_id(row: Dict[str, Any]) -> str:
    existing = first_nonempty(row, ["profile_id", "cell_profile_id", "profile_uid", "run_id", "label"])
    if existing:
        return existing
    batch_id = first_nonempty(row, ["batch_id", "batch", "batch_name"])
    protocol = first_nonempty(row, ["protocol", "protocol_id", "c_rate", "rate"])
    battery_id = first_nonempty(row, ["battery_id", "cell_id", "battery", "cell_uid"])
    if batch_id or protocol or battery_id:
        return f"{batch_id}_{protocol}_{battery_id}".strip("_")
    return "unknown_profile"


def normalize_protocol(protocol: str) -> str:
    p = str(protocol or "").strip()
    if p.lower() in {"r2.5", "r25", "r2_5"}:
        return "R2.5"
    if p.lower() == "r3":
        return "R3"
    if p.lower() == "2c":
        return "2C"
    return p


def is_target_profile(row: Dict[str, Any]) -> bool:
    batch_id = first_nonempty(row, ["batch_id", "batch", "batch_name"])
    battery_id = first_nonempty(row, ["battery_id", "battery", "cell_id", "cell_uid"])
    protocol = normalize_protocol(first_nonempty(row, ["protocol", "protocol_id", "c_rate", "rate"]))
    profile_id = infer_profile_id(row)
    checks = [
        batch_id == TARGET_BATCH_ID and battery_id == TARGET_BATTERY_ID and protocol == TARGET_PROTOCOL,
        TARGET_PROFILE_ID_CANONICAL in profile_id,
        ("Batch-1" in profile_id and "2C" in profile_id and "battery-8" in profile_id),
    ]
    return any(checks)


def build_profile_metadata(profile_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not profile_rows:
        profile_rows = [{
            "dataset_id": "XJTU",
            "batch_id": TARGET_BATCH_ID,
            "battery_id": TARGET_BATTERY_ID,
            "protocol": TARGET_PROTOCOL,
            "split": "test",
            "profile_id": TARGET_PROFILE_ID_CANONICAL,
        }]

    out: List[Dict[str, Any]] = []
    for row in profile_rows:
        profile_id = infer_profile_id(row)
        batch_id = first_nonempty(row, ["batch_id", "batch", "batch_name"])
        battery_id = first_nonempty(row, ["battery_id", "battery", "cell_id", "cell_uid"])
        protocol = normalize_protocol(first_nonempty(row, ["protocol", "protocol_id", "c_rate", "rate"]))
        split = first_nonempty(row, ["split", "profile_split"], default="unknown")
        source_file = first_nonempty(row, ["source_file", "profile_npz", "solution_npz", "path"])
        target = is_target_profile(row)
        same_b1_2c = (batch_id == TARGET_BATCH_ID and protocol == TARGET_PROTOCOL)

        if target:
            metadata_group = "flagged_late2C_boundary_target"
            role = "flagged_eval_only_not_mainline_training"
            train_allowed = False
            mainline_scope = False
        elif same_b1_2c:
            metadata_group = "same_batch_protocol_peer_B1_2C"
            role = "peer_reference_keep_in_23profile_mainline"
            train_allowed = True
            mainline_scope = True
        else:
            metadata_group = "non_boundary_profile"
            role = "normal_profile_keep_in_23profile_mainline"
            train_allowed = True
            mainline_scope = True

        out.append({
            "profile_id": profile_id,
            "dataset_id": first_nonempty(row, ["dataset_id", "dataset"], default="XJTU"),
            "batch_id": batch_id,
            "battery_id": battery_id,
            "protocol": protocol,
            "split": split,
            "source_file": source_file,
            "is_B1_2C_battery8_target": int(target),
            "flag_late2C_discharge_boundary": int(target),
            "flag_same_B1_2C_peer_group": int(same_b1_2c),
            "flag_protocol_2C": int(protocol == "2C"),
            "flag_protocol_R25": int(protocol == "R2.5"),
            "flag_protocol_R3": int(protocol == "R3"),
            "metadata_group": metadata_group,
            "d11c_role": role,
            "include_in_23profile_mainline_scope": int(mainline_scope),
            "allow_in_future_metadata_training": int(train_allowed),
            "recommended_action": "keep_flagged_excluded_from_mainline_claim" if target else "keep_as_non_outlier_mainline_profile",
        })
    # Stable sort for readability.
    out.sort(key=lambda r: (str(r.get("batch_id", "")), str(r.get("protocol", "")), str(r.get("battery_id", ""))))
    return out


def feature_group_summary(top_feature_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    group_counter: Counter[str] = Counter()
    max_abs_by_group: Dict[str, float] = defaultdict(lambda: float("nan"))
    example_by_group: Dict[str, str] = {}
    for row in top_feature_rows:
        group = first_nonempty(row, ["feature_group", "group"], default="unknown")
        feature = first_nonempty(row, ["feature"], default="")
        abs_z = to_float(first_nonempty(row, ["abs_robust_z", "abs_z", "abs robust z"]), default=float("nan"))
        group_counter[group] += 1
        prev = max_abs_by_group[group]
        if math.isnan(prev) or (not math.isnan(abs_z) and abs_z > prev):
            max_abs_by_group[group] = abs_z
            example_by_group[group] = feature
    rows: List[Dict[str, Any]] = []
    for group, count in group_counter.most_common():
        rows.append({
            "feature_group": group,
            "top_feature_count": count,
            "max_abs_robust_z_in_top_table": max_abs_by_group[group],
            "example_top_feature": example_by_group.get(group, ""),
            "d11c_metadata_use": group_to_metadata_use(group),
        })
    return rows


def group_to_metadata_use(group: str) -> str:
    g = group.lower()
    if "discharge" in g:
        return "candidate_regime_flag_context_only_do_not_fit_target"
    if "charge" in g:
        return "candidate_profile_shape_context"
    if "rest" in g:
        return "candidate_rest_voltage_relaxation_context"
    if "current" in g:
        return "candidate_current_statistics_context"
    if "voltage" in g:
        return "candidate_voltage_window_context"
    return "review_before_use"


def candidate_routes() -> List[Dict[str, Any]]:
    return [
        {
            "route_id": "D11C-0",
            "route_name": "baseline_d96_mainline_reference",
            "status": "reference_only_no_new_training",
            "allowed": True,
            "risk": "low",
            "purpose": "Keep D9.6/D9.5.1 as accepted non-outlier mainline; use as control.",
            "requires_code_change": False,
            "requires_training": False,
            "battery8_handling": "flagged_excluded_from_mainline_claim",
        },
        {
            "route_id": "D11C-1",
            "route_name": "profile_metadata_manifest_only",
            "status": "recommended_now_design_archive",
            "allowed": True,
            "risk": "low",
            "purpose": "Register profile flags, protocol groups, and target/peer roles without touching model code.",
            "requires_code_change": False,
            "requires_training": False,
            "battery8_handling": "flagged_eval_only",
        },
        {
            "route_id": "D11C-2",
            "route_name": "flag_aware_metadata_input_ablation",
            "status": "design_only_requires_separate_patch_before_training",
            "allowed": "conditional",
            "risk": "medium",
            "purpose": "Future small ablation adding metadata vector input; first run short 40ks smoke only, never overwrite D9.6.",
            "requires_code_change": True,
            "requires_training": "future_40ks_smoke_only_after_manual_approval",
            "battery8_handling": "flagged_context_not_normal_target",
        },
        {
            "route_id": "D11C-3",
            "route_name": "feature_distance_metadata_ablation",
            "status": "research_candidate_not_mainline",
            "allowed": "conditional",
            "risk": "medium_high",
            "purpose": "Use D11-B feature-distance groups as metadata; risk of leaking profile identity or overfitting boundary case.",
            "requires_code_change": True,
            "requires_training": "future_only_after_D11C2_review",
            "battery8_handling": "still_flagged",
        },
        {
            "route_id": "D11C-X",
            "route_name": "hard_voltage_or_component_clamp_repair",
            "status": "forbidden_known_failure",
            "allowed": False,
            "risk": "known_failure",
            "purpose": "D9.6.1/D9.6.2 style guards already failed and must not be revived as D11-C.",
            "requires_code_change": True,
            "requires_training": False,
            "battery8_handling": "not_allowed",
        },
        {
            "route_id": "D11C-Y",
            "route_name": "direct_24profile_200ks_mainline_claim",
            "status": "forbidden_until_new_regime_strategy_validated",
            "allowed": False,
            "risk": "misleading_mainline_claim",
            "purpose": "Battery-8 remains unresolved; direct 24-profile 200ks claim would blur mainline vs flagged regime case.",
            "requires_code_change": False,
            "requires_training": False,
            "battery8_handling": "not_allowed",
        },
    ]


def guardrail_rows(d10p5: Dict[str, Any], d11b: Dict[str, Any], profile_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    d10p5_ok = truthy(d10p5.get("ok")) and d10p5.get("verdict") == D10P5_EXPECTED_VERDICT
    d11b_ok = truthy(d11b.get("ok")) and d11b.get("verdict") in D11B_ACCEPTABLE_VERDICTS
    target_rows = [r for r in profile_meta if int(r.get("is_B1_2C_battery8_target", 0) or 0) == 1]
    target_excluded = bool(target_rows) and all(int(r.get("include_in_23profile_mainline_scope", 1) or 0) == 0 for r in target_rows)
    mainline_rows = [r for r in profile_meta if int(r.get("include_in_23profile_mainline_scope", 0) or 0) == 1]

    specs = [
        ("G01", "D10-P5 mainline freeze verdict is present", d10p5_ok),
        ("G02", "D11-B feature-distance boundary verdict supports keeping battery-8 flagged", d11b_ok),
        ("G03", "B1_2C battery-8 appears in metadata manifest", bool(target_rows)),
        ("G04", "B1_2C battery-8 is excluded from 23-profile mainline scope", target_excluded),
        ("G05", "At least 23 non-target profiles remain eligible for mainline scope", len(mainline_rows) >= 23),
        ("G06", "This package does not modify D9.6/D9.5.1 source files", True),
        ("G07", "This package does not generate a training run command for 24-profile 200ks", True),
        ("G08", "D11-C metadata input remains design-only until a separate patch is approved", True),
    ]
    rows: List[Dict[str, Any]] = []
    for check_id, desc, ok in specs:
        rows.append({
            "check_id": check_id,
            "status": "pass" if ok else "fail",
            "description": desc,
        })
    return rows


def write_patch_design(path: Path, out_dir: Path) -> None:
    text = f"""# D11-C Metadata Input Patch Design

This file is a design note only.  It is not an implementation patch.

## Objective

Test whether a small, auditable metadata vector can improve handling of profile/regime boundaries without changing the accepted D9.6/D9.5.1 non-outlier mainline claim.

## Candidate metadata vector

Recommended minimal vector for a future **D11-C-2** smoke ablation:

```text
[protocol_2C, protocol_R2p5, protocol_R3,
 batch_B1, batch_B3, batch_B4,
 flag_late2C_discharge_boundary,
 flag_same_B1_2C_peer_group]
```

The target `B1_2C battery-8` must remain flagged.  The flag is not permission to include it as a normal mainline training profile.

## Future implementation touch points

Only after manual approval, a separate package may add optional metadata inputs to:

```text
gv1/model.py
gv1/trainer.py
gv1/losses.py
scripts/gv1_train_conditioned_pinn.py
```

The implementation must be guarded by a CLI flag such as:

```text
--metadata_mode none|protocol_batch|protocol_batch_flag
```

Default must remain:

```text
--metadata_mode none
```

## Smoke-only evaluation proposal

A future D11-C-2 run, if approved, should start with 40ks smoke comparisons:

```text
C0: D9.6 mainline reference, metadata_mode=none
C1: protocol+batch metadata, metadata_mode=protocol_batch
C2: protocol+batch+battery8 flag, metadata_mode=protocol_batch_flag
```

Do not run 24-profile 200ks as a mainline claim from D11-C.

## Output archive

This design archive was generated for:

```text
{out_dir}
```
"""
    path.write_text(text, encoding="utf-8")


def write_recommendation(path: Path, summary: Dict[str, Any], routes: List[Dict[str, Any]], guardrails: List[Dict[str, Any]], feature_groups: List[Dict[str, Any]]) -> None:
    def table(rows: Sequence[Dict[str, Any]], cols: Sequence[str], max_rows: Optional[int] = None) -> str:
        shown = list(rows[:max_rows] if max_rows else rows)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in shown:
            lines.append("| " + " | ".join(str(_csv_value(r.get(c, ""))) for c in cols) + " |")
        return "\n".join(lines)

    text = f"""# D11-C Flag-aware Metadata Ablation Design

## Verdict

```text
{summary['verdict']}
```

## Recommended next action

```text
{summary['next_action']}
```

## Interpretation

- D11-C is a **design-only** stage.
- Keep D9.6/D9.5.1 frozen as the current non-outlier GV1 mainline.
- Keep B1_2C battery-8 flagged/excluded as late-2C discharge boundary/regime case.
- Do not adopt hard voltage guard, component clamp, or D10-P3 calibration.
- Do not run a direct 24-profile 200ks mainline claim from this stage.

## Context checks

- D10-P5 verdict: `{summary.get('d10p5_verdict')}`
- D11-B verdict: `{summary.get('d11b_verdict')}`
- profile_count: `{summary.get('profile_count')}`
- non_target_mainline_profile_count: `{summary.get('non_target_mainline_profile_count')}`
- target_profile_found: `{summary.get('target_profile_found')}`

## Candidate routes

{table(routes, ['route_id', 'route_name', 'status', 'allowed', 'risk'])}

## Guardrail checklist

{table(guardrails, ['check_id', 'status', 'description'])}

## D11-B top feature groups for metadata review

{table(feature_groups, ['feature_group', 'top_feature_count', 'max_abs_robust_z_in_top_table', 'example_top_feature', 'd11c_metadata_use'], max_rows=20) if feature_groups else '_No D11-B top feature table was found.  Metadata feature-group review is incomplete._'}

## Generated files

```text
recommendation_md: {summary['outputs']['recommendation_md']}
summary_json: {summary['outputs']['summary_json']}
profile_metadata_manifest_csv: {summary['outputs']['profile_metadata_manifest_csv']}
candidate_routes_csv: {summary['outputs']['candidate_routes_csv']}
guardrail_checklist_csv: {summary['outputs']['guardrail_checklist_csv']}
feature_group_summary_csv: {summary['outputs']['feature_group_summary_csv']}
metadata_patch_design_md: {summary['outputs']['metadata_patch_design_md']}
```
"""
    path.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GV1 D11-C flag-aware metadata ablation design-only audit.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--d10p5_dir", default=None)
    ap.add_argument("--d11b_dir", default=None)
    ap.add_argument("--training_ready_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    cache_root = Path(args.cache_root)
    d10p5_dir = Path(args.d10p5_dir) if args.d10p5_dir else cache_root / "xjtu_batch134_d10_p5_regime_policy_d11_plan"
    d11b_dir = Path(args.d11b_dir) if args.d11b_dir else cache_root / "xjtu_batch134_d11_b_regime_feature_distance_audit"
    training_ready_dir = Path(args.training_ready_dir) if args.training_ready_dir else cache_root / "xjtu_batch134_training_ready"
    out_dir = Path(args.out_dir) if args.out_dir else cache_root / "xjtu_batch134_d11_c_flag_aware_metadata_ablation_design"
    out_dir.mkdir(parents=True, exist_ok=True)

    d10p5_summary_path = d10p5_dir / "d10_p5_regime_policy_summary.json"
    d11b_summary_path = d11b_dir / "d11_b_regime_feature_distance_summary.json"
    profile_manifest_path = training_ready_dir / "xjtu_batch134_profile_manifest.csv"
    top_features_path = d11b_dir / "d11_b_battery8_top_distance_features.csv"

    d10p5 = read_json(d10p5_summary_path)
    d11b = read_json(d11b_summary_path)
    profile_rows = read_csv_rows(profile_manifest_path)
    top_feature_rows = read_csv_rows(top_features_path)

    profile_meta = build_profile_metadata(profile_rows)
    feature_groups = feature_group_summary(top_feature_rows)
    routes = candidate_routes()
    guardrails = guardrail_rows(d10p5, d11b, profile_meta)

    pass_count = sum(1 for r in guardrails if r["status"] == "pass")
    fail_count = sum(1 for r in guardrails if r["status"] == "fail")

    d10p5_ok = truthy(d10p5.get("ok")) and d10p5.get("verdict") == D10P5_EXPECTED_VERDICT
    d11b_ok = truthy(d11b.get("ok")) and d11b.get("verdict") in D11B_ACCEPTABLE_VERDICTS
    target_found = any(int(r.get("is_B1_2C_battery8_target", 0) or 0) == 1 for r in profile_meta)
    non_target_mainline_count = sum(int(r.get("include_in_23profile_mainline_scope", 0) or 0) == 1 for r in profile_meta)

    if d10p5_ok and d11b_ok and target_found and non_target_mainline_count >= 23 and fail_count == 0:
        verdict = "d11_c_design_only_flag_aware_metadata_ablation_plan_ready"
        next_action = "manual_review_then_optionally_prepare_separate_D11C2_metadata_input_patch"
        ok = True
    else:
        verdict = "d11_c_incomplete_context_keep_design_only_do_not_train"
        next_action = "fix_missing_context_or_manifest_before_any_metadata_ablation"
        ok = False

    outputs = {
        "recommendation_md": str(out_dir / "D11_C_RECOMMENDATION.md"),
        "summary_json": str(out_dir / "d11_c_flag_aware_metadata_ablation_summary.json"),
        "profile_metadata_manifest_csv": str(out_dir / "d11_c_profile_metadata_manifest.csv"),
        "candidate_routes_csv": str(out_dir / "d11_c_candidate_routes.csv"),
        "guardrail_checklist_csv": str(out_dir / "d11_c_guardrail_checklist.csv"),
        "feature_group_summary_csv": str(out_dir / "d11_c_feature_group_summary.csv"),
        "metadata_patch_design_md": str(out_dir / "d11_c_metadata_patch_design.md"),
    }

    summary: Dict[str, Any] = {
        "ok": ok,
        "stage": "D11-C flag-aware metadata ablation design-only",
        "verdict": verdict,
        "next_action": next_action,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "out_dir": str(out_dir),
        "d10p5_summary_path": str(d10p5_summary_path),
        "d11b_summary_path": str(d11b_summary_path),
        "profile_manifest_path": str(profile_manifest_path),
        "top_features_path": str(top_features_path),
        "d10p5_verdict": d10p5.get("verdict"),
        "d10p5_ok": d10p5_ok,
        "d11b_verdict": d11b.get("verdict"),
        "d11b_ok": d11b_ok,
        "profile_count": len(profile_meta),
        "target_profile_found": target_found,
        "non_target_mainline_profile_count": non_target_mainline_count,
        "feature_group_count": len(feature_groups),
        "guardrail_counts": {"pass": pass_count, "fail": fail_count},
        "mainline_policy": {
            "keep_mainline": "GV1 D9.6 / D9.5.1 trend-first warmup rare-regime",
            "battery8_status": "flagged_excluded_late2C_discharge_boundary_regime_case",
            "adopted_d10p3_correction": "none",
            "d11c_scope": "design_only_metadata_ablation_planning",
        },
        "outputs": outputs,
    }

    write_csv(Path(outputs["profile_metadata_manifest_csv"]), profile_meta)
    write_csv(Path(outputs["candidate_routes_csv"]), routes)
    write_csv(Path(outputs["guardrail_checklist_csv"]), guardrails)
    write_csv(Path(outputs["feature_group_summary_csv"]), feature_groups)
    write_patch_design(Path(outputs["metadata_patch_design_md"]), out_dir)
    write_json(Path(outputs["summary_json"]), summary)
    write_recommendation(Path(outputs["recommendation_md"]), summary, routes, guardrails, feature_groups)

    print(json.dumps({
        "ok": ok,
        "verdict": verdict,
        "out_dir": str(out_dir),
        "recommendation_md": outputs["recommendation_md"],
        "guardrail_counts": summary["guardrail_counts"],
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
