#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GV1 D11C2 flag-aware metadata input patch planner.

This script is intentionally design-only. It builds an enriched metadata
manifest and a patch contract that can be reviewed before any training code is
changed. It does not modify gv1/model.py, gv1/output_transform.py,
gv1/losses.py, gv1/trainer.py, or scripts/gv1_train_conditioned_pinn.py.

Default paths match the QJW-2 / PINN-for-ASSB-V1 D10/D11 workflow.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_CACHE_ROOT = Path(r"E:\XJTU battery dataset\_gv1_cache")
TARGET_BATCH = "Batch-1"
TARGET_BATTERY = "battery-8"
TARGET_PROTOCOL = "2C"
TARGET_PROFILE_ID = "Batch-1_2C_battery-8"


def _p(s: Optional[str | Path]) -> Optional[Path]:
    if s is None:
        return None
    return Path(str(s))


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"__read_error__": f"{type(exc).__name__}: {exc}"}
    return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def as_bool_str(x: bool) -> str:
    return "true" if bool(x) else "false"


def float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def profile_id_from_row(row: Dict[str, str]) -> str:
    # D8/D9 manifests usually carry cell_uid and protocol.  Prefer a stable id
    # that matches the D11 convention.
    batch = row.get("batch_id", "").strip()
    protocol = row.get("protocol", "").strip()
    battery = row.get("battery_id", "").strip()
    if batch and protocol and battery:
        return f"{batch}_{protocol}_{battery}"
    return row.get("profile_id", "") or row.get("cell_uid", "") or row.get("label", "")


def is_target(row: Dict[str, str]) -> bool:
    return (
        row.get("batch_id", "").strip() == TARGET_BATCH
        and row.get("battery_id", "").strip() == TARGET_BATTERY
        and row.get("protocol", "").strip() == TARGET_PROTOCOL
    ) or profile_id_from_row(row) == TARGET_PROFILE_ID


def default_paths(cache_root: Path) -> Dict[str, Path]:
    return {
        "training_ready_dir": cache_root / "xjtu_batch134_training_ready",
        "profile_manifest": cache_root / "xjtu_batch134_training_ready" / "xjtu_batch134_profile_manifest.csv",
        "d10p5_dir": cache_root / "xjtu_batch134_d10_p5_regime_policy_d11_plan",
        "d10p5_summary": cache_root / "xjtu_batch134_d10_p5_regime_policy_d11_plan" / "d10_p5_regime_policy_summary.json",
        "d11b_dir": cache_root / "xjtu_batch134_d11_b_regime_feature_distance_audit",
        "d11b_summary": cache_root / "xjtu_batch134_d11_b_regime_feature_distance_audit" / "d11_b_regime_feature_distance_summary.json",
        "d11b_top_features": cache_root / "xjtu_batch134_d11_b_regime_feature_distance_audit" / "d11_b_battery8_top_distance_features.csv",
        "d11c_dir": cache_root / "xjtu_batch134_d11_c_flag_aware_metadata_ablation_design",
        "d11c_summary": cache_root / "xjtu_batch134_d11_c_flag_aware_metadata_ablation_design" / "d11_c_flag_aware_metadata_ablation_summary.json",
        "d11c_manifest": cache_root / "xjtu_batch134_d11_c_flag_aware_metadata_ablation_design" / "d11_c_profile_metadata_manifest.csv",
        "out_dir": cache_root / "xjtu_batch134_d11c2_metadata_input_patch_design",
    }


def build_feature_group_context(top_feature_rows: List[Dict[str, str]], max_rows: int = 50) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    groups: Counter[str] = Counter()
    rows: List[Dict[str, Any]] = []
    for r in top_feature_rows[:max_rows]:
        group = r.get("feature_group") or r.get("group") or "unknown"
        groups[group] += 1
        z = float_or_none(r.get("abs_robust_z") or r.get("abs z") or r.get("abs_z"))
        rows.append({
            "feature": r.get("feature", ""),
            "feature_group": group,
            "target_value": r.get("target_value") or r.get("target", ""),
            "peer_median": r.get("peer_median", ""),
            "abs_robust_z": "" if z is None else f"{z:.9g}",
            "direction_vs_peer_median": r.get("direction_vs_peer_median") or r.get("direction", ""),
            "d11c2_use": feature_group_use(group),
        })
    return rows, dict(groups)


def feature_group_use(group: str) -> str:
    g = (group or "").lower()
    if "discharge" in g:
        return "regime_context_only_flag_not_target_fit"
    if "current" in g:
        return "measured_current_statistics_context"
    if "rest" in g:
        return "rest_relaxation_context"
    if "charge" in g:
        return "charge_shape_context"
    if "voltage" in g:
        return "voltage_window_context"
    return "review_only"


def build_enriched_manifest(profile_rows: List[Dict[str, str]], d11c_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    # If D11-C already generated metadata, use it as a sidecar.  Otherwise use the training-ready manifest.
    side_by_profile: Dict[str, Dict[str, str]] = {}
    for r in d11c_rows:
        pid = r.get("profile_id") or profile_id_from_row(r)
        if pid:
            side_by_profile[pid] = r
    rows: List[Dict[str, Any]] = []
    for r in profile_rows:
        pid = profile_id_from_row(r)
        side = side_by_profile.get(pid, {})
        target = is_target(r) or is_target(side)
        split = r.get("split", side.get("split", ""))
        protocol = r.get("protocol", side.get("protocol", ""))
        batch = r.get("batch_id", side.get("batch_id", ""))
        battery = r.get("battery_id", side.get("battery_id", ""))
        metadata_scope = "flagged_regime_case" if target else "mainline_non_target_profile"
        train_scope = "excluded_from_23profile_mainline" if target else "eligible_for_23profile_mainline_reference"
        # The key rule: target flag may be used to mark/review, not to claim solved mainline training.
        rows.append(OrderedDict([
            ("profile_id", pid),
            ("dataset_id", r.get("dataset_id", side.get("dataset_id", "XJTU"))),
            ("batch_id", batch),
            ("battery_id", battery),
            ("cell_uid", r.get("cell_uid", side.get("cell_uid", ""))),
            ("protocol", protocol),
            ("split", split),
            ("source_file", r.get("source_file", side.get("source_file", ""))),
            ("profile_npz", r.get("profile_npz", side.get("profile_npz", ""))),
            ("d11c2_is_b1_2c_battery8", as_bool_str(target)),
            ("d11c2_regime_flag", "late_2C_discharge_boundary_outlier" if target else "none"),
            ("d11c2_metadata_scope", metadata_scope),
            ("d11c2_training_scope", train_scope),
            ("d11c2_allow_as_input_metadata", "conditional_design_only" if target else "yes_for_ablation_reference"),
            ("d11c2_forbidden_claim", "do_not_claim_24profile_mainline_solution" if target else "none"),
            ("d11c2_notes", "target is flagged/excluded; use only in separate ablation with explicit audit" if target else "non-target profile remains D9.6/D9.5.1 mainline reference"),
        ]))
    # If profile_manifest was missing but D11-C manifest exists, still emit target-aware rows from D11-C.
    if not rows and d11c_rows:
        for r in d11c_rows:
            pid = r.get("profile_id") or profile_id_from_row(r)
            target = is_target(r)
            rows.append(OrderedDict([
                ("profile_id", pid),
                ("dataset_id", r.get("dataset_id", "XJTU")),
                ("batch_id", r.get("batch_id", "")),
                ("battery_id", r.get("battery_id", "")),
                ("cell_uid", r.get("cell_uid", "")),
                ("protocol", r.get("protocol", "")),
                ("split", r.get("split", "")),
                ("source_file", r.get("source_file", "")),
                ("profile_npz", r.get("profile_npz", "")),
                ("d11c2_is_b1_2c_battery8", as_bool_str(target)),
                ("d11c2_regime_flag", "late_2C_discharge_boundary_outlier" if target else "none"),
                ("d11c2_metadata_scope", "flagged_regime_case" if target else "mainline_non_target_profile"),
                ("d11c2_training_scope", "excluded_from_23profile_mainline" if target else "eligible_for_23profile_mainline_reference"),
                ("d11c2_allow_as_input_metadata", "conditional_design_only" if target else "yes_for_ablation_reference"),
                ("d11c2_forbidden_claim", "do_not_claim_24profile_mainline_solution" if target else "none"),
                ("d11c2_notes", "target is flagged/excluded; use only in separate ablation with explicit audit" if target else "non-target profile remains D9.6/D9.5.1 mainline reference"),
            ]))
    return rows


def build_guardrails(
    *,
    d10p5_summary: Optional[Dict[str, Any]],
    d11b_summary: Optional[Dict[str, Any]],
    d11c_summary: Optional[Dict[str, Any]],
    enriched_rows: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    d10_verdict = (d10p5_summary or {}).get("verdict")
    d11b_verdict = (d11b_summary or {}).get("verdict")
    d11c_verdict = (d11c_summary or {}).get("verdict")
    target_rows = [r for r in enriched_rows if r.get("d11c2_is_b1_2c_battery8") == "true"]
    non_target_rows = [r for r in enriched_rows if r.get("d11c2_is_b1_2c_battery8") != "true"]
    rows = [
        ("C2G01", d10_verdict == "d10_p5_mainline_freeze_and_regime_policy_ready_for_d11", "D10-P5 mainline freeze verdict is present"),
        ("C2G02", d11b_verdict == "d11_b_battery8_feature_distance_boundary_supported_keep_flagged", "D11-B supports battery-8 boundary/regime flag"),
        ("C2G03", d11c_verdict in {"d11_c_design_only_flag_aware_metadata_ablation_plan_ready", None}, "D11-C design-only context is present or will be reconstructed"),
        ("C2G04", len(target_rows) == 1, "Exactly one B1_2C battery-8 target row is present in enriched metadata"),
        ("C2G05", len(non_target_rows) >= 23, "At least 23 non-target profiles remain available as mainline reference"),
        ("C2G06", all(r.get("d11c2_training_scope") == "excluded_from_23profile_mainline" for r in target_rows), "Battery-8 remains excluded/flagged in this patch"),
        ("C2G07", True, "This script generates metadata/contract files only and does not launch training"),
        ("C2G08", True, "No D9.6/D9.5.1 source file is modified by this design-only package"),
        ("C2G09", True, "No direct 24-profile 200ks mainline training command is generated"),
        ("C2G10", True, "D11C2 is a separate ablation patch, not a mainline replacement"),
    ]
    return [{"check_id": cid, "status": "pass" if ok else "fail", "description": desc} for cid, ok, desc in rows]


def make_recommendation_md(summary: Dict[str, Any], guardrails: List[Dict[str, str]]) -> str:
    verdict = summary.get("verdict", "unknown")
    next_action = summary.get("next_action", "unknown")
    guard_counts = Counter(r["status"] for r in guardrails)
    paths = summary.get("outputs", {})
    return f"""# D11C2 Flag-aware Metadata Input Patch

## Verdict

```text
{verdict}
```

## Recommended next action

```text
{next_action}
```

## Interpretation

- D11C2 is a **separate metadata-input patch design**, not a training run.
- Keep D9.6/D9.5.1 frozen as the current non-outlier GV1 mainline.
- Keep B1_2C battery-8 flagged/excluded as a late-2C discharge boundary/regime case.
- The generated enriched metadata manifest may be reviewed before D12/D11C3-style ablation training is designed.
- This stage must not be used to claim a 24-profile 200ks mainline result.

## Context

| item | value |
|---|---:|
| D10-P5 verdict | `{summary.get('d10p5_verdict')}` |
| D11-B verdict | `{summary.get('d11b_verdict')}` |
| D11-C verdict | `{summary.get('d11c_verdict')}` |
| profile_count | `{summary.get('profile_count')}` |
| target_profile_count | `{summary.get('target_profile_count')}` |
| non_target_profile_count | `{summary.get('non_target_profile_count')}` |
| guardrail pass/warn/fail | `{dict(guard_counts)}` |

## Patch contract

- Add metadata columns only; do not alter voltage transform, loss, trainer, or model files.
- Target flag column: `d11c2_is_b1_2c_battery8`.
- Target regime label: `d11c2_regime_flag = late_2C_discharge_boundary_outlier` for B1_2C battery-8.
- Target training scope: `excluded_from_23profile_mainline`.
- Non-target profiles remain D9.6/D9.5.1 mainline references.

## Guardrails

| check_id | status | description |
|---|---|---|
""" + "\n".join(
        f"| {r['check_id']} | {r['status']} | {r['description']} |" for r in guardrails
    ) + f"""

## Generated files

```text
recommendation_md: {paths.get('recommendation_md')}
summary_json: {paths.get('summary_json')}
enriched_metadata_manifest_csv: {paths.get('enriched_metadata_manifest_csv')}
guardrail_checklist_csv: {paths.get('guardrail_checklist_csv')}
feature_context_csv: {paths.get('feature_context_csv')}
patch_contract_json: {paths.get('patch_contract_json')}
d12_training_stub_md: {paths.get('d12_training_stub_md')}
```
"""


def make_d12_stub(out_dir: Path) -> str:
    return """# D12 / D11C3 Training Stub — Not Executable

This file is intentionally a design stub. It is not a PowerShell training script.

Allowed future use:
- Build a separate experimental branch that reads `d11c2_profile_metadata_input_manifest.csv`.
- Add metadata columns as inputs only after a manual review.
- Keep battery-8 explicitly flagged and report results with and without the flagged profile.

Forbidden use:
- Do not run direct 24-profile 200ks and claim it as D9.6 mainline.
- Do not overwrite D9.6/D9.5.1 source files.
- Do not add hard voltage guards or component clamps.
- Do not use D10-P3 calibration as a mainline repair.

Minimum future validation if this patch is promoted to code:
1. 23-profile non-target sanity check remains pass.
2. Metadata-on/off ablation reports whether non-target metrics degrade.
3. Battery-8 must remain reported as flagged boundary/regime case, not silently merged into the mainline claim.
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GV1 D11C2 flag-aware metadata input patch planner")
    ap.add_argument("--cache_root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--profile_manifest", default=None)
    ap.add_argument("--d10p5_dir", default=None)
    ap.add_argument("--d11b_dir", default=None)
    ap.add_argument("--d11c_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--dry_run", action="store_true", help="Alias for design-only generation; no training is ever launched.")
    ap.add_argument("--strict", action="store_true", help="Return non-zero if any guardrail fails.")
    args = ap.parse_args(argv)

    cache_root = Path(args.cache_root)
    defaults = default_paths(cache_root)
    profile_manifest = _p(args.profile_manifest) or defaults["profile_manifest"]
    d10p5_dir = _p(args.d10p5_dir) or defaults["d10p5_dir"]
    d11b_dir = _p(args.d11b_dir) or defaults["d11b_dir"]
    d11c_dir = _p(args.d11c_dir) or defaults["d11c_dir"]
    out_dir = _p(args.out_dir) or defaults["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    d10p5_summary_path = d10p5_dir / "d10_p5_regime_policy_summary.json"
    d11b_summary_path = d11b_dir / "d11_b_regime_feature_distance_summary.json"
    d11b_top_features_path = d11b_dir / "d11_b_battery8_top_distance_features.csv"
    d11c_summary_path = d11c_dir / "d11_c_flag_aware_metadata_ablation_summary.json"
    d11c_manifest_path = d11c_dir / "d11_c_profile_metadata_manifest.csv"

    d10p5_summary = read_json(d10p5_summary_path)
    d11b_summary = read_json(d11b_summary_path)
    d11c_summary = read_json(d11c_summary_path)
    profile_rows = read_csv_dicts(profile_manifest)
    d11c_rows = read_csv_dicts(d11c_manifest_path)
    top_feature_rows = read_csv_dicts(d11b_top_features_path)

    enriched = build_enriched_manifest(profile_rows, d11c_rows)
    feature_context, group_counts = build_feature_group_context(top_feature_rows)
    guardrails = build_guardrails(
        d10p5_summary=d10p5_summary,
        d11b_summary=d11b_summary,
        d11c_summary=d11c_summary,
        enriched_rows=enriched,
    )
    guard_counts = Counter(r["status"] for r in guardrails)
    target_count = sum(1 for r in enriched if r.get("d11c2_is_b1_2c_battery8") == "true")
    non_target_count = len(enriched) - target_count
    verdict = (
        "d11c2_metadata_input_patch_ready_design_only"
        if guard_counts.get("fail", 0) == 0
        else "d11c2_incomplete_guardrail_failed_do_not_promote"
    )
    next_action = (
        "manual_review_then_prepare_D12_separate_metadata_on_off_ablation_no_mainline_overwrite"
        if guard_counts.get("fail", 0) == 0
        else "fix_failed_guardrails_before_any_D12_design"
    )

    outputs = {
        "recommendation_md": str(out_dir / "D11C2_RECOMMENDATION.md"),
        "summary_json": str(out_dir / "d11c2_metadata_input_patch_summary.json"),
        "enriched_metadata_manifest_csv": str(out_dir / "d11c2_profile_metadata_input_manifest.csv"),
        "guardrail_checklist_csv": str(out_dir / "d11c2_guardrail_checklist.csv"),
        "feature_context_csv": str(out_dir / "d11c2_feature_context_from_d11b.csv"),
        "patch_contract_json": str(out_dir / "d11c2_patch_contract.json"),
        "d12_training_stub_md": str(out_dir / "D12_TRAINING_STUB_NOT_EXECUTABLE.md"),
    }

    summary: Dict[str, Any] = OrderedDict([
        ("ok", guard_counts.get("fail", 0) == 0),
        ("stage", "D11C2 flag-aware metadata input patch"),
        ("verdict", verdict),
        ("next_action", next_action),
        ("created_at", datetime.now().isoformat(timespec="seconds")),
        ("cache_root", str(cache_root)),
        ("out_dir", str(out_dir)),
        ("dry_run", bool(args.dry_run)),
        ("d10p5_verdict", (d10p5_summary or {}).get("verdict")),
        ("d11b_verdict", (d11b_summary or {}).get("verdict")),
        ("d11c_verdict", (d11c_summary or {}).get("verdict")),
        ("profile_manifest", str(profile_manifest)),
        ("profile_count", len(enriched)),
        ("target_profile_count", target_count),
        ("non_target_profile_count", non_target_count),
        ("feature_group_counts_from_d11b_top_table", group_counts),
        ("guard_counts", dict(guard_counts)),
        ("inputs", {
            "d10p5_summary": str(d10p5_summary_path),
            "d11b_summary": str(d11b_summary_path),
            "d11b_top_features": str(d11b_top_features_path),
            "d11c_summary": str(d11c_summary_path),
            "d11c_manifest": str(d11c_manifest_path),
        }),
        ("outputs", outputs),
        ("notes", [
            "Design-only metadata patch; no training launched.",
            "D9.6/D9.5.1 mainline remains frozen.",
            "B1_2C battery-8 remains flagged/excluded.",
            "No hard voltage guard/component clamp/D10-P3 correction is adopted.",
        ]),
    ])

    patch_contract = OrderedDict([
        ("contract_version", "D11C2-v1"),
        ("design_only", True),
        ("mainline_source_files_to_keep_frozen", [
            "gv1/model.py",
            "gv1/output_transform.py",
            "gv1/profile_adaptive.py",
            "gv1/losses.py",
            "gv1/trainer.py",
            "scripts/gv1_train_conditioned_pinn.py",
        ]),
        ("metadata_columns", [
            "d11c2_is_b1_2c_battery8",
            "d11c2_regime_flag",
            "d11c2_metadata_scope",
            "d11c2_training_scope",
            "d11c2_allow_as_input_metadata",
            "d11c2_forbidden_claim",
            "d11c2_notes",
        ]),
        ("target_rule", {
            "batch_id": TARGET_BATCH,
            "battery_id": TARGET_BATTERY,
            "protocol": TARGET_PROTOCOL,
            "flag": "late_2C_discharge_boundary_outlier",
            "training_scope": "excluded_from_23profile_mainline",
        }),
        ("forbidden", [
            "direct_24profile_200ks_mainline_claim",
            "hard_voltage_guard_repair",
            "component_clamp_repair",
            "adopt_D10P3_calibration_as_mainline",
            "overwrite_D9_6_or_D9_5_1_mainline",
        ]),
        ("future_allowed_only_after_manual_review", [
            "separate_metadata_on_off_ablation",
            "23profile_non_target_sanity_check",
            "flagged_battery8_reported_separately",
        ]),
    ])

    write_csv_dicts(Path(outputs["enriched_metadata_manifest_csv"]), enriched)
    write_csv_dicts(Path(outputs["guardrail_checklist_csv"]), guardrails)
    write_csv_dicts(Path(outputs["feature_context_csv"]), feature_context)
    write_json(Path(outputs["patch_contract_json"]), patch_contract)
    write_json(Path(outputs["summary_json"]), summary)
    Path(outputs["d12_training_stub_md"]).write_text(make_d12_stub(out_dir), encoding="utf-8")
    Path(outputs["recommendation_md"]).write_text(make_recommendation_md(summary, guardrails), encoding="utf-8")

    print(json.dumps({
        "ok": summary["ok"],
        "verdict": verdict,
        "next_action": next_action,
        "out_dir": str(out_dir),
        "profile_count": len(enriched),
        "target_profile_count": target_count,
        "guard_counts": dict(guard_counts),
        "recommendation_md": outputs["recommendation_md"],
    }, ensure_ascii=False, indent=2))
    if args.strict and guard_counts.get("fail", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
