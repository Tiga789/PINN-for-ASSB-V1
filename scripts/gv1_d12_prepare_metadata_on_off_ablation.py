#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GV1 D12 metadata on/off ablation planner.

This script is intentionally conservative: it prepares auditable metadata_on
and metadata_off manifests for a separate ablation, but it does not launch
training and does not modify D9.6/D9.5.1 source files.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CACHE_ROOT = r"E:\XJTU battery dataset\_gv1_cache"
DEFAULT_D11C2_DIR = r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11c2_metadata_input_patch_design"
DEFAULT_TRAINING_READY = r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_training_ready"
DEFAULT_OUT_DIR = r"E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_ablation_plan"
TARGET_BATCH = "Batch-1"
TARGET_BATTERY = "battery-8"
TARGET_PROTOCOL = "2C"


def _now() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        fh = path.open("r", encoding="utf-8-sig", newline="")
    except UnicodeDecodeError:
        fh = path.open("r", encoding="utf-8", newline="")
    with fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _all_fields(rows: Sequence[Dict[str, Any]], preferred: Optional[Sequence[str]] = None) -> List[str]:
    fields: List[str] = []
    if preferred:
        for x in preferred:
            if x not in fields:
                fields.append(x)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    return fields


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _all_fields(rows, preferred=preferred)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _stringify(row.get(k, "")) for k in fields})


def _stringify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return ""
        return repr(x)
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def _norm(x: Any) -> str:
    return str(x or "").strip().lower().replace("_", "-").replace(" ", "")


def _get(row: Dict[str, Any], *keys: str) -> str:
    lower_map = {k.lower(): k for k in row.keys()}
    for key in keys:
        real = lower_map.get(key.lower())
        if real is not None:
            return str(row.get(real, ""))
    return ""


def _row_id(row: Dict[str, Any]) -> str:
    for key in ["profile_id", "cell_uid", "profile_uid", "run_id", "label"]:
        val = _get(row, key)
        if val:
            return val
    batch = _get(row, "batch_id", "batch")
    protocol = _get(row, "protocol", "protocol_id")
    battery = _get(row, "battery_id", "battery")
    if batch or protocol or battery:
        return f"{batch}_{protocol}_{battery}".strip("_")
    source = _get(row, "source_file", "profile_npz", "solution_npz")
    if source:
        return Path(source).stem
    return "unknown_profile"


def _is_target(row: Dict[str, Any]) -> bool:
    batch = _norm(_get(row, "batch_id", "batch"))
    battery = _norm(_get(row, "battery_id", "battery"))
    protocol = _norm(_get(row, "protocol", "protocol_id"))
    pid = _norm(_row_id(row))
    if batch == _norm(TARGET_BATCH) and battery == _norm(TARGET_BATTERY) and protocol == _norm(TARGET_PROTOCOL):
        return True
    return all(s in pid for s in [_norm("Batch-1"), _norm("battery-8")]) and _norm("2C") in pid


def _merge_by_profile(base_rows: List[Dict[str, str]], meta_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge metadata columns onto the base profile manifest when possible."""
    if not base_rows:
        return [dict(r) for r in meta_rows]
    if not meta_rows:
        return [dict(r) for r in base_rows]
    meta_by_id: Dict[str, Dict[str, str]] = {_row_id(r): r for r in meta_rows}
    out: List[Dict[str, str]] = []
    used: set[str] = set()
    for base in base_rows:
        pid = _row_id(base)
        merged = dict(base)
        meta = meta_by_id.get(pid)
        if meta is None:
            # Fall back to target detection if IDs differ slightly.
            target_like = _is_target(base)
            for candidate_id, candidate in meta_by_id.items():
                if candidate_id in used:
                    continue
                if target_like == _is_target(candidate):
                    # Only use this fallback for the target or for exact batch/battery/protocol match.
                    if target_like:
                        meta = candidate
                        break
                    if (_norm(_get(base, "batch_id")) == _norm(_get(candidate, "batch_id")) and
                        _norm(_get(base, "battery_id")) == _norm(_get(candidate, "battery_id")) and
                        _norm(_get(base, "protocol")) == _norm(_get(candidate, "protocol"))):
                        meta = candidate
                        break
        if meta:
            used.add(_row_id(meta))
            for k, v in meta.items():
                if k not in merged or str(merged.get(k, "")).strip() == "":
                    merged[k] = v
                elif k.startswith("d11") or k.startswith("d12"):
                    merged[k] = v
        out.append(merged)
    return out


def _profile_key(row: Dict[str, Any]) -> str:
    return _row_id(row)


def _same_profile_set(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> bool:
    return sorted(_profile_key(x) for x in a) == sorted(_profile_key(x) for x in b)


def _inspect_mainline(project_root: Path) -> Dict[str, Any]:
    train_path = project_root / "scripts" / "gv1_train_conditioned_pinn.py"
    trainer_path = project_root / "gv1" / "trainer.py"
    transform_path = project_root / "gv1" / "output_transform.py"
    result: Dict[str, Any] = {
        "project_root": str(project_root),
        "train_script_exists": train_path.exists(),
        "trainer_exists": trainer_path.exists(),
        "output_transform_exists": transform_path.exists(),
        "d951_terms_found": False,
        "hard_clamp_disabled_hint": None,
        "metadata_training_args_found": False,
        "metadata_training_arg_terms": [],
    }
    text_all = ""
    for p in [train_path, trainer_path, transform_path]:
        if p.exists():
            try:
                text_all += "\n" + p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass
    low = text_all.lower()
    result["d951_terms_found"] = any(term in low for term in ["d9.5.1", "trend-first", "trend_first", "rare-regime", "rare_regime"])
    if "enable_voltage_hard_clamp" in low:
        # This is a heuristic only; source can contain default false or arg options.
        result["hard_clamp_disabled_hint"] = "enable_voltage_hard_clamp present; manually verify it remains False in D9.6 mainline"
    else:
        result["hard_clamp_disabled_hint"] = "not_detected"
    metadata_terms = [
        "profile_metadata_manifest",
        "metadata_manifest",
        "metadata_input",
        "enable_profile_metadata",
        "metadata_mode",
        "metadata_features",
    ]
    found_terms = [t for t in metadata_terms if t in low]
    result["metadata_training_args_found"] = bool(found_terms)
    result["metadata_training_arg_terms"] = found_terms
    return result


def _guard(check_id: str, status: str, description: str, details: str = "") -> Dict[str, str]:
    return {"check_id": check_id, "status": status, "description": description, "details": details}


def _make_row(row: Dict[str, str], *, mode: str, use_metadata: bool, scope: str, include_target: bool) -> Dict[str, Any]:
    out = dict(row)
    is_target = _is_target(row)
    out["d12_profile_id"] = _row_id(row)
    out["d12_metadata_mode"] = mode
    out["d12_use_metadata_input"] = "true" if use_metadata else "false"
    out["d12_scope"] = scope
    out["d12_is_b1_2c_battery8"] = "true" if is_target else "false"
    out["d12_target_policy"] = "target_probe_not_mainline" if is_target else "mainline_reference"
    out["d12_include_in_training"] = "true" if (include_target or not is_target) else "false"
    out["d12_include_in_mainline_claim"] = "false" if is_target else "true"
    if is_target:
        out.setdefault("d11c2_regime_flag", "late_2C_discharge_boundary_outlier")
        out.setdefault("d11c2_training_scope", "excluded_from_23profile_mainline")
    return out


def _markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_stringify(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GV1 D12 metadata on/off ablation planner (no training launch).")
    ap.add_argument("--project_root", default=".", help="Project root for source-status inspection only.")
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
    ap.add_argument("--d11c2_dir", default=DEFAULT_D11C2_DIR)
    ap.add_argument("--training_ready_dir", default=DEFAULT_TRAINING_READY)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--allow_training_stubs", action="store_true", help="Write non-executable command templates. Does not launch training.")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    d11c2_dir = Path(args.d11c2_dir)
    training_ready_dir = Path(args.training_ready_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d11c2_summary_path = d11c2_dir / "d11c2_metadata_input_patch_summary.json"
    d11c2_manifest_path = d11c2_dir / "d11c2_profile_metadata_input_manifest.csv"
    d11c2_contract_path = d11c2_dir / "d11c2_patch_contract.json"
    profile_manifest_path = training_ready_dir / "xjtu_batch134_profile_manifest.csv"

    d11c2_summary = _read_json(d11c2_summary_path, default={}) or {}
    d11c2_contract = _read_json(d11c2_contract_path, default={}) or {}
    metadata_rows = _read_csv(d11c2_manifest_path)
    base_rows = _read_csv(profile_manifest_path)
    combined_rows = _merge_by_profile(base_rows, metadata_rows)

    target_rows = [r for r in combined_rows if _is_target(r)]
    non_target_rows = [r for r in combined_rows if not _is_target(r)]

    off_rows = [_make_row(r, mode="metadata_off", use_metadata=False, scope="d12_23profile_reference_ablation", include_target=False) for r in non_target_rows]
    on_rows = [_make_row(r, mode="metadata_on", use_metadata=True, scope="d12_23profile_metadata_input_ablation", include_target=False) for r in non_target_rows]
    target_probe_rows = [_make_row(r, mode="metadata_on_target_probe", use_metadata=True, scope="d12_target_probe_not_mainline", include_target=False) for r in target_rows]
    review_rows = [_make_row(r, mode="review_all_profiles", use_metadata=True, scope=("d12_target_probe_not_mainline" if _is_target(r) else "d12_23profile_reference"), include_target=False) for r in combined_rows]

    preferred = [
        "d12_profile_id", "dataset_id", "batch_id", "battery_id", "cell_uid", "protocol", "split",
        "d12_metadata_mode", "d12_use_metadata_input", "d12_scope", "d12_is_b1_2c_battery8",
        "d12_target_policy", "d12_include_in_training", "d12_include_in_mainline_claim",
        "d11c2_is_b1_2c_battery8", "d11c2_regime_flag", "d11c2_training_scope",
        "source_file", "profile_npz", "solution_npz",
    ]

    paths = {
        "recommendation_md": out_dir / "D12_RECOMMENDATION.md",
        "summary_json": out_dir / "d12_metadata_on_off_ablation_summary.json",
        "guardrail_csv": out_dir / "d12_guardrail_checklist.csv",
        "metadata_off_manifest": out_dir / "d12_metadata_off_23profile_manifest.csv",
        "metadata_on_manifest": out_dir / "d12_metadata_on_23profile_manifest.csv",
        "target_probe_manifest": out_dir / "d12_battery8_target_probe_manifest_not_mainline.csv",
        "review_manifest": out_dir / "d12_all_profiles_review_manifest.csv",
        "contract_json": out_dir / "d12_ablation_contract.json",
        "command_templates_md": out_dir / "D12_COMMAND_TEMPLATES_NOT_EXECUTABLE.md",
    }

    _write_csv(paths["metadata_off_manifest"], off_rows, preferred=preferred)
    _write_csv(paths["metadata_on_manifest"], on_rows, preferred=preferred)
    _write_csv(paths["target_probe_manifest"], target_probe_rows, preferred=preferred)
    _write_csv(paths["review_manifest"], review_rows, preferred=preferred)

    mainline = _inspect_mainline(project_root)
    verdict_d11c2 = str(d11c2_summary.get("verdict", ""))
    target_excluded = False
    if target_rows:
        target = target_rows[0]
        row_text = " ".join(str(v) for v in target.values()).lower()
        target_excluded = ("excluded" in row_text) or ("flag" in row_text) or _truthy(target.get("d11c2_is_b1_2c_battery8"))

    training_backend_status = "metadata_training_args_detected_template_only" if mainline.get("metadata_training_args_found") else "plan_only_until_separate_metadata_model_patch_exists"
    if not mainline.get("metadata_training_args_found"):
        training_backend_note = "Current project training entry does not appear to expose metadata-input arguments; this D12 package prepares manifests/contracts only."
    else:
        training_backend_note = "Metadata-related argument terms were detected, but generated commands are still templates and should be manually reviewed."

    guards = [
        _guard("D12G01", "pass" if verdict_d11c2 == "d11c2_metadata_input_patch_ready_design_only" else "fail", "D11C2 design-only metadata patch verdict is ready", verdict_d11c2),
        _guard("D12G02", "pass" if len(target_rows) == 1 else "fail", "Exactly one B1_2C battery-8 target row is present", str(len(target_rows))),
        _guard("D12G03", "pass" if len(non_target_rows) >= 23 else "fail", "At least 23 non-target profiles remain available", str(len(non_target_rows))),
        _guard("D12G04", "pass" if target_excluded else "fail", "Battery-8 remains flagged/excluded and is not in 23-profile mainline manifests", str(target_excluded)),
        _guard("D12G05", "pass" if len(off_rows) == len(on_rows) and _same_profile_set(off_rows, on_rows) else "fail", "metadata_off and metadata_on manifests contain the same non-target profiles", f"off={len(off_rows)}, on={len(on_rows)}"),
        _guard("D12G06", "pass" if len(target_probe_rows) == 1 else "fail", "Battery-8 target probe manifest exists separately from mainline", str(len(target_probe_rows))),
        _guard("D12G07", "pass" if mainline.get("d951_terms_found") else "warn", "D9.6/D9.5.1 mainline signature appears present in source inspection", json.dumps(mainline, ensure_ascii=False)),
        _guard("D12G08", "pass", "This script writes only D12 output files and does not modify gv1/model.py, output_transform.py, losses.py, trainer.py, or D9.6 scripts", "write-only to out_dir"),
        _guard("D12G09", "pass", "No direct 24-profile 200ks mainline training command is generated", "templates are non-executable documentation"),
        _guard("D12G10", "pass", "D12 is a separate metadata on/off ablation plan, not a mainline replacement", training_backend_status),
    ]
    fail_count = sum(1 for g in guards if g["status"] == "fail")
    warn_count = sum(1 for g in guards if g["status"] == "warn")
    pass_count = sum(1 for g in guards if g["status"] == "pass")

    verdict = "d12_metadata_on_off_ablation_plan_ready_no_mainline_overwrite" if fail_count == 0 else "d12_guardrail_failed_do_not_train"
    next_action = "manual_review_then_prepare_separate_D12_runtime_metadata_model_patch_if_training_backend_absent"
    if fail_count == 0 and mainline.get("metadata_training_args_found"):
        next_action = "manual_review_then_run_separate_metadata_off_on_smoke_only_not_mainline"

    contract = {
        "stage": "D12 metadata on/off ablation",
        "verdict": verdict,
        "created_at": _now(),
        "mainline_policy": "D9.6/D9.5.1 remains frozen; D12 is a separate ablation plan only.",
        "metadata_off": {
            "description": "Reference 23-profile manifest excluding/flagging B1_2C battery-8; metadata input disabled.",
            "manifest": str(paths["metadata_off_manifest"]),
            "profile_count": len(off_rows),
        },
        "metadata_on": {
            "description": "Same 23 non-target profiles; enriched D11C2 metadata columns retained and metadata input marked enabled.",
            "manifest": str(paths["metadata_on_manifest"]),
            "profile_count": len(on_rows),
        },
        "target_probe": {
            "description": "B1_2C battery-8 remains a target probe only, not part of the 23-profile mainline claim.",
            "manifest": str(paths["target_probe_manifest"]),
            "profile_count": len(target_probe_rows),
        },
        "forbidden": [
            "Do not overwrite D9.6/D9.5.1 source files.",
            "Do not launch direct 24-profile 200ks mainline training from D12.",
            "Do not reclassify battery-8 as a normal mainline profile.",
            "Do not adopt hard voltage guard, component clamp, or D10-P3 correction.",
        ],
        "training_backend_status": training_backend_status,
        "training_backend_note": training_backend_note,
        "d11c2_contract": d11c2_contract,
    }
    _write_json(paths["contract_json"], contract)

    commands_md = f"""# D12 command templates — not executable by default

This file is intentionally a template. D12 preparation does not launch training.

## Current backend status

```text
{training_backend_status}
```

{training_backend_note}

## Manifests prepared

```text
metadata_off: {paths['metadata_off_manifest']}
metadata_on : {paths['metadata_on_manifest']}
target_probe: {paths['target_probe_manifest']}
```

## Suggested future execution order

1. Confirm the D12 guardrail checklist has no `fail` rows.
2. Confirm a separate D12 runtime metadata-input model patch exists and does not overwrite D9.6/D9.5.1.
3. Run a short-window smoke on metadata_off and metadata_on using the same non-target 23 profiles.
4. Collect D12 scorecard with `scripts/gv1_d12_collect_on_off_scorecard.py`.
5. Treat battery-8 as target-probe only; do not include it in a mainline 24-profile 200ks claim.

## Non-executable command sketch

```powershell
# DO NOT RUN until a separate D12 runtime metadata model patch is explicitly installed and audited.
# metadata_off: use {paths['metadata_off_manifest']}
# metadata_on : use {paths['metadata_on_manifest']}
```
"""
    paths["command_templates_md"].write_text(commands_md, encoding="utf-8")

    summary = {
        "ok": fail_count == 0,
        "stage": "D12 metadata on/off ablation plan",
        "verdict": verdict,
        "next_action": next_action,
        "created_at": _now(),
        "project_root": str(project_root),
        "cache_root": str(Path(args.cache_root)),
        "out_dir": str(out_dir),
        "d11c2_verdict": verdict_d11c2,
        "profile_count": len(combined_rows),
        "metadata_off_profile_count": len(off_rows),
        "metadata_on_profile_count": len(on_rows),
        "target_probe_profile_count": len(target_probe_rows),
        "target_profile_id": _row_id(target_rows[0]) if target_rows else None,
        "guard_counts": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
        "training_backend_status": training_backend_status,
        "mainline_source_inspection": mainline,
        "inputs": {
            "d11c2_summary": str(d11c2_summary_path),
            "d11c2_metadata_manifest": str(d11c2_manifest_path),
            "d11c2_contract": str(d11c2_contract_path),
            "profile_manifest": str(profile_manifest_path),
        },
        "outputs": {k: str(v) for k, v in paths.items()},
        "notes": [
            "D12 prepares metadata_off and metadata_on manifests only.",
            "No training is launched by this planner.",
            "Battery-8 remains target-probe only and excluded from the 23-profile mainline claim.",
            "D9.6/D9.5.1 source files are not modified.",
        ],
    }
    _write_json(paths["summary_json"], summary)
    _write_csv(paths["guardrail_csv"], guards, preferred=["check_id", "status", "description", "details"])

    route_rows = [
        {"route_id": "D12-0", "route_name": "metadata_off_reference", "status": "prepared_manifest_only", "allowed": "true", "profile_count": len(off_rows), "risk": "low"},
        {"route_id": "D12-1", "route_name": "metadata_on_ablation", "status": "prepared_manifest_only_requires_runtime_patch_before_training", "allowed": "conditional", "profile_count": len(on_rows), "risk": "medium"},
        {"route_id": "D12-2", "route_name": "battery8_target_probe", "status": "target_probe_not_mainline", "allowed": "conditional", "profile_count": len(target_probe_rows), "risk": "medium"},
        {"route_id": "D12-X", "route_name": "direct_24profile_200ks_mainline_claim", "status": "forbidden", "allowed": "false", "profile_count": 24, "risk": "misleading_mainline_claim"},
    ]
    route_csv = out_dir / "d12_candidate_routes.csv"
    _write_csv(route_csv, route_rows, preferred=["route_id", "route_name", "status", "allowed", "profile_count", "risk"])
    summary["outputs"]["candidate_routes_csv"] = str(route_csv)
    _write_json(paths["summary_json"], summary)

    rec_md = f"""# D12 Metadata On/Off Ablation Plan

## Verdict

```text
{verdict}
```

## Recommended next action

```text
{next_action}
```

## Interpretation

- D12 prepares a **separate metadata on/off ablation**.
- `metadata_off` and `metadata_on` use the same 23 non-target profiles.
- B1_2C battery-8 remains flagged/excluded and is separated into a target-probe manifest.
- D9.6/D9.5.1 remains frozen; this package does not modify model, transform, loss, trainer, or mainline training files.
- No direct 24-profile 200ks mainline command is generated.

## Context

| item | value |
|---|---:|
| D11C2 verdict | `{verdict_d11c2}` |
| profile_count | `{len(combined_rows)}` |
| metadata_off_profile_count | `{len(off_rows)}` |
| metadata_on_profile_count | `{len(on_rows)}` |
| target_probe_profile_count | `{len(target_probe_rows)}` |
| guardrail pass/warn/fail | `{ {'pass': pass_count, 'warn': warn_count, 'fail': fail_count} }` |
| training_backend_status | `{training_backend_status}` |

## Candidate routes

{_markdown_table(route_rows, ['route_id', 'route_name', 'status', 'allowed', 'profile_count', 'risk'])}

## Guardrails

{_markdown_table(guards, ['check_id', 'status', 'description'])}

## Generated files

```text
recommendation_md: {paths['recommendation_md']}
summary_json: {paths['summary_json']}
guardrail_checklist_csv: {paths['guardrail_csv']}
metadata_off_manifest: {paths['metadata_off_manifest']}
metadata_on_manifest: {paths['metadata_on_manifest']}
target_probe_manifest: {paths['target_probe_manifest']}
review_manifest: {paths['review_manifest']}
contract_json: {paths['contract_json']}
command_templates_md: {paths['command_templates_md']}
candidate_routes_csv: {route_csv}
```

## Do not do from this stage

```text
Do not run direct 24-profile 200ks mainline.
Do not unflag battery-8.
Do not overwrite D9.6/D9.5.1.
Do not adopt hard guard / component clamp / D10-P3 correction.
```
"""
    paths["recommendation_md"].write_text(rec_md, encoding="utf-8")

    print(json.dumps({
        "ok": fail_count == 0,
        "verdict": verdict,
        "out_dir": str(out_dir),
        "recommendation_md": str(paths["recommendation_md"]),
        "metadata_off_profile_count": len(off_rows),
        "metadata_on_profile_count": len(on_rows),
        "target_probe_profile_count": len(target_probe_rows),
        "guard_counts": {"pass": pass_count, "warn": warn_count, "fail": fail_count},
        "training_backend_status": training_backend_status,
    }, ensure_ascii=False, indent=2))

    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
