#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GV1 D10-P5 regime/outlier policy and D11 planning package.

This script is intentionally report-only.  It does not modify gv1/model.py,
gv1/output_transform.py, gv1/losses.py, gv1/trainer.py, or the D9.6/D9.5.1
mainline training entrypoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

EXPECTED_D10P0_VERDICT = "battery8_flagged_late_2C_discharge_regime_outlier_keep_D9_6_mainline"
EXPECTED_D10P3_VERDICT = "no_safe_lightweight_correction_keep_battery8_flagged"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"__read_error__": str(exc), "__path__": str(path)}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return str(value)


def parse_markdown_verdict(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"verdict\s*=\s*([A-Za-z0-9_\-]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"Verdict:\s*\n\s*```text\s*\n\s*([^`\n]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"Verdict:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().strip("` ")
    return ""


def file_contains(path: Path, patterns: Iterable[str]) -> Dict[str, bool]:
    text = read_text(path)
    lower = text.lower()
    return {p: (p.lower() in lower) for p in patterns}


def hard_clamp_default_false(path: Path) -> bool:
    text = read_text(path)
    if not text:
        return False
    pats = [
        r"enable_voltage_hard_clamp\s*:\s*bool\s*=\s*False",
        r"enable_voltage_hard_clamp\s*=\s*False",
        r"enable_voltage_hard_clamp\s*:\s*bool\s*=\s*false",
        r"enable_voltage_hard_clamp\s*=\s*false",
    ]
    return any(re.search(p, text) for p in pats)


def status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "pass"
    if warn:
        return "warn"
    return "fail"


def make_check(
    check_id: str,
    check: str,
    observed: Any,
    expected: str,
    passed: bool,
    evidence_path: Path | str = "",
    action: str = "",
    warn_if_fail: bool = False,
) -> Dict[str, Any]:
    return {
        "check_id": check_id,
        "check": check,
        "status": status(passed, warn_if_fail),
        "observed": as_text(observed),
        "expected": expected,
        "evidence_path": str(evidence_path) if evidence_path else "",
        "action": action,
    }


def default_cache_path(cache_root: Path, name: str) -> Path:
    return cache_root / name


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="GV1 D10-P5 report-only regime policy and D11 plan generator."
    )
    ap.add_argument("--cache_root", default=r"E:\XJTU battery dataset\_gv1_cache")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--d10p0_dir", default="")
    ap.add_argument("--d10p1_dir", default="")
    ap.add_argument("--d10p3_dir", default="")
    ap.add_argument("--d10p4_dir", default="")
    ap.add_argument("--flag_batch_id", default="Batch-1")
    ap.add_argument("--flag_protocol", default="2C")
    ap.add_argument("--flag_battery_id", default="battery-8")
    ap.add_argument("--strict", action="store_true", help="Return non-zero if required checks fail.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = Path(args.cache_root)
    project_root = Path(args.project_root)
    out_dir = Path(args.out_dir) if args.out_dir else default_cache_path(cache_root, "xjtu_batch134_d10_p5_regime_policy_d11_plan")
    out_dir.mkdir(parents=True, exist_ok=True)

    d10p0_dir = Path(args.d10p0_dir) if args.d10p0_dir else default_cache_path(cache_root, "xjtu_batch134_d10_p0_battery8_regime_judgement")
    d10p1_dir = Path(args.d10p1_dir) if args.d10p1_dir else default_cache_path(cache_root, "xjtu_batch134_train_conditioned_pinn_23x200ks_d10p1_exclude_battery8")
    d10p3_dir = Path(args.d10p3_dir) if args.d10p3_dir else default_cache_path(cache_root, "xjtu_batch134_d10_p3_battery8_lightweight_correction")
    d10p4_dir = Path(args.d10p4_dir) if args.d10p4_dir else default_cache_path(cache_root, "xjtu_batch134_d10_p4_final_mainline_decision")

    d10p0_path = d10p0_dir / "d10_p0_battery8_judgement_summary.json"
    d10p1_json_path = d10p1_dir / "scorecard_d10_p1_23profile_200ks.json"
    d10p1_csv_path = d10p1_dir / "scorecard_d10_p1_23profile_200ks.csv"
    d10p3_md_path = d10p3_dir / "D10_P3_RECOMMENDATION.md"
    d10p4_json_path = d10p4_dir / "d10_p4_final_mainline_decision_summary.json"
    d10p4_md_path = d10p4_dir / "D10_P4_FINAL_MAINLINE_DECISION.md"

    d10p0 = load_json(d10p0_path) or {}
    d10p1 = load_json(d10p1_json_path) or {}
    d10p4 = load_json(d10p4_json_path) or {}
    d10p3_md = read_text(d10p3_md_path)
    d10p3_verdict = parse_markdown_verdict(d10p3_md)

    d10p1_counts = d10p1.get("counts", {}) if isinstance(d10p1, dict) else {}
    d10p1_passed = (
        d10p1.get("status") == "pass"
        and int(d10p1.get("profile_count", -1)) == 23
        and int(d10p1_counts.get("pass", -1)) == 23
        and int(d10p1_counts.get("borderline", -1)) == 0
        and int(d10p1_counts.get("fail", -1)) == 0
        and int(d10p1_counts.get("read_error", -1)) == 0
    )

    train_entry = project_root / "scripts" / "gv1_train_conditioned_pinn.py"
    trainer = project_root / "gv1" / "trainer.py"
    output_transform = project_root / "gv1" / "output_transform.py"
    train_marks = file_contains(train_entry, ["D9.5.1", "trend-first", "warmup", "rare-regime"])
    trainer_marks = file_contains(trainer, ["D9.5.1", "trend-first", "warmup"])
    clamp_false = hard_clamp_default_false(output_transform)

    checks: List[Dict[str, Any]] = []
    checks.append(make_check(
        "D10P0_VERDICT",
        "battery-8 regime/outlier judgement",
        d10p0.get("verdict", "missing"),
        EXPECTED_D10P0_VERDICT,
        d10p0.get("verdict") == EXPECTED_D10P0_VERDICT,
        d10p0_path,
        "Keep battery-8 flagged/excluded if this passes; rerun D10-P0 if missing.",
    ))
    checks.append(make_check(
        "D10P1_23PROFILE_SCORECARD",
        "23-profile 200ks excluding battery-8 scorecard",
        {"status": d10p1.get("status"), "profile_count": d10p1.get("profile_count"), "counts": d10p1_counts},
        "status=pass, profile_count=23, pass=23, borderline=0, fail=0, read_error=0",
        d10p1_passed,
        d10p1_json_path,
        "Do not advance to D11 model changes until this passes.",
    ))
    checks.append(make_check(
        "D10P3_LIGHTWEIGHT_CORRECTION",
        "battery-8 lightweight correction verdict",
        d10p3_verdict or "missing",
        EXPECTED_D10P3_VERDICT,
        d10p3_verdict == EXPECTED_D10P3_VERDICT,
        d10p3_md_path,
        "No D10-P3 correction should be adopted if this passes.",
    ))
    checks.append(make_check(
        "D10P4_FINAL_DECISION_EXISTS",
        "D10-P4 final mainline decision archive",
        {"json_exists": d10p4_json_path.exists(), "md_exists": d10p4_md_path.exists(), "ok": d10p4.get("ok")},
        "D10-P4 summary JSON and final decision MD should exist; JSON ok=true is preferred.",
        bool(d10p4_json_path.exists() and d10p4_md_path.exists() and (d10p4.get("ok", True) is True)),
        d10p4_dir,
        "Create D10-P4 archive first if this is missing.",
        warn_if_fail=True,
    ))
    checks.append(make_check(
        "CODE_TRAIN_ENTRY_MAINLINE_MARKERS",
        "train entry keeps D9.5.1 trend-first warmup rare-regime markers",
        train_marks,
        "All markers present in scripts/gv1_train_conditioned_pinn.py.",
        all(train_marks.values()),
        train_entry,
        "Do not run D11 until D9.6/D9.5.1 mainline files are restored.",
    ))
    checks.append(make_check(
        "CODE_TRAINER_MAINLINE_MARKERS",
        "trainer keeps D9.5.1 trend-first warmup markers",
        trainer_marks,
        "All markers present in gv1/trainer.py.",
        all(trainer_marks.values()),
        trainer,
        "Do not run D11 until trainer is restored.",
    ))
    checks.append(make_check(
        "CODE_NO_HARD_VOLTAGE_CLAMP_DEFAULT",
        "output transform does not default to hard voltage clamp",
        {"enable_voltage_hard_clamp_default_false": clamp_false},
        "enable_voltage_hard_clamp default should be False.",
        clamp_false,
        output_transform,
        "Avoid D9.6.1/D9.6.2-style hard clamp or component clamp.",
    ))

    required_ids = {"D10P0_VERDICT", "D10P1_23PROFILE_SCORECARD", "D10P3_LIGHTWEIGHT_CORRECTION", "CODE_TRAIN_ENTRY_MAINLINE_MARKERS", "CODE_TRAINER_MAINLINE_MARKERS", "CODE_NO_HARD_VOLTAGE_CLAMP_DEFAULT"}
    required_pass = all(row["status"] == "pass" for row in checks if row["check_id"] in required_ids)
    warn_count = sum(1 for row in checks if row["status"] == "warn")
    fail_count = sum(1 for row in checks if row["status"] == "fail")

    flag_profile_id = f"B1_2C_{args.flag_battery_id.replace('-', '')}"
    registry_rows = [
        {
            "registry_version": "D10-P5-v1",
            "dataset_id": "XJTU",
            "batch_id": args.flag_batch_id,
            "protocol": args.flag_protocol,
            "battery_id": args.flag_battery_id,
            "cell_uid": f"{args.flag_batch_id}_{args.flag_battery_id}",
            "profile_id": flag_profile_id,
            "flag_status": "flagged_excluded_from_mainline_claim",
            "flag_reason": "late_2C_discharge_regime_outlier_unresolved_by_safe_lightweight_correction",
            "evidence_d10p0_verdict": d10p0.get("verdict", "missing"),
            "evidence_d10p1_context": "23-profile 200ks excluding battery-8 passed" if d10p1_passed else "23-profile D10-P1 not confirmed",
            "evidence_d10p3_verdict": d10p3_verdict or "missing",
            "adopted_correction": "none",
            "mainline_policy": "D9.6/D9.5.1 accepted for non-outlier 23-profile medium-window verification; keep this profile flagged until a new regime-aware strategy is justified.",
            "forbidden_handling": "Do not force-pass with hard voltage clamp, component clamp, or D9.6.1/D9.6.2/D9.6.3 replacement.",
        }
    ]

    d11_rows = [
        {
            "route_id": "D11-A",
            "route_name": "mainline_report_and_flag_registry",
            "status": "recommended_now",
            "allowed": "true",
            "objective": "Freeze D9.6/D9.5.1 as the non-outlier GV1 mainline and keep battery-8 in a formal flagged registry.",
            "minimum_next_step": "Use D10-P5 outputs in README/report; do not retrain.",
            "expected_artifacts": "flagged_profile_registry.csv; D10_P5_RECOMMENDATION.md; D10/D11 report section.",
            "risk": "low",
        },
        {
            "route_id": "D11-B",
            "route_name": "regime_feature_distance_audit",
            "status": "recommended_next_analysis",
            "allowed": "true",
            "objective": "Quantify whether battery-8 is isolated by current/temperature/voltage/SOH trajectory features versus B1_2C peers.",
            "minimum_next_step": "Create feature-distance audit before model changes.",
            "expected_artifacts": "profile_feature_table.csv; battery8_peer_distance_report.json; plots.",
            "risk": "low",
        },
        {
            "route_id": "D11-C",
            "route_name": "flag_aware_metadata_input_ablation",
            "status": "design_only_until_D11B_passes",
            "allowed": "conditional",
            "objective": "Test a small metadata/regime flag input without changing D9.6 voltage transform or loss guards.",
            "minimum_next_step": "Only after D11-B shows a reproducible regime boundary; use train/val only.",
            "expected_artifacts": "one small ablation plan; no 24-profile mainline claim.",
            "risk": "medium",
        },
        {
            "route_id": "D11-D",
            "route_name": "late2C_discharge_expert_branch",
            "status": "research_candidate_not_current_mainline",
            "allowed": "conditional",
            "objective": "Design a separate discharge-regime expert for late-2C profiles if more peer data supports this regime.",
            "minimum_next_step": "Require holdout validation and clear anti-overfit criteria.",
            "expected_artifacts": "D11 expert-branch proposal; held-out validation design.",
            "risk": "medium_high",
        },
        {
            "route_id": "D11-X",
            "route_name": "hard_voltage_or_component_clamp_repair",
            "status": "forbidden",
            "allowed": "false",
            "objective": "Do not repeat D9.6.1/D9.6.2-style hard guards or component clamps.",
            "minimum_next_step": "None.",
            "expected_artifacts": "N/A",
            "risk": "known_failure",
        },
        {
            "route_id": "D11-Y",
            "route_name": "direct_24profile_200ks_mainline_claim",
            "status": "forbidden_until_new_regime_strategy",
            "allowed": "false",
            "objective": "Do not claim 24-profile 200ks as mainline while battery-8 remains unresolved.",
            "minimum_next_step": "Complete D11-B/D11-C or keep battery-8 excluded/flagged.",
            "expected_artifacts": "N/A",
            "risk": "misleading_mainline_claim",
        },
    ]

    check_cols = ["check_id", "check", "status", "observed", "expected", "evidence_path", "action"]
    write_csv(out_dir / "d10_p5_mainline_acceptance_checklist.csv", checks, check_cols)
    write_csv(out_dir / "d10_p5_flagged_profile_registry.csv", registry_rows, registry_rows[0].keys())
    write_csv(out_dir / "d10_p5_d11_candidate_routes.csv", d11_rows, d11_rows[0].keys())

    verdict = "d10_p5_mainline_freeze_and_regime_policy_ready_for_d11" if required_pass else "d10_p5_incomplete_required_checks_failed"
    summary = {
        "ok": bool(required_pass),
        "stage": "D10-P5 regime/outlier policy and D11 plan",
        "verdict": verdict,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "cache_root": str(cache_root),
        "out_dir": str(out_dir),
        "mainline_policy": {
            "accepted_mainline": "GV1 D9.6 / D9.5.1 trend-first warmup rare-regime",
            "accepted_scope": "non-outlier 23-profile 200ks excluding/flagging B1_2C battery-8",
            "battery8_status": registry_rows[0]["flag_status"],
            "adopted_battery8_correction": "none",
            "next_recommended_route": "D11-B regime_feature_distance_audit, after D10-P5 archive/report update",
        },
        "evidence": {
            "d10p0_verdict": d10p0.get("verdict"),
            "d10p1_status": d10p1.get("status"),
            "d10p1_profile_count": d10p1.get("profile_count"),
            "d10p1_counts": d10p1_counts,
            "d10p1_mean_mae_V": d10p1.get("mean_mae_V"),
            "d10p1_mean_rmse_V": d10p1.get("mean_rmse_V"),
            "d10p1_mean_corr": d10p1.get("mean_corr"),
            "d10p3_verdict": d10p3_verdict,
            "d10p4_ok": d10p4.get("ok") if isinstance(d10p4, dict) else None,
        },
        "check_counts": {
            "pass": sum(1 for row in checks if row["status"] == "pass"),
            "warn": warn_count,
            "fail": fail_count,
        },
        "outputs": {
            "recommendation_md": str(out_dir / "D10_P5_RECOMMENDATION.md"),
            "summary_json": str(out_dir / "d10_p5_regime_policy_summary.json"),
            "checklist_csv": str(out_dir / "d10_p5_mainline_acceptance_checklist.csv"),
            "flagged_registry_csv": str(out_dir / "d10_p5_flagged_profile_registry.csv"),
            "d11_routes_csv": str(out_dir / "d10_p5_d11_candidate_routes.csv"),
        },
    }
    (out_dir / "d10_p5_regime_policy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = f"""# D10-P5 Regime Policy and D11 Plan

## Verdict

```text
{verdict}
```

## Main decision

- Keep `GV1 D9.6 / D9.5.1 trend-first warmup rare-regime` as the current GV1 mainline for non-outlier profiles.
- Keep `B1_2C battery-8` flagged/excluded as a late-2C discharge regime / outlier case.
- Adopt no D10-P3 battery-8 correction.
- Do not replace the mainline with D9.6.1, D9.6.2, D9.6.3, or hard/component voltage guards.

## Evidence snapshot

| item | observed |
|---|---|
| D10-P0 verdict | `{as_text(d10p0.get('verdict', 'missing'))}` |
| D10-P1 status | `{as_text(d10p1.get('status', 'missing'))}` |
| D10-P1 profile_count | `{as_text(d10p1.get('profile_count', 'missing'))}` |
| D10-P1 counts | `{as_text(d10p1_counts)}` |
| D10-P1 mean_MAE_V | `{as_text(d10p1.get('mean_mae_V', 'missing'))}` |
| D10-P1 mean_corr | `{as_text(d10p1.get('mean_corr', 'missing'))}` |
| D10-P3 verdict | `{as_text(d10p3_verdict or 'missing')}` |
| D10-P4 archive | `json_exists={d10p4_json_path.exists()}, md_exists={d10p4_md_path.exists()}` |

## D11 recommendation

Start with **D11-B: regime feature distance audit** before changing the model.  This keeps D9.6/D9.5.1 protected while quantifying whether battery-8 is isolated by measured-current, temperature, voltage, segment, and peer-distance features.

Allowed next routes:

1. `D11-A` report and flagged registry update.
2. `D11-B` regime feature distance audit.
3. `D11-C` small flag-aware metadata ablation, only after D11-B supports a reproducible regime boundary.
4. `D11-D` late-2C discharge expert branch, only with holdout validation and a clear anti-overfit plan.

Forbidden routes:

- Hard voltage clamp / component clamp repair.
- Direct 24-profile 200ks mainline claim while battery-8 remains unresolved.
- Treating battery-8 as a normal failed profile instead of a flagged unresolved regime case.

## Generated files

```text
{out_dir / 'd10_p5_regime_policy_summary.json'}
{out_dir / 'd10_p5_mainline_acceptance_checklist.csv'}
{out_dir / 'd10_p5_flagged_profile_registry.csv'}
{out_dir / 'd10_p5_d11_candidate_routes.csv'}
```
"""
    (out_dir / "D10_P5_RECOMMENDATION.md").write_text(md, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.strict and not required_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
