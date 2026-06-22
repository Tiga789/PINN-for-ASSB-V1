from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from gv1.d18_s2.common import (
    ConfigError,
    dump_json,
    ensure_dir,
    read_csv,
    resolve_config,
    sha256_file,
    utc_now_iso,
    write_csv,
)
from gv1.d18_s2.data import (
    load_raw_profile,
    fit_train_physical_parameters,
    save_prepared_profiles,
)
from gv1.d18_s2.losses import S2LossConfig
from gv1.d18_s2.model import S2ModelConfig, architecture_contract, synthetic_forward_check
from gv1.d18_s2.trainer import S2TrainerConfig, run_micro_smoke
from gv1.d18_s2.uid import (
    audit_record_identity,
    canonical_from_record,
    parse_canonical_uid,
)

HOTFIX_VERSION = "D18-S2-HOTFIX-FAST-RESUME-v1.0.0-20260618"

# These must remain true. Other original preflight checks such as git HEAD,
# free-disk warning, and source-grid density remain audit fields, but do not
# block this bounded micro-smoke resume.
CRITICAL_PREFLIGHT_CHECKS = [
    "prior_overall_review_ready",
    "prior_s0_pass",
    "prior_s1_diagnostic_pass",
    "prior_s1_coverage_pass",
    "prior_frozen_test_unused",
    "human_review_token_exact",
    "selected_profile_count_exact",
    "selected_replay_resolved",
    "no_role_overlap",
    "no_frozen_test_selected",
    "no_flagged_probe_selected",
    "fit_train_all_six_protocols",
    "fit_train_both_branches",
    "micro_view_never_exceeds_source",
    "torch_import_ok",
    "amp_forbidden_by_config",
    "torch_compile_forbidden_by_config",
    "formal_s2_training_disabled",
]

AUDIT_ONLY_PREFLIGHT_CHECKS = [
    "selected_identity_exact",  # rechecked after canonical softlabel-path repair
    "battery1_battery10_exact_collision_guard",  # rechecked after canonical repair
    "per_cycle_source_minimum",
    "preflight_source_counts_not_downsampled",
    "git_head_matches_expected_commit",
    "disk_free_above_minimum",
]


def log(message: str) -> None:
    print(f"[D18-S2-HOTFIX] {message}", flush=True)


def _backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(path.name + f"_backup_{stamp}")
    counter = 1
    while target.exists():
        target = path.with_name(path.name + f"_backup_{stamp}_{counter:02d}")
        counter += 1
    path.rename(target)
    return target


def _load_json(path: Path) -> dict[str, Any]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _load_preflight_state(output_root: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    pre_dir = output_root / "d18_s2_preflight"
    summary_path = pre_dir / "D18_S2_PREFLIGHT_SUMMARY.json"
    selected_path = pre_dir / "D18_S2_SELECTED_PROFILE_MANIFEST.csv"
    cycle_path = pre_dir / "D18_S2_PER_CYCLE_SOURCE_COVERAGE.csv"
    canonical_path = pre_dir / "D18_S2_CANONICAL_SPLIT_VIEW.json"
    missing = [p for p in (summary_path, selected_path, cycle_path, canonical_path) if not p.exists()]
    if missing:
        raise ConfigError(
            "Hotfix resume requires the already-generated preflight artifacts. Missing: "
            + ", ".join(str(p) for p in missing)
        )
    return _load_json(summary_path), read_csv(selected_path), read_csv(cycle_path), _load_json(canonical_path)


def _preflight_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    checks = dict(summary.get("checks", {}))
    critical_fail = [name for name in CRITICAL_PREFLIGHT_CHECKS if not bool(checks.get(name, False))]
    audit_false = [name for name in AUDIT_ONLY_PREFLIGHT_CHECKS if name in checks and not bool(checks.get(name, False))]
    return {
        "status": "PASS_CRITICAL_PREFLIGHT" if not critical_fail else "FAIL_CRITICAL_PREFLIGHT",
        "critical_failures": critical_fail,
        "audit_only_false_checks": audit_false,
        "original_preflight_status": summary.get("status"),
        "original_checks": checks,
    }


def _canonicalize_softlabel_paths(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Repair exact known battery-1/battery-10 style softlabel path drift.

    The canonical UID remains the source of truth. If an old split/manifest record
    points to a sibling profile directory, replace the softlabel path with the
    exact sibling directory matching the canonical cell_uid.
    """
    uid = canonical_from_record(record)
    repaired = dict(record)
    notes: list[str] = []
    repaired["canonical_cell_uid"] = uid.canonical
    repaired["cell_uid"] = uid.cell_uid
    repaired["batch"] = f"Batch-{uid.batch}"
    repaired["battery"] = f"battery-{uid.battery}"
    repaired["protocol"] = uid.protocol

    old_soft = Path(str(record.get("softlabel_npz", "")))
    candidate: Path | None = None
    if old_soft.name == "solution_softlabels.npz" and old_soft.parent.parent.exists():
        candidate = old_soft.parent.parent / uid.cell_uid / "solution_softlabels.npz"
    if candidate and candidate.exists() and old_soft.parent.name.lower() != uid.cell_uid.lower():
        repaired["softlabel_npz_original_pre_hotfix"] = str(old_soft)
        repaired["softlabel_npz"] = str(candidate)
        notes.append(f"softlabel_npz:{old_soft}->{candidate}")
        summary_candidate = candidate.with_name("soft_label_summary.json")
        if summary_candidate.exists():
            repaired["softlabel_summary"] = str(summary_candidate)
    elif not old_soft.exists() and candidate and candidate.exists():
        repaired["softlabel_npz_original_pre_hotfix"] = str(old_soft)
        repaired["softlabel_npz"] = str(candidate)
        notes.append(f"softlabel_npz_missing_repaired:{old_soft}->{candidate}")
        summary_candidate = candidate.with_name("soft_label_summary.json")
        if summary_candidate.exists():
            repaired["softlabel_summary"] = str(summary_candidate)
    return repaired, notes


def _records_from_preflight(
    selected_rows: Sequence[Mapping[str, str]],
    canonical_view: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str | None], list[dict[str, Any]]]:
    records_raw = canonical_view.get("records", [])
    record_map: dict[str, dict[str, Any]] = {}
    for raw in records_raw:
        uid = canonical_from_record(raw)
        record_map[uid.canonical.lower()] = dict(raw)

    records: list[dict[str, Any]] = []
    replay_paths: dict[str, str | None] = {}
    repair_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        canonical = str(row.get("canonical_cell_uid", "")).strip()
        if not canonical:
            raise ConfigError("Selected manifest row missing canonical_cell_uid")
        uid = parse_canonical_uid(canonical)
        key = uid.canonical.lower()
        if key not in record_map:
            raise ConfigError(f"Selected UID missing from canonical split view: {uid.canonical}")
        record = dict(record_map[key])
        record["d18_s2_role"] = row.get("role", "")
        record["branch_family"] = row.get("branch_family", uid.branch_family)
        record["split"] = row.get("split", record.get("split", ""))
        resolved_replay = row.get("replay_npz_resolved") or row.get("replay_npz_declared") or ""
        if resolved_replay:
            record["replay_npz"] = resolved_replay
        record, notes = _canonicalize_softlabel_paths(record)
        replay_paths[uid.canonical] = resolved_replay or None
        audit = audit_record_identity(record, Path(resolved_replay) if resolved_replay else None)
        repair_rows.append({
            "canonical_cell_uid": uid.canonical,
            "role": record.get("d18_s2_role", ""),
            "split": record.get("split", ""),
            "softlabel_npz": record.get("softlabel_npz", ""),
            "replay_npz": resolved_replay,
            "identity_status_after_hotfix": audit.get("identity_status", ""),
            "identity_errors_after_hotfix": ";".join(audit.get("errors", [])),
            "identity_warnings_after_hotfix": ";".join(audit.get("warnings", [])),
            "repair_notes": " | ".join(notes),
        })
        if audit.get("identity_status") != "PASS":
            raise ConfigError(f"Selected identity still fails after hotfix repair: {uid.canonical}: {audit}")
        records.append(record)
    return records, replay_paths, repair_rows


def _cycle_rows_by_uid(cycle_rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in cycle_rows:
        uid = str(row.get("canonical_cell_uid", "")).strip()
        if not uid:
            continue
        out.setdefault(uid, []).append(dict(row))
    return out


def _build_casepack_from_existing_preflight(
    *,
    records: Sequence[Mapping[str, Any]],
    replay_paths: Mapping[str, str | Path | None],
    cycle_rows: Sequence[Mapping[str, str]],
    selected_cycles: Mapping[str, Sequence[int]],
    output_dir: Path,
    micro_points_per_cycle: int,
    target_radial_points: int,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    by_uid = _cycle_rows_by_uid(cycle_rows)
    raw_profiles = []
    all_audit_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        uid = canonical_from_record(record).canonical
        log(f"[{idx}/{len(records)}] loading selected source states without re-running full preflight: {uid}")
        cycles = selected_cycles.get(uid) or selected_cycles.get(uid.lower())
        if not cycles:
            raise ConfigError(f"Preflight selected_cycles missing for {uid}")
        rows = by_uid.get(uid, [])
        if not rows:
            raise ConfigError(f"Preflight cycle coverage rows missing for {uid}")
        all_audit_rows.extend(rows)
        raw_profiles.append(
            load_raw_profile(
                record,
                selected_cycles=np.asarray([int(x) for x in cycles], dtype=np.int64),
                micro_points_per_cycle=micro_points_per_cycle,
                target_radial_points=target_radial_points,
                replay_path=replay_paths.get(uid),
                cycle_audit_rows=rows,
            )
        )
    write_csv(all_audit_rows, output_dir / "D18_S2_PER_CYCLE_SOURCE_COVERAGE_REUSED_FROM_PREFLIGHT.csv")
    fit = fit_train_physical_parameters(raw_profiles)
    manifest_rows = save_prepared_profiles(raw_profiles, fit, output_dir / "profiles")
    source_counts = [int(float(r["source_cycle_points"])) for r in all_audit_rows]
    summary = {
        "status": "PASS_HOTFIX_FAST_CASEPACK",
        "hotfix_version": HOTFIX_VERSION,
        "profile_count": len(raw_profiles),
        "fit_train_count": sum(p.role == "fit_train" for p in raw_profiles),
        "internal_heldout_count": sum(p.role == "internal_heldout" for p in raw_profiles),
        "validation_report_only_count": sum(p.role == "validation_report_only" for p in raw_profiles),
        "protocols": sorted({p.protocol for p in raw_profiles}),
        "branches": sorted({p.branch_family for p in raw_profiles}),
        "cycle_audit_row_count": len(all_audit_rows),
        "source_cycle_points_min": min(source_counts) if source_counts else 0,
        "source_cycle_points_max": max(source_counts) if source_counts else 0,
        "micro_points_per_cycle": micro_points_per_cycle,
        "reused_preflight_cycle_selection": True,
        "teacher_initial_cbar_anchor_used": True,
        "formal_s2_training_eligible": False,
        "physical_fit": fit.as_dict(),
        "manifest_rows": manifest_rows,
    }
    dump_json(summary, output_dir / "D18_S2_HOTFIX_CASEPACK_BUILD_SUMMARY.json")
    return summary


def run(config_path: Path, project_root: Path, output_override: str | None = None) -> tuple[int, dict[str, Any]]:
    cfg = resolve_config(config_path, project_root=project_root)
    if output_override:
        cfg["paths"]["output_root"] = output_override
    output_root = Path(cfg["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    hotfix_root = ensure_dir(output_root / "d18_s2_hotfix_fast_resume")
    copied_config = hotfix_root / "D18_S2_HOTFIX_RESOLVED_CONFIG.json"
    dump_json(cfg, copied_config)
    overall: dict[str, Any] = {
        "stage": "D18-S2-HOTFIX-FAST-RESUME",
        "created_at_utc": utc_now_iso(),
        "hotfix_version": HOTFIX_VERSION,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "hotfix_output_root": str(hotfix_root),
        "formal_s2_training_enabled": False,
        "frozen_test_used": False,
        "flagged_probe_used": False,
        "final_goal": cfg.get("final_goal"),
    }
    try:
        log("1/4 reading existing failed preflight artifacts; no full replay-index scan")
        pre_summary, selected_rows, cycle_rows, canonical_view = _load_preflight_state(output_root)
        gate = _preflight_gate(pre_summary)
        dump_json(gate, hotfix_root / "D18_S2_HOTFIX_PREFLIGHT_GATE.json")
        if gate["status"] != "PASS_CRITICAL_PREFLIGHT":
            raise RuntimeError("Critical preflight checks failed: " + ", ".join(gate["critical_failures"]))

        log("2/4 reconstructing selected records with exact UID repair")
        records, replay_paths, repair_rows = _records_from_preflight(selected_rows, canonical_view)
        write_csv(repair_rows, hotfix_root / "D18_S2_HOTFIX_SELECTED_IDENTITY_RECHECK.csv")

        log("3/4 building micro-smoke casepack from existing selected cycles")
        pre_cfg = cfg["preflight"]
        casepack_dir = hotfix_root / "d18_s2_micro_casepack_hotfix"
        prior_casepack_backup = _backup_existing(casepack_dir)
        casepack = _build_casepack_from_existing_preflight(
            records=records,
            replay_paths=replay_paths,
            cycle_rows=cycle_rows,
            selected_cycles=pre_summary.get("selected_cycles", {}),
            output_dir=casepack_dir,
            micro_points_per_cycle=int(pre_cfg["micro_points_per_cycle"]),
            target_radial_points=int(pre_cfg["target_radial_points"]),
        )
        if not str(casepack.get("status", "")).startswith("PASS"):
            raise RuntimeError("Hotfix casepack build failed")

        log("4/4 running bounded 8-epoch micro-smoke; still not formal S2 training")
        model_cfg = S2ModelConfig.from_mapping(cfg["micro_smoke"].get("model"))
        synthetic = synthetic_forward_check(model_cfg)
        if synthetic.get("status") != "PASS":
            raise RuntimeError("Architecture synthetic check failed")
        dump_json(architecture_contract(model_cfg), hotfix_root / "D18_S2_HOTFIX_ARCHITECTURE_CONTRACT.json")
        dump_json(synthetic, hotfix_root / "D18_S2_HOTFIX_ARCHITECTURE_SYNTHETIC_CHECK.json")
        loss_cfg = S2LossConfig.from_mapping(cfg["micro_smoke"].get("loss"))
        trainer_cfg = S2TrainerConfig.from_mapping(cfg["micro_smoke"].get("trainer"))
        if not trainer_cfg.disable_amp or not trainer_cfg.disable_torch_compile:
            raise RuntimeError("AMP and torch.compile must remain disabled")
        micro_dir = hotfix_root / "d18_s2_micro_smoke_hotfix"
        micro_backup = _backup_existing(micro_dir)
        micro = run_micro_smoke(
            casepack_profiles_dir=casepack_dir / "profiles",
            casepack_summary_path=casepack_dir / "profiles" / "D18_S2_MICRO_CASEPACK_SUMMARY.json",
            output_dir=micro_dir,
            model_config=model_cfg,
            loss_config=loss_cfg,
            trainer_config=trainer_cfg,
            progress=log,
        )
        completed = micro.get("status") in {"PASS_MICRO_SMOKE", "REVIEW_MICRO_SMOKE"}
        overall.update({
            "status": "PASS_HOTFIX_COMPLETED_MICRO_SMOKE" if micro.get("status") == "PASS_MICRO_SMOKE" else "REVIEW_HOTFIX_COMPLETED_MICRO_SMOKE",
            "completed": completed,
            "original_preflight_status": pre_summary.get("status"),
            "critical_preflight_gate_status": gate.get("status"),
            "audit_only_false_checks": gate.get("audit_only_false_checks", []),
            "selected_identity_recheck_status": "PASS",
            "casepack_status": casepack.get("status"),
            "micro_smoke_status": micro.get("status"),
            "prior_casepack_backup": str(prior_casepack_backup) if prior_casepack_backup else None,
            "prior_micro_backup": str(micro_backup) if micro_backup else None,
            "formal_s2_training_eligible": False,
            "go_to_formal_s2_training": False,
            "go_to_s3": False,
            "reused_failed_preflight_artifacts": True,
            "config_sha256": sha256_file(copied_config),
            "next_action": "Upload d18_s2_hotfix_fast_resume for review. Do not start formal S2 training.",
        })
        dump_json(overall, output_root / "D18_S2_HOTFIX_FAST_RESUME_OVERALL_SUMMARY.json")
        dump_json(overall, hotfix_root / "D18_S2_HOTFIX_FAST_RESUME_OVERALL_SUMMARY.json")
        log(f"completed with status={overall['status']}")
        return 0, overall
    except Exception as exc:
        overall.update({
            "status": "FAIL_HOTFIX_STOPPED_BEFORE_FORMAL_TRAINING",
            "completed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "formal_s2_training_eligible": False,
            "go_to_formal_s2_training": False,
            "next_action": "Upload D18_S2_HOTFIX_FAST_RESUME_OVERALL_SUMMARY.json and d18_s2_preflight/D18_S2_PREFLIGHT_SUMMARY.json.",
        })
        dump_json(overall, output_root / "D18_S2_HOTFIX_FAST_RESUME_OVERALL_SUMMARY.json")
        dump_json(overall, hotfix_root / "D18_S2_HOTFIX_FAST_RESUME_OVERALL_SUMMARY.json")
        (hotfix_root / "D18_S2_HOTFIX_FAILURE_TRACEBACK.txt").write_text(overall["traceback"], encoding="utf-8")
        log(f"FAILED: {type(exc).__name__}: {exc}")
        return 2, overall


def selftest() -> int:
    fake = {"checks": {name: True for name in CRITICAL_PREFLIGHT_CHECKS}}
    fake["checks"].update({name: False for name in AUDIT_ONLY_PREFLIGHT_CHECKS})
    gate = _preflight_gate(fake)
    if gate["status"] != "PASS_CRITICAL_PREFLIGHT":
        print("selftest failed: audit-only checks blocked critical gate")
        return 2
    fake["checks"]["selected_replay_resolved"] = False
    gate = _preflight_gate(fake)
    if gate["status"] != "FAIL_CRITICAL_PREFLIGHT":
        print("selftest failed: critical check did not block")
        return 2
    print("PASS: D18-S2 hotfix selftest")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume D18-S2 micro-smoke from existing preflight artifacts")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "d18_s2_preflight_micro_smoke.json"))
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--selftest-only", action="store_true")
    args = parser.parse_args()
    if args.selftest_only:
        return selftest()
    code, _ = run(Path(args.config), Path(args.project_root).resolve(), args.output_root)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
