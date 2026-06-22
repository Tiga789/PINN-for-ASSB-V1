from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .common import ConfigError, dump_json, ensure_dir, load_json, utc_now_iso, write_csv
from .data import inspect_profile_cycles
from .uid import (
    audit_record_identity,
    build_replay_index,
    canonical_from_record,
    load_role_index,
    load_split_index,
    resolve_replay_path,
    select_exact_records,
)


def _git_info(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> tuple[int, str, str]:
        proc = subprocess.run(
            ["git", *args], cwd=project_root, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    if not (project_root / ".git").exists():
        return {"available": False, "reason": "no .git directory"}
    rc, head, err = run("rev-parse", "HEAD")
    rc2, status, err2 = run("status", "--short")
    return {
        "available": rc == 0,
        "head": head if rc == 0 else "",
        "status_short": status if rc2 == 0 else "",
        "dirty": bool(status) if rc2 == 0 else None,
        "error": err or err2,
    }


def _prior_status(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    overall = load_json(paths["prior_s0_s1_overall_summary"])
    s0 = load_json(paths["prior_s0_validation"])
    s1 = load_json(paths["prior_s1_summary"])
    coverage = load_json(paths["prior_s1_coverage"])
    checks = {
        "prior_overall_review_ready": overall.get("status") == "PASS_READY_FOR_HUMAN_ARCHITECTURE_REVIEW",
        "prior_s0_pass": s0.get("status") == "PASS",
        "prior_s1_diagnostic_pass": s1.get("status") == "PASS_VALID_DIAGNOSTIC_COVERAGE",
        "prior_s1_coverage_pass": coverage.get("status") == "PASS",
        "prior_frozen_test_unused": not bool(s1.get("frozen_test_used", False)),
        "human_review_token_exact": config.get("human_review_token")
        == "D18_S1_REVIEW_ACCEPTED_FOR_S2_PREFLIGHT_ONLY_20260618",
    }
    return {
        "checks": checks,
        "overall_status": overall.get("status"),
        "s0_status": s0.get("status"),
        "s1_status": s1.get("status"),
        "coverage_status": coverage.get("status"),
    }


def _selected_manifest_rows(records: Sequence[Mapping[str, Any]], replay_resolution: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        uid = canonical_from_record(record)
        resolved = replay_resolution[uid.canonical]
        rows.append(
            {
                "canonical_cell_uid": uid.canonical,
                "cell_uid": uid.cell_uid,
                "role": record.get("d18_s2_role", ""),
                "split": record.get("split", ""),
                "protocol": uid.protocol,
                "branch_family": uid.branch_family,
                "softlabel_npz": record.get("softlabel_npz", ""),
                "replay_npz_declared": record.get("replay_npz", ""),
                "replay_npz_resolved": resolved.get("path", ""),
                "replay_resolution": resolved.get("method", ""),
                "is_flagged_probe": bool(record.get("is_flagged_probe", False)),
                "g2_stage": record.get("g2_stage", ""),
            }
        )
    return rows


def run_preflight(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
    output_dir: str | Path,
    progress: callable | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = ensure_dir(output_dir)
    paths = config["paths"]
    pre_cfg = config["preflight"]
    prior = _prior_status(config)
    if progress:
        progress("Loading locked D17 split and G2 role manifests")
    split_manifest, split_index = load_split_index(paths["d17_split_manifest"])
    role_index = load_role_index(paths["d17_g2_prediction_manifest"])
    requested = {
        "fit_train": pre_cfg["fit_train_uids"],
        "internal_heldout": pre_cfg["internal_heldout_uids"],
        "validation_report_only": pre_cfg["validation_report_only_uids"],
    }
    records = select_exact_records(
        split_index=split_index, role_index=role_index, requested=requested
    )

    replay_roots = list(split_manifest.get("replay_roots", []))
    replay_roots.extend(pre_cfg.get("additional_replay_roots", []))
    if progress:
        progress(f"Building exact replay index across {len(replay_roots)} configured roots")
    replay_index = build_replay_index(replay_roots)
    replay_resolution: dict[str, dict[str, Any]] = {}
    all_replay_resolution: dict[str, dict[str, Any]] = {}
    identity_rows: list[dict[str, Any]] = []
    canonical_view_records: list[dict[str, Any]] = []
    selected_keys = {canonical_from_record(r).canonical.lower() for r in records}

    # Build a corrected canonical view for all 55 split records. Only selected-record
    # identity errors are hard blockers; global warnings remain auditable.
    for record in split_index.values():
        uid = canonical_from_record(record)
        replay_path, method = resolve_replay_path(record, replay_index)
        audit = audit_record_identity(record, replay_path)
        audit["selected_for_s2_micro"] = uid.canonical.lower() in selected_keys
        audit["replay_resolution"] = method
        identity_rows.append(audit)
        corrected = dict(record)
        corrected["replay_npz_original"] = record.get("replay_npz", "")
        corrected["replay_npz"] = str(replay_path) if replay_path else ""
        corrected["replay_resolution"] = method
        corrected["identity_status"] = audit["identity_status"]
        canonical_view_records.append(corrected)
        all_replay_resolution[uid.canonical.lower()] = {
            "path": str(replay_path) if replay_path else "",
            "method": method,
            "identity_status": audit["identity_status"],
        }
        if uid.canonical.lower() in selected_keys:
            replay_resolution[uid.canonical] = {"path": str(replay_path) if replay_path else "", "method": method}

    if progress:
        progress("Checking per-cycle source generator-grid coverage")
    cycle_rows: list[dict[str, Any]] = []
    selected_cycles: dict[str, list[int]] = {}
    for record in records:
        uid = canonical_from_record(record)
        resolution = replay_resolution[uid.canonical]
        rows, cycles = inspect_profile_cycles(
            record,
            min_source_points_per_cycle=int(pre_cfg["min_source_points_per_cycle"]),
            cycles_per_position=int(pre_cfg["cycles_per_position"]),
            micro_points_per_cycle=int(pre_cfg["micro_points_per_cycle"]),
            replay_path=resolution.get("path") or None,
        )
        cycle_rows.extend(rows)
        selected_cycles[uid.canonical] = [int(x) for x in cycles]

    selected_identity = [r for r in identity_rows if r["selected_for_s2_micro"]]
    selected_manifest = _selected_manifest_rows(records, replay_resolution)
    train_rows = [r for r in selected_manifest if r["role"] == "fit_train"]
    report_rows = [r for r in selected_manifest if r["role"] != "fit_train"]
    role_uid_sets = {
        role: {r["canonical_cell_uid"] for r in selected_manifest if r["role"] == role}
        for role in ("fit_train", "internal_heldout", "validation_report_only")
    }
    overlap = (
        role_uid_sets["fit_train"] & role_uid_sets["internal_heldout"]
        | role_uid_sets["fit_train"] & role_uid_sets["validation_report_only"]
        | role_uid_sets["internal_heldout"] & role_uid_sets["validation_report_only"]
    )
    protocols = {r["protocol"] for r in train_rows}
    branches = {r["branch_family"] for r in train_rows}
    selected_replay_missing = [r["canonical_cell_uid"] for r in selected_manifest if not r["replay_npz_resolved"]]
    selected_identity_fail = [r["canonical_cell_uid"] for r in selected_identity if r["identity_status"] != "PASS"]
    source_min = min(int(r["source_cycle_points"]) for r in cycle_rows) if cycle_rows else 0
    micro_all_downsampled_or_equal = all(
        int(r["micro_smoke_exported_points"]) <= int(r["source_cycle_points"]) for r in cycle_rows
    )

    battery1_key = "batch-2_3c_battery-1"
    battery10_key = "batch-2_3c_battery-10"
    battery1 = split_index.get(battery1_key)
    battery10 = split_index.get(battery10_key)
    battery1_resolved = all_replay_resolution.get(battery1_key, {})
    battery10_resolved = all_replay_resolution.get(battery10_key, {})
    collision_audit = {
        "battery1_record_found": battery1 is not None,
        "battery10_record_found": battery10 is not None,
        "records_are_distinct_objects": battery1 is not battery10,
        "softlabel_paths_are_distinct": bool(
            battery1 and battery10 and battery1.get("softlabel_npz") != battery10.get("softlabel_npz")
        ),
        "resolved_replay_paths_present": bool(
            battery1_resolved.get("path") and battery10_resolved.get("path")
        ),
        "resolved_replay_paths_are_distinct": bool(
            battery1_resolved.get("path")
            and battery10_resolved.get("path")
            and battery1_resolved.get("path") != battery10_resolved.get("path")
        ),
        "resolved_identity_battery1_pass": battery1_resolved.get("identity_status") == "PASS",
        "resolved_identity_battery10_pass": battery10_resolved.get("identity_status") == "PASS",
        "exact_lookup_keys_distinct": battery1_key != battery10_key,
    }
    collision_details = {
        "battery1_resolved_path": battery1_resolved.get("path", ""),
        "battery1_resolution_method": battery1_resolved.get("method", ""),
        "battery10_resolved_path": battery10_resolved.get("path", ""),
        "battery10_resolution_method": battery10_resolved.get("method", ""),
    }

    try:
        disk = shutil.disk_usage(out)
    except OSError:
        disk_target = Path(paths["output_root"]).anchor or out.anchor or "."
        disk = shutil.disk_usage(disk_target)
    git = _git_info(root)
    expected_commit = str(config.get("expected_github_commit", "")).strip()
    checks = {
        **prior["checks"],
        "selected_profile_count_exact": len(records) == (
            len(pre_cfg["fit_train_uids"])
            + len(pre_cfg["internal_heldout_uids"])
            + len(pre_cfg["validation_report_only_uids"])
        ),
        "selected_identity_exact": not selected_identity_fail,
        "selected_replay_resolved": not selected_replay_missing,
        "no_role_overlap": not overlap,
        "no_frozen_test_selected": all(r["split"] not in {"frozen_test", "test"} for r in selected_manifest),
        "no_flagged_probe_selected": all(not bool(r["is_flagged_probe"]) for r in selected_manifest),
        "fit_train_all_six_protocols": protocols == {"2C", "3C", "R2.5", "R3", "random_walk", "GEO"},
        "fit_train_both_branches": branches == {"RG", "P4D"},
        "per_cycle_source_minimum": source_min >= int(pre_cfg["min_source_points_per_cycle"]),
        "preflight_source_counts_not_downsampled": all(not bool(r["preflight_downsampled"]) for r in cycle_rows),
        "micro_view_never_exceeds_source": micro_all_downsampled_or_equal,
        "battery1_battery10_exact_collision_guard": all(collision_audit.values()),
        "torch_import_ok": True,
        "amp_forbidden_by_config": bool(config["micro_smoke"]["trainer"].get("disable_amp", True)),
        "torch_compile_forbidden_by_config": bool(config["micro_smoke"]["trainer"].get("disable_torch_compile", True)),
        "formal_s2_training_disabled": not bool(config.get("formal_s2_training_enabled", False)),
        "git_head_matches_expected_commit": bool(
            git.get("available") and expected_commit and str(git.get("head", "")).startswith(expected_commit)
        ),
        "disk_free_above_minimum": disk.free >= int(pre_cfg.get("minimum_free_disk_bytes", 1_000_000_000)),
    }
    hard_pass = all(checks.values())
    status = "PASS_PREFLIGHT_FOR_MICRO_SMOKE" if hard_pass else "FAIL_PREFLIGHT"

    write_csv(identity_rows, out / "D18_S2_EXACT_UID_AUDIT.csv")
    write_csv(selected_manifest, out / "D18_S2_SELECTED_PROFILE_MANIFEST.csv")
    write_csv(cycle_rows, out / "D18_S2_PER_CYCLE_SOURCE_COVERAGE.csv")
    dump_json(
        {
            "protocol": "D18-S2-CANONICAL-SPLIT-VIEW",
            "source_manifest": paths["d17_split_manifest"],
            "records": canonical_view_records,
        },
        out / "D18_S2_CANONICAL_SPLIT_VIEW.json",
    )
    decision_text = f"""# D18-S2 preflight decision\n\n- Status: **{status}**\n- Micro-smoke allowed: **{str(hard_pass)}**\n- Formal S2 training allowed: **False**\n- Selected fit profiles: **{len(train_rows)}**\n- Selected report-only profiles: **{len(report_rows)}**\n- Source cycle minimum points: **{source_min}**\n\nThe source generator-grid counts are inspected without downsampling. The micro-smoke view is explicitly capped at {pre_cfg['micro_points_per_cycle']} points per cycle and cannot be used as a full-cycle accuracy claim. Validation profiles remain report-only; frozen test and the flagged probe are not loaded.\n"""
    (out / "D18_S2_PREFLIGHT_DECISION.md").write_text(decision_text, encoding="utf-8")

    summary = {
        "stage": "D18-S2-PREFLIGHT",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "checks": checks,
        "prior_review": prior,
        "selected_profile_count": len(records),
        "selected_profile_roles": {role: len(values) for role, values in role_uid_sets.items()},
        "fit_train_protocols": sorted(protocols),
        "fit_train_branches": sorted(branches),
        "selected_identity_failures": selected_identity_fail,
        "selected_replay_missing": selected_replay_missing,
        "role_overlap": sorted(overlap),
        "selected_cycles": selected_cycles,
        "cycle_audit_row_count": len(cycle_rows),
        "source_cycle_points_min": source_min,
        "source_cycle_points_max": max(int(r["source_cycle_points"]) for r in cycle_rows) if cycle_rows else 0,
        "micro_points_per_cycle": int(pre_cfg["micro_points_per_cycle"]),
        "source_grid_and_micro_view_are_separately_labeled": True,
        "battery1_battery10_collision_audit": collision_audit,
        "battery1_battery10_collision_details": collision_details,
        "git": git,
        "expected_github_commit": expected_commit,
        "github_commit_exact": bool(expected_commit and str(git.get("head", "")).startswith(expected_commit)),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "disk_free_bytes": disk.free,
        "micro_smoke_allowed": hard_pass,
        "formal_s2_training_allowed": False,
        "frozen_test_used": False,
        "flagged_probe_used": False,
        "next_action": "Run only the bounded micro-smoke after this preflight passes; do not start formal S2 training.",
    }
    dump_json(summary, out / "D18_S2_PREFLIGHT_SUMMARY.json")
    return {
        "summary": summary,
        "records": records,
        "replay_paths": {uid: data["path"] or None for uid, data in replay_resolution.items()},
    }
