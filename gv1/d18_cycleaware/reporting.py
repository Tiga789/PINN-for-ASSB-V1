from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import dump_json, utc_now_iso, write_csv
from .cycle_features import cumulative_ah, cycle_segments
from .diagnostics import CaseDiagnostic, build_recommendation
from .metrics import volume_mean
from .schema import ArrayCase, RADIAL_STATES, normalize_step_labels


BLOCKED_SPLITS = {"frozen_test", "test", "flagged_probe", "flagged", "probe", "unknown"}


def _flatten(items: Sequence[CaseDiagnostic], attr: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        out.extend(getattr(item, attr))
    return out


def _norm_protocol(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    aliases = {
        "r2_5": "R2.5",
        "r2.5": "R2.5",
        "r3": "R3",
        "2c": "2C",
        "3c": "3C",
        "randomwalk": "random_walk",
        "random_walk": "random_walk",
        "geo": "GEO",
    }
    compact = text.replace("-", "_")
    return aliases.get(compact, str(value))


def _split_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"val": "validation", "valid": "validation", "heldout": "internal_heldout"}
    return aliases.get(text, text)


def _branch_family(case: ArrayCase) -> str:
    explicit = str(case.metadata.get("branch_family", "")).strip().upper()
    if explicit in {"RG", "P4D"}:
        return explicit
    text = f"{case.branch} {case.protocol}".upper()
    return "P4D" if any(x in text for x in ("P4D", "CURRENT_INTEGRAL", "GEO", "RANDOM_WALK")) else "RG"


def _coverage_audit(
    cases: Sequence[ArrayCase],
    boundary_rows: Sequence[Mapping[str, Any]],
    s1_config: Mapping[str, Any],
) -> dict[str, Any]:
    required_splits = {str(x) for x in s1_config.get("coverage_required_splits", ["train", "internal_heldout", "validation"])}
    required_protocols = {_norm_protocol(x) for x in s1_config.get("coverage_required_protocols", ["2C", "3C", "R2.5", "R3", "random_walk", "GEO"])}
    required_branches = {str(x).upper() for x in s1_config.get("coverage_required_branch_families", ["RG", "P4D"])}
    required_positions = {str(x) for x in s1_config.get("coverage_required_cycle_positions", ["early", "middle", "late"])}
    required_phases = {str(x) for x in s1_config.get("coverage_required_phases", ["charge", "rest", "discharge"])}
    required_case_ids = {str(x) for x in s1_config.get("coverage_required_case_ids", [])}
    minimum_cases = int(s1_config.get("coverage_min_cases", max(1, len(required_case_ids))))
    minimum_cycles_per_case = int(s1_config.get("coverage_min_cycles_per_case", 3))
    dense_min_points = int(s1_config.get("dense_min_time_points", 768))

    splits = {_split_name(c.split) for c in cases}
    protocols = {_norm_protocol(c.protocol) for c in cases}
    branches = {_branch_family(c) for c in cases}
    case_ids = {c.case_id for c in cases}
    blocked_cases = [c.case_id for c in cases if _split_name(c.split) in BLOCKED_SPLITS]
    per_case: list[dict[str, Any]] = []
    positions: set[str] = set()
    phases: set[str] = set()
    missing_current: list[str] = []
    zero_ah: list[str] = []
    too_few_cycles: list[str] = []
    not_dense: list[str] = []
    no_casepack_marker: list[str] = []

    for case in cases:
        segments = cycle_segments(case.cycle_id, case.n_time)
        positions.update(seg.position for seg in segments)
        labels = normalize_step_labels(case.step_type, case.current_A, case.n_time)
        phases.update(str(x) for x in np.unique(labels))
        current_ok = case.current_A is not None and np.any(np.isfinite(case.current_A)) and float(np.nanmax(np.abs(case.current_A))) > 1e-10
        abs_ah = cumulative_ah(case.time_s, case.current_A, absolute=True)
        abs_ah_end = float(abs_ah[-1]) if abs_ah.size else 0.0
        if not current_ok:
            missing_current.append(case.case_id)
        if not np.isfinite(abs_ah_end) or abs_ah_end <= 0:
            zero_ah.append(case.case_id)
        if len(segments) < minimum_cycles_per_case:
            too_few_cycles.append(case.case_id)
        if case.n_time < dense_min_points:
            not_dense.append(case.case_id)
        marker = str(case.metadata.get("casepack_version", ""))
        if not marker.startswith("D18-S1-DENSE-CASEPACK-FIX-v2"):
            no_casepack_marker.append(case.case_id)
        per_case.append(
            {
                "case_id": case.case_id,
                "canonical_cell_uid": case.canonical_cell_uid,
                "split": _split_name(case.split),
                "protocol": _norm_protocol(case.protocol),
                "branch_family": _branch_family(case),
                "case_role": str(case.metadata.get("case_role", "")),
                "n_time": case.n_time,
                "cycle_count": len(segments),
                "cycle_positions": sorted({seg.position for seg in segments}),
                "phases": sorted(set(str(x) for x in np.unique(labels))),
                "current_signal_present": current_ok,
                "cumulative_abs_Ah_end": abs_ah_end,
                "casepack_marker": marker,
            }
        )

    checks = {
        "minimum_case_count": len(cases) >= minimum_cases,
        "required_splits_present": required_splits.issubset(splits),
        "required_protocols_present": required_protocols.issubset(protocols),
        "required_branch_families_present": required_branches.issubset(branches),
        "required_cycle_positions_present": required_positions.issubset(positions),
        "required_phases_present": required_phases.issubset(phases),
        "required_case_ids_present": required_case_ids.issubset(case_ids),
        "no_blocked_or_unknown_split": not blocked_cases,
        "all_cases_have_minimum_cycles": not too_few_cycles,
        "all_cases_are_dense": not not_dense,
        "cycle_boundary_rows_present": len(boundary_rows) > 0,
        "current_signal_present_all_cases": not missing_current,
        "cumulative_Ah_nonzero_all_cases": not zero_ah,
        "all_cases_from_explicit_fixed_casepack": not no_casepack_marker,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "required": {
            "minimum_case_count": minimum_cases,
            "splits": sorted(required_splits),
            "protocols": sorted(required_protocols),
            "branch_families": sorted(required_branches),
            "cycle_positions": sorted(required_positions),
            "phases": sorted(required_phases),
            "case_ids": sorted(required_case_ids),
            "minimum_cycles_per_case": minimum_cycles_per_case,
            "dense_min_time_points": dense_min_points,
        },
        "observed": {
            "case_count": len(cases),
            "splits": sorted(splits),
            "protocols": sorted(protocols),
            "branch_families": sorted(branches),
            "cycle_positions": sorted(positions),
            "phases": sorted(phases),
            "case_ids": sorted(case_ids),
            "boundary_row_count": len(boundary_rows),
        },
        "violations": {
            "blocked_cases": blocked_cases,
            "missing_current_cases": missing_current,
            "zero_cumulative_Ah_cases": zero_ah,
            "too_few_cycle_cases": too_few_cycles,
            "not_dense_cases": not_dense,
            "missing_casepack_marker_cases": no_casepack_marker,
            "missing_required_case_ids": sorted(required_case_ids - case_ids),
            "missing_splits": sorted(required_splits - splits),
            "missing_protocols": sorted(required_protocols - protocols),
            "missing_branch_families": sorted(required_branches - branches),
            "missing_cycle_positions": sorted(required_positions - positions),
            "missing_phases": sorted(required_phases - phases),
        },
        "case_rows": per_case,
    }


def _write_rows(rows: Sequence[Mapping[str, Any]], path: Path, fallback_fields: Sequence[str]) -> None:
    write_csv(list(rows), path, fieldnames=None if rows else list(fallback_fields))


def write_s1_reports(
    *,
    output_dir: str | Path,
    cases: Sequence[ArrayCase],
    diagnostics: Sequence[CaseDiagnostic],
    inventory_rows: Sequence[Mapping[str, Any]],
    warnings: Sequence[Mapping[str, Any]],
    s1_config: Mapping[str, Any],
    make_plots: bool = True,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_rows = _flatten(diagnostics, "state_rows")
    phase_rows = _flatten(diagnostics, "phase_rows")
    cycle_rows = _flatten(diagnostics, "cycle_rows")
    boundary_rows = _flatten(diagnostics, "boundary_rows")
    rank_rows = _flatten(diagnostics, "rank_rows")
    radial_rows = _flatten(diagnostics, "radial_rows")
    cycle_summary_rows = _flatten(diagnostics, "cycle_summary_rows")
    consistency_rows = _flatten(diagnostics, "consistency_rows")
    recommendation = build_recommendation(state_rows, boundary_rows, consistency_rows, s1_config)
    coverage = _coverage_audit(cases, boundary_rows, s1_config)

    _write_rows(inventory_rows, out / "d18_s1_case_inventory.csv", ["prediction_path", "status", "case_id", "canonical_cell_uid", "split", "protocol", "branch", "n_time"])
    _write_rows(state_rows, out / "d18_s1_state_metrics.csv", ["case_id", "canonical_cell_uid", "split", "protocol", "branch", "state", "r2", "mae", "rmse"])
    _write_rows(phase_rows, out / "d18_s1_phase_metrics.csv", ["case_id", "canonical_cell_uid", "state", "group_dimension", "group_value", "r2", "mae"])
    _write_rows(cycle_rows, out / "d18_s1_error_components_by_cycle.csv", ["case_id", "state", "cycle_id", "cycle_position", "n_points", "bias", "mae", "rmse"])
    _write_rows(boundary_rows, out / "d18_s1_cycle_boundary_audit.csv", ["case_id", "state", "left_cycle_id", "right_cycle_id", "boundary_index", "true_jump", "pred_jump", "jump_error", "normalized_abs_jump_error"])
    _write_rows(rank_rows, out / "d18_s1_residual_rank.csv", ["case_id", "state", "rank_at_90", "rank_at_95", "rank_at_99"])
    _write_rows(radial_rows, out / "d18_s1_radial_components.csv", ["case_id", "canonical_cell_uid", "state", "component", "r2", "mae", "rmse"])
    _write_rows(cycle_summary_rows, out / "d18_s1_cycle_features.csv", ["case_id", "cycle_id", "cycle_position", "n_points", "duration_s", "q_signed_Ah", "q_abs_Ah", "cumulative_abs_Ah_end", "efc_proxy_end"])
    _write_rows(consistency_rows, out / "d18_s1_theta_cs_consistency.csv", ["case_id", "canonical_cell_uid", "electrode", "true_relation_rmse", "pred_relation_rmse"])
    _write_rows(warnings, out / "d18_s1_warnings.csv", ["prediction_path", "type", "message"])
    _write_rows(coverage["case_rows"], out / "d18_s1_coverage_by_case.csv", ["case_id", "canonical_cell_uid", "split", "protocol", "branch_family", "n_time", "cycle_count"])
    dump_json(coverage, out / "d18_s1_coverage_audit.json")

    if coverage["status"] != "PASS":
        status = "REVIEW_INVALID_DIAGNOSTIC_COVERAGE"
    elif int(recommendation.get("failed_state_count", 0)) <= 0:
        status = "REVIEW_NO_STRUCTURAL_FAILURE_DETECTED"
    else:
        status = "PASS_VALID_DIAGNOSTIC_COVERAGE"

    summary = {
        "stage": "D18-S1-FIX",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "training_launched": False,
        "go_to_s2": False,
        "case_count": len(cases),
        "state_metric_row_count": len(state_rows),
        "phase_metric_row_count": len(phase_rows),
        "cycle_metric_row_count": len(cycle_rows),
        "boundary_row_count": len(boundary_rows),
        "rank_row_count": len(rank_rows),
        "radial_row_count": len(radial_rows),
        "warning_count": len(warnings),
        "case_summaries": [item.summary for item in diagnostics],
        "coverage_audit": coverage,
        "recommendation": recommendation,
        "frozen_test_used": any(_split_name(c.split) in BLOCKED_SPLITS for c in cases),
        "goal": "55_cells_all_cycles_high_accuracy_full_cycle_surrogate",
        "next_action": "Human review of S0/S1 fix outputs; D18-S2 remains blocked in this package.",
    }
    dump_json(summary, out / "d18_s1_array_latent_summary.json")
    _write_recommendation_md(summary, out / "d18_s1_recommendation.md")
    if make_plots and cases:
        _write_plots(cases, out / "plots", max_plot_points=int(s1_config.get("max_plot_points", 3000)))
    return summary


def _write_recommendation_md(summary: Mapping[str, Any], path: Path) -> None:
    rec = summary.get("recommendation", {})
    coverage = summary.get("coverage_audit", {})
    labels = rec.get("labels", [])
    lines = [
        "# D18-S1-FIX Array-level Diagnostic Recommendation",
        "",
        f"- Status: **{summary.get('status')}**",
        f"- Coverage: **{coverage.get('status')}**",
        f"- Selected cases: **{summary.get('case_count')}**",
        f"- Training launched: **{summary.get('training_launched')}**",
        f"- Frozen/test/flagged profile used: **{summary.get('frozen_test_used')}**",
        f"- Go to S2: **{summary.get('go_to_s2')}**",
        "",
        "## Architecture decision labels",
        "",
    ]
    lines.extend([f"- `{label}`" for label in labels] or ["- No label generated."])
    lines += [
        "",
        "## Coverage checks",
        "",
    ]
    for key, value in coverage.get("checks", {}).items():
        lines.append(f"- `{key}`: **{value}**")
    lines += [
        "",
        "## Decision",
        "",
        str(rec.get("reason", "No recommendation available.")),
        "",
        "This fix package never launches D18-S2. Promotion requires explicit human review of the real dense-case arrays and coverage audit.",
        "",
        "## Final target",
        "",
        "The final target remains 55 cells × all cycles, but the next model must pass sampled-grid, dense selected-cycle, and streaming full-cycle metrics before that claim is made.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(cases: Sequence[ArrayCase], plot_dir: Path, max_plot_points: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        n = case.n_time
        if n <= 0:
            continue
        step = max(1, int(np.ceil(n / max(1, max_plot_points))))
        idx = np.arange(0, n, step)
        time = case.time_s[idx]
        for state in case.available_states:
            true = case.true[state]
            pred = case.pred[state]
            if state in RADIAL_STATES:
                grid = case.radial_grid_a if state.endswith("_a") else case.radial_grid_c
                true_line = volume_mean(true, grid)[idx]
                pred_line = volume_mean(pred, grid)[idx]
                ylabel = f"{state} volume mean"
            else:
                true_line = true[idx, 0]
                pred_line = pred[idx, 0]
                ylabel = state
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(time, true_line, label="true", linewidth=1.2)
            ax.plot(time, pred_line, label="pred", linewidth=1.0)
            ax.set_xlabel("time (s)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{case.canonical_cell_uid} | {state}")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{case.case_id}__{state}.png", dpi=130)
            plt.close(fig)

        cycle = case.cycle_id
        if cycle is None:
            continue
        cycle_numeric = np.asarray(cycle).astype(float)
        rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for state in case.available_states:
            true = case.true[state]
            pred = case.pred[state]
            for c in np.unique(cycle_numeric[np.isfinite(cycle_numeric)]):
                mask = cycle_numeric == c
                if np.count_nonzero(mask) < 2:
                    continue
                rows[state].append((float(c), float(np.nanmean(np.abs(pred[mask] - true[mask])))))
        if rows:
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for state, values in rows.items():
                values = sorted(values)
                ax.plot([v[0] for v in values], [v[1] for v in values], label=state, linewidth=1.0)
            ax.set_xlabel("cycle id")
            ax.set_ylabel("mean absolute error")
            ax.set_title(f"{case.canonical_cell_uid} | cycle-wise error")
            ax.legend(ncol=3)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{case.case_id}__cycle_error.png", dpi=130)
            plt.close(fig)
