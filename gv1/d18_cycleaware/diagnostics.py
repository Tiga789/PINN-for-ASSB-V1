from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .cycle_features import (
    assign_cycle_position,
    boundary_rows,
    cumulative_ah,
    cycle_bias_rows,
    cycle_bias_trend,
    cycle_summary_rows,
    sanitize_cycle_id,
)
from .metrics import (
    best_integer_lag,
    constant_shift_correct,
    fit_affine,
    radial_deviation,
    regression_metrics,
    residual_svd_summary,
    safe_corr,
    volume_mean,
)
from .schema import ArrayCase, POTENTIAL_STATES, RADIAL_STATES, normalize_step_labels


@dataclass
class CaseDiagnostic:
    summary: dict[str, Any]
    state_rows: list[dict[str, Any]]
    phase_rows: list[dict[str, Any]]
    cycle_rows: list[dict[str, Any]]
    boundary_rows: list[dict[str, Any]]
    rank_rows: list[dict[str, Any]]
    radial_rows: list[dict[str, Any]]
    cycle_summary_rows: list[dict[str, Any]]
    consistency_rows: list[dict[str, Any]]


def _state_grid(case: ArrayCase, state: str) -> np.ndarray | None:
    if state.endswith("_a"):
        return case.radial_grid_a
    if state.endswith("_c"):
        return case.radial_grid_c
    return None


def _metrics_row(case: ArrayCase, state: str, true: np.ndarray, pred: np.ndarray, s1: Mapping[str, Any]) -> dict[str, Any]:
    base = regression_metrics(true, pred)
    shift, shifted = constant_shift_correct(true, pred)
    shifted_metrics = regression_metrics(true, shifted)
    slope, intercept, affine = fit_affine(true, pred)
    affine_metrics = regression_metrics(true, affine)
    lag_input_true = volume_mean(true, _state_grid(case, state)) if state in RADIAL_STATES else true[:, 0]
    lag_input_pred = volume_mean(pred, _state_grid(case, state)) if state in RADIAL_STATES else pred[:, 0]
    lag = best_integer_lag(lag_input_true, lag_input_pred, max_lag=int(s1.get("max_lag_samples", 64)))
    err = pred - true
    svd = residual_svd_summary(err)
    cycle_rows_state = cycle_bias_rows(case.cycle_id, case.n_time, true, pred, state, case.case_id)
    trend = cycle_bias_trend(cycle_rows_state)
    return {
        "case_id": case.case_id,
        "canonical_cell_uid": case.canonical_cell_uid,
        "split": case.split,
        "protocol": case.protocol,
        "branch": case.branch,
        "state": state,
        **base.as_dict(),
        "constant_shift": shift,
        "shifted_r2": shifted_metrics.r2,
        "shifted_mae": shifted_metrics.mae,
        "affine_slope": slope,
        "affine_intercept": intercept,
        "affine_r2": affine_metrics.r2,
        "affine_mae": affine_metrics.mae,
        **lag,
        **trend,
        "residual_rank_at_90": svd["rank_at_90"],
        "residual_rank_at_95": svd["rank_at_95"],
        "residual_rank_at_99": svd["rank_at_99"],
        "residual_energy_rank1": svd["energy_rank1"],
        "residual_energy_rank2": svd["energy_rank2"],
        "residual_energy_rank4": svd["energy_rank4"],
        "residual_energy_rank8": svd["energy_rank8"],
        "nonfinite_true_fraction": float(np.mean(~np.isfinite(true))),
        "nonfinite_pred_fraction": float(np.mean(~np.isfinite(pred))),
    }


def _phase_metrics(case: ArrayCase, state: str, true: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    labels = normalize_step_labels(case.step_type, case.current_A, case.n_time)
    positions = assign_cycle_position(case.cycle_id, case.n_time)
    rows: list[dict[str, Any]] = []
    for dimension, values in (("step", labels), ("cycle_position", positions)):
        for label in sorted({str(v) for v in values}):
            mask = values == label
            if np.count_nonzero(mask) < 4:
                continue
            metrics = regression_metrics(true[mask], pred[mask])
            rows.append(
                {
                    "case_id": case.case_id,
                    "canonical_cell_uid": case.canonical_cell_uid,
                    "state": state,
                    "group_dimension": dimension,
                    "group_value": label,
                    **metrics.as_dict(),
                }
            )
    return rows


def _radial_diagnostics(case: ArrayCase, state: str, true: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    if state not in RADIAL_STATES or true.ndim != 2 or true.shape[1] < 2:
        return []
    grid = _state_grid(case, state)
    cbar_true = volume_mean(true, grid)
    cbar_pred = volume_mean(pred, grid)
    delta_true = radial_deviation(true, grid)
    delta_pred = radial_deviation(pred, grid)
    sc_true = true[:, -1] - true[:, 0]
    sc_pred = pred[:, -1] - pred[:, 0]
    grad_true = np.nanmax(true, axis=1) - np.nanmin(true, axis=1)
    grad_pred = np.nanmax(pred, axis=1) - np.nanmin(pred, axis=1)
    active = np.abs(sc_true) > max(1e-12, 0.01 * np.nanpercentile(np.abs(sc_true), 95))
    direction_acc = float(np.mean(np.sign(sc_true[active]) == np.sign(sc_pred[active]))) if np.any(active) else float("nan")
    rows: list[dict[str, Any]] = []
    for component, yt, yp in (
        ("volume_mean", cbar_true, cbar_pred),
        ("radial_deviation", delta_true, delta_pred),
        ("surface_minus_center", sc_true, sc_pred),
        ("radial_amplitude", grad_true, grad_pred),
    ):
        metrics = regression_metrics(yt, yp)
        row = {
            "case_id": case.case_id,
            "canonical_cell_uid": case.canonical_cell_uid,
            "state": state,
            "component": component,
            **metrics.as_dict(),
        }
        if component == "surface_minus_center":
            row["direction_accuracy"] = direction_acc
        rows.append(row)
    zero_mean_true = volume_mean(delta_true, grid)
    zero_mean_pred = volume_mean(delta_pred, grid)
    rows.append(
        {
            "case_id": case.case_id,
            "canonical_cell_uid": case.canonical_cell_uid,
            "state": state,
            "component": "zero_volume_mean_audit",
            "true_abs_mean": float(np.nanmean(np.abs(zero_mean_true))),
            "pred_abs_mean": float(np.nanmean(np.abs(zero_mean_pred))),
            "direction_accuracy": direction_acc,
        }
    )
    return rows


def _rank_row(case: ArrayCase, state: str, true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    summary = residual_svd_summary(pred - true)
    return {
        "case_id": case.case_id,
        "canonical_cell_uid": case.canonical_cell_uid,
        "state": state,
        **summary,
    }


def _theta_cs_consistency(case: ArrayCase, electrode: str) -> dict[str, Any] | None:
    theta_key = f"theta_{electrode}"
    cs_key = f"cs_{electrode}"
    if theta_key not in case.true or cs_key not in case.true or theta_key not in case.pred or cs_key not in case.pred:
        return None

    def fit_relation(cs: np.ndarray, theta: np.ndarray) -> tuple[float, float, float]:
        x = np.asarray(cs, dtype=np.float64).reshape(-1)
        y = np.asarray(theta, dtype=np.float64).reshape(-1)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if x.size < 2 or np.std(x) < 1e-12:
            return float("nan"), float("nan"), float("nan")
        coeff, *_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)
        yhat = coeff[0] * x + coeff[1]
        return float(coeff[0]), float(coeff[1]), regression_metrics(y, yhat).rmse

    ts, ti, trmse = fit_relation(case.true[cs_key], case.true[theta_key])
    ps, pi, prmse = fit_relation(case.pred[cs_key], case.pred[theta_key])
    return {
        "case_id": case.case_id,
        "canonical_cell_uid": case.canonical_cell_uid,
        "electrode": electrode,
        "true_theta_per_cs_slope": ts,
        "true_intercept": ti,
        "true_relation_rmse": trmse,
        "pred_theta_per_cs_slope": ps,
        "pred_intercept": pi,
        "pred_relation_rmse": prmse,
        "slope_relative_error": abs(ps - ts) / max(abs(ts), 1e-12) if np.isfinite(ts) and np.isfinite(ps) else float("nan"),
    }


def _potential_common_mode(case: ArrayCase) -> dict[str, Any] | None:
    if not all(key in case.pred and key in case.true for key in ("phie", "phis_c")):
        return None
    e_phie = (case.pred["phie"] - case.true["phie"]).reshape(-1)
    e_phis = (case.pred["phis_c"] - case.true["phis_c"]).reshape(-1)
    common = 0.5 * (e_phie + e_phis)
    differential = e_phis - e_phie
    return {
        "case_id": case.case_id,
        "common_mode_mae": float(np.nanmean(np.abs(common))),
        "common_mode_rmse": float(np.sqrt(np.nanmean(common**2))),
        "common_mode_bias": float(np.nanmean(common)),
        "differential_mae": float(np.nanmean(np.abs(differential))),
        "differential_rmse": float(np.sqrt(np.nanmean(differential**2))),
        "phie_phis_error_corr": safe_corr(e_phie, e_phis),
    }


def diagnose_case(case: ArrayCase, s1: Mapping[str, Any]) -> CaseDiagnostic:
    state_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    cycle_rows_all: list[dict[str, Any]] = []
    boundary_rows_all: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for state in case.available_states:
        true = np.asarray(case.true[state], dtype=np.float64)
        pred = np.asarray(case.pred[state], dtype=np.float64)
        row = _metrics_row(case, state, true, pred, s1)
        state_rows.append(row)
        phase_rows.extend(_phase_metrics(case, state, true, pred))
        cycle_rows = cycle_bias_rows(case.cycle_id, case.n_time, true, pred, state, case.case_id)
        cycle_rows_all.extend(cycle_rows)
        boundary_rows_all.extend(boundary_rows(case.cycle_id, case.n_time, true, pred, state, case.case_id))
        rank_rows.append(_rank_row(case, state, true, pred))
        radial_rows.extend(_radial_diagnostics(case, state, true, pred))
        summaries[state] = row

    for electrode in ("a", "c"):
        item = _theta_cs_consistency(case, electrode)
        if item is not None:
            consistency_rows.append(item)

    common_mode = _potential_common_mode(case)
    cycle_summaries = cycle_summary_rows(
        case.time_s,
        case.cycle_id,
        case.current_A,
        case.voltage_V,
        case.temperature_C,
        case.step_type,
    )
    cum_abs = cumulative_ah(case.time_s, case.current_A, absolute=True)
    inventory_correlations: dict[str, float] = {}
    for state in case.available_states:
        true_mean = volume_mean(case.true[state], _state_grid(case, state)) if state in RADIAL_STATES else case.true[state][:, 0]
        pred_mean = volume_mean(case.pred[state], _state_grid(case, state)) if state in RADIAL_STATES else case.pred[state][:, 0]
        bias_t = pred_mean - true_mean
        inventory_correlations[state] = safe_corr(cum_abs, bias_t) if cum_abs.size == bias_t.size else float("nan")

    summary = {
        "case_id": case.case_id,
        "prediction_path": case.prediction_path,
        "truth_path": case.truth_path,
        "canonical_cell_uid": case.canonical_cell_uid,
        "cell_uid": case.cell_uid,
        "split": case.split,
        "protocol": case.protocol,
        "branch": case.branch,
        "n_time": case.n_time,
        "cycle_count": len({int(float(x)) for x in sanitize_cycle_id(case.cycle_id, case.n_time)}),
        "available_states": case.available_states,
        "states": summaries,
        "potential_common_mode": common_mode,
        "inventory_bias_vs_cumulative_abs_Ah_corr": inventory_correlations,
        "metadata": case.metadata,
    }
    return CaseDiagnostic(
        summary=summary,
        state_rows=state_rows,
        phase_rows=phase_rows,
        cycle_rows=cycle_rows_all,
        boundary_rows=boundary_rows_all,
        rank_rows=rank_rows,
        radial_rows=radial_rows,
        cycle_summary_rows=[{"case_id": case.case_id, **row} for row in cycle_summaries],
        consistency_rows=consistency_rows,
    )


def _finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def build_recommendation(
    state_rows: list[dict[str, Any]],
    boundary_rows_all: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
    s1: Mapping[str, Any],
) -> dict[str, Any]:
    failure_r2 = float(s1.get("r2_failure_threshold", 0.90))
    recovery_r2 = float(s1.get("r2_recovery_threshold", 0.95))
    rank95_low_dim_max = int(s1.get("low_dim_rank95_max", 4))
    lag_gain_threshold = float(s1.get("lag_gain_threshold", 0.08))
    boundary_threshold = float(s1.get("boundary_jump_error_threshold", 0.02))
    failed = [row for row in state_rows if np.isfinite(float(row.get("r2", np.nan))) and float(row["r2"]) < failure_r2]
    if not failed:
        return {
            "labels": ["NO_STRUCTURAL_FAILURE_DETECTED_IN_SELECTED_CASES"],
            "go_to_s2": False,
            "reason": "S1 selected arrays do not reproduce the known dense-cycle failure; inspect discovery/config before training.",
            "failed_state_count": 0,
        }

    low_dim_recovered = [
        row
        for row in failed
        if max(float(row.get("shifted_r2", -np.inf)), float(row.get("affine_r2", -np.inf))) >= recovery_r2
        and int(row.get("residual_rank_at_95", 999)) <= rank95_low_dim_max
    ]
    sequence_signals = [
        row
        for row in failed
        if float(row.get("lag_r2_gain", 0.0)) >= lag_gain_threshold
        or abs(float(row.get("cycle_bias_corr", 0.0))) >= 0.5
        or int(row.get("residual_rank_at_95", 0)) > rank95_low_dim_max
    ]
    boundary_bad = [row for row in boundary_rows_all if np.isfinite(float(row.get("normalized_abs_jump_error", np.nan))) and float(row["normalized_abs_jump_error"]) > boundary_threshold]
    teacher_bad = [
        row
        for row in failed
        if float(row.get("nonfinite_true_fraction", 0.0)) > 0.0 or float(row.get("nonfinite_pred_fraction", 0.0)) > 0.0
    ]
    teacher_bad.extend(
        row
        for row in consistency_rows
        if np.isfinite(float(row.get("true_relation_rmse", np.nan)))
        and np.isfinite(float(row.get("pred_relation_rmse", np.nan)))
        and float(row.get("pred_relation_rmse", 0.0)) > 10.0 * max(float(row.get("true_relation_rmse", 0.0)), 1e-8)
    )

    branch_groups: dict[str, list[float]] = {}
    for row in failed:
        branch_groups.setdefault(str(row.get("branch", "UNKNOWN")), []).append(float(row.get("r2", np.nan)))
    branch_means = {k: _finite_mean(v) for k, v in branch_groups.items()}
    finite_branch_means = [v for v in branch_means.values() if np.isfinite(v)]
    branch_gap = max(finite_branch_means) - min(finite_branch_means) if len(finite_branch_means) >= 2 else 0.0

    labels: list[str] = []
    low_dim_fraction = len(low_dim_recovered) / max(1, len(failed))
    if low_dim_fraction >= 0.75:
        labels.append("LOW_DIMENSIONAL_LATENT_SUFFICIENT")
    if sequence_signals or boundary_bad:
        labels.append("SEQUENCE_MODEL_REQUIRED")
    has_p4d_failure = any(any(token in k.upper() for token in ("P4D", "GEO", "CURRENT_INTEGRAL")) for k in branch_groups)
    if branch_gap >= float(s1.get("branch_mean_r2_gap_threshold", 0.20)) or has_p4d_failure:
        labels.append("BRANCH_SPECIFIC_OPERATOR_REQUIRED")
    if teacher_bad:
        labels.append("TEACHER_OR_DATA_INCONSISTENCY")
    if not labels:
        labels.append("STRUCTURAL_OPERATOR_REDESIGN_REQUIRED")

    # D18-S1 is diagnostic-only by design. Promotion to S2 requires manual review of this report.
    go_to_s2 = False
    reason = (
        "S1 found dense-cycle failures. Build the S2 architecture from the indicated labels, then require explicit manual promotion; "
        "this package intentionally does not launch training."
    )
    return {
        "labels": labels,
        "go_to_s2": go_to_s2,
        "reason": reason,
        "failed_state_count": len(failed),
        "low_dim_recovered_count": len(low_dim_recovered),
        "low_dim_recovered_fraction": low_dim_fraction,
        "sequence_signal_count": len(sequence_signals),
        "bad_cycle_boundary_count": len(boundary_bad),
        "teacher_or_data_warning_count": len(teacher_bad),
        "branch_mean_failed_r2": branch_means,
        "branch_failed_r2_gap": branch_gap,
        "thresholds": {
            "r2_failure_threshold": failure_r2,
            "r2_recovery_threshold": recovery_r2,
            "low_dim_rank95_max": rank95_low_dim_max,
            "lag_gain_threshold": lag_gain_threshold,
            "boundary_jump_error_threshold": boundary_threshold,
        },
    }
