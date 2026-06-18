# -*- coding: utf-8 -*-
"""D17-P4-mini adaptation diagnostic.

Purpose
-------
This is not a new training stage and it does not modify the P3/P4 candidate.
It runs the existing P4 report-only state audit on the same first frozen-test
profile with a short adaptation budget and a formal adaptation budget, then
compares whether the poor P4 smoke state R2 was caused mainly by too few
observed-only latent adaptation steps.

Soft-label boundary
-------------------
The script reuses D17-P4 audit code.  Predictions are generated first from
observed replay fields (I, V, T, time, metadata); P2Dlite-RG soft labels are
loaded only afterward for report-only metrics.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_pinn.config import cfg_get, load_config
from gv1.d17_pinn.p4_state_audit import run_p4_report_only_state_audit

STATE_R2_KEYS = [
    "theta_a_r2",
    "theta_c_r2",
    "cs_a_r2",
    "cs_c_r2",
    "phie_r2",
    "phis_c_r2",
]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_first_csv_row(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return dict(row)
    return {}


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    return x


def _set_path_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    cfg.setdefault("paths", {})
    for key in [
        "candidate_p34_dir",
        "candidate_p34v_dir",
        "split_manifest",
        "softlabel_root",
        "resolved_spec",
        "checkpoint",
        "no_state_label_audit",
    ]:
        val = getattr(args, key, None)
        if val:
            cfg["paths"][key] = val


def _prepare_run_cfg(base_cfg: Mapping[str, Any], args: argparse.Namespace, *, steps: int) -> Dict[str, Any]:
    cfg: Dict[str, Any] = copy.deepcopy(dict(base_cfg))
    cfg["d17_protocol_version"] = 4
    cfg["experiment_name"] = f"d17_p4mini_adaptation_diagnostic_{steps}steps"
    _set_path_overrides(cfg, args)
    cfg.setdefault("p4", {})

    # In this diagnostic we deliberately use only one normal frozen-test profile.
    # No train/validation/flagged profile is audited, so this cannot become a
    # hidden validation/test-tuning loop.
    cfg["p4"]["train_profile_limit"] = 0
    cfg["p4"]["validation_profile_limit"] = 0
    cfg["p4"]["frozen_test_profile_limit"] = 1
    cfg["p4"]["flagged_probe_profile_limit"] = 0

    cfg["p4"]["adaptation_steps"] = int(steps)
    cfg["p4"]["train_adaptation_steps"] = int(steps)
    cfg["p4"]["validation_adaptation_steps"] = int(steps)
    cfg["p4"]["frozen_test_adaptation_steps"] = int(steps)
    cfg["p4"]["flagged_probe_adaptation_steps"] = int(steps)
    cfg["p4"]["no_candidate_modification"] = True
    cfg["p4"]["diagnostic_only"] = True

    # CLI overrides for shared runtime controls.
    for key in ["n_r", "max_time_points", "time_window_s", "adaptation_lr", "device"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg["p4"][key] = val
    return cfg


def _extract_run_summary(run_dir: Path, steps: int) -> Dict[str, Any]:
    scorecard = _read_json(run_dir / "D17_P4_SCORECARD.json")
    state_row = _read_first_csv_row(run_dir / "D17_P4_STATE_AUDIT_PROFILE_METRICS.csv")
    voltage_row = _read_first_csv_row(run_dir / "D17_P4_VOLTAGE_STATE_DECOMPOSITION.csv")
    out: Dict[str, Any] = {
        "steps": int(steps),
        "run_dir": str(run_dir),
        "status": scorecard.get("status"),
        "promotion_status": scorecard.get("promotion_status"),
        "p5_ready": bool(scorecard.get("p5_ready")),
        "promotion_reasons": scorecard.get("promotion_reasons", []),
        "profile": {
            "split": state_row.get("split"),
            "canonical_cell_uid": state_row.get("canonical_cell_uid"),
            "cell_uid": state_row.get("cell_uid"),
            "protocol": state_row.get("protocol"),
            "pred_npz": state_row.get("pred_npz"),
            "softlabel_npz_report_only": state_row.get("softlabel_npz_report_only"),
        },
        "state_r2": {k: _to_float(state_row.get(k)) for k in STATE_R2_KEYS},
        "state_mae": {k.replace("_r2", "_mae"): _to_float(state_row.get(k.replace("_r2", "_mae"))) for k in STATE_R2_KEYS},
        "voltage": {
            "corrected_voltage_mae": _to_float(voltage_row.get("corrected_voltage_mae")),
            "forward_voltage_mae": _to_float(voltage_row.get("forward_voltage_mae")),
            "residual_total_abs_mean_V": _to_float(voltage_row.get("residual_total_abs_mean_V")),
            "residual_total_abs_max_V": _to_float(voltage_row.get("residual_total_abs_max_V")),
        },
        "scorecard_json": str(run_dir / "D17_P4_SCORECARD.json"),
        "state_profile_metrics_csv": str(run_dir / "D17_P4_STATE_AUDIT_PROFILE_METRICS.csv"),
        "voltage_state_decomposition_csv": str(run_dir / "D17_P4_VOLTAGE_STATE_DECOMPOSITION.csv"),
    }
    return out


def _classify(short_run: Mapping[str, Any], long_run: Mapping[str, Any], *, min_recovery_r2: float, target_r2: float) -> Dict[str, Any]:
    deltas: Dict[str, Optional[float]] = {}
    long_vals: Dict[str, float] = dict(long_run.get("state_r2", {}))  # type: ignore[arg-type]
    short_vals: Dict[str, float] = dict(short_run.get("state_r2", {}))  # type: ignore[arg-type]
    for key in STATE_R2_KEYS:
        a = short_vals.get(key, float("nan"))
        b = long_vals.get(key, float("nan"))
        deltas[key] = None if (not math.isfinite(a) or not math.isfinite(b)) else float(b - a)

    finite_long = [v for v in long_vals.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    min_long = float(min(finite_long)) if finite_long else float("nan")
    mean_long = float(sum(finite_long) / len(finite_long)) if finite_long else float("nan")

    # A deliberately conservative diagnostic.  It is not promotion: it only says
    # whether running full P4 is worth it.
    hard_fail_keys = [k for k, v in long_vals.items() if not (isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= min_recovery_r2)]
    target_fail_keys = [k for k, v in long_vals.items() if not (isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) >= target_r2)]

    if not hard_fail_keys:
        decision = "RECOVERED_ENOUGH_TO_RUN_FORMAL_P4"
        recommendation = "120-step adaptation recovered the frozen-test smoke profile. Running formal P4 is now reasonable, but still report-only."
    elif len(hard_fail_keys) < len(STATE_R2_KEYS) and mean_long >= min_recovery_r2:
        decision = "PARTIAL_RECOVERY_DIAGNOSE_BEFORE_FORMAL_P4"
        recommendation = "120-step adaptation improved enough to justify one more diagnosis, but not enough for P5 promotion. Inspect failed targets before formal P4."
    else:
        decision = "STOP_FORMAL_P4_FIX_STATE_ALIGNMENT"
        recommendation = "120-step adaptation did not rescue state alignment. Do not run formal P4 for promotion; fix theta/OCP phase and phie gauge/state alignment first."

    return {
        "decision": decision,
        "recommendation": recommendation,
        "r2_delta_long_minus_short": deltas,
        "long_step_r2_mean": mean_long,
        "long_step_r2_min": min_long,
        "failed_min_recovery_keys": hard_fail_keys,
        "failed_target_keys": target_fail_keys,
        "min_recovery_r2": min_recovery_r2,
        "target_r2": target_r2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P4-mini same-profile adaptation-step diagnostic")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p4mini_adaptation_diagnostic.json")
    ap.add_argument("--candidate_p34_dir", default=None)
    ap.add_argument("--candidate_p34v_dir", default=None)
    ap.add_argument("--split_manifest", default=None)
    ap.add_argument("--softlabel_root", default=None)
    ap.add_argument("--resolved_spec", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no_state_label_audit", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--short_steps", type=int, default=10)
    ap.add_argument("--long_steps", type=int, default=120)
    ap.add_argument("--n_r", type=int, default=None)
    ap.add_argument("--max_time_points", type=int, default=None)
    ap.add_argument("--time_window_s", type=float, default=None)
    ap.add_argument("--adaptation_lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--min_recovery_r2", type=float, default=0.90, help="Diagnostic recovery threshold; not a promotion threshold.")
    ap.add_argument("--target_r2", type=float, default=0.98, help="P5-style target threshold for reporting only.")
    args = ap.parse_args()

    base_cfg: Dict[str, Any] = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    for label, steps in [("short", int(args.short_steps)), ("long", int(args.long_steps))]:
        run_dir = out_dir / f"{label}_{steps}steps"
        cfg = _prepare_run_cfg(base_cfg, args, steps=steps)
        scorecard = run_p4_report_only_state_audit(cfg, run_dir)
        summary = _extract_run_summary(run_dir, steps)
        summary["scorecard_status_inline"] = {
            "status": scorecard.get("status"),
            "promotion_status": scorecard.get("promotion_status"),
            "p5_ready": scorecard.get("p5_ready"),
        }
        runs.append(summary)

    diagnostic = _classify(runs[0], runs[1], min_recovery_r2=float(args.min_recovery_r2), target_r2=float(args.target_r2))
    summary_out: Dict[str, Any] = {
        "protocol": "D17-P4MINI_ADAPTATION_DIAGNOSTIC",
        "status": "PASS",
        "purpose": "Check whether the poor P4 smoke state audit is mainly due to too few observed-only adaptation steps.",
        "candidate_is_modified": False,
        "training_or_checkpoint_selection_performed": False,
        "softlabels_report_only": True,
        "state_softlabels_used_for_adaptation": False,
        "profile_scope": "first normal frozen_test profile only",
        "runs": {"short": runs[0], "long": runs[1]},
        "diagnostic": diagnostic,
        "outputs": {
            "summary_json": str(out_dir / "D17_P4MINI_ADAPTATION_DIAGNOSTIC_SUMMARY.json"),
            "short_run_dir": runs[0].get("run_dir"),
            "long_run_dir": runs[1].get("run_dir"),
        },
    }
    (out_dir / "D17_P4MINI_ADAPTATION_DIAGNOSTIC_SUMMARY.json").write_text(json.dumps(_jsonable(summary_out), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": summary_out["status"],
        "decision": diagnostic["decision"],
        "recommendation": diagnostic["recommendation"],
        "short_steps": int(args.short_steps),
        "long_steps": int(args.long_steps),
        "long_step_r2_mean": diagnostic["long_step_r2_mean"],
        "long_step_r2_min": diagnostic["long_step_r2_min"],
        "failed_min_recovery_keys": diagnostic["failed_min_recovery_keys"],
        "summary_json": str(out_dir / "D17_P4MINI_ADAPTATION_DIAGNOSTIC_SUMMARY.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
