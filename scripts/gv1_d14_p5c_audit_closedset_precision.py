#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D14-P5C XJTU P2Dlite 8-cell closed-set precision audit.

This script does not train and does not regenerate soft labels. It reads the
existing D14-P5B-v2 output directory and freezes the result as an auditable
closed-set calibration benchmark.

It produces:
  - compact metrics tables;
  - pass/warn/fail precision checks;
  - training convergence summary;
  - a Markdown/JSON report suitable for project archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def status_rank(status: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(str(status).upper(), 1)


def combine_status(checks: List[dict]) -> str:
    worst = "PASS"
    for check in checks:
        st = str(check.get("status", "WARN")).upper()
        if status_rank(st) > status_rank(worst):
            worst = st
    return worst


def threshold_check(check_id: str, name: str, value: float, pass_thr: float, warn_thr: Optional[float], direction: str, unit: str = "") -> dict:
    if not math.isfinite(value):
        return {"check_id": check_id, "name": name, "status": "FAIL", "value": value, "detail": "nonfinite metric"}
    if direction == "le":
        if value <= pass_thr:
            status = "PASS"
        elif warn_thr is not None and value <= warn_thr:
            status = "WARN"
        else:
            status = "FAIL"
        detail = f"value={value:.8g}{unit}, pass<={pass_thr}{unit}" + (f", warn<={warn_thr}{unit}" if warn_thr is not None else "")
    elif direction == "ge":
        if value >= pass_thr:
            status = "PASS"
        elif warn_thr is not None and value >= warn_thr:
            status = "WARN"
        else:
            status = "FAIL"
        detail = f"value={value:.8g}{unit}, pass>={pass_thr}{unit}" + (f", warn>={warn_thr}{unit}" if warn_thr is not None else "")
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return {"check_id": check_id, "name": name, "status": status, "value": value, "detail": detail}


def compact_profile_rows(metrics_by_profile: List[dict]) -> List[dict]:
    rows = []
    for row in metrics_by_profile:
        rows.append({
            "cell_uid": row.get("cell_uid", ""),
            "batch": row.get("batch", ""),
            "protocol": row.get("protocol", ""),
            "n_points": row.get("n_points", ""),
            "theta_a_mae": row.get("theta_a_mae", ""),
            "theta_c_mae": row.get("theta_c_mae", ""),
            "theta_mean_mae": row.get("theta_mean_mae", ""),
            "phie_mae": row.get("phie_mae", ""),
            "phis_c_mae_V": row.get("phis_c_mae", ""),
            "theta_a_corr": row.get("theta_a_corr", ""),
            "theta_c_corr": row.get("theta_c_corr", ""),
            "phie_corr": row.get("phie_corr", ""),
            "phis_c_corr": row.get("phis_c_corr", ""),
            "cs_a_mae_mol_m3": row.get("cs_a_mae", ""),
            "cs_c_mae_mol_m3": row.get("cs_c_mae", ""),
        })
    return rows


def summarize_loss(loss_rows: List[dict]) -> dict:
    if not loss_rows:
        return {"status": "WARN", "detail": "loss_history.csv missing or empty"}
    first = loss_rows[0]
    last = loss_rows[-1]
    def f(row, key):
        return as_float(row.get(key))
    best_eval = min([f(r, "closed_eval_loss") for r in loss_rows if math.isfinite(f(r, "closed_eval_loss"))], default=float("nan"))
    best_rows = [r for r in loss_rows if math.isfinite(f(r, "closed_eval_loss")) and f(r, "closed_eval_loss") == best_eval]
    best_epoch = best_rows[0].get("epoch", "") if best_rows else ""
    return {
        "status": "PASS",
        "epoch_count": len(loss_rows),
        "first_epoch": first.get("epoch", ""),
        "last_epoch": last.get("epoch", ""),
        "first_train_loss": f(first, "train_loss"),
        "last_train_loss": f(last, "train_loss"),
        "first_eval_loss": f(first, "closed_eval_loss"),
        "last_eval_loss": f(last, "closed_eval_loss"),
        "best_eval_loss": best_eval,
        "best_epoch_from_loss_history": best_epoch,
        "last_points_per_s": f(last, "points_per_s"),
        "last_epoch_time_s": f(last, "epoch_time_s"),
        "last_cuda_max_memory_allocated_MB": f(last, "cuda_max_memory_allocated_MB"),
        "last_cuda_memory_reserved_MB": f(last, "cuda_memory_reserved_MB"),
    }


def md_table(rows: List[dict], cols: List[str]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p5b_output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_warn", action="store_true")
    args = parser.parse_args()

    p5b = Path(args.p5b_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_json(Path(args.config)) or {}
    req = cfg.get("required_inputs", {})
    thr = cfg.get("precision_thresholds", {})

    paths = {
        "training_summary": p5b / req.get("training_summary", "ModelFin_D14_P5B_8cell_closedset_precision/training_summary.json"),
        "loss_history": p5b / req.get("loss_history", "ModelFin_D14_P5B_8cell_closedset_precision/loss_history.csv"),
        "eval_report": p5b / req.get("eval_report", "EvalFin_D14_P5B_8cell_closedset_precision/D14_P5B_EVAL_REPORT.json"),
        "metrics_by_profile": p5b / req.get("metrics_by_profile", "EvalFin_D14_P5B_8cell_closedset_precision/metrics_by_profile.csv"),
        "metrics_global": p5b / req.get("metrics_global", "EvalFin_D14_P5B_8cell_closedset_precision/metrics_global.json"),
        "verify_report": p5b / req.get("verify_report", "D14_P5B_VERIFY_REPORT.json"),
    }

    training = read_json(paths["training_summary"]) or {}
    eval_report = read_json(paths["eval_report"]) or {}
    metrics_global = read_json(paths["metrics_global"]) or eval_report.get("global_metrics", {})
    verify_report = read_json(paths["verify_report"]) or {}
    profile_rows_raw = read_csv(paths["metrics_by_profile"])
    loss_rows = read_csv(paths["loss_history"])

    profile_rows = compact_profile_rows(profile_rows_raw)
    loss_summary = summarize_loss(loss_rows)

    write_csv(output_dir / "D14_P5C_PROFILE_METRICS_COMPACT.csv", profile_rows)
    write_json(output_dir / "D14_P5C_LOSS_SUMMARY.json", loss_summary)

    # Batch/protocol summaries can be copied from eval report if available.
    write_json(output_dir / "D14_P5C_BATCH_METRICS.json", eval_report.get("batch_metrics", []))
    write_json(output_dir / "D14_P5C_PROTOCOL_METRICS.json", eval_report.get("protocol_metrics", []))

    checks: List[dict] = []
    missing = [name for name, path in paths.items() if not path.exists()]
    checks.append({
        "check_id": "P5C-C00",
        "name": "required input files present",
        "status": "PASS" if not missing else "FAIL",
        "detail": f"missing={missing}",
    })
    checks.append({
        "check_id": "P5C-C01",
        "name": "training summary status",
        "status": "PASS" if training.get("status") == "PASS" else "FAIL",
        "detail": f"status={training.get('status')} best_epoch={training.get('best_epoch')} best_loss={training.get('best_loss')}",
    })
    checks.append({
        "check_id": "P5C-C02",
        "name": "eval report status",
        "status": "PASS" if eval_report.get("overall_status") == "PASS" else ("WARN" if eval_report.get("overall_status") == "WARN" else "FAIL"),
        "detail": f"eval_status={eval_report.get('overall_status')} warn={eval_report.get('warn_reasons')} fail={eval_report.get('fail_reasons')}",
    })
    checks.append({
        "check_id": "P5C-C03",
        "name": "profile count",
        "status": "PASS" if int(eval_report.get("profile_count", len(profile_rows_raw)) or 0) >= int(thr.get("profile_count_required", 8)) else "FAIL",
        "detail": f"profile_count={eval_report.get('profile_count', len(profile_rows_raw))} required={thr.get('profile_count_required', 8)}",
    })
    checks.append({
        "check_id": "P5C-C04",
        "name": "n_r consistency",
        "status": "PASS" if int(training.get("feature_stats", {}).get("n_r", -1)) == int(thr.get("n_r_required", 17)) else "FAIL",
        "detail": f"n_r={training.get('feature_stats', {}).get('n_r')} required={thr.get('n_r_required', 17)}",
    })

    checks.append(threshold_check("P5C-M00", "mean phis_c MAE", as_float(metrics_global.get("mean_phis_c_mae")), as_float(thr.get("mean_phis_c_mae_pass_V", 0.010)), as_float(thr.get("mean_phis_c_mae_warn_V", 0.015)), "le", " V"))
    checks.append(threshold_check("P5C-M01", "max phis_c MAE", as_float(metrics_global.get("max_phis_c_mae")), as_float(thr.get("max_phis_c_mae_pass_V", 0.015)), as_float(thr.get("max_phis_c_mae_warn_V", 0.025)), "le", " V"))
    checks.append(threshold_check("P5C-M02", "mean phie MAE", as_float(metrics_global.get("mean_phie_mae")), as_float(thr.get("mean_phie_mae_pass", 0.010)), None, "le", ""))
    checks.append(threshold_check("P5C-M03", "max phie MAE", as_float(metrics_global.get("max_phie_mae")), as_float(thr.get("max_phie_mae_pass", 0.015)), None, "le", ""))
    checks.append(threshold_check("P5C-M04", "mean theta_mean MAE", as_float(metrics_global.get("mean_theta_mean_mae")), as_float(thr.get("mean_theta_mean_mae_pass", 0.010)), as_float(thr.get("mean_theta_mean_mae_warn", 0.020)), "le", ""))
    checks.append(threshold_check("P5C-M05", "max theta_mean MAE", as_float(metrics_global.get("max_theta_mean_mae")), as_float(thr.get("max_theta_mean_mae_pass", 0.015)), as_float(thr.get("max_theta_mean_mae_warn", 0.035)), "le", ""))
    checks.append(threshold_check("P5C-M06", "mean theta_a MAE", as_float(metrics_global.get("mean_theta_a_mae")), as_float(thr.get("mean_theta_a_mae_pass", 0.010)), None, "le", ""))
    checks.append(threshold_check("P5C-M07", "mean theta_c MAE", as_float(metrics_global.get("mean_theta_c_mae")), as_float(thr.get("mean_theta_c_mae_pass", 0.010)), None, "le", ""))
    checks.append(threshold_check("P5C-M08", "min phis_c corr", as_float(metrics_global.get("min_phis_c_corr")), as_float(thr.get("min_phis_c_corr_pass", 0.999)), None, "ge", ""))
    min_theta_corr = min(as_float(metrics_global.get("min_theta_a_corr")), as_float(metrics_global.get("min_theta_c_corr")))
    checks.append(threshold_check("P5C-M09", "min theta corr", min_theta_corr, as_float(thr.get("min_theta_corr_pass", 0.999)), None, "ge", ""))

    overall = combine_status(checks)
    if overall == "PASS":
        recommendation = "Accept D14-P5B-v2 as the 8-cell closed-set precision benchmark. Next step can be a controlled ablation or leave-one-profile-out experiment."
    elif overall == "WARN":
        recommendation = "Accept as warning-level closed-set benchmark only after reviewing WARN checks."
    else:
        recommendation = "Do not archive P5B-v2 as precision benchmark until FAIL checks are resolved."

    report = {
        "package": "D14-P5C XJTU P2Dlite 8-cell closed-set precision audit",
        "overall_status": overall,
        "recommendation": recommendation,
        "p5b_output_dir": str(p5b),
        "output_dir": str(output_dir),
        "checks": checks,
        "global_metrics": metrics_global,
        "training_summary_compact": {
            "device": training.get("device"),
            "compiled": training.get("compiled"),
            "amp": training.get("amp"),
            "gpu_resident_tensors": training.get("gpu_resident_tensors"),
            "epochs": training.get("epochs"),
            "batch_size": training.get("batch_size"),
            "point_count": training.get("point_count"),
            "profile_count": training.get("profile_count"),
            "best_epoch": training.get("best_epoch"),
            "best_loss": training.get("best_loss"),
        },
        "loss_summary": loss_summary,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(output_dir / "D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.json", report)
    write_csv(output_dir / "D14_P5C_CHECKS.csv", checks)

    md = []
    md.append("# D14-P5C XJTU P2Dlite 8-cell Closed-set Precision Audit\n")
    md.append(f"Overall status: **{overall}**\n")
    md.append(f"Recommendation: {recommendation}\n")
    md.append("## Key global metrics\n")
    md.append(f"- mean phis_c MAE: `{metrics_global.get('mean_phis_c_mae')}` V\n")
    md.append(f"- max phis_c MAE: `{metrics_global.get('max_phis_c_mae')}` V\n")
    md.append(f"- mean phie MAE: `{metrics_global.get('mean_phie_mae')}`\n")
    md.append(f"- mean theta_mean MAE: `{metrics_global.get('mean_theta_mean_mae')}`\n")
    md.append(f"- max theta_mean MAE: `{metrics_global.get('max_theta_mean_mae')}`\n")
    md.append("## Checks\n")
    md.append(md_table(checks, ["check_id", "name", "status", "detail"]))
    md.append("\n## Compact profile metrics\n")
    md.append(md_table(profile_rows, ["cell_uid", "batch", "protocol", "theta_mean_mae", "phie_mae", "phis_c_mae_V", "phis_c_corr"]))
    md.append("\n## Boundary\n")
    md.append("- This is closed-set calibration, not a generalization claim.\n")
    md.append("- No training is run by P5C.\n")
    md.append("- No soft labels or SOH labels are generated by P5C.\n")
    (output_dir / "D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    outputs = [
        "D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.json",
        "D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.md",
        "D14_P5C_CHECKS.csv",
        "D14_P5C_PROFILE_METRICS_COMPACT.csv",
        "D14_P5C_LOSS_SUMMARY.json",
        "D14_P5C_BATCH_METRICS.json",
        "D14_P5C_PROTOCOL_METRICS.json",
        "D14_P5C_OUTPUT_INDEX.json",
    ]
    # Write index last.
    write_json(output_dir / "D14_P5C_OUTPUT_INDEX.json", {
        "overall_status": overall,
        "files": [{"name": name, "exists": True if name == "D14_P5C_OUTPUT_INDEX.json" else (output_dir / name).exists()} for name in outputs],
    })

    print(f"[P5C audit] overall_status={overall}")
    print(f"[P5C audit] recommendation={recommendation}")
    return 0 if overall == "PASS" or (overall == "WARN" and args.allow_warn) else (2 if overall == "WARN" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
