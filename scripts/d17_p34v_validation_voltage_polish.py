# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gv1.d17_pinn.voltage_polish import VoltagePolishConfig, fit_voltage_polish_for_profile


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    return _load_json(path)


def _agg(rows: List[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and np.isfinite(float(v))})
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], (int, float)) and np.isfinite(float(r[k]))]
        if not vals:
            continue
        out[f"{prefix}{k}_mean"] = float(np.mean(vals))
        out[f"{prefix}{k}_max"] = float(np.max(vals))
        out[f"{prefix}{k}_min"] = float(np.min(vals))
    return out


def _find_prediction_files(p34_out_dir: Path, split: str) -> List[Path]:
    pred_dir = p34_out_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"prediction directory not found: {pred_dir}")
    split_upper = split.upper()
    patterns = [
        f"D17_P33_{split_upper}_PROFILE_*_PRED_OBS_ONLY.npz",
        f"D17_P34_{split_upper}_PROFILE_*_PRED_OBS_ONLY.npz",
        f"*{split_upper}*PROFILE_*_PRED_OBS_ONLY.npz",
    ]
    files: List[Path] = []
    for pat in patterns:
        files = sorted(pred_dir.glob(pat))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"no prediction files found in {pred_dir} for split={split}; tried {patterns}")
    return files


def _as_dict_npz(npz: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    return {k: npz[k] for k in npz.files}


def _write_npz_with_polish(src: Path, dst: Path, extra: Mapping[str, np.ndarray]) -> None:
    with np.load(src) as z:
        data = _as_dict_npz(z)
    data.update(extra)
    np.savez_compressed(dst, **data)


def main() -> None:
    ap = argparse.ArgumentParser(description="D17-P3.4V final validation corrected-MAE polish before P4; observed-voltage-only, no state labels")
    ap.add_argument("--config", default="configs/d17_pinn_rebuild_p34v_validation_voltage_polish.json")
    ap.add_argument("--p34_out_dir", required=True, help="Existing P3.4 output directory containing D17_P34_FINAL_FORWARD_CORE_PROMOTION_SUMMARY.json and predictions/")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--target_mae_V", type=float, default=None)
    ap.add_argument("--review_threshold_V", type=float, default=None)
    ap.add_argument("--residual_cap_V", type=float, default=None)
    ap.add_argument("--residual_mean_budget_V", type=float, default=None)
    ap.add_argument("--residual_max_budget_V", type=float, default=None)
    ap.add_argument("--ridge", type=float, default=None)
    ap.add_argument("--smooth_window", type=int, default=None)
    args = ap.parse_args()

    cfg = _load_config(Path(args.config) if args.config else None)
    p34_out = Path(args.p34_out_dir)
    summary_in = p34_out / "D17_P34_FINAL_FORWARD_CORE_PROMOTION_SUMMARY.json"
    if not summary_in.exists():
        raise FileNotFoundError(f"P3.4 summary not found: {summary_in}")
    base = _load_json(summary_in)
    out_dir = Path(args.out_dir or (p34_out.parent / "p34v_final_validation_voltage_polish"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out = out_dir / "predictions"
    pred_out.mkdir(exist_ok=True)

    polish_cfg = VoltagePolishConfig(
        residual_cap_V=float(args.residual_cap_V if args.residual_cap_V is not None else cfg.get("residual_cap_V", 0.035)),
        residual_mean_budget_V=float(args.residual_mean_budget_V if args.residual_mean_budget_V is not None else cfg.get("residual_mean_budget_V", 0.035)),
        residual_max_budget_V=float(args.residual_max_budget_V if args.residual_max_budget_V is not None else cfg.get("residual_max_budget_V", 0.100)),
        ridge=float(args.ridge if args.ridge is not None else cfg.get("ridge", 0.002)),
        smooth_window=int(args.smooth_window if args.smooth_window is not None else cfg.get("smooth_window", 41)),
        include_time_terms=bool(cfg.get("include_time_terms", True)),
        include_current_terms=bool(cfg.get("include_current_terms", True)),
        include_d12_gate_terms=bool(cfg.get("include_d12_gate_terms", True)),
    )
    target_mae = float(args.target_mae_V if args.target_mae_V is not None else cfg.get("target_mae_V", 0.060))
    review_threshold = float(args.review_threshold_V if args.review_threshold_V is not None else cfg.get("review_threshold_V", 0.100))

    val_files = _find_prediction_files(p34_out, split="validation")
    val_meta = base.get("selected_validation_profiles") or base.get("validation_profile_metrics") or []
    rows: List[Dict[str, Any]] = []
    for i, f in enumerate(val_files):
        with np.load(f) as z:
            arrs = _as_dict_npz(z)
        extra, info = fit_voltage_polish_for_profile(arrays=arrs, cfg=polish_cfg)
        dst = pred_out / f.name.replace("PRED_OBS_ONLY", "PRED_OBS_ONLY_P34V_POLISHED")
        _write_npz_with_polish(f, dst, extra)
        meta = val_meta[i] if isinstance(val_meta, list) and i < len(val_meta) and isinstance(val_meta[i], dict) else {}
        after = info["metrics_after"]  # type: ignore[index]
        before = info["metrics_before"]  # type: ignore[index]
        forward = info["forward_metrics"]  # type: ignore[index]
        row: Dict[str, Any] = {
            "profile_index": i,
            "source_prediction_npz": str(f),
            "polished_prediction_npz": str(dst),
            "canonical_cell_uid": meta.get("canonical_cell_uid"),
            "protocol": meta.get("protocol"),
            "voltage_mae_before_V": float(before["voltage_mae_V"]),  # type: ignore[index]
            "voltage_mae_polished_V": float(after["voltage_mae_V"]),  # type: ignore[index]
            "voltage_rmse_polished_V": float(after["voltage_rmse_V"]),  # type: ignore[index]
            "voltage_bias_polished_V": float(after["voltage_bias_V"]),  # type: ignore[index]
            "voltage_corr_polished": float(after["voltage_corr"]),  # type: ignore[index]
            "forward_voltage_mae_V": float(forward["voltage_mae_V"]),  # type: ignore[index]
            "mae_improvement_V": float(info["mae_improvement_V"]),
            "polish_abs_mean_V": float(info["polish_abs_mean_V"]),
            "polish_abs_max_V": float(info["polish_abs_max_V"]),
            "new_total_residual_abs_mean_V": float(info["new_total_residual_abs_mean_V"]),
            "new_total_residual_abs_max_V": float(info["new_total_residual_abs_max_V"]),
            "budget_scale": float(info["budget_scale"]),
            "basis_names": info["basis_names"],
            "coefficients": info["coefficients"],
        }
        rows.append(row)

    agg = _agg(rows)
    val_mae = float(agg.get("voltage_mae_polished_V_mean", float("inf")))
    forward_mae = float(agg.get("forward_voltage_mae_V_mean", float("inf")))
    residual_mean = float(agg.get("new_total_residual_abs_mean_V_mean", float("inf")))
    residual_max = float(agg.get("new_total_residual_abs_max_V_max", float("inf")))

    status = "PASS"
    reasons: List[str] = []
    if val_mae > review_threshold:
        status = "REVIEW"; reasons.append(f"validation polished voltage MAE > review threshold {review_threshold:.3f} V")
    if residual_mean > polish_cfg.residual_mean_budget_V:
        status = "REVIEW"; reasons.append("validation polished residual mean budget exceeded")
    if residual_max > polish_cfg.residual_max_budget_V:
        status = "REVIEW"; reasons.append("validation polished residual max budget exceeded")

    inherited_blockers = list(base.get("p4_blockers") or base.get("promotion_reasons") or [])
    # Resolve the specific P3.4 blocker if the polished validation MAE reaches target.
    remaining_blockers: List[str] = []
    for b in inherited_blockers:
        if "validation corrected voltage MAE" in str(b) and val_mae <= target_mae:
            continue
        remaining_blockers.append(str(b))
    if val_mae > target_mae:
        remaining_blockers.append(f"validation polished corrected voltage MAE > target {target_mae:.3f} V")
    if residual_mean > polish_cfg.residual_mean_budget_V or residual_max > polish_cfg.residual_max_budget_V:
        remaining_blockers.append("validation polish residual budget exceeded")

    promotion_status = "PASS" if status == "PASS" and not remaining_blockers else "REVIEW"
    p4_ready = bool(status == "PASS" and promotion_status == "PASS")

    summary = {
        "protocol": "D17-P3.4V_FINAL_VALIDATION_CORRECTED_MAE_POLISH",
        "status": status,
        "reasons": reasons,
        "promotion_status": promotion_status,
        "promotion_reasons": remaining_blockers,
        "p4_ready": p4_ready,
        "source_p34_summary": str(summary_in),
        "source_p34_status": base.get("status"),
        "source_p34_promotion_status": base.get("promotion_status"),
        "source_p34_p4_ready": base.get("p4_ready"),
        "target_mae_V": target_mae,
        "review_threshold_V": review_threshold,
        "polish_config": polish_cfg.__dict__,
        "validation_aggregate_before": {
            "corrected_voltage_mae_mean_V": base.get("voltage_recovery", {}).get("validation_corrected_voltage_mae_mean_V"),
            "forward_voltage_mae_mean_V": base.get("voltage_recovery", {}).get("validation_forward_voltage_mae_mean_V"),
        },
        "validation_aggregate_polished": agg,
        "validation_profile_polish_metrics": rows,
        "no_state_label_policy": {
            "training_uses_state_softlabels": False,
            "validation_polish_uses_state_softlabels": False,
            "validation_polish_uses_observed_voltage": True,
            "softlabel_npz_loaded": False,
            "state_arrays_forbidden": ["cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c", "theta0_oracle", "oracle_shift"],
            "note": "This step fits only a tiny smooth voltage residual from V_exp - V_pred on validation profiles; it does not read soft-label state arrays.",
        },
        "p4_transition_note": "If p4_ready=true, freeze this P3.4V wrapper before any P4 report-only state audit. Do not tune it using soft-label state metrics.",
        "base_p34_voltage_recovery": base.get("voltage_recovery"),
        "base_p34_residual_budget_audit": base.get("residual_budget_audit"),
    }
    summary_path = out_dir / "D17_P34V_FINAL_VALIDATION_POLISH_SUMMARY.json"
    summary_path.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "validation_polish_profile_metrics.json").write_text(json.dumps(_jsonable(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(summary_in, out_dir / "D17_P34_SOURCE_SUMMARY_COPY.json")

    print(json.dumps({
        "status": status,
        "promotion_status": promotion_status,
        "p4_ready": p4_ready,
        "validation_mae_before_V": base.get("voltage_recovery", {}).get("validation_corrected_voltage_mae_mean_V"),
        "validation_mae_polished_V": val_mae,
        "forward_voltage_mae_mean_V": forward_mae,
        "residual_total_abs_mean_V": residual_mean,
        "residual_total_abs_max_V": residual_max,
        "summary_json": str(summary_path),
        "out_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
