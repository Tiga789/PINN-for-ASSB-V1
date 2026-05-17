# -*- coding: utf-8 -*-
r"""Aggregate ASSB-111 saturating_v2_stable multi-seed results.

This script is post-evaluation only. It may read held-out test metrics because it
is not called from the training loop and is not used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd


def _json_clean(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_json_clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_soh_score(eval_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"eval_dir": str(eval_dir), "available": False}
    score = eval_dir / "five_state_scorecard.csv"
    if not score.exists():
        row["failure"] = f"missing {score}"
        return row
    df = pd.read_csv(score)
    soh = df[df.get("variable", "").astype(str).str.upper() == "SOH"] if "variable" in df.columns else pd.DataFrame()
    if soh.empty:
        row["failure"] = "SOH row not found"
        return row
    row.update(soh.iloc[0].to_dict())
    for key in ["n", "MAE", "RMSE", "NMAE", "NRMSE", "R2", "corr"]:
        if key in row:
            try:
                row[key] = float(row[key])
            except Exception:
                pass
    row["available"] = True
    diag = _load_json(eval_dir / "soh_overdecay_diagnostic.json")
    if diag:
        row["active_clamp_count_all"] = diag.get("active_clamp_count_all")
        row["active_clamp_fraction_all"] = diag.get("active_clamp_fraction_all")
        test = (diag.get("split_summary") or {}).get("test", {}) if isinstance(diag.get("split_summary"), dict) else {}
        row["test_pred_min_diag"] = test.get("SOH_pred_min")
        row["test_pred_max_diag"] = test.get("SOH_pred_max")
        row["test_bias_diag"] = (test.get("metrics") or {}).get("BIAS") if isinstance(test.get("metrics"), dict) else None
        for seg in diag.get("segments", []) if isinstance(diag.get("segments"), list) else []:
            name = str(seg.get("segment", ""))
            if name:
                prefix = "tail" if name == "401-521" else f"seg_{name.replace('-', '_')}"
                row[f"{prefix}_R2"] = seg.get("metric_R2")
                row[f"{prefix}_MAE"] = seg.get("metric_MAE")
                row[f"{prefix}_BIAS"] = seg.get("metric_BIAS")
    return row


def _read_training(model_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"model_dir": str(model_dir)}
    summ = _load_json(model_dir / "train_summary.json") or _load_json(model_dir / "training_summary.json")
    cfg = _load_json(model_dir / "soh_head_config.json")
    if summ:
        out["best_epoch"] = summ.get("best_epoch")
        out["best_selection_status"] = summ.get("best_selection_status")
        final = summ.get("final_visible_metrics") or {}
        if isinstance(final, dict):
            out.update({f"visible_{k}": v for k, v in final.items()})
        init = summ.get("initializer_applied") or {}
        if isinstance(init, dict):
            out.update({f"init_{k}": v for k, v in init.items() if k in {"floor", "soh0", "k_per_cycle", "freeze_soh_floor", "freeze_soh0", "rate_head_bias_lam_window"}})
    if isinstance(cfg, dict):
        c = cfg.get("config") or {}
        if isinstance(c, dict):
            for k in ["model_variant", "rate_correction_bound", "init_k_per_cycle", "soh_floor_prior", "floor_min", "floor_max", "freeze_soh_floor", "residual_bound"]:
                out[f"cfg_{k}"] = c.get(k)
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dirs", nargs="+", required=True)
    p.add_argument("--model_dirs", nargs="*", default=[])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--target_r2", type=float, default=0.98)
    p.add_argument("--max_r2_std", type=float, default=0.01)
    p.add_argument("--max_mae_mean", type=float, default=0.006)
    p.add_argument("--require_no_clamp", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_read_soh_score(Path(p)) for p in args.eval_dirs]
    df = pd.DataFrame(rows)
    df.to_csv(out / "seed_stability_summary.csv", index=False, encoding="utf-8-sig")
    train_rows = [_read_training(Path(p)) for p in args.model_dirs]
    train_df = pd.DataFrame(train_rows)
    if not train_df.empty:
        train_df.to_csv(out / "v2stable_training_diagnostic.csv", index=False, encoding="utf-8-sig")

    valid = df[df.get("available", False) == True].copy()  # noqa: E712
    failures: List[str] = []
    summary: Dict[str, Any] = {
        "eval_dirs": list(args.eval_dirs),
        "model_dirs": list(args.model_dirs),
        "n_eval_dirs": len(args.eval_dirs),
        "n_available": int(len(valid)),
        "target_r2": float(args.target_r2),
        "max_r2_std": float(args.max_r2_std),
        "max_mae_mean": float(args.max_mae_mean),
        "seed_stability_summary_csv": str(out / "seed_stability_summary.csv"),
        "training_diagnostic_csv": str(out / "v2stable_training_diagnostic.csv") if not train_df.empty else "",
    }
    if valid.empty:
        failures.append("No available SOH score rows.")
    else:
        r2 = pd.to_numeric(valid["R2"], errors="coerce").to_numpy(dtype=float)
        mae = pd.to_numeric(valid["MAE"], errors="coerce").to_numpy(dtype=float)
        summary.update(R2_min=float(np.nanmin(r2)), R2_mean=float(np.nanmean(r2)), R2_std=float(np.nanstd(r2)), MAE_mean=float(np.nanmean(mae)), MAE_max=float(np.nanmax(mae)))
        if np.any(r2 < float(args.target_r2)):
            failures.append("At least one seed is below target_r2.")
        if float(summary["R2_std"]) > float(args.max_r2_std):
            failures.append(f"R2_std={summary['R2_std']:.6g} exceeds max_r2_std={args.max_r2_std}")
        if float(summary["MAE_mean"]) > float(args.max_mae_mean):
            failures.append(f"MAE_mean={summary['MAE_mean']:.6g} exceeds max_mae_mean={args.max_mae_mean}")
        if args.require_no_clamp and "active_clamp_count_all" in valid.columns:
            clamp = pd.to_numeric(valid["active_clamp_count_all"], errors="coerce").fillna(0).to_numpy(dtype=float)
            if np.any(clamp > 0):
                failures.append("At least one seed has active clamp hits.")
    summary["ok"] = len(failures) == 0
    summary["failures"] = failures
    with (out / "seed_stability_summary.json").open("w", encoding="utf-8") as f:
        json.dump(_json_clean(summary), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
