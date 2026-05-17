# -*- coding: utf-8 -*-
"""Diagnose why ASSB-111 SOH seeds differ.

Reads model directories and optional evaluation directories, then summarizes:
- visible train/val fit quality from training summaries/history;
- learned floor/soh0 from soh_pred_by_cycle.csv or config;
- held-out test SOH metrics when an eval directory is provided;
- whether a seed should be rejected before looking at test metrics.

This script is intended for debugging and reporting. It does not train a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        val = float(x)
        return None if not np.isfinite(val) else val
    if isinstance(x, float):
        return None if not np.isfinite(x) else x
    return x


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _scorecard_soh(eval_dir: Optional[Path]) -> Dict[str, Any]:
    if eval_dir is None:
        return {}
    p = eval_dir / "five_state_scorecard.csv"
    if not p.exists():
        return {"eval_available": False, "eval_failure": f"missing {p}"}
    df = pd.read_csv(p)
    if "variable" not in df.columns:
        return {"eval_available": False, "eval_failure": "no variable column"}
    row = df[df["variable"].astype(str).str.upper() == "SOH"]
    if row.empty:
        return {"eval_available": False, "eval_failure": "SOH row missing"}
    d = row.iloc[0].to_dict()
    return {"eval_available": True, **{f"test_{k}": _safe_float(v) if k in {"MAE", "RMSE", "NMAE", "NRMSE", "R2", "corr"} else v for k, v in d.items()}}


def _pred_split_metrics(model_dir: Path) -> Dict[str, Any]:
    p = model_dir / "soh_pred_by_cycle.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"].astype(str).str.lower() == split]
        if sub.empty:
            continue
        obs = pd.to_numeric(sub.get("SOH_obs"), errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(sub.get("SOH_pred"), errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(obs) & np.isfinite(pred)
        if not mask.any():
            continue
        err = pred[mask] - obs[mask]
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((obs[mask] - np.mean(obs[mask])) ** 2))
        out[f"{split}_mae_from_pred_csv"] = float(np.mean(np.abs(err)))
        out[f"{split}_rmse_from_pred_csv"] = float(np.sqrt(np.mean(err ** 2)))
        out[f"{split}_r2_from_pred_csv"] = float("nan") if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot
        out[f"{split}_pred_min"] = float(np.min(pred[mask]))
        out[f"{split}_pred_max"] = float(np.max(pred[mask]))
    for col in ["soh_floor", "soh0"]:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if arr.size:
                out[f"{col}_median"] = float(np.median(arr))
                out[f"{col}_min"] = float(np.min(arr))
                out[f"{col}_max"] = float(np.max(arr))
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dirs", nargs="+", required=True)
    p.add_argument("--eval_dirs", nargs="*", default=[])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--min_train_r2", type=float, default=0.985)
    p.add_argument("--max_train_mae", type=float, default=0.004)
    p.add_argument("--max_val_mae", type=float, default=0.0025)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dirs = [Path(x) for x in args.eval_dirs]
    rows: List[Dict[str, Any]] = []
    for i, m_text in enumerate(args.model_dirs):
        model_dir = Path(m_text)
        row: Dict[str, Any] = {"model_dir": str(model_dir), "model_exists": model_dir.exists()}
        cfg = _load_json(model_dir / "soh_head_config.json")
        train_summary = _load_json(model_dir / "train_summary.json") or _load_json(model_dir / "training_summary.json")
        row["config_model_variant"] = cfg.get("config", {}).get("model_variant")
        row["config_seed"] = cfg.get("extra", {}).get("seed")
        row["best_epoch_config"] = cfg.get("extra", {}).get("best_epoch")
        row["best_val_mae_config"] = cfg.get("extra", {}).get("best_val_mae")
        row["train_summary_present"] = bool(train_summary)
        if train_summary:
            row["best_epoch_summary"] = train_summary.get("best_epoch")
            row["best_val_mae_summary"] = train_summary.get("best_val_mae")
            row["leakage_ok"] = train_summary.get("leakage_ok")
        row.update(_pred_split_metrics(model_dir))
        if i < len(eval_dirs):
            row.update(_scorecard_soh(eval_dirs[i]))
        train_r2 = _safe_float(row.get("train_r2_from_pred_csv"))
        train_mae = _safe_float(row.get("train_mae_from_pred_csv"))
        val_mae = _safe_float(row.get("val_mae_from_pred_csv"))
        visible_ok = True
        reasons: List[str] = []
        if np.isfinite(train_r2) and train_r2 < float(args.min_train_r2):
            visible_ok = False
            reasons.append(f"train_r2 {train_r2:.6g} < {args.min_train_r2}")
        if np.isfinite(train_mae) and train_mae > float(args.max_train_mae):
            visible_ok = False
            reasons.append(f"train_mae {train_mae:.6g} > {args.max_train_mae}")
        if np.isfinite(val_mae) and val_mae > float(args.max_val_mae):
            visible_ok = False
            reasons.append(f"val_mae {val_mae:.6g} > {args.max_val_mae}")
        row["visible_guard_ok"] = visible_ok
        row["visible_guard_failures"] = "; ".join(reasons)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "saturating_v3_stability_diagnostic.csv"
    json_path = out_dir / "saturating_v3_stability_diagnostic.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {
        "csv": str(csv_path),
        "n_models": len(rows),
        "visible_guard_pass_count": int(df.get("visible_guard_ok", pd.Series(dtype=bool)).fillna(False).sum()) if len(df) else 0,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(summary), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(_json_clean(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
