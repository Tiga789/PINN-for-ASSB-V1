#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D14-P5A robust evaluator for XJTU P2Dlite soft-label NN smoke.

This replaces the D14-P5 evaluator's fragile split aggregation. It can:
  1. perform a full re-evaluation from checkpoint and manifest; or
  2. repair missing `metrics_by_split.csv` and `D14_P5_EVAL_REPORT.json` from
     an existing `metrics_by_profile.csv` if predictions already exist.

The repair mode is useful when the original D14-P5 run produced all
`prediction_sampled.npz` files and `metrics_by_profile.csv`, but crashed while
aggregating by split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels_nn.xjtu_p2dlite_dataset import (
    read_json,
    write_json,
    write_csv,
    load_profile_sample,
    stats_from_json,
)
from gv1.softlabels_nn.xjtu_p2dlite_model import build_model_from_config
from gv1.softlabels_nn.xjtu_p2dlite_metrics import compact_profile_metrics, aggregate_metrics


def load_manifest(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def predict_profile(model, sample, stats, device):
    X = torch.from_numpy(sample["X"]).to(device)
    preds = []
    batch_size = 8192
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            out = model(X[start:start + batch_size])
            preds.append({k: v.detach().cpu().numpy() for k, v in out.items()})
    theta_a = np.concatenate([p["theta_a"] for p in preds], axis=0)
    theta_c = np.concatenate([p["theta_c"] for p in preds], axis=0)
    phie = np.concatenate([p["phie_norm"] for p in preds], axis=0).reshape(-1) * stats.phie_std + stats.phie_mean
    phis_c = np.concatenate([p["phis_c_norm"] for p in preds], axis=0).reshape(-1) * stats.phis_c_std + stats.phis_c_mean
    return {
        "theta_a": theta_a.astype(np.float32),
        "theta_c": theta_c.astype(np.float32),
        "phie": phie.astype(np.float32),
        "phis_c": phis_c.astype(np.float32),
        "cs_a": (theta_a * stats.cmax_a_est).astype(np.float32),
        "cs_c": (theta_c * stats.cmax_c_est).astype(np.float32),
    }


def status_from_metrics(global_rows: List[dict], cfg: dict) -> tuple[str, list, list]:
    thresholds = cfg.get("eval", {}).get("pass_thresholds", {})
    warn_reasons = []
    fail_reasons = []
    by_split = {r["split"]: r for r in global_rows}
    for required in ["train", "val", "test"]:
        if required not in by_split:
            fail_reasons.append(f"missing_{required}_metrics")

    def split_val(split, key, default=np.nan):
        try:
            return float(by_split.get(split, {}).get(key, default))
        except Exception:
            return default

    if split_val("train", "mean_phis_c_mae") > float(thresholds.get("train_phis_c_mae_warn_V", 0.12)):
        warn_reasons.append("train_phis_c_mae_high")
    if split_val("val", "mean_phis_c_mae") > float(thresholds.get("val_phis_c_mae_warn_V", 0.16)):
        warn_reasons.append("val_phis_c_mae_high")
    if split_val("test", "mean_phis_c_mae") > float(thresholds.get("test_phis_c_mae_warn_V", 0.22)):
        warn_reasons.append("test_phis_c_mae_high")
    if split_val("val", "mean_theta_mean_mae") > float(thresholds.get("val_theta_mean_mae_warn", 0.10)):
        warn_reasons.append("val_theta_mean_mae_high")
    if split_val("test", "mean_theta_mean_mae") > float(thresholds.get("test_theta_mean_mae_warn", 0.14)):
        warn_reasons.append("test_theta_mean_mae_high")

    status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    return status, warn_reasons, fail_reasons


def write_eval_report(eval_dir: Path, model_dir: Path, checkpoint_path: Path, metrics_rows: List[dict], cfg: dict, mode: str) -> dict:
    global_rows = aggregate_metrics(metrics_rows)
    write_csv(eval_dir / "metrics_by_split.csv", global_rows)

    status, warn_reasons, fail_reasons = status_from_metrics(global_rows, cfg)
    report = {
        "package": "D14-P5A XJTU P2Dlite soft-label NN smoke evaluation fix",
        "overall_status": status,
        "mode": mode,
        "warn_reasons": warn_reasons,
        "fail_reasons": fail_reasons,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "eval_dir": str(eval_dir),
        "metrics_by_profile": str(eval_dir / "metrics_by_profile.csv"),
        "metrics_by_split": str(eval_dir / "metrics_by_split.csv"),
        "global_metrics": global_rows,
        "profile_count": len(metrics_rows),
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(eval_dir / "D14_P5_EVAL_REPORT.json", report)
    return report


def repair_from_existing(eval_dir: Path, model_dir: Path, checkpoint_path: Path, cfg: dict) -> dict:
    metrics_path = eval_dir / "metrics_by_profile.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Cannot repair: missing {metrics_path}")
    metrics_rows = read_csv_rows(metrics_path)
    return write_eval_report(eval_dir, model_dir, checkpoint_path, metrics_rows, cfg, mode="repair_from_existing_metrics_by_profile")


def full_eval(project_root: Path, cfg: dict, manifest_csv: Path, model_dir: Path, output_dir: Path) -> dict:
    eval_dir = output_dir / "EvalFin_D14_P5_p2dlite_nn_smoke"
    eval_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best.pt"
    stats_path = model_dir / "feature_stats.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing feature stats: {stats_path}")

    stats = stats_from_json(stats_path)
    device = choose_device(str(cfg.get("training", {}).get("device", "auto")))
    ckpt = safe_torch_load(checkpoint_path, device)
    model = build_model_from_config(stats.feature_dim, stats.n_r, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    rows = load_manifest(manifest_csv)
    metrics_rows = []
    prediction_root = eval_dir / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        if row.get("status", "PASS") == "FAIL":
            continue
        sample = load_profile_sample(row, cfg, stats=stats)
        pred = predict_profile(model, sample, stats, device)
        true = {
            "theta_a": sample["arrays"]["theta_a"],
            "theta_c": sample["arrays"]["theta_c"],
            "phie": sample["arrays"]["phie"],
            "phis_c": sample["arrays"]["phis_c"],
            "cs_a": sample["arrays"]["cs_a"],
            "cs_c": sample["arrays"]["cs_c"],
        }
        m = compact_profile_metrics(row["cell_uid"], row["split"], pred, true)
        m.update({"batch": row.get("batch", ""), "protocol": row.get("protocol", ""), "softlabel_npz": row.get("softlabel_npz", "")})
        metrics_rows.append(m)

        if cfg.get("eval", {}).get("save_prediction_npz", True):
            out_dir = prediction_root / row["cell_uid"]
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / "prediction_sampled.npz",
                t_global_s=sample["arrays"]["t_global_s"],
                I_profile=sample["arrays"]["I_profile"],
                voltage_exp=sample["arrays"]["voltage_exp"],
                theta_a_true=true["theta_a"],
                theta_a_pred=pred["theta_a"],
                theta_c_true=true["theta_c"],
                theta_c_pred=pred["theta_c"],
                cs_a_true=true["cs_a"],
                cs_a_pred=pred["cs_a"],
                cs_c_true=true["cs_c"],
                cs_c_pred=pred["cs_c"],
                phie_true=true["phie"],
                phie_pred=pred["phie"],
                phis_c_true=true["phis_c"],
                phis_c_pred=pred["phis_c"],
                split=row["split"],
                batch=row.get("batch", ""),
                protocol=row.get("protocol", ""),
                cell_uid=row["cell_uid"],
            )

    write_csv(eval_dir / "metrics_by_profile.csv", metrics_rows)
    return write_eval_report(eval_dir, model_dir, checkpoint_path, metrics_rows, cfg, mode="full_eval")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--repair_only", action="store_true")
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root)
    cfg = read_json(Path(args.config))
    output_dir = Path(args.output_dir)
    eval_dir = output_dir / "EvalFin_D14_P5_p2dlite_nn_smoke"
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "best.pt"

    try:
        if args.repair_only:
            report = repair_from_existing(eval_dir, model_dir, checkpoint_path, cfg)
        else:
            report = full_eval(project_root, cfg, Path(args.manifest_csv), model_dir, output_dir)
    except Exception as exc:
        err_report = {
            "package": "D14-P5A XJTU P2Dlite soft-label NN smoke evaluation fix",
            "overall_status": "FAIL",
            "mode": "full_eval" if not args.repair_only else "repair_only",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "eval_dir": str(eval_dir),
        }
        write_json(eval_dir / "D14_P5_EVAL_REPORT.json", err_report)
        print(f"[P5A eval] FAIL {err_report['error']}")
        return 1

    print(f"[P5A eval] status={report['overall_status']} mode={report['mode']} profile_count={report['profile_count']}")
    if report["overall_status"] == "FAIL":
        return 1
    if report["overall_status"] == "WARN" and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
