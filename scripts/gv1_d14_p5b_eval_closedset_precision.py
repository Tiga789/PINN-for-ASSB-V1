#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate D14-P5B 8-cell closed-set precision model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels_nn.xjtu_p2dlite_closedset_dataset import (
    read_json,
    write_json,
    write_csv,
    load_profile_sample,
    stats_from_json,
)
from gv1.softlabels_nn.xjtu_p2dlite_closedset_model import build_model
from gv1.softlabels_nn.xjtu_p2dlite_closedset_metrics import profile_metrics, aggregate, by_group


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


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def predict_profile(model, sample, stats, device, amp: bool, batch_size: int = 65536):
    X_np = sample["X"]
    out_parts = []
    model.eval()
    for start in range(0, X_np.shape[0], batch_size):
        X = torch.from_numpy(X_np[start:start + batch_size]).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            pred = model(X)
        out_parts.append({k: v.detach().cpu().numpy() for k, v in pred.items()})
    theta_a = np.concatenate([p["theta_a"] for p in out_parts], axis=0)
    theta_c = np.concatenate([p["theta_c"] for p in out_parts], axis=0)
    phie = np.concatenate([p["phie_norm"] for p in out_parts], axis=0).reshape(-1) * stats.phie_std + stats.phie_mean
    phis_c = np.concatenate([p["phis_c_norm"] for p in out_parts], axis=0).reshape(-1) * stats.phis_c_std + stats.phis_c_mean
    return {
        "theta_a": theta_a.astype(np.float32),
        "theta_c": theta_c.astype(np.float32),
        "phie": phie.astype(np.float32),
        "phis_c": phis_c.astype(np.float32),
        "cs_a": (theta_a * stats.cmax_a_est).astype(np.float32),
        "cs_c": (theta_c * stats.cmax_c_est).astype(np.float32),
    }


def decide_status(global_metrics: dict, profile_rows: list, cfg: dict) -> tuple[str, list, list]:
    thresholds = cfg.get("eval", {}).get("pass_thresholds", {})
    warn = []
    fail = []
    phis_mae = float(global_metrics.get("mean_phis_c_mae", float("nan")))
    phie_mae = float(global_metrics.get("mean_phie_mae", float("nan")))
    theta_mae = float(global_metrics.get("mean_theta_mean_mae", float("nan")))

    if not np.isfinite(phis_mae) or not np.isfinite(phie_mae) or not np.isfinite(theta_mae):
        fail.append("nonfinite_global_metrics")
    if phis_mae > float(thresholds.get("closed_phis_c_mae_warn_V", 0.025)):
        warn.append("closed_phis_c_mae_above_warn")
    if phie_mae > float(thresholds.get("closed_phie_mae_warn", 0.025)):
        warn.append("closed_phie_mae_above_warn")
    if theta_mae > float(thresholds.get("closed_theta_mean_mae_warn", 0.035)):
        warn.append("closed_theta_mean_mae_above_warn")

    bad_profile_phis = [r["cell_uid"] for r in profile_rows if float(r.get("phis_c_mae", 0)) > float(thresholds.get("profile_phis_c_mae_warn_V", 0.035))]
    bad_profile_theta = [r["cell_uid"] for r in profile_rows if float(r.get("theta_mean_mae", 0)) > float(thresholds.get("profile_theta_mean_mae_warn", 0.050))]
    if bad_profile_phis:
        warn.append("profile_phis_c_mae_warn:" + ",".join(bad_profile_phis[:8]))
    if bad_profile_theta:
        warn.append("profile_theta_mean_mae_warn:" + ",".join(bad_profile_theta[:8]))

    status = "FAIL" if fail else ("WARN" if warn else "PASS")
    return status, warn, fail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_warn", action="store_true")
    args = parser.parse_args()

    cfg = read_json(Path(args.config))
    out = Path(args.output_dir)
    eval_dir = out / "EvalFin_D14_P5B_8cell_closedset_precision"
    pred_root = eval_dir / "predictions"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    checkpoint = model_dir / "best.pt"
    stats_path = model_dir / "feature_stats.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing feature_stats: {stats_path}")

    stats = stats_from_json(stats_path)
    device = choose_device(str(cfg.get("training", {}).get("device", "auto")))
    amp = bool(cfg.get("training", {}).get("amp", True)) and device.type == "cuda"

    ckpt = safe_torch_load(checkpoint, device)
    model = build_model(stats.feature_dim, stats.n_r, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    rows = [r for r in load_manifest(Path(args.manifest_csv)) if r.get("status", "PASS") != "FAIL"]
    profile_rows = []
    for i, row in enumerate(rows):
        sample = load_profile_sample(row, cfg, stats=stats, profile_index=i, profile_ids=stats.profile_ids)
        pred = predict_profile(model, sample, stats, device, amp=amp, batch_size=int(cfg.get("training", {}).get("batch_size", 65536)))
        true = {
            "theta_a": sample["arrays"]["theta_a"],
            "theta_c": sample["arrays"]["theta_c"],
            "phie": sample["arrays"]["phie"],
            "phis_c": sample["arrays"]["phis_c"],
            "cs_a": sample["arrays"]["cs_a"],
            "cs_c": sample["arrays"]["cs_c"],
        }
        m = profile_metrics(row["cell_uid"], row.get("batch", ""), row.get("protocol", ""), pred, true)
        m["softlabel_npz"] = row.get("softlabel_npz", "")
        profile_rows.append(m)

        if cfg.get("eval", {}).get("save_prediction_npz", True):
            pdir = pred_root / row["cell_uid"]
            pdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pdir / "prediction_sampled.npz",
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
                batch=row.get("batch", ""),
                protocol=row.get("protocol", ""),
                cell_uid=row["cell_uid"],
            )

    global_metrics = aggregate(profile_rows)
    batch_metrics = by_group(profile_rows, "batch")
    protocol_metrics = by_group(profile_rows, "protocol")
    write_csv(eval_dir / "metrics_by_profile.csv", profile_rows)
    write_csv(eval_dir / "metrics_by_batch.csv", batch_metrics)
    write_csv(eval_dir / "metrics_by_protocol.csv", protocol_metrics)
    write_json(eval_dir / "metrics_global.json", global_metrics)

    status, warn, fail = decide_status(global_metrics, profile_rows, cfg)
    report = {
        "package": "D14-P5B XJTU P2Dlite 8-cell closed-set precision evaluation",
        "overall_status": status,
        "warn_reasons": warn,
        "fail_reasons": fail,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "eval_dir": str(eval_dir),
        "profile_count": len(profile_rows),
        "global_metrics": global_metrics,
        "batch_metrics": batch_metrics,
        "protocol_metrics": protocol_metrics,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(eval_dir / "D14_P5B_EVAL_REPORT.json", report)
    print(f"[P5B eval] status={status} profiles={len(profile_rows)} phis_c_mae={global_metrics.get('mean_phis_c_mae')} theta_mean_mae={global_metrics.get('mean_theta_mean_mae')}")
    if status == "FAIL":
        return 1
    if status == "WARN" and not args.allow_warn:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
