#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train D14-P5B-v2 8-cell closed-set precision model.

Stability changes for GTX 1080 Ti / Pascal:
  - torch.compile disabled by default;
  - AMP disabled by default because GTX 1080 Ti has no tensor cores;
  - GPU-resident tensors remain enabled;
  - large batch remains enabled;
  - write training_failed.json if training crashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

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
    estimate_stats,
    concatenate_tensors,
    memory_summary,
)
from gv1.softlabels_nn.xjtu_p2dlite_closedset_model import build_model
from gv1.softlabels_nn.xjtu_p2dlite_closedset_losses import closedset_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(mode: str) -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def batch_slice(tensors: Dict[str, torch.Tensor], idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in tensors.items() if k != "profile_id"}


@torch.no_grad()
def eval_full(model, tensors: Dict[str, torch.Tensor], cfg: dict, batch_size: int, device: torch.device, amp: bool):
    model.eval()
    n = tensors["X"].shape[0]
    losses = []
    parts = {"loss_theta_a": [], "loss_theta_c": [], "loss_phie": [], "loss_phis_c": [], "loss_surface": [], "loss_shape": []}
    for start in range(0, n, batch_size):
        sl = slice(start, min(n, start + batch_size))
        batch = {k: v[sl] for k, v in tensors.items() if k != "profile_id"}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            pred = model(batch["X"])
            loss_dict = closedset_loss(pred, batch, cfg)
        losses.append(float(loss_dict["loss"].detach().cpu()))
        for k in parts:
            parts[k].append(float(loss_dict[k].detach().cpu()))
    out = {"loss": float(np.mean(losses)) if losses else float("nan")}
    for k, vals in parts.items():
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_training(args) -> int:
    cfg = read_json(Path(args.config))
    train_cfg = cfg.get("training", {})
    set_seed(int(train_cfg.get("seed", 42)))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    device = choose_device(str(train_cfg.get("device", "auto")))
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    gpu_resident = bool(train_cfg.get("gpu_resident_tensors", True)) and device.type == "cuda"

    out = Path(args.output_dir)
    model_dir = out / "ModelFin_D14_P5B_8cell_closedset_precision"
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_manifest(Path(args.manifest_csv)) if r.get("status", "PASS") != "FAIL"]
    profile_ids = [r["cell_uid"] for r in rows]
    expected = int(cfg.get("profile_policy", {}).get("expected_profile_count", 8))
    if len(rows) < expected:
        raise RuntimeError(f"Expected at least {expected} profiles for P5B closed-set, got {len(rows)}")

    raw_samples = [load_profile_sample(r, cfg, stats=None, profile_index=i, profile_ids=profile_ids) for i, r in enumerate(rows)]
    prior_hashes = sorted(set(r.get("prior_hash", "") for r in rows if r.get("prior_hash", "")))
    prior_hash = prior_hashes[0] if prior_hashes else ""
    stats = estimate_stats(raw_samples, prior_hash=prior_hash, profile_ids=profile_ids)
    write_json(model_dir / "feature_stats.json", stats.to_dict())

    samples = [load_profile_sample(r, cfg, stats=stats, profile_index=i, profile_ids=profile_ids) for i, r in enumerate(rows)]
    tensors = concatenate_tensors(samples, stats, device=device, gpu_resident=gpu_resident)
    write_json(model_dir / "tensor_memory_summary.json", memory_summary(tensors))

    model = build_model(stats.feature_dim, stats.n_r, cfg).to(device)
    compile_mode = str(train_cfg.get("torch_compile", "false")).lower()
    compiled = False
    if device.type == "cuda" and compile_mode in {"true", "1", "yes"} and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            compiled = True
        except Exception as exc:
            print(f"[P5B-v2 train] torch.compile skipped: {type(exc).__name__}: {exc}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 8e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-7)),
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    epochs = int(args.epochs or train_cfg.get("epochs", 500))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 65536))
    grad_clip = float(train_cfg.get("grad_clip_norm", 10.0))
    log_every = int(train_cfg.get("log_every_epochs", 10))

    n = tensors["X"].shape[0]
    indices = torch.arange(n, device=device if gpu_resident else torch.device("cpu"))
    history = []
    best_loss = float("inf")
    best_epoch = -1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        print(f"[P5B-v2 train] device={torch.cuda.get_device_name(0)} amp={amp} gpu_resident={gpu_resident} compiled={compiled}", flush=True)
    else:
        print(f"[P5B-v2 train] device=cpu amp={amp} gpu_resident={gpu_resident}", flush=True)
    print(f"[P5B-v2 train] points={n} batch_size={batch_size} steps_per_epoch={int(np.ceil(n / batch_size))}", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = indices[torch.randperm(n, device=indices.device)]
        losses = []
        parts = {"loss_theta_a": [], "loss_theta_c": [], "loss_phie": [], "loss_phis_c": [], "loss_surface": [], "loss_shape": []}

        for start in range(0, n, batch_size):
            idx = perm[start:min(n, start + batch_size)]
            if not gpu_resident:
                idx_cpu = idx.cpu()
                batch = {k: v[idx_cpu].to(device, non_blocking=True) for k, v in tensors.items() if k != "profile_id"}
            else:
                batch = batch_slice(tensors, idx)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                pred = model(batch["X"])
                loss_dict = closedset_loss(pred, batch, cfg)
                loss = loss_dict["loss"]
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            for k in parts:
                parts[k].append(float(loss_dict[k].detach().cpu()))

        eval_metrics = eval_full(model, tensors, cfg, batch_size=batch_size, device=device, amp=amp)
        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "closed_eval_loss": eval_metrics["loss"],
            "epoch_time_s": epoch_time,
            "points_per_s": float(n / max(epoch_time, 1e-6)),
        }
        for k, vals in parts.items():
            row[f"train_{k}"] = float(np.mean(vals)) if vals else float("nan")
        for k, v in eval_metrics.items():
            row[f"eval_{k}"] = v
        if device.type == "cuda":
            row["cuda_max_memory_allocated_MB"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 3)
            row["cuda_memory_reserved_MB"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 3)
        history.append(row)

        if eval_metrics["loss"] < best_loss:
            best_loss = eval_metrics["loss"]
            best_epoch = epoch
            state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                "model_state_dict": state_model.state_dict(),
                "model_config": state_model.config_dict(),
                "feature_stats": stats.to_dict(),
                "config": cfg,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
            }, model_dir / "best.pt")

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"[P5B-v2 train] epoch={epoch}/{epochs} loss={row['train_loss']:.6g} "
                f"eval={eval_metrics['loss']:.6g} pts/s={row['points_per_s']:.0f} time={epoch_time:.2f}s",
                flush=True,
            )

    write_csv(model_dir / "loss_history.csv", history)
    summary = {
        "status": "PASS",
        "model_dir": str(model_dir),
        "device": str(device),
        "compiled": compiled,
        "amp": amp,
        "gpu_resident_tensors": gpu_resident,
        "epochs": epochs,
        "batch_size": batch_size,
        "point_count": int(n),
        "profile_count": len(rows),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "checkpoint": str(model_dir / "best.pt"),
        "feature_stats": stats.to_dict(),
        "profiles": rows,
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(model_dir / "training_summary.json", summary)
    print(f"[P5B-v2 train] done best_epoch={best_epoch} best_loss={best_loss:.6g}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--allow_warn", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.output_dir) / "ModelFin_D14_P5B_8cell_closedset_precision"
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        return run_training(args)
    except Exception as exc:
        fail = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=12),
            "hint": "If CUDA OOM, retry with -BatchSize 32768. If this mentions compile/triton/inductor, P5B-v2 already disables compile by default.",
        }
        write_json(model_dir / "training_failed.json", fail)
        print(f"[P5B-v2 train] FAIL {fail['error']}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
