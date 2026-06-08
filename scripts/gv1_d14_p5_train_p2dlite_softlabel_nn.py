#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train D14-P5 XJTU P2Dlite soft-label NN smoke model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from gv1.softlabels_nn.xjtu_p2dlite_dataset import (
    read_json,
    write_json,
    write_csv,
    load_profile_sample,
    estimate_feature_stats,
    P2DliteTensorDataset,
)
from gv1.softlabels_nn.xjtu_p2dlite_model import build_model_from_config
from gv1.softlabels_nn.xjtu_p2dlite_losses import supervised_loss


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


def load_manifest(path: str | Path) -> List[dict]:
    import csv
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def eval_loader(model, loader, cfg, device):
    model.eval()
    losses = []
    parts = {"loss_theta_a": [], "loss_theta_c": [], "loss_phie": [], "loss_phis_c": []}
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["X"])
        loss_dict = supervised_loss(pred, batch, cfg)
        losses.append(float(loss_dict["loss"].detach().cpu()))
        for k in parts:
            parts[k].append(float(loss_dict[k].detach().cpu()))
    out = {"loss": float(np.mean(losses)) if losses else float("nan")}
    for k, vals in parts.items():
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


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

    cfg = read_json(Path(args.config))
    train_cfg = cfg.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(str(train_cfg.get("device", "auto")))
    output_dir = Path(args.output_dir)
    model_dir = output_dir / "ModelFin_D14_P5_p2dlite_nn_smoke"
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(args.manifest_csv)
    train_rows = [r for r in rows if r.get("split") == "train" and r.get("status", "PASS") != "FAIL"]
    val_rows = [r for r in rows if r.get("split") == "val" and r.get("status", "PASS") != "FAIL"]
    if not train_rows or not val_rows:
        raise RuntimeError(f"Need nonempty train and val rows. train={len(train_rows)} val={len(val_rows)}")

    # First load train samples without stats, estimate stats, then reload all with stats.
    train_samples_raw = [load_profile_sample(r, cfg, stats=None) for r in train_rows]
    prior_hash = sorted(set(r.get("prior_hash", "") for r in train_rows if r.get("prior_hash", "")))[0]
    stats = estimate_feature_stats(train_samples_raw, prior_hash=prior_hash)
    write_json(model_dir / "feature_stats.json", stats.to_dict())

    train_samples = [load_profile_sample(r, cfg, stats=stats) for r in train_rows]
    val_samples = [load_profile_sample(r, cfg, stats=stats) for r in val_rows]

    train_ds = P2DliteTensorDataset(train_samples, stats)
    val_ds = P2DliteTensorDataset(val_samples, stats)

    batch_size = int(args.batch_size or train_cfg.get("batch_size", 2048))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = build_model_from_config(stats.feature_dim, stats.n_r, cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )
    epochs = int(args.epochs or train_cfg.get("epochs", 120))
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))

    history = []
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        parts = {"loss_theta_a": [], "loss_theta_c": [], "loss_phie": [], "loss_phis_c": []}
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            pred = model(batch["X"])
            loss_dict = supervised_loss(pred, batch, cfg)
            loss = loss_dict["loss"]
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
            for k in parts:
                parts[k].append(float(loss_dict[k].detach().cpu()))

        val_metrics = eval_loader(model, val_loader, cfg, device)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "train_loss_theta_a": float(np.mean(parts["loss_theta_a"])) if parts["loss_theta_a"] else float("nan"),
            "train_loss_theta_c": float(np.mean(parts["loss_theta_c"])) if parts["loss_theta_c"] else float("nan"),
            "train_loss_phie": float(np.mean(parts["loss_phie"])) if parts["loss_phie"] else float("nan"),
            "train_loss_phis_c": float(np.mean(parts["loss_phis_c"])) if parts["loss_phis_c"] else float("nan"),
            "val_loss_theta_a": val_metrics["loss_theta_a"],
            "val_loss_theta_c": val_metrics["loss_theta_c"],
            "val_loss_phie": val_metrics["loss_phie"],
            "val_loss_phis_c": val_metrics["loss_phis_c"],
        }
        history.append(row)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model.config_dict(),
                "feature_stats": stats.to_dict(),
                "config": cfg,
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
            }, model_dir / "best.pt")

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f"[P5 train] epoch={epoch}/{epochs} train_loss={train_loss:.6g} val_loss={val_metrics['loss']:.6g}", flush=True)

    write_csv(model_dir / "loss_history.csv", history)
    summary = {
        "status": "PASS",
        "model_dir": str(model_dir),
        "device": str(device),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "train_points": len(train_ds),
        "val_points": len(val_ds),
        "train_profiles": [r["cell_uid"] for r in train_rows],
        "val_profiles": [r["cell_uid"] for r in val_rows],
        "feature_stats": stats.to_dict(),
        "checkpoint": str(model_dir / "best.pt"),
        "boundaries": cfg.get("boundaries", {}),
    }
    write_json(model_dir / "training_summary.json", summary)
    print(f"[P5 train] done best_epoch={best_epoch} best_val_loss={best_val:.6g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
