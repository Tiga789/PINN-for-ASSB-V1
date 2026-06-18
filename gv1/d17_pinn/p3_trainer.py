# -*- coding: utf-8 -*-
"""D17-P3 6-profile mechanism smoke trainer."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .config import cfg_get
from .dataset import D17ProfileDataset, load_observed_profile
from .latent_adapter import LATENT_NAMES
from .losses import audit_numbers, total_d17_loss
from .model import D17MechanisticPINN, make_batch_from_profile
from .p2dlite_prior import load_p2dlite_prior, prior_to_jsonable
from .trainer import FORBIDDEN_PROFILE_KEYS, assert_no_state_profile_keys, choose_device, crop_time_window, set_seed


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Mapping): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    return x


def normalize_protocol(record: Mapping[str, Any]) -> str:
    raw = str(record.get("protocol", "") or "").strip()
    cid = str(record.get("canonical_cell_uid", record.get("cell_uid", "")) or "")
    text = (raw + " " + cid).replace("\\", "/")
    low = text.lower()
    if "r2.5" in low: return "R2.5"
    if "r3" in low: return "R3"
    if "2c" in low: return "2C"
    if "batch-2" in low or "3c" in low: return "3C"
    if "batch-5" in low or "random" in low: return "RW"
    if "batch-6" in low or "geo" in low: return "GEO"
    return raw if raw and raw != "protocol-UNKNOWN" else "UNKNOWN"


def select_balanced_records(ds: D17ProfileDataset, profile_count: int = 6, preferred_protocols: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    preferred = list(preferred_protocols or ["2C", "R2.5", "R3", "3C", "RW", "GEO", "UNKNOWN"])
    buckets: Dict[str, List[Dict[str, Any]]] = {p: [] for p in preferred}
    for rec in ds.records:
        if rec.get("is_flagged_probe") or rec.get("split") == "flagged_probe":
            continue
        if not rec.get("replay_npz"):
            continue
        proto = normalize_protocol(rec)
        buckets.setdefault(proto, []).append(rec)
    for vals in buckets.values():
        vals.sort(key=lambda r: str(r.get("canonical_cell_uid", r.get("cell_uid", ""))))
    selected: List[Dict[str, Any]] = []
    while len(selected) < profile_count:
        added = False
        for proto in preferred:
            if len(selected) >= profile_count:
                break
            vals = buckets.get(proto, [])
            if vals:
                selected.append(vals.pop(0))
                added = True
        if not added:
            break
    if len(selected) < profile_count:
        rest: List[Dict[str, Any]] = []
        used = {id(x) for x in selected}
        for vals in buckets.values():
            for r in vals:
                if id(r) not in used:
                    rest.append(r)
        rest.sort(key=lambda r: str(r.get("canonical_cell_uid", r.get("cell_uid", ""))))
        selected.extend(rest[: max(0, profile_count - len(selected))])
    if len(selected) < profile_count:
        raise RuntimeError(f"Only selected {len(selected)} replay-ready profiles; requested {profile_count}")
    return selected[:profile_count]


def load_profiles(records: Sequence[Mapping[str, Any]], time_window_s: float, max_time_points: int) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        replay_npz = rec.get("replay_npz")
        if not replay_npz:
            raise RuntimeError(f"Selected record lacks replay_npz: {rec}")
        profile = load_observed_profile(replay_npz)
        assert_no_state_profile_keys(profile)
        profile = crop_time_window(profile, time_window_s=time_window_s, max_time_points=max_time_points)
        profile["_manifest_record"] = dict(rec)
        profile["_p3_profile_index"] = i
        profiles.append(profile)
    return profiles


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    numeric_keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and math.isfinite(float(v))})
    out: Dict[str, float] = {}
    for k in numeric_keys:
        vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], (int, float)) and math.isfinite(float(r[k]))]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_max"] = float(np.max(vals))
            out[f"{k}_min"] = float(np.min(vals))
    return out


def train_p3_mechanism_smoke(cfg: Mapping[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    seed = int(cfg_get(cfg, "seed", 20260615))
    set_seed(seed)
    device = choose_device(str(cfg_get(cfg, "train.device", "auto")))
    split_manifest = cfg_get(cfg, "paths.split_manifest")
    resolved_spec = cfg_get(cfg, "paths.resolved_spec")
    split = str(cfg_get(cfg, "train.split", "train"))
    profile_count = int(cfg_get(cfg, "train.profile_count", 6))
    time_window_s = float(cfg_get(cfg, "train.time_window_s", 40000.0))
    max_time_points = int(cfg_get(cfg, "train.max_time_points", 512))
    n_r = int(cfg_get(cfg, "train.n_r", 17))

    ds = D17ProfileDataset(split_manifest=split_manifest, split=split, allow_softlabel_npz_profile_source=False)
    selected_records = select_balanced_records(ds, profile_count=profile_count)
    profiles = load_profiles(selected_records, time_window_s=time_window_s, max_time_points=max_time_points)
    prior = load_p2dlite_prior(resolved_spec, allow_smoke_defaults=True)
    batches = [make_batch_from_profile(p, n_r=n_r, device=device) for p in profiles]

    feature_dim = int(batches[0]["features"].shape[-1])
    model = D17MechanisticPINN(
        prior=prior,
        feature_dim=feature_dim,
        n_r=n_r,
        hidden_dim=int(cfg_get(cfg, "model.hidden_dim", 64)),
        latent_hidden_dim=int(cfg_get(cfg, "model.latent_hidden_dim", 64)),
        delta_layers=int(cfg_get(cfg, "model.delta_layers", 3)),
        delta_amp_fraction=float(cfg_get(cfg, "model.delta_amp_fraction", 0.018)),
        enable_low_transition_residual=bool(cfg_get(cfg, "model.enable_low_transition_residual", True)),
        use_observed_voltage_for_gate=bool(cfg_get(cfg, "model.use_observed_voltage_for_gate", True)),
        enable_voltage_inverse_residual=bool(cfg_get(cfg, "model.enable_voltage_inverse_residual", True)),
        voltage_inverse_residual_amp_V=float(cfg_get(cfg, "model.voltage_inverse_residual_amp_V", 0.12)),
    ).to(device)
    latent_offsets = torch.nn.Parameter(torch.zeros(len(batches), len(LATENT_NAMES), device=device))
    params = list(model.parameters()) + [latent_offsets]
    opt = torch.optim.AdamW(params, lr=float(cfg_get(cfg, "train.lr", 8e-4)), weight_decay=float(cfg_get(cfg, "train.weight_decay", 1e-6)))
    epochs = int(cfg_get(cfg, "train.epochs", 80))
    warmup_epochs = int(cfg_get(cfg, "train.warmup_epochs", 20))
    grad_clip = float(cfg_get(cfg, "train.gradient_clip_norm", 10.0))
    weights_main = cfg.get("loss_weights", {}) if isinstance(cfg.get("loss_weights", {}), Mapping) else {}
    weights_warm = cfg.get("loss_weights_warmup", weights_main) if isinstance(cfg.get("loss_weights_warmup", {}), Mapping) else weights_main

    history: List[Dict[str, Any]] = []
    best = {"score": float("inf"), "epoch": -1, "state_dict": None, "latent_offsets": None, "aggregate": None}

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        weights = weights_warm if epoch <= warmup_epochs else weights_main
        total_loss = torch.zeros((), device=device)
        for i, batch in enumerate(batches):
            b = dict(batch)
            b["latent_raw_offset"] = latent_offsets[i : i + 1]
            pred = model(b)
            loss, _ = total_d17_loss(pred, b, prior, weights=weights)
            total_loss = total_loss + loss / float(len(batches))
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite D17-P3 loss at epoch {epoch}: {total_loss}")
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip) if grad_clip > 0 else torch.tensor(0.0)
        opt.step()

        model.eval()
        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                b = dict(batch)
                b["latent_raw_offset"] = latent_offsets[i : i + 1]
                pred = model(b)
                m = audit_numbers(pred, b)
                m["profile_index"] = i
                rows.append(m)
        agg = aggregate(rows)
        row = {"epoch": epoch, "phase": "warmup" if epoch <= warmup_epochs else "mechanism", "total_loss": float(total_loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu())}
        row.update(agg)
        history.append(row)
        score = (
            float(row.get("voltage_mae_V_mean", 1e9))
            + 0.05 * float(row["total_loss"])
            + 1.0e-6 * max(
                float(row.get("zero_mean_max_abs_a_mol_m3_max", 0.0)),
                float(row.get("zero_mean_max_abs_c_mol_m3_max", 0.0)),
            )
        )
        if score < best["score"]:
            best = {"score": score, "epoch": epoch, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "latent_offsets": latent_offsets.detach().cpu().clone(), "aggregate": dict(row)}

    hist_path = out_dir / "training_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)
    torch.save({"model": model.state_dict(), "latent_offsets": latent_offsets.detach().cpu()}, out_dir / "model" / "last_model_and_latents.pt")
    if best["state_dict"] is not None:
        torch.save({"model": best["state_dict"], "latent_offsets": best["latent_offsets"]}, out_dir / "model" / "best_model_and_latents.pt")

    final_rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for i, (rec, batch) in enumerate(zip(selected_records, batches)):
            b = dict(batch)
            b["latent_raw_offset"] = latent_offsets[i : i + 1]
            pred = model(b)
            metrics = audit_numbers(pred, b)
            item = {"profile_index": i, "canonical_cell_uid": rec.get("canonical_cell_uid"), "protocol": normalize_protocol(rec), "replay_npz": rec.get("replay_npz"), "n_time_points": int(batch["t_s"].numel())}
            item.update(metrics)
            final_rows.append(item)
            np.savez_compressed(
                out_dir / "predictions" / f"D17_P3_PROFILE_{i:02d}_PRED_OBS_ONLY.npz",
                t_s=batch["t_s"].detach().cpu().numpy(),
                I_profile=batch["current_A"].detach().cpu().numpy(),
                voltage_exp=batch["voltage_exp"].detach().cpu().numpy(),
                V_pred=pred["V_pred"].detach().cpu().numpy(),
                V_base=pred["V_base"].detach().cpu().numpy(),
                V_pred_forward=pred.get("V_pred_forward", pred["V_pred"]).detach().cpu().numpy(),
                V_residual_local=pred["V_residual_local"].detach().cpu().numpy(),
                V_residual_inverse=pred.get("V_residual_inverse", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                low_transition_gate=pred["low_transition_gate"].detach().cpu().numpy(),
                cbar_a=pred["cbar_a"].detach().cpu().numpy(),
                cbar_c=pred["cbar_c"].detach().cpu().numpy(),
                theta_a_surface=pred["theta_a_surface"].detach().cpu().numpy(),
                theta_c_surface=pred["theta_c_surface"].detach().cpu().numpy(),
                phie=pred["phie"].detach().cpu().numpy(),
                phis_c=pred["phis_c"].detach().cpu().numpy(),
            )
    final_agg = aggregate(final_rows)
    selected_manifest = [{"profile_index": i, "canonical_cell_uid": rec.get("canonical_cell_uid"), "cell_uid": rec.get("cell_uid"), "protocol": normalize_protocol(rec), "split": rec.get("split"), "replay_npz": rec.get("replay_npz"), "softlabel_npz_report_only": rec.get("softlabel_npz")} for i, rec in enumerate(selected_records)]
    (out_dir / "selected_profiles.json").write_text(json.dumps(selected_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS"; reasons: List[str] = []
    mean_mae = float(final_agg.get("voltage_mae_V_mean", float("inf")))
    initial_mae = float(history[0].get("voltage_mae_V_mean", float("inf")))
    if len(final_rows) != profile_count:
        status = "FAIL"; reasons.append(f"profile count {len(final_rows)} != requested {profile_count}")
    if not math.isfinite(float(history[-1].get("total_loss", float("nan")))):
        status = "FAIL"; reasons.append("non-finite final total loss")
    if mean_mae > 0.25:
        status = "REVIEW" if status == "PASS" else status; reasons.append("mean voltage MAE > 0.25 V; mechanism ran but voltage inversion needs P3.1")
    if math.isfinite(initial_mae) and math.isfinite(mean_mae) and mean_mae > initial_mae * 0.98:
        status = "REVIEW" if status == "PASS" else status; reasons.append("voltage MAE did not improve by at least 2 percent")
    zero_max = max(float(final_agg.get("zero_mean_max_abs_a_mol_m3_max", 0.0)), float(final_agg.get("zero_mean_max_abs_c_mol_m3_max", 0.0)))
    if zero_max > 1e-2:
        status = "REVIEW" if status == "PASS" else status; reasons.append("zero-volume-mean audit exceeded 1e-2 mol/m3")
    theta_a_min = float(final_agg.get("theta_a_min_min", 0.0))
    theta_c_min = float(final_agg.get("theta_c_min_min", 0.0))
    theta_a_max = float(final_agg.get("theta_a_max_max", 1.0))
    theta_c_max = float(final_agg.get("theta_c_max_max", 1.0))
    theta_bounds_a = (float(prior.negative.theta_min), float(prior.negative.theta_max))
    theta_bounds_c = (float(prior.positive.theta_min), float(prior.positive.theta_max))
    bounds_eps = 1.0e-3
    if (theta_a_min < theta_bounds_a[0] - bounds_eps or theta_a_max > theta_bounds_a[1] + bounds_eps
        or theta_c_min < theta_bounds_c[0] - bounds_eps or theta_c_max > theta_bounds_c[1] + bounds_eps):
        status = "REVIEW" if status == "PASS" else status; reasons.append("theta/cs physical bounds audit failed")

    summary = {
        "protocol": "D17-P3_6PROFILE_MECHANISM_SMOKE",
        "status": status,
        "reasons": reasons,
        "seed": seed,
        "device": str(device),
        "split_manifest": str(split_manifest),
        "resolved_spec": str(resolved_spec),
        "split": split,
        "profile_count": len(final_rows),
        "n_r": n_r,
        "max_time_points": max_time_points,
        "time_window_s": time_window_s,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]),
        "initial_aggregate": _jsonable(history[0]),
        "final_aggregate": _jsonable(history[-1]),
        "final_profile_metrics": _jsonable(final_rows),
        "theta_bounds_used_for_audit": {
            "negative_a": {"theta_min": theta_bounds_a[0], "theta_max": theta_bounds_a[1]},
            "positive_c": {"theta_min": theta_bounds_c[0], "theta_max": theta_bounds_c[1]},
            "eps": bounds_eps,
        },
        "best_aggregate": _jsonable(best["aggregate"]),
        "selected_profiles": selected_manifest,
        "no_state_label_policy": {
            "training_uses_state_softlabels": False,
            "profile_loader": "replay_npz observed-only",
            "softlabel_npz": "report-only path in manifest; not loaded by P3 trainer",
            "forbidden_state_keys": sorted(FORBIDDEN_PROFILE_KEYS),
            "checkpoint_selection_uses_state_softlabels": False,
            "checkpoint_selection_uses_frozen_test": False,
            "voltage_exp_usage": "observed voltage is used as inverse observation and voltage loss; no cs/theta/phie/phis labels are used"
        },
        "mechanism_notes": {
            "cbar_core": "I(t)-integrated hard inventory baseline",
            "radial_core": "cs=cbar+zero-volume-mean delta_c",
            "closure": "OCP/BV/Ohm/gauge voltage closure",
            "p3_additions": "profile-wise latent offsets + bounded low/transition voltage inverse residual",
            "voltage_not_copied": "V(t) is not assigned to V_pred; residual is bounded, gated, and reported separately"
        },
        "prior_snapshot": prior_to_jsonable(prior),
        "outputs": {"summary_json": str(out_dir / "D17_P3_6PROFILE_SMOKE_SUMMARY.json"), "training_history_csv": str(hist_path), "selected_profiles_json": str(out_dir / "selected_profiles.json"), "best_model_and_latents_pt": str(out_dir / "model" / "best_model_and_latents.pt"), "last_model_and_latents_pt": str(out_dir / "model" / "last_model_and_latents.pt"), "predictions_dir": str(out_dir / "predictions")}
    }
    (out_dir / "D17_P3_6PROFILE_SMOKE_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
