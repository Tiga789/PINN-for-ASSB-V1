# -*- coding: utf-8 -*-
"""D17-P3.2 aggressive voltage recovery mechanism smoke trainer.

P3.2 fixes the P3.2 voltage-gate wiring bug and adds an aggressive smooth voltage-basis residual stage.  It keeps the D17 boundary:
only observed replay fields are loaded for training; state soft labels are never
loaded as inputs or losses.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from .config import cfg_get
from .dataset import D17ProfileDataset
from .latent_adapter import LATENT_NAMES
from .losses import audit_numbers, total_d17_loss
from .model import D17MechanisticPINN, make_batch_from_profile
from .p2dlite_prior import load_p2dlite_prior, prior_to_jsonable
from .trainer import FORBIDDEN_PROFILE_KEYS, choose_device, set_seed
from .p3_trainer import aggregate, load_profiles, normalize_protocol, select_balanced_records


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _phase_for_epoch(epoch: int, warmup_epochs: int, recovery_until_epoch: int) -> str:
    if epoch <= warmup_epochs:
        return "warmup"
    if epoch <= recovery_until_epoch:
        return "voltage_recovery"
    return "mechanism"


def _weights_for_phase(cfg: Mapping[str, Any], phase: str) -> Mapping[str, float]:
    if phase == "warmup":
        return cfg.get("loss_weights_warmup", cfg.get("loss_weights", {}))  # type: ignore[return-value]
    if phase == "voltage_recovery":
        return cfg.get("loss_weights_recovery", cfg.get("loss_weights", {}))  # type: ignore[return-value]
    return cfg.get("loss_weights", {})  # type: ignore[return-value]


def _loss_terms_as_float(loss_terms: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in loss_terms.items():
        if isinstance(v, torch.Tensor):
            val = float(v.detach().cpu())
        else:
            val = float(v)  # type: ignore[arg-type]
        if math.isfinite(val):
            out[f"loss_{k}"] = val
    return out


def _selected_manifest(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "profile_index": i,
            "canonical_cell_uid": rec.get("canonical_cell_uid"),
            "cell_uid": rec.get("cell_uid"),
            "protocol": normalize_protocol(rec),
            "split": rec.get("split"),
            "replay_npz": rec.get("replay_npz"),
            "is_flagged_probe": bool(rec.get("is_flagged_probe") or rec.get("split") == "flagged_probe"),
            "softlabel_npz_report_only": rec.get("softlabel_npz"),
        }
        for i, rec in enumerate(records)
    ]


def train_p32_mechanism_smoke(cfg: Mapping[str, Any], out_dir: str | Path) -> Dict[str, Any]:
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
    profile_count = int(cfg_get(cfg, "train.profile_count", 12))
    time_window_s = float(cfg_get(cfg, "train.time_window_s", 40000.0))
    max_time_points = int(cfg_get(cfg, "train.max_time_points", 512))
    n_r = int(cfg_get(cfg, "train.n_r", 17))

    ds = D17ProfileDataset(split_manifest=split_manifest, split=split, allow_softlabel_npz_profile_source=False)
    selected_records = select_balanced_records(ds, profile_count=profile_count)
    selected_manifest = _selected_manifest(selected_records)
    flagged_selected = [x for x in selected_manifest if x.get("is_flagged_probe")]
    missing_replay = [x for x in selected_manifest if not x.get("replay_npz")]
    if flagged_selected:
        raise RuntimeError(f"D17-P3.2 selected flagged_probe records, refusing training: {flagged_selected}")
    if missing_replay:
        raise RuntimeError(f"D17-P3.2 selected records without replay_npz, refusing training: {missing_replay}")

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
        delta_amp_fraction=float(cfg_get(cfg, "model.delta_amp_fraction", 0.016)),
        enable_low_transition_residual=bool(cfg_get(cfg, "model.enable_low_transition_residual", True)),
        use_observed_voltage_for_gate=bool(cfg_get(cfg, "model.use_observed_voltage_for_gate", True)),
        enable_voltage_inverse_residual=bool(cfg_get(cfg, "model.enable_voltage_inverse_residual", True)),
        voltage_inverse_residual_amp_V=float(cfg_get(cfg, "model.voltage_inverse_residual_amp_V", 0.24)),
        voltage_inverse_residual_gate_mode=str(cfg_get(cfg, "model.voltage_inverse_residual_gate_mode", "all_bounded")),
        enable_voltage_basis_residual=bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True)),
        voltage_basis_residual_amp_V=float(cfg_get(cfg, "model.voltage_basis_residual_amp_V", 0.12)),
        voltage_basis_count=int(cfg_get(cfg, "model.voltage_basis_count", 10)),
    ).to(device)
    latent_offsets = torch.nn.Parameter(torch.zeros(len(batches), len(LATENT_NAMES), device=device))
    voltage_basis_count = int(cfg_get(cfg, "model.voltage_basis_count", 10))
    enable_basis = bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True))
    voltage_basis_coeffs = torch.nn.Parameter(torch.zeros(len(batches), voltage_basis_count, device=device)) if enable_basis else None
    params = list(model.parameters()) + [latent_offsets]
    if voltage_basis_coeffs is not None:
        params.append(voltage_basis_coeffs)
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg_get(cfg, "train.lr", 7.0e-4)),
        weight_decay=float(cfg_get(cfg, "train.weight_decay", 1.0e-6)),
    )
    epochs = int(cfg_get(cfg, "train.epochs", 120))
    warmup_epochs = int(cfg_get(cfg, "train.warmup_epochs", 20))
    recovery_until_epoch = int(cfg_get(cfg, "train.voltage_recovery_until_epoch", max(warmup_epochs, int(0.65 * epochs))))
    grad_clip = float(cfg_get(cfg, "train.gradient_clip_norm", 10.0))

    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"score": float("inf"), "epoch": -1, "state_dict": None, "latent_offsets": None, "aggregate": None}

    for epoch in range(1, epochs + 1):
        phase = _phase_for_epoch(epoch, warmup_epochs, recovery_until_epoch)
        weights = _weights_for_phase(cfg, phase)
        model.train()
        opt.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_loss_terms: List[Dict[str, float]] = []
        # Memory-safe micro-batch gradient accumulation over profiles.
        for i, batch in enumerate(batches):
            b = dict(batch)
            b["latent_raw_offset"] = latent_offsets[i : i + 1]
            if voltage_basis_coeffs is not None:
                b["voltage_basis_raw_coeffs"] = voltage_basis_coeffs[i : i + 1]
            pred = model(b)
            loss, loss_terms = total_d17_loss(pred, b, prior, weights=weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite D17-P3.2 loss at epoch {epoch}, profile {i}: {loss}")
            (loss / float(len(batches))).backward()
            train_loss_sum += float(loss.detach().cpu()) / float(len(batches))
            train_loss_terms.append(_loss_terms_as_float(loss_terms))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip) if grad_clip > 0 else torch.tensor(0.0)
        opt.step()

        model.eval()
        rows: List[Dict[str, Any]] = []
        eval_loss_terms: List[Dict[str, float]] = []
        with torch.no_grad():
            for i, batch in enumerate(batches):
                b = dict(batch)
                b["latent_raw_offset"] = latent_offsets[i : i + 1]
                if voltage_basis_coeffs is not None:
                    b["voltage_basis_raw_coeffs"] = voltage_basis_coeffs[i : i + 1]
                pred = model(b)
                metrics = audit_numbers(pred, b)
                metrics["profile_index"] = i
                rows.append(metrics)
                _, lt = total_d17_loss(pred, b, prior, weights=weights)
                eval_loss_terms.append(_loss_terms_as_float(lt))
        agg = aggregate(rows)
        loss_agg = aggregate(eval_loss_terms)
        row: Dict[str, Any] = {
            "epoch": epoch,
            "phase": phase,
            "total_loss": float(train_loss_sum),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        row.update(agg)
        # Prefix component losses to keep CSV clear.
        for k, v in loss_agg.items():
            row[f"component_{k}"] = v
        history.append(row)

        zero_max = max(
            float(row.get("zero_mean_max_abs_a_mol_m3_max", 0.0)),
            float(row.get("zero_mean_max_abs_c_mol_m3_max", 0.0)),
        )
        bounds_penalty = float(row.get("component_loss_state_bounds_mean", 0.0))
        # Select by voltage first, but never reward breaking inventory or bounds.
        score = (
            float(row.get("voltage_mae_V_mean", 1e9))
            + 0.02 * float(row.get("total_loss", 0.0))
            + 1.0e-6 * zero_max
            + 2.0 * bounds_penalty
        )
        if score < best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "latent_offsets": latent_offsets.detach().cpu().clone(),
                "voltage_basis_coeffs": voltage_basis_coeffs.detach().cpu().clone() if voltage_basis_coeffs is not None else None,
                "aggregate": dict(row),
            }

    hist_path = out_dir / "training_history.csv"
    fieldnames = sorted({k for r in history for k in r.keys()})
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    last_payload = {"model": model.state_dict(), "latent_offsets": latent_offsets.detach().cpu()}
    if voltage_basis_coeffs is not None:
        last_payload["voltage_basis_coeffs"] = voltage_basis_coeffs.detach().cpu()
    torch.save(last_payload, out_dir / "model" / "last_model_and_latents.pt")
    if best["state_dict"] is not None:
        best_payload = {"model": best["state_dict"], "latent_offsets": best["latent_offsets"]}
        if best.get("voltage_basis_coeffs") is not None:
            best_payload["voltage_basis_coeffs"] = best["voltage_basis_coeffs"]
        torch.save(best_payload, out_dir / "model" / "best_model_and_latents.pt")

    # P3.2 reports the best voltage-safe checkpoint, not the last epoch.
    # This avoids over-regularization in the final mechanism epochs hiding a
    # successful recovery checkpoint.  Selection still uses only voltage,
    # zero-mean and bounds audits; no state soft-label metrics are used.
    if best.get("state_dict") is not None:
        model.load_state_dict(best["state_dict"])
        with torch.no_grad():
            latent_offsets.copy_(best["latent_offsets"].to(device))
            if voltage_basis_coeffs is not None and best.get("voltage_basis_coeffs") is not None:
                voltage_basis_coeffs.copy_(best["voltage_basis_coeffs"].to(device))

    final_rows: List[Dict[str, Any]] = []
    final_loss_rows: List[Dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for i, (rec, batch) in enumerate(zip(selected_records, batches)):
            b = dict(batch)
            b["latent_raw_offset"] = latent_offsets[i : i + 1]
            if voltage_basis_coeffs is not None:
                b["voltage_basis_raw_coeffs"] = voltage_basis_coeffs[i : i + 1]
            pred = model(b)
            metrics = audit_numbers(pred, b)
            item: Dict[str, Any] = {
                "profile_index": i,
                "canonical_cell_uid": rec.get("canonical_cell_uid"),
                "protocol": normalize_protocol(rec),
                "replay_npz": rec.get("replay_npz"),
                "n_time_points": int(batch["t_s"].numel()),
            }
            item.update(metrics)
            final_rows.append(item)
            _, lt = total_d17_loss(pred, b, prior, weights=_weights_for_phase(cfg, "mechanism"))
            final_loss_rows.append(_loss_terms_as_float(lt))
            np.savez_compressed(
                out_dir / "predictions" / f"D17_P32_PROFILE_{i:02d}_PRED_OBS_ONLY.npz",
                t_s=batch["t_s"].detach().cpu().numpy(),
                I_profile=batch["current_A"].detach().cpu().numpy(),
                voltage_exp=batch["voltage_exp"].detach().cpu().numpy(),
                V_pred=pred["V_pred"].detach().cpu().numpy(),
                V_base=pred["V_base"].detach().cpu().numpy(),
                V_pred_forward=pred.get("V_pred_forward", pred["V_pred"]).detach().cpu().numpy(),
                V_residual_local=pred["V_residual_local"].detach().cpu().numpy(),
                V_residual_inverse=pred.get("V_residual_inverse", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                V_residual_basis=pred.get("V_residual_basis", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                V_residual_total=pred.get("V_residual_total", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                voltage_inverse_gate=pred.get("voltage_inverse_gate", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                low_transition_gate=pred["low_transition_gate"].detach().cpu().numpy(),
                cbar_a=pred["cbar_a"].detach().cpu().numpy(),
                cbar_c=pred["cbar_c"].detach().cpu().numpy(),
                theta_a_surface=pred["theta_a_surface"].detach().cpu().numpy(),
                theta_c_surface=pred["theta_c_surface"].detach().cpu().numpy(),
                theta_a_min=float(torch.min(pred["theta_a"]).detach().cpu()),
                theta_a_max=float(torch.max(pred["theta_a"]).detach().cpu()),
                theta_c_min=float(torch.min(pred["theta_c"]).detach().cpu()),
                theta_c_max=float(torch.max(pred["theta_c"]).detach().cpu()),
                phie=pred["phie"].detach().cpu().numpy(),
                phis_c=pred["phis_c"].detach().cpu().numpy(),
            )

    final_agg = aggregate(final_rows)
    final_loss_agg = aggregate(final_loss_rows)
    (out_dir / "selected_profiles.json").write_text(json.dumps(selected_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    loss_audit = {
        "component_loss_aggregate_final": final_loss_agg,
        "history_component_columns": sorted([k for k in history[-1].keys() if k.startswith("component_loss_")]),
        "notes": "Loss terms are computed with observed voltage and physics losses only; no state soft-label losses are present.",
    }
    (out_dir / "D17_P32_LOSS_SCALE_AUDIT.json").write_text(json.dumps(_jsonable(loss_audit), ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS"
    reasons: List[str] = []
    mean_mae = float(final_agg.get("voltage_mae_V_mean", float("inf")))
    initial_mae = float(history[0].get("voltage_mae_V_mean", float("inf")))
    voltage_review_threshold = float(cfg_get(cfg, "audit.voltage_mae_review_threshold_V", 0.16))
    voltage_target = float(cfg_get(cfg, "audit.voltage_mae_target_V", 0.09))
    if len(final_rows) != profile_count:
        status = "FAIL"
        reasons.append(f"profile count {len(final_rows)} != requested {profile_count}")
    if not math.isfinite(float(history[-1].get("total_loss", float("nan")))):
        status = "FAIL"
        reasons.append("non-finite final total loss")
    if mean_mae > voltage_review_threshold:
        status = "REVIEW" if status == "PASS" else status
        reasons.append(f"mean voltage MAE > {voltage_review_threshold:.3f} V")
    if math.isfinite(initial_mae) and math.isfinite(mean_mae) and mean_mae > initial_mae * 0.95:
        status = "REVIEW" if status == "PASS" else status
        reasons.append("voltage MAE did not improve by at least 5 percent")
    zero_max = max(float(final_agg.get("zero_mean_max_abs_a_mol_m3_max", 0.0)), float(final_agg.get("zero_mean_max_abs_c_mol_m3_max", 0.0)))
    if zero_max > 1e-2:
        status = "REVIEW" if status == "PASS" else status
        reasons.append("zero-volume-mean audit exceeded 1e-2 mol/m3")
    theta_a_min = float(final_agg.get("theta_a_min_min", 0.0))
    theta_c_min = float(final_agg.get("theta_c_min_min", 0.0))
    theta_a_max = float(final_agg.get("theta_a_max_max", 1.0))
    theta_c_max = float(final_agg.get("theta_c_max_max", 1.0))
    theta_bounds_a = (float(prior.negative.theta_min), float(prior.negative.theta_max))
    theta_bounds_c = (float(prior.positive.theta_min), float(prior.positive.theta_max))
    bounds_eps = 1.0e-3
    if (
        theta_a_min < theta_bounds_a[0] - bounds_eps
        or theta_a_max > theta_bounds_a[1] + bounds_eps
        or theta_c_min < theta_bounds_c[0] - bounds_eps
        or theta_c_max > theta_bounds_c[1] + bounds_eps
    ):
        status = "REVIEW" if status == "PASS" else status
        reasons.append("theta/cs physical bounds audit failed")
    if flagged_selected:
        status = "FAIL"
        reasons.append("flagged_probe profile selected")

    summary = {
        "protocol": "D17-P3.2_12PROFILE_AGGRESSIVE_VOLTAGE_RECOVERY_SMOKE",
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
        "voltage_recovery_until_epoch": recovery_until_epoch,
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]),
        "initial_aggregate": _jsonable(history[0]),
        "last_epoch_aggregate": _jsonable(history[-1]),
        "final_aggregate": _jsonable(final_agg),
        "final_loss_component_aggregate": _jsonable(final_loss_agg),
        "final_loss_scale_audit": _jsonable(loss_audit),
        "final_profile_metrics": _jsonable(final_rows),
        "voltage_recovery": {
            "initial_voltage_mae_mean_V": initial_mae,
            "final_voltage_mae_mean_V": mean_mae,
            "final_voltage_corr_mean": float(final_agg.get("voltage_corr_mean", float("nan"))),
            "target_V": voltage_target,
            "review_threshold_V": voltage_review_threshold,
            "target_met": bool(mean_mae <= voltage_target),
        },
        "theta_bounds_used_for_audit": {
            "negative_a": {"theta_min": theta_bounds_a[0], "theta_max": theta_bounds_a[1]},
            "positive_c": {"theta_min": theta_bounds_c[0], "theta_max": theta_bounds_c[1]},
            "eps": bounds_eps,
        },
        "best_aggregate": _jsonable(best["aggregate"]),
        "reported_checkpoint": "best_voltage_safe_checkpoint",
        "selected_profiles": selected_manifest,
        "no_state_label_policy": {
            "training_uses_state_softlabels": False,
            "profile_loader": "replay_npz observed-only",
            "softlabel_npz": "report-only path in manifest; not loaded by P3.2 trainer",
            "forbidden_state_keys": sorted(FORBIDDEN_PROFILE_KEYS),
            "checkpoint_selection_uses_state_softlabels": False,
            "checkpoint_selection_uses_frozen_test": False,
            "voltage_exp_usage": "observed voltage is inverse observation and voltage loss; cs/theta/phie/phis labels are not used",
        },
        "mechanism_notes": {
            "cbar_core": "I(t)-integrated hard inventory baseline",
            "inventory_bounds": "theta0 feasible projection + bounded zero-mean radial residual scaling",
            "radial_core": "cs=cbar+zero-volume-mean delta_c; no full-field cs clamp",
            "closure": "OCP/BV/Ohm/gauge voltage closure",
            "p32_additions": "fixed gate-mode wiring + all-bounded inverse residual + smooth voltage-basis residual + 12-profile loss-scale audit",
            "voltage_not_copied": "V(t) is not assigned to V_pred; inverse residual is bounded and smooth-basis residual has only low-dimensional profile coefficients",
        },
        "p32_model_voltage_recovery_config": {
            "voltage_inverse_residual_gate_mode": str(cfg_get(cfg, "model.voltage_inverse_residual_gate_mode", "all_bounded")),
            "voltage_inverse_residual_amp_V": float(cfg_get(cfg, "model.voltage_inverse_residual_amp_V", 0.24)),
            "enable_voltage_basis_residual": bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True)),
            "voltage_basis_residual_amp_V": float(cfg_get(cfg, "model.voltage_basis_residual_amp_V", 0.12)),
            "voltage_basis_count": int(cfg_get(cfg, "model.voltage_basis_count", 10)),
        },
        "prior_snapshot": prior_to_jsonable(prior),
        "outputs": {
            "summary_json": str(out_dir / "D17_P32_12PROFILE_VOLTAGE_RECOVERY_SUMMARY.json"),
            "loss_scale_audit_json": str(out_dir / "D17_P32_LOSS_SCALE_AUDIT.json"),
            "training_history_csv": str(hist_path),
            "selected_profiles_json": str(out_dir / "selected_profiles.json"),
            "best_model_and_latents_pt": str(out_dir / "model" / "best_model_and_latents.pt"),
            "last_model_and_latents_pt": str(out_dir / "model" / "last_model_and_latents.pt"),
            "predictions_dir": str(out_dir / "predictions"),
        },
    }
    (out_dir / "D17_P32_12PROFILE_VOLTAGE_RECOVERY_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
