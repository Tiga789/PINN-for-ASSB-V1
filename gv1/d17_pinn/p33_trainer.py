# -*- coding: utf-8 -*-
"""D17-P3.3 forward-core reliability audit and D12 formula migration.

P3.3 is intentionally an audit/diagnostic stage, not a promotion stage.  It
trains a 12-profile D17 mechanism PINN, evaluates 6 validation profiles with
observed-only profile adaptation, and reports three voltages separately:

  V_forward: electrochemical core only, no inverse residual.
  V_residual: bounded D12-S1K-style transition-fade correction.
  V_pred: V_forward + declared residual channels.

No cs/theta/phie/phis soft-label arrays are loaded as inputs or losses.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .config import cfg_get
from .dataset import D17ProfileDataset
from .d12_transition_fade import D12TransitionFadeConfig, gate_audit_numbers
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


def _loss_terms_as_float(loss_terms: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in loss_terms.items():
        try:
            val = float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)  # type: ignore[arg-type]
        except Exception:
            continue
        if math.isfinite(val):
            out[f"loss_{k}"] = val
    return out


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


def _selected_manifest(records: Sequence[Mapping[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    return [
        {
            "profile_index": i,
            "canonical_cell_uid": rec.get("canonical_cell_uid"),
            "cell_uid": rec.get("cell_uid"),
            "protocol": normalize_protocol(rec),
            "split": rec.get("split", split_name),
            "replay_npz": rec.get("replay_npz"),
            "is_flagged_probe": bool(rec.get("is_flagged_probe") or rec.get("split") == "flagged_probe"),
            "softlabel_npz_report_only": rec.get("softlabel_npz"),
        }
        for i, rec in enumerate(records)
    ]


def _make_model(cfg: Mapping[str, Any], prior, feature_dim: int, n_r: int, device: torch.device) -> D17MechanisticPINN:
    return D17MechanisticPINN(
        prior=prior,
        feature_dim=feature_dim,
        n_r=n_r,
        hidden_dim=int(cfg_get(cfg, "model.hidden_dim", 64)),
        latent_hidden_dim=int(cfg_get(cfg, "model.latent_hidden_dim", 64)),
        delta_layers=int(cfg_get(cfg, "model.delta_layers", 3)),
        delta_amp_fraction=float(cfg_get(cfg, "model.delta_amp_fraction", 0.014)),
        enable_low_transition_residual=bool(cfg_get(cfg, "model.enable_low_transition_residual", True)),
        use_observed_voltage_for_gate=bool(cfg_get(cfg, "model.use_observed_voltage_for_gate", True)),
        enable_voltage_inverse_residual=bool(cfg_get(cfg, "model.enable_voltage_inverse_residual", True)),
        voltage_inverse_residual_amp_V=float(cfg_get(cfg, "model.voltage_inverse_residual_amp_V", 0.12)),
        voltage_inverse_residual_gate_mode=str(cfg_get(cfg, "model.voltage_inverse_residual_gate_mode", "d12_transition_fade")),
        enable_voltage_basis_residual=bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True)),
        voltage_basis_residual_amp_V=float(cfg_get(cfg, "model.voltage_basis_residual_amp_V", 0.07)),
        voltage_basis_count=int(cfg_get(cfg, "model.voltage_basis_count", 10)),
        voltage_basis_formula_mode=str(cfg_get(cfg, "model.voltage_basis_formula_mode", "d12_transition_fade")),
        d12_low_v=float(cfg_get(cfg, "d12_transition_fade.low_v", 2.75)),
        d12_normal_v=float(cfg_get(cfg, "d12_transition_fade.normal_v", 3.05)),
        d12_low_width_v=float(cfg_get(cfg, "d12_transition_fade.low_width_v", 0.055)),
        d12_transition_width_v=float(cfg_get(cfg, "d12_transition_fade.transition_width_v", 0.080)),
        d12_transition_gain=float(cfg_get(cfg, "d12_transition_fade.transition_gain", 0.70)),
        d12_non_low_preservation_floor=float(cfg_get(cfg, "d12_transition_fade.non_low_preservation_floor", 0.02)),
    ).to(device)


def _append_d12_gate_metrics(metrics: Dict[str, float], pred: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], cfg: Mapping[str, Any]) -> Dict[str, float]:
    d12_cfg = D12TransitionFadeConfig(
        low_v=float(cfg_get(cfg, "d12_transition_fade.low_v", 2.75)),
        normal_v=float(cfg_get(cfg, "d12_transition_fade.normal_v", 3.05)),
        low_width_v=float(cfg_get(cfg, "d12_transition_fade.low_width_v", 0.055)),
        transition_width_v=float(cfg_get(cfg, "d12_transition_fade.transition_width_v", 0.080)),
        transition_gain=float(cfg_get(cfg, "d12_transition_fade.transition_gain", 0.70)),
        non_low_preservation_floor=float(cfg_get(cfg, "d12_transition_fade.non_low_preservation_floor", 0.02)),
    )
    residual = pred.get("V_residual_total", torch.zeros_like(pred["V_pred"]))
    metrics.update(gate_audit_numbers(batch["voltage_exp"], batch["current_A"], residual, d12_cfg))
    return metrics


def _evaluate_records(
    *,
    split_name: str,
    records: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, torch.Tensor]],
    model: D17MechanisticPINN,
    prior: Any,
    cfg: Mapping[str, Any],
    latent_offsets: torch.Tensor,
    voltage_basis_coeffs: torch.Tensor | None,
    out_dir: Path,
    pred_prefix: str,
    save_predictions: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for i, (rec, batch) in enumerate(zip(records, batches)):
            b = dict(batch)
            b["latent_raw_offset"] = latent_offsets[i : i + 1]
            if voltage_basis_coeffs is not None:
                b["voltage_basis_raw_coeffs"] = voltage_basis_coeffs[i : i + 1]
            pred = model(b)
            metrics = audit_numbers(pred, b)
            metrics = _append_d12_gate_metrics(metrics, pred, b, cfg)
            item: Dict[str, Any] = {
                "profile_index": i,
                "split": split_name,
                "canonical_cell_uid": rec.get("canonical_cell_uid"),
                "protocol": normalize_protocol(rec),
                "replay_npz": rec.get("replay_npz"),
                "n_time_points": int(batch["t_s"].numel()),
            }
            item.update(metrics)
            rows.append(item)
            _, lt = total_d17_loss(pred, b, prior, weights=_weights_for_phase(cfg, "mechanism"))
            loss_rows.append(_loss_terms_as_float(lt))
            if save_predictions:
                np.savez_compressed(
                    out_dir / "predictions" / f"{pred_prefix}_PROFILE_{i:02d}_PRED_OBS_ONLY.npz",
                    t_s=batch["t_s"].detach().cpu().numpy(),
                    I_profile=batch["current_A"].detach().cpu().numpy(),
                    voltage_exp=batch["voltage_exp"].detach().cpu().numpy(),
                    V_pred=pred["V_pred"].detach().cpu().numpy(),
                    V_pred_forward=pred.get("V_pred_forward", pred["V_pred"]).detach().cpu().numpy(),
                    V_base=pred["V_base"].detach().cpu().numpy(),
                    V_residual_local=pred["V_residual_local"].detach().cpu().numpy(),
                    V_residual_inverse=pred.get("V_residual_inverse", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                    V_residual_basis=pred.get("V_residual_basis", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                    V_residual_total=pred.get("V_residual_total", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                    d12_fade_gate=pred.get("d12_fade_gate", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                    d12_low_core_gate=pred.get("d12_low_core_gate", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                    d12_transition_gate=pred.get("d12_transition_gate", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                    d12_preserve_gate=pred.get("d12_preserve_gate", torch.ones_like(pred["V_pred"])).detach().cpu().numpy(),
                    voltage_inverse_gate=pred.get("voltage_inverse_gate", torch.zeros_like(pred["V_pred"])).detach().cpu().numpy(),
                    low_transition_gate=pred["low_transition_gate"].detach().cpu().numpy(),
                    cbar_a=pred["cbar_a"].detach().cpu().numpy(),
                    cbar_c=pred["cbar_c"].detach().cpu().numpy(),
                    theta_a_surface=pred["theta_a_surface"].detach().cpu().numpy(),
                    theta_c_surface=pred["theta_c_surface"].detach().cpu().numpy(),
                    phie=pred["phie"].detach().cpu().numpy(),
                    phis_c=pred["phis_c"].detach().cpu().numpy(),
                )
    return rows, aggregate(rows), aggregate(loss_rows)


def _score_row(row: Mapping[str, Any], residual_mean_budget: float, residual_max_budget: float) -> float:
    mae = float(row.get("voltage_mae_V_mean", 1e9))
    fmae = float(row.get("forward_voltage_mae_V_mean", 1e9))
    res_mean = float(row.get("V_residual_total_abs_mean_V_mean", 0.0))
    res_max = float(row.get("V_residual_total_abs_max_V_max", 0.0))
    zero = max(float(row.get("zero_mean_max_abs_a_mol_m3_max", 0.0)), float(row.get("zero_mean_max_abs_c_mol_m3_max", 0.0)))
    bounds = float(row.get("component_loss_state_bounds_mean", 0.0))
    budget_penalty = max(0.0, res_mean - residual_mean_budget) + 0.25 * max(0.0, res_max - residual_max_budget)
    # Forward voltage is included lightly to avoid selecting a checkpoint that
    # hides all electrochemical mismatch behind residual channels.
    return mae + 0.10 * fmae + 0.35 * budget_penalty + 1.0e-6 * zero + 2.0 * bounds


def _validate_records(records: Sequence[Mapping[str, Any]], label: str) -> None:
    bad_flag = [r.get("canonical_cell_uid") for r in records if bool(r.get("is_flagged_probe") or r.get("split") == "flagged_probe")]
    bad_replay = [r.get("canonical_cell_uid") for r in records if not r.get("replay_npz")]
    if bad_flag:
        raise RuntimeError(f"D17-P3.3 selected flagged_probe records for {label}: {bad_flag}")
    if bad_replay:
        raise RuntimeError(f"D17-P3.3 selected records without replay_npz for {label}: {bad_replay}")


def _adapt_validation_latents(
    *,
    model: D17MechanisticPINN,
    prior: Any,
    cfg: Mapping[str, Any],
    batches: Sequence[Mapping[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor | None, List[Dict[str, Any]]]:
    steps = int(cfg_get(cfg, "validation.adaptation_steps", 80))
    lr = float(cfg_get(cfg, "validation.adaptation_lr", 1.0e-2))
    voltage_basis_count = int(cfg_get(cfg, "model.voltage_basis_count", 10))
    enable_basis = bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True))
    val_latents = torch.nn.Parameter(torch.zeros(len(batches), len(LATENT_NAMES), device=device))
    val_basis = torch.nn.Parameter(torch.zeros(len(batches), voltage_basis_count, device=device)) if enable_basis else None
    params: List[torch.nn.Parameter] = [val_latents]
    if val_basis is not None:
        params.append(val_basis)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    weights = cfg.get("validation_loss_weights", cfg.get("loss_weights", {}))  # type: ignore[assignment]
    hist: List[Dict[str, Any]] = []
    # Freeze model weights.  We only infer profile-specific observed-only latent
    # variables on validation profiles, using V(t) and physics penalties.
    old_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            rows: List[Dict[str, float]] = []
            for i, batch in enumerate(batches):
                b = dict(batch)
                b["latent_raw_offset"] = val_latents[i : i + 1]
                if val_basis is not None:
                    b["voltage_basis_raw_coeffs"] = val_basis[i : i + 1]
                pred = model(b)
                loss, _ = total_d17_loss(pred, b, prior, weights=weights)  # observed V + physics only
                loss_sum = loss_sum + loss / float(len(batches))
                if step == 1 or step == steps or step % max(1, steps // 5) == 0:
                    rows.append(audit_numbers(pred, b))
            if not torch.isfinite(loss_sum):
                raise RuntimeError(f"Non-finite validation adaptation loss at step {step}: {loss_sum}")
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg_get(cfg, "validation.gradient_clip_norm", 10.0)))
            opt.step()
            if rows:
                agg = aggregate(rows)
                agg.update({"adapt_step": step, "adapt_loss": float(loss_sum.detach().cpu())})
                hist.append(agg)
    finally:
        for p, flag in zip(model.parameters(), old_flags):
            p.requires_grad_(flag)
    return val_latents.detach(), val_basis.detach() if val_basis is not None else None, hist


def _formula_alignment_audit(cfg: Mapping[str, Any], resolved_spec: str, summary_bits: Mapping[str, Any]) -> Dict[str, Any]:
    formula_mode = str(cfg_get(cfg, "model.voltage_basis_formula_mode", ""))
    gate_mode = str(cfg_get(cfg, "model.voltage_inverse_residual_gate_mode", ""))
    out = {
        "protocol": "D17-P3.3_FORMULA_ALIGNMENT_AUDIT",
        "pass": True,
        "resolved_spec": str(resolved_spec),
        "uses_placeholder_spec": "placeholder" in str(resolved_spec).lower(),
        "d12_s1k_transition_fade_formula_migrated": formula_mode.lower() in {"d12", "d12_s1k", "d12_transition_fade", "s1k"},
        "voltage_inverse_gate_mode": gate_mode,
        "voltage_basis_formula_mode": formula_mode,
        "d12_transition_fade_config": {
            "low_v": float(cfg_get(cfg, "d12_transition_fade.low_v", 2.75)),
            "normal_v": float(cfg_get(cfg, "d12_transition_fade.normal_v", 3.05)),
            "low_width_v": float(cfg_get(cfg, "d12_transition_fade.low_width_v", 0.055)),
            "transition_width_v": float(cfg_get(cfg, "d12_transition_fade.transition_width_v", 0.080)),
            "transition_gain": float(cfg_get(cfg, "d12_transition_fade.transition_gain", 0.70)),
        },
        "declared_mechanisms": [
            "I(t)-cbar hard inventory baseline",
            "cs=cbar+zero-volume-mean delta_c",
            "hard feasible theta0 projection and radial residual scaling",
            "OCP/BV/Ohm/gauge voltage closure",
            "D12-S1K-style low/transition fade residual basis",
            "non-low preservation and residual budget audit",
        ],
        "warnings": [],
    }
    if out["uses_placeholder_spec"]:
        out["warnings"].append("resolved_spec path still looks like a placeholder; forward-core promotion requires D15 generator resolved P2Dlite-RG prior alignment.")
    if not out["d12_s1k_transition_fade_formula_migrated"]:
        out["pass"] = False
        out["warnings"].append("D12 transition-fade formula mode is not enabled.")
    out.update(summary_bits)
    return out


def _residual_budget_audit(cfg: Mapping[str, Any], train_agg: Mapping[str, float], val_agg: Mapping[str, float]) -> Dict[str, Any]:
    mean_budget = float(cfg_get(cfg, "audit.residual_abs_mean_budget_V", 0.06))
    max_budget = float(cfg_get(cfg, "audit.residual_abs_max_budget_V", 0.16))
    fwd_target = float(cfg_get(cfg, "audit.forward_voltage_mae_target_V", 0.09))
    corr_target = float(cfg_get(cfg, "audit.corrected_voltage_mae_target_V", 0.06))
    def check(prefix: str, agg: Mapping[str, float]) -> Dict[str, Any]:
        res_mean = float(agg.get("V_residual_total_abs_mean_V_mean", float("inf")))
        res_max = float(agg.get("V_residual_total_abs_max_V_max", float("inf")))
        fwd = float(agg.get("forward_voltage_mae_V_mean", float("inf")))
        corr = float(agg.get("voltage_mae_V_mean", float("inf")))
        return {
            "split": prefix,
            "corrected_voltage_mae_mean_V": corr,
            "forward_voltage_mae_mean_V": fwd,
            "residual_total_abs_mean_V": res_mean,
            "residual_total_abs_max_V": res_max,
            "corrected_voltage_target_met": bool(corr <= corr_target),
            "forward_core_target_met": bool(fwd <= fwd_target),
            "residual_mean_budget_met": bool(res_mean <= mean_budget),
            "residual_max_budget_met": bool(res_max <= max_budget),
        }
    tr = check("train", train_agg)
    va = check("validation", val_agg)
    return {
        "protocol": "D17-P3.3_RESIDUAL_BUDGET_AUDIT",
        "budgets": {
            "residual_abs_mean_budget_V": mean_budget,
            "residual_abs_max_budget_V": max_budget,
            "forward_voltage_mae_target_V": fwd_target,
            "corrected_voltage_mae_target_V": corr_target,
        },
        "train": tr,
        "validation": va,
        "forward_core_reliability_status": "PASS" if tr["forward_core_target_met"] and va["forward_core_target_met"] else "REVIEW",
        "residual_budget_status": "PASS" if tr["residual_mean_budget_met"] and va["residual_mean_budget_met"] and tr["residual_max_budget_met"] and va["residual_max_budget_met"] else "REVIEW",
        "notes": "Corrected voltage can pass while forward core remains REVIEW; this is expected until resolved prior and formula alignment are fully confirmed.",
    }


def train_p33_forward_core_reliability(cfg: Mapping[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    seed = int(cfg_get(cfg, "seed", 20260615))
    set_seed(seed)
    device = choose_device(str(cfg_get(cfg, "train.device", "auto")))
    split_manifest = cfg_get(cfg, "paths.split_manifest")
    resolved_spec = cfg_get(cfg, "paths.resolved_spec")
    train_split = str(cfg_get(cfg, "train.split", "train"))
    validation_split = str(cfg_get(cfg, "validation.split", "validation"))
    train_profile_count = int(cfg_get(cfg, "train.profile_count", 12))
    validation_profile_count = int(cfg_get(cfg, "validation.profile_count", 6))
    time_window_s = float(cfg_get(cfg, "train.time_window_s", 40000.0))
    max_time_points = int(cfg_get(cfg, "train.max_time_points", 512))
    n_r = int(cfg_get(cfg, "train.n_r", 17))

    train_ds = D17ProfileDataset(split_manifest=split_manifest, split=train_split, allow_softlabel_npz_profile_source=False)
    val_ds = D17ProfileDataset(split_manifest=split_manifest, split=validation_split, allow_softlabel_npz_profile_source=False)
    train_records = select_balanced_records(train_ds, profile_count=train_profile_count)
    val_records = select_balanced_records(val_ds, profile_count=validation_profile_count)
    _validate_records(train_records, "train")
    _validate_records(val_records, "validation")
    train_manifest = _selected_manifest(train_records, train_split)
    val_manifest = _selected_manifest(val_records, validation_split)

    train_profiles = load_profiles(train_records, time_window_s=time_window_s, max_time_points=max_time_points)
    val_profiles = load_profiles(val_records, time_window_s=time_window_s, max_time_points=max_time_points)
    prior = load_p2dlite_prior(resolved_spec, allow_smoke_defaults=True)
    train_batches = [make_batch_from_profile(p, n_r=n_r, device=device) for p in train_profiles]
    val_batches = [make_batch_from_profile(p, n_r=n_r, device=device) for p in val_profiles]

    feature_dim = int(train_batches[0]["features"].shape[-1])
    model = _make_model(cfg, prior, feature_dim, n_r, device)
    voltage_basis_count = int(cfg_get(cfg, "model.voltage_basis_count", 10))
    enable_basis = bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True))
    train_latents = torch.nn.Parameter(torch.zeros(len(train_batches), len(LATENT_NAMES), device=device))
    train_basis = torch.nn.Parameter(torch.zeros(len(train_batches), voltage_basis_count, device=device)) if enable_basis else None
    params: List[torch.nn.Parameter] = list(model.parameters()) + [train_latents]
    if train_basis is not None:
        params.append(train_basis)
    opt = torch.optim.AdamW(params, lr=float(cfg_get(cfg, "train.lr", 8e-4)), weight_decay=float(cfg_get(cfg, "train.weight_decay", 1e-6)))
    epochs = int(cfg_get(cfg, "train.epochs", 160))
    warmup_epochs = int(cfg_get(cfg, "train.warmup_epochs", 25))
    recovery_until_epoch = int(cfg_get(cfg, "train.voltage_recovery_until_epoch", 110))
    grad_clip = float(cfg_get(cfg, "train.gradient_clip_norm", 10.0))
    residual_mean_budget = float(cfg_get(cfg, "audit.residual_abs_mean_budget_V", 0.06))
    residual_max_budget = float(cfg_get(cfg, "audit.residual_abs_max_budget_V", 0.16))

    history: List[Dict[str, Any]] = []
    best: Dict[str, Any] = {"score": float("inf"), "epoch": -1, "state_dict": None, "latents": None, "basis": None, "aggregate": None}
    for epoch in range(1, epochs + 1):
        phase = _phase_for_epoch(epoch, warmup_epochs, recovery_until_epoch)
        weights = _weights_for_phase(cfg, phase)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        for i, batch in enumerate(train_batches):
            b = dict(batch)
            b["latent_raw_offset"] = train_latents[i : i + 1]
            if train_basis is not None:
                b["voltage_basis_raw_coeffs"] = train_basis[i : i + 1]
            pred = model(b)
            loss, _ = total_d17_loss(pred, b, prior, weights=weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite D17-P3.3 train loss at epoch {epoch}, profile {i}: {loss}")
            (loss / float(len(train_batches))).backward()
            loss_sum = loss_sum + loss.detach() / float(len(train_batches))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip) if grad_clip > 0 else torch.tensor(0.0)
        opt.step()

        rows, agg, loss_agg = _evaluate_records(
            split_name="train",
            records=train_records,
            batches=train_batches,
            model=model,
            prior=prior,
            cfg=cfg,
            latent_offsets=train_latents.detach(),
            voltage_basis_coeffs=train_basis.detach() if train_basis is not None else None,
            out_dir=out_dir,
            pred_prefix="TMP",
            save_predictions=False,
        )
        row: Dict[str, Any] = {"epoch": epoch, "phase": phase, "total_loss": float(loss_sum.cpu()), "grad_norm": float(grad_norm.detach().cpu())}
        row.update(agg)
        for k, v in loss_agg.items():
            row[f"component_{k}"] = v
        history.append(row)
        score = _score_row(row, residual_mean_budget, residual_max_budget)
        if score < best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "latents": train_latents.detach().cpu().clone(),
                "basis": train_basis.detach().cpu().clone() if train_basis is not None else None,
                "aggregate": dict(row),
            }

    fieldnames = sorted({k for r in history for k in r.keys()})
    hist_path = out_dir / "training_history_train.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(history)
    torch.save({"model": model.state_dict(), "latent_offsets": train_latents.detach().cpu(), "voltage_basis_coeffs": train_basis.detach().cpu() if train_basis is not None else None}, out_dir / "model" / "last_model_and_latents.pt")
    if best["state_dict"] is not None:
        torch.save({"model": best["state_dict"], "latent_offsets": best["latents"], "voltage_basis_coeffs": best["basis"]}, out_dir / "model" / "best_model_and_latents.pt")
        model.load_state_dict(best["state_dict"])
        with torch.no_grad():
            train_latents.copy_(best["latents"].to(device))
            if train_basis is not None and best["basis"] is not None:
                train_basis.copy_(best["basis"].to(device))

    train_rows, train_agg, train_loss_agg = _evaluate_records(
        split_name="train",
        records=train_records,
        batches=train_batches,
        model=model,
        prior=prior,
        cfg=cfg,
        latent_offsets=train_latents.detach(),
        voltage_basis_coeffs=train_basis.detach() if train_basis is not None else None,
        out_dir=out_dir,
        pred_prefix="D17_P33_TRAIN",
        save_predictions=True,
    )

    val_latents, val_basis, val_adapt_history = _adapt_validation_latents(model=model, prior=prior, cfg=cfg, batches=val_batches, device=device)
    val_rows, val_agg, val_loss_agg = _evaluate_records(
        split_name="validation",
        records=val_records,
        batches=val_batches,
        model=model,
        prior=prior,
        cfg=cfg,
        latent_offsets=val_latents,
        voltage_basis_coeffs=val_basis,
        out_dir=out_dir,
        pred_prefix="D17_P33_VALIDATION",
        save_predictions=True,
    )
    val_hist_path = out_dir / "validation_adaptation_history.csv"
    if val_adapt_history:
        val_fields = sorted({k for r in val_adapt_history for k in r.keys()})
        with val_hist_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=val_fields)
            writer.writeheader(); writer.writerows(val_adapt_history)

    (out_dir / "selected_profiles_train.json").write_text(json.dumps(_jsonable(train_manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "selected_profiles_validation.json").write_text(json.dumps(_jsonable(val_manifest), ensure_ascii=False, indent=2), encoding="utf-8")

    formula_audit = _formula_alignment_audit(cfg, str(resolved_spec), {"train_profile_count": len(train_rows), "validation_profile_count": len(val_rows)})
    budget_audit = _residual_budget_audit(cfg, train_agg, val_agg)
    (out_dir / "D17_P33_FORMULA_ALIGNMENT_AUDIT.json").write_text(json.dumps(_jsonable(formula_audit), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "D17_P33_RESIDUAL_BUDGET_AUDIT.json").write_text(json.dumps(_jsonable(budget_audit), ensure_ascii=False, indent=2), encoding="utf-8")

    status = "PASS"
    reasons: List[str] = []
    corrected_target = float(cfg_get(cfg, "audit.corrected_voltage_mae_target_V", 0.06))
    corrected_review = float(cfg_get(cfg, "audit.corrected_voltage_mae_review_threshold_V", 0.10))
    zero_max = max(
        float(train_agg.get("zero_mean_max_abs_a_mol_m3_max", 0.0)), float(train_agg.get("zero_mean_max_abs_c_mol_m3_max", 0.0)),
        float(val_agg.get("zero_mean_max_abs_a_mol_m3_max", 0.0)), float(val_agg.get("zero_mean_max_abs_c_mol_m3_max", 0.0)),
    )
    if zero_max > 1.0e-2:
        status = "REVIEW"
        reasons.append("zero-volume-mean audit exceeded 1e-2 mol/m3")
    # Theta bounds against prior windows.
    ta_min = min(float(train_agg.get("theta_a_min_min", 1.0)), float(val_agg.get("theta_a_min_min", 1.0)))
    ta_max = max(float(train_agg.get("theta_a_max_max", 0.0)), float(val_agg.get("theta_a_max_max", 0.0)))
    tc_min = min(float(train_agg.get("theta_c_min_min", 1.0)), float(val_agg.get("theta_c_min_min", 1.0)))
    tc_max = max(float(train_agg.get("theta_c_max_max", 0.0)), float(val_agg.get("theta_c_max_max", 0.0)))
    if ta_min < prior.negative.theta_min - 1e-3 or ta_max > prior.negative.theta_max + 1e-3 or tc_min < prior.positive.theta_min - 1e-3 or tc_max > prior.positive.theta_max + 1e-3:
        status = "REVIEW"
        reasons.append("theta/cs physical bounds audit failed")
    val_mae = float(val_agg.get("voltage_mae_V_mean", float("inf")))
    if val_mae > corrected_review:
        status = "REVIEW"
        reasons.append(f"validation corrected voltage MAE > {corrected_review:.3f} V")
    if not formula_audit.get("pass", False):
        status = "REVIEW"
        reasons.append("formula alignment audit not pass")

    promotion_status = "PASS"
    promotion_reasons: List[str] = []
    if budget_audit["forward_core_reliability_status"] != "PASS":
        promotion_status = "REVIEW"
        promotion_reasons.append("forward electrochemical core still above target without residual")
    if budget_audit["residual_budget_status"] != "PASS":
        promotion_status = "REVIEW"
        promotion_reasons.append("voltage residual budget still too large for promotion")
    if val_mae > corrected_target:
        promotion_status = "REVIEW"
        promotion_reasons.append(f"validation corrected voltage MAE > target {corrected_target:.3f} V")

    summary = {
        "protocol": "D17-P3.3_FORWARD_CORE_RELIABILITY_AUDIT",
        "status": status,
        "reasons": reasons,
        "promotion_status": promotion_status,
        "promotion_reasons": promotion_reasons,
        "seed": seed,
        "device": str(device),
        "split_manifest": str(split_manifest),
        "resolved_spec": str(resolved_spec),
        "train_split": train_split,
        "validation_split": validation_split,
        "train_profile_count": len(train_rows),
        "validation_profile_count": len(val_rows),
        "n_r": n_r,
        "max_time_points": max_time_points,
        "time_window_s": time_window_s,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "voltage_recovery_until_epoch": recovery_until_epoch,
        "best_epoch": int(best["epoch"]),
        "best_score": float(best["score"]),
        "initial_train_aggregate": _jsonable(history[0]),
        "last_epoch_train_aggregate": _jsonable(history[-1]),
        "best_train_aggregate": _jsonable(best.get("aggregate")),
        "final_train_aggregate": _jsonable(train_agg),
        "final_validation_aggregate": _jsonable(val_agg),
        "final_train_loss_component_aggregate": _jsonable(train_loss_agg),
        "final_validation_loss_component_aggregate": _jsonable(val_loss_agg),
        "validation_adaptation": {
            "steps": int(cfg_get(cfg, "validation.adaptation_steps", 80)),
            "uses_state_softlabels": False,
            "uses_observed_voltage": True,
            "history_csv": str(val_hist_path),
        },
        "formula_alignment_audit": _jsonable(formula_audit),
        "residual_budget_audit": _jsonable(budget_audit),
        "train_profile_metrics": _jsonable(train_rows),
        "validation_profile_metrics": _jsonable(val_rows),
        "voltage_recovery": {
            "train_corrected_voltage_mae_mean_V": float(train_agg.get("voltage_mae_V_mean", float("nan"))),
            "train_forward_voltage_mae_mean_V": float(train_agg.get("forward_voltage_mae_V_mean", float("nan"))),
            "validation_corrected_voltage_mae_mean_V": float(val_agg.get("voltage_mae_V_mean", float("nan"))),
            "validation_forward_voltage_mae_mean_V": float(val_agg.get("forward_voltage_mae_V_mean", float("nan"))),
            "corrected_target_V": corrected_target,
            "corrected_target_met": bool(float(val_agg.get("voltage_mae_V_mean", float("inf"))) <= corrected_target),
        },
        "no_state_label_policy": {
            "training_uses_state_softlabels": False,
            "validation_adaptation_uses_state_softlabels": False,
            "profile_loader": "replay_npz observed-only",
            "softlabel_npz": "report-only path in manifest; not loaded by P3.3 trainer",
            "forbidden_state_keys": sorted(FORBIDDEN_PROFILE_KEYS),
            "checkpoint_selection_uses_state_softlabels": False,
            "checkpoint_selection_uses_frozen_test": False,
        },
        "mechanism_notes": {
            "p33_goal": "separate forward electrochemical core from bounded voltage residual and audit the residual budget",
            "d12_formula_migration": "low/transition fade basis and non-low preservation are explicit formulae in gv1/d17_pinn/d12_transition_fade.py",
            "not_promotion_by_voltage_alone": "Corrected V(t) can be excellent while forward core remains REVIEW if residual is large.",
        },
        "prior_snapshot": prior_to_jsonable(prior),
        "selected_train_profiles": train_manifest,
        "selected_validation_profiles": val_manifest,
        "outputs": {
            "summary_json": str(out_dir / "D17_P33_FORWARD_CORE_RELIABILITY_SUMMARY.json"),
            "formula_alignment_audit_json": str(out_dir / "D17_P33_FORMULA_ALIGNMENT_AUDIT.json"),
            "residual_budget_audit_json": str(out_dir / "D17_P33_RESIDUAL_BUDGET_AUDIT.json"),
            "training_history_train_csv": str(hist_path),
            "validation_adaptation_history_csv": str(val_hist_path),
            "selected_profiles_train_json": str(out_dir / "selected_profiles_train.json"),
            "selected_profiles_validation_json": str(out_dir / "selected_profiles_validation.json"),
            "best_model_and_latents_pt": str(out_dir / "model" / "best_model_and_latents.pt"),
            "last_model_and_latents_pt": str(out_dir / "model" / "last_model_and_latents.pt"),
            "predictions_dir": str(out_dir / "predictions"),
        },
    }
    (out_dir / "D17_P33_FORWARD_CORE_RELIABILITY_SUMMARY.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
