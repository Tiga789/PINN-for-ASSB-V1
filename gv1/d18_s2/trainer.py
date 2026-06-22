from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .common import ConfigError, choose_device, dump_json, ensure_dir, read_csv, seed_everything, utc_now_iso, write_csv
from .data import load_prepared_npz
from .losses import S2LossConfig, compute_loss
from .model import CycleAwareS2Operator, S2ModelConfig, architecture_contract


@dataclass(frozen=True)
class S2TrainerConfig:
    seed: int = 1802
    device: str = "auto"
    epochs: int = 8
    batch_size_profiles: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    grad_clip_norm: float = 5.0
    internal_selection_weight: float = 0.25
    min_relative_train_improvement: float = 0.005
    save_predictions: bool = True
    disable_amp: bool = True
    disable_torch_compile: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "S2TrainerConfig":
        if not value:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: value[k] for k in value if k in allowed})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


TENSOR_KEYS = [
    "cycle_features",
    "local_features",
    "cycle_index",
    "cbar_baseline",
    "cbar_true_report_only",
    "potential_baseline",
    "theta_offset",
    "theta_scale",
    "selected_cycle_ids",
    "cs_a_true",
    "cs_c_true",
    "theta_a_true",
    "theta_c_true",
    "phie_true",
    "phis_c_true",
]


@dataclass
class ProfileTensor:
    uid: str
    role: str
    protocol: str
    branch_family: str
    path: str
    arrays: dict[str, torch.Tensor]

    @property
    def branch_id(self) -> int:
        return 0 if self.branch_family == "RG" else 1


def _scalar_text(value: np.ndarray) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="replace")
    return str(item)


def load_profile_tensor(path: str | Path) -> ProfileTensor:
    data = load_prepared_npz(path)
    arrays: dict[str, torch.Tensor] = {}
    for key in TENSOR_KEYS:
        if key not in data:
            raise ConfigError(f"Prepared profile missing {key}: {path}")
        array = data[key]
        if key in {"cycle_index", "selected_cycle_ids"}:
            arrays[key] = torch.as_tensor(array, dtype=torch.long)
        else:
            arrays[key] = torch.as_tensor(array, dtype=torch.float32)
    return ProfileTensor(
        uid=_scalar_text(data["canonical_cell_uid"]),
        role=_scalar_text(data["role"]),
        protocol=_scalar_text(data["protocol"]),
        branch_family=_scalar_text(data["branch_family"]),
        path=str(path),
        arrays=arrays,
    )


def load_casepack(casepack_profiles_dir: str | Path) -> list[ProfileTensor]:
    root = Path(casepack_profiles_dir)
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise ConfigError(f"No prepared profiles found under {root}")
    profiles = [load_profile_tensor(path) for path in paths]
    shapes = {(tuple(p.arrays["cycle_features"].shape), tuple(p.arrays["local_features"].shape)) for p in profiles}
    if len(shapes) != 1:
        raise ConfigError(f"Prepared profiles do not share fixed micro-smoke shapes: {shapes}")
    return profiles


def collate(profiles: Sequence[ProfileTensor], device: str) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    for key in TENSOR_KEYS:
        batch[key] = torch.stack([p.arrays[key] for p in profiles], dim=0).to(device)
    batch["branch_id"] = torch.tensor([p.branch_id for p in profiles], dtype=torch.long, device=device)
    return batch


def _forward(model: CycleAwareS2Operator, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        cycle_features=batch["cycle_features"],
        local_features=batch["local_features"],
        cycle_index=batch["cycle_index"],
        cbar_baseline=batch["cbar_baseline"],
        potential_baseline=batch["potential_baseline"],
        branch_id=batch["branch_id"],
        theta_offset=batch["theta_offset"],
        theta_scale=batch["theta_scale"],
    )


def _iter_batches(profiles: Sequence[ProfileTensor], batch_size: int, rng: random.Random, shuffle: bool) -> list[list[ProfileTensor]]:
    items = list(profiles)
    if shuffle:
        rng.shuffle(items)
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _evaluate_loss(
    model: CycleAwareS2Operator,
    profiles: Sequence[ProfileTensor],
    device: str,
    batch_size: int,
    target_scales: Mapping[str, float],
    loss_cfg: S2LossConfig,
) -> tuple[float, dict[str, float]]:
    if not profiles:
        return float("nan"), {}
    model.eval()
    totals: list[float] = []
    component_acc: dict[str, list[float]] = {}
    with torch.no_grad():
        for group in _iter_batches(profiles, batch_size, random.Random(0), False):
            batch = collate(group, device)
            out = _forward(model, batch)
            loss, components = compute_loss(out, batch, target_scales, loss_cfg)
            totals.append(float(loss.detach().cpu()))
            for key, value in components.items():
                component_acc.setdefault(key, []).append(float(value.detach().cpu()))
    return float(np.mean(totals)), {k: float(np.mean(v)) for k, v in component_acc.items()}


def _adapter_grad_norm(model: CycleAwareS2Operator, branch_id: int) -> float:
    total = 0.0
    for p in model.branch_adapters.adapters[branch_id].parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).cpu())
    return math.sqrt(total)


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(true, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if y.size < 2:
        return float("nan")
    ss = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss > 1e-20 else float("nan")


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(true, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if y.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan")}
    err = p - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r2": _r2(y, p),
    }


def _physical_audit(model: CycleAwareS2Operator, outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    mean_a = model.basis_a.weighted_mean(outputs["delta_cs_a"])
    mean_c = model.basis_c.weighted_mean(outputs["delta_cs_c"])
    relation_a = batch["theta_offset"][:, None, 0:1] + batch["theta_scale"][:, None, 0:1] * outputs["cs_a"]
    relation_c = batch["theta_offset"][:, None, 1:2] + batch["theta_scale"][:, None, 1:2] * outputs["cs_c"]
    checks = {
        "finite": all(bool(torch.isfinite(outputs[k]).all()) for k in ("cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c")),
        "zero_mean": float(torch.max(torch.abs(mean_a)).cpu()) < 1e-4 and float(torch.max(torch.abs(mean_c)).cpu()) < 1e-4,
        "theta_relation": float(torch.max(torch.abs(outputs["theta_a"] - relation_a)).cpu()) < 1e-5
        and float(torch.max(torch.abs(outputs["theta_c"] - relation_c)).cpu()) < 1e-5,
        "theta_bounds": bool(((outputs["theta_a"] >= -1e-6) & (outputs["theta_a"] <= 1 + 1e-6)).all())
        and bool(((outputs["theta_c"] >= -1e-6) & (outputs["theta_c"] <= 1 + 1e-6)).all()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "zero_mean_max_a": float(torch.max(torch.abs(mean_a)).cpu()),
        "zero_mean_max_c": float(torch.max(torch.abs(mean_c)).cpu()),
        "theta_relation_max_a": float(torch.max(torch.abs(outputs["theta_a"] - relation_a)).cpu()),
        "theta_relation_max_c": float(torch.max(torch.abs(outputs["theta_c"] - relation_c)).cpu()),
    }


def evaluate_profiles(
    model: CycleAwareS2Operator,
    profiles: Sequence[ProfileTensor],
    device: str,
    output_dir: str | Path,
    save_predictions: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    physical: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for profile in profiles:
            batch = collate([profile], device)
            out = _forward(model, batch)
            physical.append({"canonical_cell_uid": profile.uid, **_physical_audit(model, out, batch)})
            pred_arrays: dict[str, np.ndarray] = {}
            for state in ("cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c"):
                pred = out[state][0].detach().cpu().numpy()
                true = batch[f"{state}_true"][0].detach().cpu().numpy()
                pred_arrays[state] = pred
                metric = _metrics(true, pred)
                rows.append(
                    {
                        "canonical_cell_uid": profile.uid,
                        "role": profile.role,
                        "protocol": profile.protocol,
                        "branch_family": profile.branch_family,
                        "state": state,
                        **metric,
                    }
                )
            if save_predictions:
                target = ensure_dir(out_dir / profile.role) / f"{profile.uid}_PRED.npz"
                np.savez_compressed(
                    target,
                    canonical_cell_uid=np.array(profile.uid),
                    role=np.array(profile.role),
                    protocol=np.array(profile.protocol),
                    branch_family=np.array(profile.branch_family),
                    selected_cycle_ids=profile.arrays["selected_cycle_ids"].numpy(),
                    **{f"{k}_pred": v for k, v in pred_arrays.items()},
                    **{f"{k}_true_report_only": profile.arrays[f"{k}_true"].numpy() for k in pred_arrays},
                )
    role_summary: dict[str, Any] = {}
    for role in sorted({r["role"] for r in rows}):
        subset = [r for r in rows if r["role"] == role]
        role_summary[role] = {
            "row_count": len(subset),
            "mean_r2": float(np.nanmean([r["r2"] for r in subset])),
            "min_r2": float(np.nanmin([r["r2"] for r in subset])),
            "mean_mae": float(np.nanmean([r["mae"] for r in subset])),
        }
    return rows, {"roles": role_summary, "physical_audits": physical}


def run_micro_smoke(
    *,
    casepack_profiles_dir: str | Path,
    casepack_summary_path: str | Path,
    output_dir: str | Path,
    model_config: S2ModelConfig,
    loss_config: S2LossConfig,
    trainer_config: S2TrainerConfig,
    progress: callable | None = None,
) -> dict[str, Any]:
    seed_everything(trainer_config.seed)
    device = choose_device(trainer_config.device)
    output = ensure_dir(output_dir)
    profiles = load_casepack(casepack_profiles_dir)
    fit = [p for p in profiles if p.role == "fit_train"]
    internal = [p for p in profiles if p.role == "internal_heldout"]
    validation = [p for p in profiles if p.role == "validation_report_only"]
    if not fit or not internal or not validation:
        raise ConfigError("Micro-smoke requires fit_train, internal_heldout, and validation_report_only profiles")
    if {p.branch_family for p in fit} != {"RG", "P4D"}:
        raise ConfigError("fit_train must cover both RG and P4D branches")
    if {p.uid for p in fit} & {p.uid for p in internal + validation}:
        raise ConfigError("Profile leakage between fit and report-only roles")

    import json
    with Path(casepack_summary_path).open("r", encoding="utf-8") as f:
        case_summary = json.load(f)
    target_scales = case_summary["physical_fit"]["target_scales"]
    model = CycleAwareS2Operator(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer_config.learning_rate, weight_decay=trainer_config.weight_decay
    )
    rng = random.Random(trainer_config.seed)
    initial_train_loss, _ = _evaluate_loss(
        model, fit, device, trainer_config.batch_size_profiles, target_scales, loss_config
    )
    initial_internal_loss, _ = _evaluate_loss(
        model, internal, device, trainer_config.batch_size_profiles, target_scales, loss_config
    )
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = -1
    best_path = ensure_dir(output / "model") / "best_micro_smoke.pt"
    branch_grad_max = {"RG": 0.0, "P4D": 0.0}

    for epoch in range(1, trainer_config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for group in _iter_batches(fit, trainer_config.batch_size_profiles, rng, True):
            batch = collate(group, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward(model, batch)
            loss, _ = compute_loss(outputs, batch, target_scales, loss_config)
            if not torch.isfinite(loss):
                raise ConfigError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            present = {p.branch_id for p in group}
            for branch_id in present:
                name = "RG" if branch_id == 0 else "P4D"
                branch_grad_max[name] = max(branch_grad_max[name], _adapter_grad_norm(model, branch_id))
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        train_loss, train_components = _evaluate_loss(
            model, fit, device, trainer_config.batch_size_profiles, target_scales, loss_config
        )
        internal_loss, internal_components = _evaluate_loss(
            model, internal, device, trainer_config.batch_size_profiles, target_scales, loss_config
        )
        score = train_loss + trainer_config.internal_selection_weight * internal_loss
        if math.isfinite(score) and score < best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "stage": "D18-S2-MICRO-SMOKE",
                    "formal_training_eligible": False,
                    "epoch": epoch,
                    "selection_score": score,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config.as_dict(),
                    "loss_config": loss_config.as_dict(),
                    "trainer_config": trainer_config.as_dict(),
                    "target_scales": target_scales,
                    "fit_uids": [p.uid for p in fit],
                    "internal_uids": [p.uid for p in internal],
                    "validation_uids_report_only": [p.uid for p in validation],
                },
                best_path,
            )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "internal_heldout_loss": internal_loss,
            "selection_score": score,
            "best_score_so_far": best_score,
            "selected": epoch == best_epoch,
            "rg_adapter_grad_max": branch_grad_max["RG"],
            "p4d_adapter_grad_max": branch_grad_max["P4D"],
        }
        for key, value in train_components.items():
            row[f"train_{key}"] = value
        for key, value in internal_components.items():
            row[f"internal_{key}"] = value
        history.append(row)
        if progress:
            progress(
                f"epoch={epoch:02d}/{trainer_config.epochs} train={train_loss:.6g} "
                f"internal={internal_loss:.6g} best={best_score:.6g}"
            )

    if best_epoch < 0 or not best_path.exists():
        raise ConfigError("No finite checkpoint was selected")
    try:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only keyword
        checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_train_loss, _ = _evaluate_loss(
        model, fit, device, trainer_config.batch_size_profiles, target_scales, loss_config
    )
    final_internal_loss, _ = _evaluate_loss(
        model, internal, device, trainer_config.batch_size_profiles, target_scales, loss_config
    )
    # Validation is evaluated only after the checkpoint is fixed.
    final_validation_loss, _ = _evaluate_loss(
        model, validation, device, trainer_config.batch_size_profiles, target_scales, loss_config
    )
    metric_rows, metric_summary = evaluate_profiles(
        model, profiles, device, output / "predictions", trainer_config.save_predictions
    )
    write_csv(history, output / "D18_S2_training_history.csv")
    write_csv(metric_rows, output / "D18_S2_metrics_by_profile_state.csv")
    dump_json(metric_summary, output / "D18_S2_metrics_summary.json")
    dump_json(architecture_contract(model_config), output / "D18_S2_MODEL_CONTRACT.json")

    physical_ok = all(x.get("status") == "PASS" for x in metric_summary["physical_audits"])
    train_improvement = (
        (initial_train_loss - final_train_loss) / max(abs(initial_train_loss), 1e-12)
        if math.isfinite(initial_train_loss) and math.isfinite(final_train_loss)
        else float("nan")
    )
    checks = {
        "finite_initial_and_final_loss": all(
            math.isfinite(v) for v in (initial_train_loss, initial_internal_loss, final_train_loss, final_internal_loss, final_validation_loss)
        ),
        "train_loss_improved": train_improvement >= trainer_config.min_relative_train_improvement,
        "rg_adapter_received_gradient": branch_grad_max["RG"] > 0.0,
        "p4d_adapter_received_gradient": branch_grad_max["P4D"] > 0.0,
        "physical_constraints_pass": physical_ok,
        "checkpoint_selection_excluded_validation": True,
        "frozen_test_used": False,
        "amp_disabled": trainer_config.disable_amp,
        "torch_compile_disabled": trainer_config.disable_torch_compile,
        "formal_s2_training_disabled": True,
    }
    status = "PASS_MICRO_SMOKE" if all(checks.values()) else "REVIEW_MICRO_SMOKE"
    summary = {
        "stage": "D18-S2-PREFLIGHT-PLUS-MICRO-SMOKE",
        "created_at_utc": utc_now_iso(),
        "status": status,
        "checks": checks,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "profile_counts": {
            "fit_train": len(fit),
            "internal_heldout": len(internal),
            "validation_report_only": len(validation),
        },
        "fit_train_protocols": sorted({p.protocol for p in fit}),
        "fit_train_branches": sorted({p.branch_family for p in fit}),
        "initial_train_loss": initial_train_loss,
        "initial_internal_loss": initial_internal_loss,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "final_train_loss": final_train_loss,
        "final_internal_loss": final_internal_loss,
        "final_validation_report_only_loss": final_validation_loss,
        "relative_train_improvement": train_improvement,
        "branch_adapter_gradient_max": branch_grad_max,
        "checkpoint_path": str(best_path),
        "checkpoint_selection_policy": "fit_train + internal_heldout only; validation report-only after checkpoint fixed",
        "teacher_initial_cbar_anchor_used": True,
        "micro_smoke_downsampled_view": True,
        "formal_s2_training_eligible": False,
        "go_to_formal_s2_training": False,
        "go_to_s3": False,
        "next_action": "Human review of preflight provenance, source-grid coverage, training history, physical audits, and report-only metrics.",
        "model_config": model_config.as_dict(),
        "loss_config": loss_config.as_dict(),
        "trainer_config": trainer_config.as_dict(),
    }
    dump_json(summary, output / "D18_S2_MICRO_SMOKE_SUMMARY.json")
    return summary
