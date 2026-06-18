# -*- coding: utf-8 -*-
"""D17-P4 report-only internal-state audit.

P4 is deliberately not a training stage.  It freezes the P3.4/P3.4V
candidate, runs observed-only inference/adaptation on train/validation/
frozen-test/flagged-probe replay profiles, saves predictions, and only then
loads P2Dlite-RG soft labels for report-only MAE/R2 audits.

State soft-label arrays must never enter the forward/adaptation loss.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import cfg_get
from .dataset import D17ProfileDataset, load_observed_profile
from .latent_adapter import LATENT_NAMES
from .losses import audit_numbers, total_d17_loss
from .model import D17MechanisticPINN, make_batch_from_profile
from .p2dlite_prior import D17P2DlitePrior, load_p2dlite_prior, prior_to_jsonable
from .radial_fv_core import radial_gradient_audit, radial_volume_weights, zero_mean_error
from .trainer import assert_no_state_profile_keys, choose_device, set_seed
try:
    from .p3_trainer import aggregate, normalize_protocol
except Exception:  # pragma: no cover - fallback for older trees
    def normalize_protocol(record: Mapping[str, Any]) -> str:
        text = (str(record.get("protocol", "")) + " " + str(record.get("canonical_cell_uid", record.get("cell_uid", "")))).lower()
        if "r2.5" in text:
            return "R2.5"
        if "r3" in text:
            return "R3"
        if "2c" in text:
            return "2C"
        if "batch-2" in text or "3c" in text:
            return "3C"
        if "batch-5" in text:
            return "RW"
        if "batch-6" in text:
            return "GEO"
        return str(record.get("protocol", "UNKNOWN"))

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and math.isfinite(float(v))})
        out: Dict[str, float] = {}
        for k in keys:
            vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], (int, float)) and math.isfinite(float(r[k]))]
            if vals:
                out[f"{k}_mean"] = float(np.mean(vals))
                out[f"{k}_max"] = float(np.max(vals))
                out[f"{k}_min"] = float(np.min(vals))
        return out


STATE_KEYS = ("cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c")
FIELD_STATE_KEYS = ("cs_a", "cs_c", "theta_a", "theta_c")
POTENTIAL_STATE_KEYS = ("phie", "phis_c")


def jsonable(x: Any) -> Any:
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
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def safe_read_json(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha256_file(path: str | Path | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def crop_indices_from_time(t: np.ndarray, time_window_s: Optional[float], max_time_points: Optional[int]) -> np.ndarray:
    tt = np.asarray(t, dtype=np.float64).reshape(-1)
    if tt.size == 0:
        return np.asarray([], dtype=int)
    t0 = float(tt[0])
    mask = np.ones_like(tt, dtype=bool)
    if time_window_s is not None and float(time_window_s) > 0:
        mask &= (tt - t0) <= float(time_window_s)
    idx = np.where(mask)[0]
    if idx.size < 8:
        idx = np.arange(min(tt.size, 8), dtype=int)
    if max_time_points is not None and int(max_time_points) > 0 and idx.size > int(max_time_points):
        idx = np.linspace(idx[0], idx[-1], int(max_time_points)).round().astype(int)
    return idx.astype(int)


def apply_time_indices(profile: Mapping[str, Any], idx: np.ndarray, n_full: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in profile.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_full:
            out[k] = v[idx]
        else:
            out[k] = v
    out["_crop_indices"] = idx.astype(int)
    out["_crop_indices_count"] = int(idx.size)
    return out


def load_observed_profile_with_crop(record: Mapping[str, Any], time_window_s: float, max_time_points: int) -> Tuple[Dict[str, Any], np.ndarray, int]:
    replay_npz = record.get("replay_npz")
    if not replay_npz:
        raise RuntimeError(f"P4 record has no replay_npz: {record.get('canonical_cell_uid', record.get('cell_uid'))}")
    profile_full = load_observed_profile(replay_npz)
    assert_no_state_profile_keys(profile_full)
    t_key = "t_global_s" if "t_global_s" in profile_full else "time_s"
    t_full = np.asarray(profile_full[t_key], dtype=np.float64).reshape(-1)
    idx = crop_indices_from_time(t_full, time_window_s=time_window_s, max_time_points=max_time_points)
    profile = apply_time_indices(profile_full, idx, n_full=len(t_full))
    profile["_manifest_record"] = dict(record)
    return profile, idx, len(t_full)


def make_model_from_config(cfg: Mapping[str, Any], prior: D17P2DlitePrior, feature_dim: int, n_r: int, device: torch.device) -> D17MechanisticPINN:
    return D17MechanisticPINN(
        prior=prior,
        feature_dim=feature_dim,
        n_r=n_r,
        hidden_dim=int(cfg_get(cfg, "model.hidden_dim", 96)),
        latent_hidden_dim=int(cfg_get(cfg, "model.latent_hidden_dim", 96)),
        delta_layers=int(cfg_get(cfg, "model.delta_layers", 3)),
        delta_amp_fraction=float(cfg_get(cfg, "model.delta_amp_fraction", 0.014)),
        enable_low_transition_residual=bool(cfg_get(cfg, "model.enable_low_transition_residual", True)),
        use_observed_voltage_for_gate=bool(cfg_get(cfg, "model.use_observed_voltage_for_gate", True)),
        enable_voltage_inverse_residual=bool(cfg_get(cfg, "model.enable_voltage_inverse_residual", True)),
        voltage_inverse_residual_amp_V=float(cfg_get(cfg, "model.voltage_inverse_residual_amp_V", 0.055)),
        voltage_inverse_residual_gate_mode=str(cfg_get(cfg, "model.voltage_inverse_residual_gate_mode", "d12_transition_fade")),
        enable_voltage_basis_residual=bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True)),
        voltage_basis_residual_amp_V=float(cfg_get(cfg, "model.voltage_basis_residual_amp_V", 0.035)),
        voltage_basis_count=int(cfg_get(cfg, "model.voltage_basis_count", 10)),
        voltage_basis_formula_mode=str(cfg_get(cfg, "model.voltage_basis_formula_mode", "d12_transition_fade")),
        d12_low_v=float(cfg_get(cfg, "d12_transition_fade.low_v", 2.75)),
        d12_normal_v=float(cfg_get(cfg, "d12_transition_fade.normal_v", 3.05)),
        d12_low_width_v=float(cfg_get(cfg, "d12_transition_fade.low_width_v", 0.055)),
        d12_transition_width_v=float(cfg_get(cfg, "d12_transition_fade.transition_width_v", 0.080)),
        d12_transition_gain=float(cfg_get(cfg, "d12_transition_fade.transition_gain", 0.70)),
        d12_non_low_preservation_floor=float(cfg_get(cfg, "d12_transition_fade.non_low_preservation_floor", 0.02)),
    ).to(device)


def load_candidate_model(
    cfg: Mapping[str, Any],
    checkpoint_path: str | Path,
    prior: D17P2DlitePrior,
    feature_dim: int,
    n_r: int,
    device: torch.device,
) -> D17MechanisticPINN:
    model = make_model_from_config(cfg, prior, feature_dim, n_r, device)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"P4 checkpoint not found: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, Mapping) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def adaptation_weights(cfg: Mapping[str, Any]) -> Mapping[str, float]:
    w = cfg.get("p4_adaptation_loss_weights")
    if isinstance(w, Mapping):
        return w  # type: ignore[return-value]
    w = cfg.get("validation_loss_weights")
    if isinstance(w, Mapping):
        return w  # type: ignore[return-value]
    w = cfg.get("loss_weights")
    return w if isinstance(w, Mapping) else {}


def adapt_one_profile(
    *,
    model: D17MechanisticPINN,
    prior: D17P2DlitePrior,
    cfg: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    steps: int,
    lr: float,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor], List[Dict[str, Any]]]:
    enable_basis = bool(cfg_get(cfg, "model.enable_voltage_basis_residual", True))
    k_basis = int(cfg_get(cfg, "model.voltage_basis_count", 10))
    latent = torch.nn.Parameter(torch.zeros(1, len(LATENT_NAMES), device=device))
    basis = torch.nn.Parameter(torch.zeros(1, k_basis, device=device)) if enable_basis else None
    params: List[torch.nn.Parameter] = [latent]
    if basis is not None:
        params.append(basis)
    opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
    weights = adaptation_weights(cfg)
    hist: List[Dict[str, Any]] = []
    old_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    try:
        for step in range(1, int(steps) + 1):
            opt.zero_grad(set_to_none=True)
            b = dict(batch)
            b["latent_raw_offset"] = latent
            if basis is not None:
                b["voltage_basis_raw_coeffs"] = basis
            pred = model(b)
            loss, terms = total_d17_loss(pred, b, prior, weights=weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"P4 non-finite observed-only adaptation loss at step {step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg_get(cfg, "p4.gradient_clip_norm", 10.0)))
            opt.step()
            log_every = max(1, int(steps) // 5)
            if step == 1 or step == int(steps) or step % log_every == 0:
                with torch.no_grad():
                    m = audit_numbers(pred, b)
                    row: Dict[str, Any] = {"adapt_step": step, "adapt_loss": float(loss.detach().cpu())}
                    row.update({k: float(v.detach().cpu()) for k, v in terms.items() if isinstance(v, torch.Tensor) and v.numel() == 1 and torch.isfinite(v)})
                    row.update(m)
                    hist.append(row)
        with torch.no_grad():
            b = dict(batch)
            b["latent_raw_offset"] = latent
            if basis is not None:
                b["voltage_basis_raw_coeffs"] = basis
            pred_final = model(b)
    finally:
        for p, flag in zip(model.parameters(), old_flags):
            p.requires_grad_(flag)
    return pred_final, latent.detach(), basis.detach() if basis is not None else None, hist


def tensor_to_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_prediction_npz(
    path: str | Path,
    record: Mapping[str, Any],
    profile: Mapping[str, Any],
    batch: Mapping[str, torch.Tensor],
    pred: Mapping[str, torch.Tensor],
    latent: torch.Tensor,
    basis: Optional[torch.Tensor],
) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, Any] = {
        "t_s": tensor_to_np(batch["t_s"]),
        "I_profile": tensor_to_np(batch["current_A"]),
        "voltage_exp": tensor_to_np(batch["voltage_exp"]),
        "temperature_C": tensor_to_np(batch["temperature_C"]),
        "r_norm": tensor_to_np(batch["r_norm"]),
        "cs_a": tensor_to_np(pred["cs_a"]),
        "cs_c": tensor_to_np(pred["cs_c"]),
        "theta_a": tensor_to_np(pred["theta_a"]),
        "theta_c": tensor_to_np(pred["theta_c"]),
        "phie": tensor_to_np(pred["phie"]),
        "phis_c": tensor_to_np(pred["phis_c"]),
        "cbar_a": tensor_to_np(pred["cbar_a"]),
        "cbar_c": tensor_to_np(pred["cbar_c"]),
        "delta_a": tensor_to_np(pred["delta_a"]),
        "delta_c": tensor_to_np(pred["delta_c"]),
        "V_pred": tensor_to_np(pred["V_pred"]),
        "V_pred_forward": tensor_to_np(pred.get("V_pred_forward", pred["V_pred"])),
        "V_residual_total": tensor_to_np(pred.get("V_residual_total", torch.zeros_like(pred["V_pred"]))),
        "V_residual_local": tensor_to_np(pred.get("V_residual_local", torch.zeros_like(pred["V_pred"]))),
        "V_residual_inverse": tensor_to_np(pred.get("V_residual_inverse", torch.zeros_like(pred["V_pred"]))),
        "V_residual_basis": tensor_to_np(pred.get("V_residual_basis", torch.zeros_like(pred["V_pred"]))),
        "low_transition_gate": tensor_to_np(pred.get("low_transition_gate", torch.zeros_like(pred["V_pred"]))),
        "voltage_inverse_gate": tensor_to_np(pred.get("voltage_inverse_gate", torch.zeros_like(pred["V_pred"]))),
        "latent_raw_offset": tensor_to_np(latent),
        "voltage_basis_raw_coeffs": tensor_to_np(basis) if basis is not None else np.zeros((1, 0), dtype=np.float32),
        "crop_indices": np.asarray(profile.get("_crop_indices", []), dtype=np.int64),
    }
    meta = {
        "canonical_cell_uid": record.get("canonical_cell_uid"),
        "cell_uid": record.get("cell_uid"),
        "split": record.get("split"),
        "protocol": normalize_protocol(record),
        "replay_npz": record.get("replay_npz"),
        "softlabel_npz_report_only": record.get("softlabel_npz"),
        "is_flagged_probe": bool(record.get("is_flagged_probe") or record.get("split") == "flagged_probe"),
    }
    arrays["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(p, **arrays)
    return meta


def finite_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    a = np.asarray(y_true, dtype=np.float64).reshape(-1)
    b = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan"), "corr": float("nan")}
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r2": float("nan"), "corr": float("nan")}
    a, b = a[m], b[m]
    err = b - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(np.mean(err))
    denom = float(np.sum((a - np.mean(a)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / denom) if denom > 1e-18 else float("nan")
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size >= 3 and np.std(a) > 1e-12 and np.std(b) > 1e-12 else float("nan")
    return {"n": int(a.size), "mae": mae, "rmse": rmse, "bias": bias, "r2": r2, "corr": corr}


def _as_numeric_array(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S", "O"}:
        try:
            arr = arr.astype(float)
        except Exception:
            return np.asarray([], dtype=float)
    return np.asarray(arr, dtype=np.float64)


def _downsample_or_align_time(
    arr: np.ndarray,
    *,
    pred_n: int,
    replay_full_n: int,
    crop_idx: np.ndarray,
    soft_time: Optional[np.ndarray],
    pred_time: np.ndarray,
) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        return a
    n0 = a.shape[0]
    if n0 == pred_n:
        return a
    if n0 == replay_full_n and crop_idx.size == pred_n and int(np.max(crop_idx)) < n0:
        return a[crop_idx]
    if soft_time is not None and len(soft_time) == n0 and pred_time.size == pred_n:
        st = np.asarray(soft_time, dtype=np.float64).reshape(-1)
        pt = np.asarray(pred_time, dtype=np.float64).reshape(-1)
        if st.size == n0 and pt.size:
            # Both time arrays may be absolute or re-zeroed.  Compare after zeroing.
            stz = st - st[0]
            ptz = pt - pt[0]
            inds = np.searchsorted(stz, ptz, side="left")
            inds = np.clip(inds, 0, n0 - 1)
            left = np.clip(inds - 1, 0, n0 - 1)
            choose_left = np.abs(stz[left] - ptz) < np.abs(stz[inds] - ptz)
            inds = np.where(choose_left, left, inds)
            return a[inds]
    if n0 > 0 and pred_n > 0:
        inds = np.linspace(0, n0 - 1, pred_n).round().astype(int)
        return a[inds]
    return a


def _interpolate_radial(arr: np.ndarray, target_r: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] == target_r:
        return a
    if a.shape[1] < 2 or target_r < 2:
        return a[:, :target_r]
    x_old = np.linspace(0.0, 1.0, a.shape[1])
    x_new = np.linspace(0.0, 1.0, target_r)
    return np.vstack([np.interp(x_new, x_old, row) for row in a])


def load_softlabel_targets_report_only(
    softlabel_npz: str | Path,
    pred_npz: str | Path,
    replay_full_n: int,
    crop_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    p = Path(softlabel_npz)
    if not p.exists():
        raise FileNotFoundError(f"Report-only softlabel file not found: {p}")
    pred = np.load(pred_npz, allow_pickle=True)
    pred_time = np.asarray(pred["t_s"], dtype=np.float64).reshape(-1)
    pred_n = int(pred_time.size)
    r_n = int(np.asarray(pred["r_norm"]).reshape(-1).size)
    out: Dict[str, np.ndarray] = {}
    with np.load(p, allow_pickle=True) as data:
        soft_time: Optional[np.ndarray] = None
        for tk in ("t_global_s", "time_s", "t_s", "time"):
            if tk in data.files:
                soft_time = _as_numeric_array(data[tk]).reshape(-1)
                break
        for key in STATE_KEYS:
            if key not in data.files:
                continue
            arr = _as_numeric_array(data[key])
            if arr.size == 0:
                continue
            arr = np.squeeze(arr)
            arr = _downsample_or_align_time(arr, pred_n=pred_n, replay_full_n=replay_full_n, crop_idx=crop_idx, soft_time=soft_time, pred_time=pred_time)
            if key in FIELD_STATE_KEYS and arr.ndim == 2:
                arr = _interpolate_radial(arr, r_n)
            elif key in POTENTIAL_STATE_KEYS and arr.ndim > 1:
                # phie/phis_c may be [time,1] or [time, x].  Use the first/effective channel.
                arr = arr.reshape(arr.shape[0], -1)[:, 0]
            out[key] = np.asarray(arr, dtype=np.float64)
    pred.close()
    return out


def add_prefixed_metrics(row: MutableMapping[str, Any], prefix: str, metrics: Mapping[str, float]) -> None:
    for k, v in metrics.items():
        row[f"{prefix}_{k}"] = v


def state_report_metrics_for_profile(
    *,
    pred_npz: str | Path,
    record: Mapping[str, Any],
    replay_full_n: int,
    crop_idx: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    softlabel_npz = record.get("softlabel_npz")
    if not softlabel_npz:
        raise RuntimeError(f"No softlabel_npz in manifest record: {record.get('canonical_cell_uid')}")
    pred = np.load(pred_npz, allow_pickle=True)
    targets = load_softlabel_targets_report_only(softlabel_npz, pred_npz, replay_full_n, crop_idx)
    row: Dict[str, Any] = {
        "split": record.get("split"),
        "canonical_cell_uid": record.get("canonical_cell_uid"),
        "cell_uid": record.get("cell_uid"),
        "protocol": normalize_protocol(record),
        "is_flagged_probe": bool(record.get("is_flagged_probe") or record.get("split") == "flagged_probe"),
        "softlabel_npz_report_only": softlabel_npz,
        "pred_npz": str(pred_npz),
    }
    radial_row: Dict[str, Any] = dict(row)
    volt_row: Dict[str, Any] = dict(row)
    for key in STATE_KEYS:
        if key not in targets or key not in pred.files:
            row[f"{key}_missing"] = True
            continue
        true = np.asarray(targets[key], dtype=np.float64)
        ypred = np.asarray(pred[key], dtype=np.float64)
        if key in FIELD_STATE_KEYS:
            if not (true.ndim == 2 and ypred.ndim == 2):
                row[f"{key}_missing"] = True
                row[f"{key}_invalid_shape"] = f"target_shape={true.shape}, pred_shape={ypred.shape}"
                continue
            true = _interpolate_radial(true, ypred.shape[1])
        if key in POTENTIAL_STATE_KEYS and ypred.ndim > 1:
            ypred = ypred.reshape(ypred.shape[0], -1)[:, 0]
        add_prefixed_metrics(row, key, finite_metrics(true, ypred))
        if key in FIELD_STATE_KEYS and true.ndim == 2 and ypred.ndim == 2:
            w = radial_volume_weights(np.linspace(0.0, 1.0, ypred.shape[1]))
            true_bar = np.sum(true * w.reshape(1, -1), axis=1)
            pred_bar = np.sum(ypred * w.reshape(1, -1), axis=1)
            add_prefixed_metrics(row, f"{key}_radial_mean", finite_metrics(true_bar, pred_bar))
            add_prefixed_metrics(row, f"{key}_surface", finite_metrics(true[:, -1], ypred[:, -1]))
            add_prefixed_metrics(row, f"{key}_center", finite_metrics(true[:, 0], ypred[:, 0]))
            true_grad = true[:, -1] - true[:, 0]
            pred_grad = ypred[:, -1] - ypred[:, 0]
            add_prefixed_metrics(row, f"{key}_surface_minus_center", finite_metrics(true_grad, pred_grad))
            radial_row[f"{key}_pred_surface_minus_center_abs_mean"] = float(np.mean(np.abs(pred_grad)))
            radial_row[f"{key}_true_surface_minus_center_abs_mean"] = float(np.mean(np.abs(true_grad)))
            radial_row[f"{key}_gradient_sign_agreement_fraction"] = float(np.mean(np.sign(true_grad) == np.sign(pred_grad))) if true_grad.size else float("nan")
    # mechanism audits from predictions only
    r = np.asarray(pred["r_norm"], dtype=np.float64).reshape(-1)
    for key in ("delta_a", "delta_c"):
        if key in pred.files:
            radial_row[f"{key}_zero_mean_max_abs"] = zero_mean_error(np.asarray(pred[key], dtype=np.float64), r)
    for key in ("cs_a", "cs_c"):
        if key in pred.files:
            try:
                aud = radial_gradient_audit(np.asarray(pred[key], dtype=np.float64), r)
                for ak, av in aud.items():
                    radial_row[f"{key}_{ak}"] = av
            except Exception:
                pass
    if "voltage_exp" in pred.files:
        vexp = np.asarray(pred["voltage_exp"], dtype=np.float64)
        vpred = np.asarray(pred["V_pred"], dtype=np.float64) if "V_pred" in pred.files else np.full_like(vexp, np.nan)
        vfwd = np.asarray(pred["V_pred_forward"], dtype=np.float64) if "V_pred_forward" in pred.files else vpred
        vres = np.asarray(pred["V_residual_total"], dtype=np.float64) if "V_residual_total" in pred.files else np.zeros_like(vexp)
        add_prefixed_metrics(volt_row, "corrected_voltage", finite_metrics(vexp, vpred))
        add_prefixed_metrics(volt_row, "forward_voltage", finite_metrics(vexp, vfwd))
        volt_row["residual_total_abs_mean_V"] = float(np.mean(np.abs(vres))) if vres.size else float("nan")
        volt_row["residual_total_abs_max_V"] = float(np.max(np.abs(vres))) if vres.size else float("nan")
    pred.close()
    return row, radial_row, volt_row


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: jsonable(v) for k, v in r.items()})


def split_records_from_manifest(split_manifest: str | Path, split: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = D17ProfileDataset(split_manifest=split_manifest, split=split, allow_softlabel_npz_profile_source=False)
    records = [dict(r) for r in ds.records if r.get("replay_npz")]
    records.sort(key=lambda r: str(r.get("canonical_cell_uid", r.get("cell_uid", ""))))
    if limit is not None and int(limit) >= 0:
        records = records[: int(limit)]
    return records


def build_freeze_manifest(
    *,
    cfg: Mapping[str, Any],
    out_dir: Path,
    p34_dir: Path,
    p34v_dir: Optional[Path],
    split_manifest: Path,
    resolved_spec: Path,
    checkpoint: Path,
) -> Dict[str, Any]:
    p34_summary_path = p34_dir / "D17_P34_FINAL_FORWARD_CORE_PROMOTION_SUMMARY.json"
    p34v_summary_path = (p34v_dir / "D17_P34V_FINAL_VALIDATION_POLISH_SUMMARY.json") if p34v_dir else None
    no_state_path = Path(str(cfg_get(cfg, "paths.no_state_label_audit", ""))) if str(cfg_get(cfg, "paths.no_state_label_audit", "")) else None
    p34 = safe_read_json(p34_summary_path)
    p34v = safe_read_json(p34v_summary_path) if p34v_summary_path else {}
    manifest = {
        "protocol": "D17-P4_FREEZE_CANDIDATE",
        "candidate_id": "D17-P3.4V" if p34v else "D17-P3.4",
        "p4_entry_mode": "report_only_state_audit_after_p34v_review",
        "p34_status": p34.get("status"),
        "p34_promotion_status": p34.get("promotion_status"),
        "p34_p4_ready": p34.get("p4_ready"),
        "p34v_status": p34v.get("status") if p34v else None,
        "p34v_promotion_status": p34v.get("promotion_status") if p34v else None,
        "p34v_p4_ready": p34v.get("p4_ready") if p34v else None,
        "p4_entry_reason": "P3 stopped; P4 is report-only state audit. No further validation voltage/state tuning is allowed.",
        "softlabels_report_only": True,
        "state_softlabels_used_for_inference_or_adaptation": False,
        "candidate_paths": {
            "p34_dir": str(p34_dir),
            "p34v_dir": str(p34v_dir) if p34v_dir else "",
            "split_manifest": str(split_manifest),
            "resolved_spec": str(resolved_spec),
            "checkpoint": str(checkpoint),
            "no_state_label_audit": str(no_state_path) if no_state_path else "",
        },
        "sha256": {
            "p34_summary": sha256_file(p34_summary_path),
            "p34v_summary": sha256_file(p34v_summary_path) if p34v_summary_path else "",
            "split_manifest": sha256_file(split_manifest),
            "resolved_spec": sha256_file(resolved_spec),
            "checkpoint": sha256_file(checkpoint),
            "no_state_label_audit": sha256_file(no_state_path) if no_state_path else "",
        },
        "no_state_label_recheck": safe_read_json(no_state_path) if no_state_path else {},
    }
    (out_dir / "00_freeze_candidate").mkdir(parents=True, exist_ok=True)
    (out_dir / "00_freeze_candidate" / "D17_P4_FREEZE_MANIFEST.json").write_text(json.dumps(jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_p4_report_only_state_audit(cfg: Mapping[str, Any], out_dir: str | Path) -> Dict[str, Any]:
    out = Path(out_dir)
    for sub in ["00_freeze_candidate", "01_inference_predictions", "02_state_report_only_audit", "03_scorecards", "04_figures", "05_decision"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg_get(cfg, "seed", 20260615)))
    device = choose_device(str(cfg_get(cfg, "p4.device", cfg_get(cfg, "train.device", "auto"))))
    split_manifest = Path(str(cfg_get(cfg, "paths.split_manifest")))
    p34_dir = Path(str(cfg_get(cfg, "paths.candidate_p34_dir")))
    p34v_raw = str(cfg_get(cfg, "paths.candidate_p34v_dir", ""))
    p34v_dir = Path(p34v_raw) if p34v_raw else None
    resolved_spec = Path(str(cfg_get(cfg, "paths.resolved_spec", "")))
    if not str(resolved_spec) or not resolved_spec.exists():
        p34_summary = safe_read_json(p34_dir / "D17_P34_FINAL_FORWARD_CORE_PROMOTION_SUMMARY.json")
        resolved_spec = Path(str(p34_summary.get("resolved_spec") or cfg_get(cfg, "paths.resolved_spec")))
    checkpoint = Path(str(cfg_get(cfg, "paths.checkpoint", p34_dir / "model" / "best_model_and_latents.pt")))
    freeze_manifest = build_freeze_manifest(cfg=cfg, out_dir=out, p34_dir=p34_dir, p34v_dir=p34v_dir, split_manifest=split_manifest, resolved_spec=resolved_spec, checkpoint=checkpoint)

    prior = load_p2dlite_prior(resolved_spec, allow_smoke_defaults=True)
    n_r = int(cfg_get(cfg, "p4.n_r", cfg_get(cfg, "train.n_r", 17)))
    time_window_s = float(cfg_get(cfg, "p4.time_window_s", cfg_get(cfg, "train.time_window_s", 40000.0)))
    max_time_points = int(cfg_get(cfg, "p4.max_time_points", cfg_get(cfg, "train.max_time_points", 512)))
    # Load first available profile to determine feature_dim.
    first_records: List[Dict[str, Any]] = []
    for sp in ["train", "validation", "frozen_test", "flagged_probe"]:
        first_records.extend(split_records_from_manifest(split_manifest, sp, limit=1))
        if first_records:
            break
    if not first_records:
        raise RuntimeError("No replay-ready records found in split manifest for P4.")
    first_profile, _, _ = load_observed_profile_with_crop(first_records[0], time_window_s, max_time_points)
    first_batch = make_batch_from_profile(first_profile, n_r=n_r, device=device)
    model = load_candidate_model(cfg, checkpoint, prior, int(first_batch["features"].shape[-1]), n_r, device)

    default_steps = int(cfg_get(cfg, "p4.adaptation_steps", 120))
    split_steps = {
        "train": int(cfg_get(cfg, "p4.train_adaptation_steps", default_steps)),
        "validation": int(cfg_get(cfg, "p4.validation_adaptation_steps", default_steps)),
        "frozen_test": int(cfg_get(cfg, "p4.frozen_test_adaptation_steps", default_steps)),
        "flagged_probe": int(cfg_get(cfg, "p4.flagged_probe_adaptation_steps", max(20, min(default_steps, 80)))),
    }
    lr = float(cfg_get(cfg, "p4.adaptation_lr", cfg_get(cfg, "validation.adaptation_lr", 0.010)))
    limits = {
        "train": cfg_get(cfg, "p4.train_profile_limit", -1),
        "validation": cfg_get(cfg, "p4.validation_profile_limit", -1),
        "frozen_test": cfg_get(cfg, "p4.frozen_test_profile_limit", -1),
        "flagged_probe": cfg_get(cfg, "p4.flagged_probe_profile_limit", -1),
    }
    limits = {k: (None if v is None or int(v) < 0 else int(v)) for k, v in limits.items()}

    inference_rows: List[Dict[str, Any]] = []
    adapt_rows: List[Dict[str, Any]] = []
    pred_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for split in ["train", "validation", "frozen_test", "flagged_probe"]:
        records = split_records_from_manifest(split_manifest, split, limit=limits[split])
        split_dir = out / "01_inference_predictions" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for local_i, rec in enumerate(records):
            profile, crop_idx, replay_full_n = load_observed_profile_with_crop(rec, time_window_s, max_time_points)
            batch = make_batch_from_profile(profile, n_r=n_r, device=device)
            pred, latent, basis, hist = adapt_one_profile(model=model, prior=prior, cfg=cfg, batch=batch, steps=split_steps[split], lr=lr, device=device)
            uid = str(rec.get("canonical_cell_uid", rec.get("cell_uid", f"profile_{local_i}"))).replace("\\", "_").replace("/", "_").replace(":", "_")
            pred_path = split_dir / f"D17_P4_{split}_{local_i:03d}_{uid}_PRED_OBS_ONLY.npz"
            meta = save_prediction_npz(pred_path, rec, profile, batch, pred, latent, basis)
            metrics = audit_numbers(pred, batch)
            row: Dict[str, Any] = {
                "split": split,
                "profile_index": local_i,
                "canonical_cell_uid": rec.get("canonical_cell_uid"),
                "cell_uid": rec.get("cell_uid"),
                "protocol": normalize_protocol(rec),
                "replay_npz": rec.get("replay_npz"),
                "softlabel_npz_report_only": rec.get("softlabel_npz"),
                "pred_npz": str(pred_path),
                "n_time_points": int(batch["t_s"].numel()),
                "adaptation_steps": split_steps[split],
                "report_only_state_softlabels_loaded_during_inference": False,
            }
            row.update(metrics)
            inference_rows.append(row)
            pred_index[(split, str(rec.get("canonical_cell_uid", rec.get("cell_uid"))))] = {
                "pred_npz": str(pred_path),
                "record": rec,
                "replay_full_n": replay_full_n,
                "crop_idx": crop_idx,
            }
            for h in hist:
                hr = {"split": split, "profile_index": local_i, "canonical_cell_uid": row["canonical_cell_uid"]}
                hr.update(h)
                adapt_rows.append(hr)
    write_csv(out / "D17_P4_INFERENCE_MANIFEST.csv", inference_rows)
    write_csv(out / "01_inference_predictions" / "D17_P4_ADAPTATION_HISTORY.csv", adapt_rows)

    # Report-only state audit starts here.  Soft labels are loaded only after all prediction NPZs exist.
    state_rows: List[Dict[str, Any]] = []
    radial_rows: List[Dict[str, Any]] = []
    voltage_rows: List[Dict[str, Any]] = []
    for item in pred_index.values():
        rec = item["record"]
        try:
            srow, rrow, vrow = state_report_metrics_for_profile(pred_npz=item["pred_npz"], record=rec, replay_full_n=int(item["replay_full_n"]), crop_idx=np.asarray(item["crop_idx"], dtype=int))
            state_rows.append(srow)
            radial_rows.append(rrow)
            voltage_rows.append(vrow)
        except Exception as exc:
            state_rows.append({
                "split": rec.get("split"),
                "canonical_cell_uid": rec.get("canonical_cell_uid"),
                "protocol": normalize_protocol(rec),
                "is_flagged_probe": bool(rec.get("is_flagged_probe") or rec.get("split") == "flagged_probe"),
                "audit_error": str(exc),
                "pred_npz": item["pred_npz"],
            })
    write_csv(out / "D17_P4_STATE_AUDIT_PROFILE_METRICS.csv", state_rows)
    write_csv(out / "D17_P4_RADIAL_MECHANISM_AUDIT.csv", radial_rows)
    write_csv(out / "D17_P4_VOLTAGE_STATE_DECOMPOSITION.csv", voltage_rows)

    split_metric_rows: List[Dict[str, Any]] = []
    for split in ["train", "validation", "frozen_test", "flagged_probe"]:
        rows = [r for r in state_rows if r.get("split") == split]
        if not rows:
            continue
        agg = aggregate(rows)
        agg.update({"split": split, "profile_count": len(rows)})
        split_metric_rows.append(agg)
    write_csv(out / "D17_P4_STATE_AUDIT_SPLIT_METRICS.csv", split_metric_rows)

    flagged_rows = [r for r in state_rows if r.get("split") == "flagged_probe" or r.get("is_flagged_probe")]
    (out / "D17_P4_FLAGGED_PROBE_REPORT.json").write_text(json.dumps(jsonable({"profile_count": len(flagged_rows), "profiles": flagged_rows}), ensure_ascii=False, indent=2), encoding="utf-8")

    # Promotion scorecard: normal frozen_test only.  Soft labels are report-only; no changes to candidate are made.
    ft_rows = [r for r in state_rows if r.get("split") == "frozen_test" and not r.get("is_flagged_probe") and not r.get("audit_error")]
    ft_agg = aggregate(ft_rows)
    r2_targets = ["theta_a_r2", "theta_c_r2", "cs_a_r2", "cs_c_r2", "phie_r2", "phis_c_r2"]
    mean_threshold = float(cfg_get(cfg, "p4.promotion_r2_mean_threshold", 0.98))
    min_threshold = float(cfg_get(cfg, "p4.promotion_r2_min_profile_threshold", 0.95))
    target_report: Dict[str, Any] = {}
    promotion = True
    promotion_reasons: List[str] = []
    for target in r2_targets:
        vals = [float(r[target]) for r in ft_rows if target in r and isinstance(r.get(target), (int, float)) and math.isfinite(float(r[target]))]
        if not vals:
            target_report[target] = {"available": False, "mean": None, "min": None, "pass": False}
            promotion = False
            promotion_reasons.append(f"missing frozen-test metric {target}")
            continue
        m, mn = float(np.mean(vals)), float(np.min(vals))
        ok = bool(m >= mean_threshold and mn >= min_threshold)
        target_report[target] = {"available": True, "mean": m, "min": mn, "pass": ok}
        if not ok:
            promotion = False
            promotion_reasons.append(f"{target} below threshold: mean={m:.6g}, min={mn:.6g}")
    errors = [r for r in state_rows if r.get("audit_error")]
    if errors:
        promotion = False
        promotion_reasons.append(f"state audit errors: {len(errors)} profiles")
    status = "PASS" if not errors else "REVIEW"
    scorecard = {
        "protocol": "D17-P4_REPORT_ONLY_STATE_AUDIT",
        "status": status,
        "promotion_status": "PASS" if promotion else "REVIEW",
        "p5_ready": bool(promotion),
        "candidate": freeze_manifest.get("candidate_id"),
        "p4_entry_note": "P3.4V was promotion REVIEW by validation voltage, but P4 is allowed as frozen report-only state audit.",
        "no_state_label_protocol": True,
        "softlabels_report_only": True,
        "softlabels_loaded_after_prediction_npz_written": True,
        "state_softlabels_used_for_inference_or_adaptation": False,
        "frozen_test_used_for_training_or_checkpoint_selection": False,
        "normal_frozen_test_profile_count": len(ft_rows),
        "target_r2_thresholds": {"mean": mean_threshold, "min_profile": min_threshold},
        "normal_frozen_test_state_r2": target_report,
        "normal_frozen_test_aggregate": ft_agg,
        "split_aggregates": split_metric_rows,
        "promotion_reasons": promotion_reasons,
        "outputs": {
            "freeze_manifest": str(out / "00_freeze_candidate" / "D17_P4_FREEZE_MANIFEST.json"),
            "inference_manifest_csv": str(out / "D17_P4_INFERENCE_MANIFEST.csv"),
            "state_profile_metrics_csv": str(out / "D17_P4_STATE_AUDIT_PROFILE_METRICS.csv"),
            "state_split_metrics_csv": str(out / "D17_P4_STATE_AUDIT_SPLIT_METRICS.csv"),
            "radial_audit_csv": str(out / "D17_P4_RADIAL_MECHANISM_AUDIT.csv"),
            "voltage_state_decomposition_csv": str(out / "D17_P4_VOLTAGE_STATE_DECOMPOSITION.csv"),
            "flagged_probe_report_json": str(out / "D17_P4_FLAGGED_PROBE_REPORT.json"),
            "scorecard_json": str(out / "D17_P4_SCORECARD.json"),
        },
        "prior_snapshot": prior_to_jsonable(prior),
    }
    (out / "D17_P4_SCORECARD.json").write_text(json.dumps(jsonable(scorecard), ensure_ascii=False, indent=2), encoding="utf-8")
    decision_lines = [
        "# D17-P4 Decision Report",
        "",
        f"- status: {scorecard['status']}",
        f"- promotion_status: {scorecard['promotion_status']}",
        f"- p5_ready: {scorecard['p5_ready']}",
        f"- normal_frozen_test_profile_count: {len(ft_rows)}",
        "",
        "## Promotion reasons",
    ]
    if promotion_reasons:
        decision_lines += [f"- {x}" for x in promotion_reasons]
    else:
        decision_lines.append("- All P4 report-only state R² gates passed.")
    decision_lines += ["", "## Boundary", "Soft labels were loaded only after observed-only prediction files were written. No P4 output may be used to modify P3/P4 candidate structure without starting a new protocol branch."]
    (out / "D17_P4_DECISION_REPORT.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    return scorecard
