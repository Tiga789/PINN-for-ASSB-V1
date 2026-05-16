# -*- coding: utf-8 -*-
"""Capacity and SOH utilities for ASSB aging-fix1.

Capacity/SOH labels are experimental observations derived from
``ZHB_ASSB_NCM811.xlsx -> step sheet -> discharge capacity``.  They are used by
Stage-B/Stage-C aging mechanism losses only; they are not routed into the
original PINN soft-label ``DATA_LOSS``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import math

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None  # type: ignore

from util.assb_aging_fix1_config import AgingFix1Config

PathLike = Union[str, Path]


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for ASSB capacity utilities")


def _find_col(frame: "pd.DataFrame", candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in normalized:
            return normalized[key]
    # relaxed: remove spaces, underscores, brackets
    def relax(s: str) -> str:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    relaxed = {relax(c): c for c in frame.columns}
    for name in candidates:
        key = relax(name)
        if key in relaxed:
            return relaxed[key]
    return None


def _to_numeric(series, default=np.nan) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    vals = np.array(vals, dtype=float, copy=True)
    vals[~np.isfinite(vals)] = default
    return vals


def standardize_capacity_targets(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Return a canonical cycle-level capacity target table.

    Expected output columns:
    ``cycle_id``, ``Q_obs_Ah``, ``SOH_obs``, ``Q_obs_mAh``, ``complete_cycle``.
    The function accepts several historical column names from earlier packages.
    """
    _require_pandas()
    frame = frame.copy()
    cycle_col = _find_col(frame, ["cycle_id", "cycle", "循环", "循环号", "Cycle"])
    q_col = _find_col(
        frame,
        [
            "Q_obs_Ah",
            "Q_obs_mAh",
            "Q_dis_Ah",
            "Q_dis_mAh",
            "Q_discharge_Ah",
            "Q_discharge_mAh",
            "q_dis_ah",
            "q_dis_mah",
            "q_discharge_ah",
            "q_discharge_mah",
            "discharge_capacity_Ah",
            "discharge_capacity_mAh",
            "放电容量(Ah)",
            "放电容量（Ah）",
            "放电容量(mAh)",
            "放电容量（mAh）",
            "容量(Ah)",
            "容量（Ah）",
            "容量(mAh)",
            "容量（mAh）",
            "capacity_Ah",
            "capacity_mAh",
        ],
    )
    soh_col = _find_col(frame, ["SOH_obs", "SOH", "soh", "soh_target", "capacity_norm", "SOH_clipped"])
    q_ref_col = _find_col(frame, ["q_ref_Ah", "Q_ref_Ah", "q_ref_ah", "Q_ref_mAh", "q_ref_mAh"])
    complete_col = _find_col(frame, ["complete_cycle", "is_complete", "complete", "cycle_complete", "train_mask"])
    if cycle_col is None:
        raise KeyError(f"Cannot find cycle_id column in capacity target CSV. Columns: {list(frame.columns)}")
    out = pd.DataFrame()
    out["cycle_id"] = pd.to_numeric(frame[cycle_col], errors="coerce").astype("Int64")
    if q_col is not None:
        q_raw = _to_numeric(frame[q_col])
    else:
        q_raw = np.full(len(frame), np.nan)
    # Heuristic: files may store mAh under a generic capacity name.  ASSB values
    # are around 0.0002-0.0004 Ah or 0.2-0.4 mAh.  If the column name says mAh,
    # or the median magnitude is > 0.02, treat values as mAh.
    finite = q_raw[np.isfinite(q_raw)]
    q_col_name = "" if q_col is None else str(q_col).lower()
    looks_mAh = ("mah" in q_col_name) or ("毫安" in q_col_name)
    if finite.size and (looks_mAh or np.nanmedian(np.abs(finite)) > 0.02):
        q_ah = q_raw / 1000.0
    else:
        q_ah = q_raw
    out["Q_obs_Ah"] = q_ah
    out["Q_obs_mAh"] = q_ah * 1000.0
    if soh_col is not None:
        soh = _to_numeric(frame[soh_col])
    else:
        # Infer reference from the maximum/first valid discharge capacity.
        ref = np.nanmax(q_ah) if np.isfinite(q_ah).any() else np.nan
        soh = q_ah / ref if np.isfinite(ref) and ref > 0 else np.full(len(q_ah), np.nan)
    out["SOH_obs"] = soh
    if complete_col is not None:
        out["complete_cycle"] = frame[complete_col].map(lambda x: str(x).strip().lower() not in {"0", "false", "no", "nan", "none", ""}).astype(bool)
    else:
        out["complete_cycle"] = np.isfinite(out["Q_obs_Ah"].to_numpy(dtype=float))
    out = out.dropna(subset=["cycle_id"]).copy()
    out["cycle_id"] = out["cycle_id"].astype(int)
    out = out.sort_values("cycle_id").drop_duplicates("cycle_id", keep="last").reset_index(drop=True)

    # Infer q_ref_Ah.  Prefer an explicit Q_ref column from the original
    # capacity target CSV, then Q/SOH, and finally max observed capacity.
    q = out["Q_obs_Ah"].to_numpy(dtype=float)
    soh = out["SOH_obs"].to_numpy(dtype=float)
    q_ref = np.nan
    if q_ref_col is not None:
        qref_raw = _to_numeric(frame[q_ref_col])
        qref_finite = qref_raw[np.isfinite(qref_raw) & (qref_raw > 0)]
        if qref_finite.size:
            qref_col_name = str(q_ref_col).lower()
            q_ref = float(np.nanmedian(qref_finite))
            if ("mah" in qref_col_name) or q_ref > 0.02:
                q_ref /= 1000.0
    valid = np.isfinite(q) & np.isfinite(soh) & (soh > 0)
    if (not np.isfinite(q_ref) or q_ref <= 0) and valid.any():
        q_ref = float(np.nanmedian(q[valid] / soh[valid]))
    if not np.isfinite(q_ref) or q_ref <= 0:
        q_finite = q[np.isfinite(q) & (q > 0)]
        q_ref = float(np.nanmax(q_finite)) if q_finite.size else 1.0
    out["q_ref_Ah"] = q_ref
    out["q_ref_mAh"] = q_ref * 1000.0
    soh_arr = out["SOH_obs"].to_numpy(dtype=float)
    needs_soh = ~np.isfinite(soh_arr)
    if np.any(needs_soh):
        out.loc[needs_soh, "SOH_obs"] = out.loc[needs_soh, "Q_obs_Ah"] / q_ref
    return out


def load_capacity_targets(csv_path: PathLike, *, cycle_from: Optional[int] = None, cycle_to: Optional[int] = None) -> "pd.DataFrame":
    _require_pandas()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"capacity target CSV not found: {path}")
    frame = pd.read_csv(path)
    out = standardize_capacity_targets(frame)
    if cycle_from is not None:
        out = out[out["cycle_id"] >= int(cycle_from)]
    if cycle_to is not None:
        out = out[out["cycle_id"] <= int(cycle_to)]
    if out.empty:
        raise RuntimeError(f"No capacity target rows after filtering: {path}")
    return out.reset_index(drop=True)


def q_ref_from_targets(frame: "pd.DataFrame") -> float:
    """Return Q_ref_Ah from a canonical or historical capacity target table.

    This function is deliberately defensive because earlier ModelFin108 target
    files used columns such as Q_dis_Ah / Q_ref_Ah / SOH, whereas ModelFin110
    Stage-B expects Q_obs_Ah / SOH_obs.  A missing match should produce a clear
    error instead of a NumPy zero-size reduction error.
    """
    _require_pandas()
    if frame is None or len(frame) == 0:
        raise RuntimeError("No rows available to infer q_ref_Ah from capacity targets")
    work = frame.copy()
    if "Q_obs_Ah" not in work.columns or "SOH_obs" not in work.columns:
        work = standardize_capacity_targets(work)
    for col in ["q_ref_Ah", "Q_ref_Ah", "q_ref_ah", "Q_ref_mAh", "q_ref_mAh"]:
        if col in work.columns:
            vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                qref = float(np.nanmedian(vals))
                if "mah" in col.lower() or qref > 0.02:
                    qref /= 1000.0
                return qref
    q = pd.to_numeric(work["Q_obs_Ah"], errors="coerce").to_numpy(dtype=float)
    soh = pd.to_numeric(work["SOH_obs"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(q) & np.isfinite(soh) & (soh > 0)
    if valid.any():
        return float(np.nanmedian(q[valid] / soh[valid]))
    q_valid = q[np.isfinite(q) & (q > 0)]
    if q_valid.size:
        return float(np.nanmax(q_valid))
    raise RuntimeError(
        "Cannot infer q_ref_Ah: no finite positive Q_obs_Ah/SOH_obs rows. "
        f"Available columns: {list(frame.columns)}"
    )


def soh_struct_from_profiles(profiles) -> Any:
    return profiles.f_LAM_c * profiles.theta_window_scale_c


def q_from_soh(soh: Any, q_ref_ah: float) -> Any:
    if torch is not None and torch.is_tensor(soh):
        return torch.as_tensor(float(q_ref_ah), dtype=soh.dtype, device=soh.device) * soh
    return np.asarray(soh, dtype=float) * float(q_ref_ah)


def _huber_torch(x, delta: float):
    if delta <= 0:
        return x.pow(2)
    return F.huber_loss(x, torch.zeros_like(x), delta=float(delta), reduction="none")


def capacity_loss(
    q_obs_ah,
    q_pred_ah,
    soh_obs,
    soh_pred,
    cfg: AgingFix1Config,
    *,
    train_mask=None,
    complete_mask=None,
    lam_rate=None,
    window_rate=None,
    f_lam=None,
    window_scale=None,
) -> Tuple[Any, Dict[str, float]]:
    """Capacity/SOH mechanism loss.

    Supports torch tensors and returns ``(loss, scalar_log_dict)``.
    ``train_mask`` controls which cycles contribute to observation terms.
    ``complete_mask`` is used by the final-amplitude term so incomplete terminal
    cycles such as cycle 522 do not dominate the fit.
    """
    if torch is None or not torch.is_tensor(q_pred_ah):
        raise RuntimeError("capacity_loss is intended for PyTorch Stage-B/Stage-C training")
    dtype = q_pred_ah.dtype
    device = q_pred_ah.device
    q_obs_ah = torch.as_tensor(q_obs_ah, dtype=dtype, device=device)
    soh_obs = torch.as_tensor(soh_obs, dtype=dtype, device=device)
    soh_pred = torch.as_tensor(soh_pred, dtype=dtype, device=device)
    if train_mask is None:
        mask = torch.ones_like(soh_pred, dtype=torch.bool)
    else:
        mask = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    valid = mask & torch.isfinite(q_obs_ah) & torch.isfinite(soh_obs)
    if not torch.any(valid):
        raise RuntimeError("No valid cycles for capacity_loss")
    q_ref = torch.clamp(torch.nanmedian(q_obs_ah[valid] / torch.clamp(soh_obs[valid], min=1e-12)), min=1e-12)
    q_err = (q_pred_ah[valid] - q_obs_ah[valid]) / q_ref
    soh_err = soh_pred[valid] - soh_obs[valid]
    loss_q = _huber_torch(q_err, cfg.huber_delta).mean()
    loss_soh = _huber_torch(soh_err, cfg.huber_delta).mean()

    if soh_pred.numel() >= 3:
        second = soh_pred[2:] - 2.0 * soh_pred[1:-1] + soh_pred[:-2]
        loss_smooth = second.pow(2).mean()
    else:
        loss_smooth = torch.zeros((), dtype=dtype, device=device)
    rate_terms = []
    for rate in (lam_rate, window_rate):
        if rate is not None and torch.as_tensor(rate).numel() >= 2:
            r = torch.as_tensor(rate, dtype=dtype, device=device)
            rate_terms.append((r[1:] - r[:-1]).pow(2).mean())
    loss_rate = sum(rate_terms) / max(len(rate_terms), 1) if rate_terms else torch.zeros((), dtype=dtype, device=device)

    bound_terms = []
    for arr in (f_lam, window_scale, soh_pred):
        if arr is not None:
            a = torch.as_tensor(arr, dtype=dtype, device=device)
            bound_terms.append(torch.relu(-a).pow(2).mean() + torch.relu(a - 1.05).pow(2).mean())
    loss_bounds = sum(bound_terms) / max(len(bound_terms), 1) if bound_terms else torch.zeros((), dtype=dtype, device=device)

    # Final-amplitude term uses the last complete valid cycle in the whole table,
    # not necessarily the training split, to avoid the 109 flat profile.  It is a
    # weak observation anchoring the terminal degradation amplitude.
    if complete_mask is None:
        complete_valid = torch.isfinite(soh_obs)
    else:
        complete_valid = torch.as_tensor(complete_mask, dtype=torch.bool, device=device) & torch.isfinite(soh_obs)
    idx = torch.nonzero(complete_valid, as_tuple=False).flatten()
    if idx.numel() > 0:
        j = idx[-1]
        loss_final = _huber_torch((soh_pred[j] - soh_obs[j]).reshape(1), cfg.huber_delta).mean()
    else:
        loss_final = torch.zeros((), dtype=dtype, device=device)

    if f_lam is not None and window_scale is not None:
        loss_balance = (torch.as_tensor(f_lam, dtype=dtype, device=device) - torch.as_tensor(window_scale, dtype=dtype, device=device)).pow(2).mean()
    else:
        loss_balance = torch.zeros((), dtype=dtype, device=device)

    total = (
        float(cfg.w_q) * loss_q
        + float(cfg.w_soh) * loss_soh
        + float(cfg.w_smooth) * loss_smooth
        + float(cfg.w_rate) * loss_rate
        + float(cfg.w_bounds) * loss_bounds
        + float(cfg.w_final) * loss_final
        + float(cfg.w_lam_window_balance) * loss_balance
    )
    logs = {
        "loss_total": float(total.detach().cpu()),
        "loss_q": float(loss_q.detach().cpu()),
        "loss_soh": float(loss_soh.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_rate": float(loss_rate.detach().cpu()),
        "loss_bounds": float(loss_bounds.detach().cpu()),
        "loss_final": float(loss_final.detach().cpu()),
        "loss_balance": float(loss_balance.detach().cpu()),
        "n_valid": int(valid.detach().cpu().sum().item()),
    }
    return total, logs


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) <= 1e-15 or np.std(bb) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _safe_r2(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 2:
        return float("nan")
    o = obs[mask]
    p = pred[mask]
    ss_res = float(np.sum((p - o) ** 2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    if ss_tot <= 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def capacity_metrics(obs_q_ah: Sequence[float], pred_q_ah: Sequence[float], obs_soh: Sequence[float], pred_soh: Sequence[float]) -> Dict[str, float]:
    q_obs = np.asarray(obs_q_ah, dtype=float)
    q_pred = np.asarray(pred_q_ah, dtype=float)
    soh_obs = np.asarray(obs_soh, dtype=float)
    soh_pred = np.asarray(pred_soh, dtype=float)
    mask_q = np.isfinite(q_obs) & np.isfinite(q_pred)
    mask_s = np.isfinite(soh_obs) & np.isfinite(soh_pred)
    out: Dict[str, float] = {"n": int(np.sum(mask_s & mask_q))}
    if mask_q.any():
        dq_mAh = (q_pred[mask_q] - q_obs[mask_q]) * 1000.0
        out.update(
            Q_MAE_mAh=float(np.mean(np.abs(dq_mAh))),
            Q_RMSE_mAh=float(np.sqrt(np.mean(dq_mAh**2))),
            Q_BIAS_mAh=float(np.mean(dq_mAh)),
            Q_R2=_safe_r2(q_obs, q_pred),
            Q_corr=_safe_corr(q_obs, q_pred),
        )
    else:
        out.update(Q_MAE_mAh=float("nan"), Q_RMSE_mAh=float("nan"), Q_BIAS_mAh=float("nan"), Q_R2=float("nan"), Q_corr=float("nan"))
    if mask_s.any():
        ds = soh_pred[mask_s] - soh_obs[mask_s]
        out.update(
            SOH_MAE=float(np.mean(np.abs(ds))),
            SOH_RMSE=float(np.sqrt(np.mean(ds**2))),
            SOH_BIAS=float(np.mean(ds)),
            SOH_R2=_safe_r2(soh_obs, soh_pred),
            SOH_corr=_safe_corr(soh_obs, soh_pred),
            SOH_obs_min=float(np.nanmin(soh_obs[mask_s])),
            SOH_pred_min=float(np.nanmin(soh_pred[mask_s])),
        )
    else:
        out.update(SOH_MAE=float("nan"), SOH_RMSE=float("nan"), SOH_BIAS=float("nan"), SOH_R2=float("nan"), SOH_corr=float("nan"), SOH_obs_min=float("nan"), SOH_pred_min=float("nan"))
    return out


def capacity_metrics_by_split(frame: "pd.DataFrame", *, split_col: str = "split", complete_only: bool = False) -> Dict[str, Dict[str, float]]:
    _require_pandas()
    data = frame.copy()
    if complete_only and "complete_cycle" in data.columns:
        data = data[data["complete_cycle"].astype(bool)]
    result: Dict[str, Dict[str, float]] = {}
    splits = ["all"]
    if split_col in data.columns:
        splits += [str(s) for s in sorted(data[split_col].dropna().unique())]
    for split in splits:
        sub = data if split == "all" or split_col not in data.columns else data[data[split_col].astype(str) == split]
        if sub.empty:
            continue
        result[split] = capacity_metrics(sub["Q_obs_Ah"], sub["Q_pred_Ah"], sub["SOH_obs"], sub["SOH_pred"])
    return result


def save_json(obj: Dict[str, Any], path: PathLike) -> None:
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return [clean(v) for v in x.tolist()]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            val = float(x)
            return None if not math.isfinite(val) else val
        if isinstance(x, float):
            return None if not math.isfinite(x) else x
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "standardize_capacity_targets",
    "load_capacity_targets",
    "q_ref_from_targets",
    "soh_struct_from_profiles",
    "q_from_soh",
    "capacity_loss",
    "capacity_metrics",
    "capacity_metrics_by_split",
    "save_json",
]
