# -*- coding: utf-8 -*-
"""
Capacity/SOH derivation utilities for ASSB ModelFin_109.

SOH is not treated as an independent network output.  The first 109A path uses
mechanism variables to derive Q_pred and SOH_pred.  Optional endpoint helpers
are provided for later validation against cs_a/cs_c endpoints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import json
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

PathLike = Union[str, Path]


def capacity_from_mechanism(aging_profiles, q_ref_ah):
    """Return (Q_pred_Ah, SOH_pred) from AgingProfiles."""
    soh = aging_profiles.soh_mech
    if torch is not None and torch.is_tensor(soh):
        q_ref = torch.as_tensor(float(q_ref_ah), dtype=soh.dtype, device=soh.device)
        return q_ref * soh, soh
    soh_np = np.asarray(soh, dtype=float)
    return float(q_ref_ah) * soh_np, soh_np


def soh_from_capacity(q_ah, q_ref_ah):
    if torch is not None and torch.is_tensor(q_ah):
        q_ref = torch.as_tensor(float(q_ref_ah), dtype=q_ah.dtype, device=q_ah.device)
        return q_ah / torch.clamp(q_ref, min=1.0e-30)
    return np.asarray(q_ah, dtype=float) / max(float(q_ref_ah), 1.0e-30)


def capacity_from_endpoints(
    theta_c_start,
    theta_c_end,
    *,
    q_ref_ah: float,
    theta_window_ref: Optional[float] = None,
):
    """Approximate capacity from positive-electrode endpoint theta span.

    This is a diagnostic helper, not the default loss path.  It helps detect
    whether endpoint state windows are consistent with the mechanism-derived
    SOH.  If theta_window_ref is omitted, the first cycle's theta span is used.
    """
    if torch is not None and torch.is_tensor(theta_c_start):
        span = torch.abs(theta_c_end - theta_c_start)
        if theta_window_ref is None:
            ref = torch.clamp(span[0].detach(), min=torch.as_tensor(1.0e-12, dtype=span.dtype, device=span.device))
        else:
            ref = torch.as_tensor(float(theta_window_ref), dtype=span.dtype, device=span.device)
        soh = span / torch.clamp(ref, min=1.0e-12)
        return torch.as_tensor(float(q_ref_ah), dtype=span.dtype, device=span.device) * soh, soh
    span_np = np.abs(np.asarray(theta_c_end, dtype=float) - np.asarray(theta_c_start, dtype=float))
    ref_np = float(theta_window_ref) if theta_window_ref is not None else max(float(span_np[0]), 1.0e-12)
    soh_np = span_np / max(ref_np, 1.0e-12)
    return float(q_ref_ah) * soh_np, soh_np


def _require_pandas():
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for capacity target CSV loading.")


def load_capacity_targets_simple(csv_path: PathLike) -> "pd.DataFrame":
    """Load cycle-level capacity/SOH targets.

    Expected columns are cycle_id and either Q_dis_Ah or Q_discharge_Ah.  If SOH
    is missing, it is computed from Q_ref_Ah or max(Q_dis_Ah).
    """
    _require_pandas()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Capacity target CSV not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "cycle_id" not in frame.columns:
        raise KeyError(f"{path} does not contain cycle_id")
    if "Q_dis_Ah" not in frame.columns:
        if "Q_discharge_Ah" in frame.columns:
            frame = frame.rename(columns={"Q_discharge_Ah": "Q_dis_Ah"})
        elif "Q_discharge_mAh" in frame.columns:
            frame["Q_dis_Ah"] = pd.to_numeric(frame["Q_discharge_mAh"], errors="coerce") / 1000.0
        else:
            raise KeyError(f"{path} must contain Q_dis_Ah/Q_discharge_Ah/Q_discharge_mAh")
    frame = frame.copy()
    frame["cycle_id"] = pd.to_numeric(frame["cycle_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["cycle_id"]).copy()
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    frame["Q_dis_Ah"] = pd.to_numeric(frame["Q_dis_Ah"], errors="coerce")
    frame = frame.dropna(subset=["Q_dis_Ah"]).sort_values("cycle_id").reset_index(drop=True)
    if "Q_ref_Ah" in frame.columns and pd.to_numeric(frame["Q_ref_Ah"], errors="coerce").notna().any():
        q_ref = float(pd.to_numeric(frame["Q_ref_Ah"], errors="coerce").dropna().iloc[0])
    else:
        q_ref = float(frame["Q_dis_Ah"].max())
        frame["Q_ref_Ah"] = q_ref
    if "SOH" not in frame.columns:
        frame["SOH"] = frame["Q_dis_Ah"] / max(q_ref, 1.0e-30)
    else:
        frame["SOH"] = pd.to_numeric(frame["SOH"], errors="coerce")
        frame["SOH"] = frame["SOH"].fillna(frame["Q_dis_Ah"] / max(q_ref, 1.0e-30))
    return frame


def capacity_metrics(q_obs, q_pred, soh_obs=None, soh_pred=None) -> Dict[str, float]:
    q_obs_np = np.asarray(q_obs, dtype=float)
    q_pred_np = np.asarray(q_pred, dtype=float)
    mask = np.isfinite(q_obs_np) & np.isfinite(q_pred_np)
    out: Dict[str, float] = {"n": int(mask.sum())}
    if mask.sum() == 0:
        return {**out, "Q_MAE_Ah": float("nan"), "Q_RMSE_Ah": float("nan"), "Q_R2": float("nan")}
    resid = q_pred_np[mask] - q_obs_np[mask]
    out["Q_MAE_Ah"] = float(np.mean(np.abs(resid)))
    out["Q_MAE_mAh"] = float(out["Q_MAE_Ah"] * 1000.0)
    out["Q_RMSE_Ah"] = float(np.sqrt(np.mean(resid ** 2)))
    out["Q_RMSE_mAh"] = float(out["Q_RMSE_Ah"] * 1000.0)
    denom = float(np.sum((q_obs_np[mask] - np.mean(q_obs_np[mask])) ** 2))
    out["Q_R2"] = float(1.0 - np.sum(resid ** 2) / denom) if denom > 1.0e-30 else float("nan")
    if soh_obs is not None and soh_pred is not None:
        s_obs = np.asarray(soh_obs, dtype=float)
        s_pred = np.asarray(soh_pred, dtype=float)
        s_mask = np.isfinite(s_obs) & np.isfinite(s_pred)
        if s_mask.any():
            s_resid = s_pred[s_mask] - s_obs[s_mask]
            out["SOH_MAE"] = float(np.mean(np.abs(s_resid)))
            out["SOH_RMSE"] = float(np.sqrt(np.mean(s_resid ** 2)))
            denom_s = float(np.sum((s_obs[s_mask] - np.mean(s_obs[s_mask])) ** 2))
            out["SOH_R2"] = float(1.0 - np.sum(s_resid ** 2) / denom_s) if denom_s > 1.0e-30 else float("nan")
    return out


def split_capacity_metrics(frame: "pd.DataFrame", q_pred_col: str = "Q_pred_Ah", soh_pred_col: str = "SOH_pred") -> Dict[str, Dict[str, float]]:
    _require_pandas()
    if "split" not in frame.columns:
        frame = frame.copy()
        frame["split"] = "all"
    out: Dict[str, Dict[str, float]] = {}
    for split in ["all", "train", "val", "test"]:
        part = frame if split == "all" else frame[frame["split"].astype(str).str.lower() == split]
        if len(part) == 0:
            continue
        out[split] = capacity_metrics(
            part["Q_obs_Ah"].to_numpy(dtype=float) if "Q_obs_Ah" in part.columns else part["Q_dis_Ah"].to_numpy(dtype=float),
            part[q_pred_col].to_numpy(dtype=float),
            part["SOH_obs"].to_numpy(dtype=float) if "SOH_obs" in part.columns else part["SOH"].to_numpy(dtype=float),
            part[soh_pred_col].to_numpy(dtype=float),
        )
    return out


def save_json(obj: Dict[str, object], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
