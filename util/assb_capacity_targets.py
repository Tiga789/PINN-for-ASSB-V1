# -*- coding: utf-8 -*-
"""
Shared capacity/SOH target utilities for ASSB ModelFin_108.

This module is deliberately small and import-safe. It can be used by:
- scripts/prepare_assb_capacity_soh_targets.py
- util/aging_assb_capacity.py
- evaluate_assb_capacity_curve.py
- later patches to util/init_pinn.py and util/_losses.py

Important: Q_dis_Ah/SOH are labels, not input features. The default feature
constructor uses only cycle/protocol columns so the capacity head does not
cheat by receiving the target capacity as input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


DEFAULT_FEATURE_COLUMNS = (
    "cycle_norm",
    "d_tau",
    "I_dis_abs_A_norm",
    "discharge_step_time_s_norm",
    "discharge_step_count_norm",
)


@dataclass
class CapacityTargetBundle:
    csv_path: str
    frame: "pd.DataFrame"
    feature_columns: Tuple[str, ...]
    q_ref_ah: float
    cycle_id: np.ndarray
    features_np: np.ndarray
    d_tau_np: np.ndarray
    q_dis_ah_np: np.ndarray
    soh_np: np.ndarray
    train_mask_np: np.ndarray
    complete_cycle_np: np.ndarray

    def as_torch(self, device=None, dtype=None) -> Dict[str, "torch.Tensor"]:
        if torch is None:
            raise RuntimeError("PyTorch is required for as_torch().")
        if dtype is None:
            dtype = torch.float64
        device = device or torch.device("cpu")
        return {
            "cycle_id": torch.as_tensor(self.cycle_id, dtype=torch.long, device=device),
            "features": torch.as_tensor(self.features_np, dtype=dtype, device=device),
            "d_tau": torch.as_tensor(self.d_tau_np, dtype=dtype, device=device),
            "Q_ref_Ah": torch.as_tensor(float(self.q_ref_ah), dtype=dtype, device=device),
            "Q_dis_Ah": torch.as_tensor(self.q_dis_ah_np, dtype=dtype, device=device),
            "SOH": torch.as_tensor(self.soh_np, dtype=dtype, device=device),
            "train_mask": torch.as_tensor(self.train_mask_np, dtype=torch.bool, device=device),
            "complete_cycle": torch.as_tensor(self.complete_cycle_np, dtype=torch.bool, device=device),
        }


def _require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required to load ASSB capacity target CSV files.")


def _bool_array(values: Iterable[object], default: bool = True) -> np.ndarray:
    out: List[bool] = []
    for v in values:
        if v is None:
            out.append(default)
        elif isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
        else:
            s = str(v).strip().lower()
            if s in {"true", "1", "yes", "y", "t"}:
                out.append(True)
            elif s in {"false", "0", "no", "n", "f"}:
                out.append(False)
            else:
                out.append(default)
    return np.asarray(out, dtype=bool)


def _safe_numeric(frame: "pd.DataFrame", col: str, default: float = 0.0) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), default, dtype=float)
    # Some pandas/openpyxl/pyarrow-backed columns expose read-only NumPy views.
    # Use an owned writable array before replacing NaN/inf in-place.
    vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float, copy=True)
    vals = np.array(vals, dtype=float, copy=True)
    vals.setflags(write=True)
    vals[~np.isfinite(vals)] = default
    return vals


def _normalize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if hi - lo < 1.0e-12:
        return np.zeros_like(x, dtype=float)
    y = (x - lo) / (hi - lo)
    y[~np.isfinite(y)] = 0.0
    return y


def _ensure_default_feature_columns(frame: "pd.DataFrame") -> "pd.DataFrame":
    out = frame.copy()
    cycle = _safe_numeric(out, "cycle_id", default=0.0)
    if "cycle_norm" not in out.columns:
        out["cycle_norm"] = _normalize_feature(cycle)
    if "d_tau" not in out.columns:
        d = np.diff(cycle, prepend=cycle[0] if len(cycle) else 0.0)
        if len(d) > 1 and d[0] == 0:
            d[0] = d[1]
        total = float(np.nansum(np.maximum(d, 0.0)))
        if total <= 0:
            out["d_tau"] = np.full(len(out), 1.0 / max(len(out), 1), dtype=float)
        else:
            out["d_tau"] = np.maximum(d / total, 1.0e-8)
    if "I_dis_abs_A_norm" not in out.columns:
        out["I_dis_abs_A_norm"] = _normalize_feature(_safe_numeric(out, "I_dis_abs_A", default=0.0))
    if "discharge_step_time_s_norm" not in out.columns:
        out["discharge_step_time_s_norm"] = _normalize_feature(_safe_numeric(out, "discharge_step_time_s", default=0.0))
    if "discharge_step_count_norm" not in out.columns:
        out["discharge_step_count_norm"] = _normalize_feature(_safe_numeric(out, "discharge_step_count", default=0.0))
    return out


def load_capacity_targets(
    csv_path: Union[str, Path],
    *,
    cycle_from: Optional[int] = None,
    cycle_to: Optional[int] = None,
    feature_columns: Optional[Sequence[str]] = None,
    train_mask_col: str = "train_mask",
) -> CapacityTargetBundle:
    """Load a capacity target CSV produced by prepare_assb_capacity_soh_targets.py."""
    _require_pandas()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Capacity target CSV not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "cycle_id" not in frame.columns:
        raise KeyError(f"{path} does not contain cycle_id")
    if "Q_dis_Ah" not in frame.columns:
        raise KeyError(f"{path} does not contain Q_dis_Ah")

    frame = frame.copy()
    frame["cycle_id"] = pd.to_numeric(frame["cycle_id"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["cycle_id"])
    frame["cycle_id"] = frame["cycle_id"].astype(int)
    if cycle_from is not None:
        frame = frame[frame["cycle_id"] >= int(cycle_from)]
    if cycle_to is not None:
        frame = frame[frame["cycle_id"] <= int(cycle_to)]
    frame = frame.sort_values("cycle_id").reset_index(drop=True)
    if len(frame) == 0:
        raise RuntimeError("No capacity target rows remain after cycle filtering.")

    frame = _ensure_default_feature_columns(frame)
    q_ref = float(pd.to_numeric(frame.get("Q_ref_Ah", np.nan), errors="coerce").dropna().iloc[0]) if "Q_ref_Ah" in frame.columns and pd.to_numeric(frame["Q_ref_Ah"], errors="coerce").notna().any() else float(pd.to_numeric(frame["Q_dis_Ah"], errors="coerce").max())
    q_dis = pd.to_numeric(frame["Q_dis_Ah"], errors="coerce").to_numpy(dtype=float)
    soh = pd.to_numeric(frame["SOH"], errors="coerce").to_numpy(dtype=float) if "SOH" in frame.columns else q_dis / q_ref
    cycle = frame["cycle_id"].to_numpy(dtype=int)
    complete = _bool_array(frame["complete_cycle"].values, default=True) if "complete_cycle" in frame.columns else np.ones(len(frame), dtype=bool)
    mask = _bool_array(frame[train_mask_col].values, default=True) if train_mask_col in frame.columns else complete.copy()

    feature_cols = tuple(feature_columns) if feature_columns else DEFAULT_FEATURE_COLUMNS
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing capacity feature columns {missing}. Available={list(frame.columns)}")
    features = frame.loc[:, list(feature_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    d_tau = pd.to_numeric(frame["d_tau"], errors="coerce").fillna(1.0 / max(len(frame), 1)).to_numpy(dtype=float)
    d_tau = np.maximum(d_tau, 1.0e-8)

    return CapacityTargetBundle(
        csv_path=str(path),
        frame=frame,
        feature_columns=feature_cols,
        q_ref_ah=float(q_ref),
        cycle_id=cycle,
        features_np=features,
        d_tau_np=d_tau,
        q_dis_ah_np=q_dis,
        soh_np=soh,
        train_mask_np=mask,
        complete_cycle_np=complete,
    )


def summarize_capacity_targets(bundle: CapacityTargetBundle) -> Dict[str, float]:
    q = bundle.q_dis_ah_np
    soh = bundle.soh_np
    mask = bundle.train_mask_np.astype(bool)
    return {
        "n_cycles": int(len(q)),
        "n_train_cycles": int(mask.sum()),
        "cycle_min": int(np.min(bundle.cycle_id)),
        "cycle_max": int(np.max(bundle.cycle_id)),
        "q_ref_mAh": float(bundle.q_ref_ah * 1000.0),
        "q_min_mAh": float(np.nanmin(q) * 1000.0),
        "q_max_mAh": float(np.nanmax(q) * 1000.0),
        "soh_min": float(np.nanmin(soh)),
        "soh_max": float(np.nanmax(soh)),
    }


def make_capacity_torch_batch(csv_path: Union[str, Path], *, device=None, dtype=None, **kwargs) -> Dict[str, "torch.Tensor"]:
    bundle = load_capacity_targets(csv_path, **kwargs)
    return bundle.as_torch(device=device, dtype=dtype)
