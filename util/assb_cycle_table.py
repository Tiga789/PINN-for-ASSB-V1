# -*- coding: utf-8 -*-
"""
Cycle-level table utilities for ASSB ModelFin_109 aging mechanism experiments.

This module is intentionally import-safe and independent from the existing
PINN core.  It is used by the data-preparation script, the aging mechanism
head, later training patches, and the ModelFin_109 evaluator.

The central convention is:
- cycle_id is the experimental cycle number, normally 5..522.
- t_start_s/t_end_s are absolute times in the continuous soft-label timeline.
- q_net_cycle_C is the signed integral of I(t) over each cycle in Coulombs.
- throughput_cycle_C is the integral of |I(t)| over each cycle in Coulombs.

The helper functions accept either numpy arrays/DataFrames or torch tensors
where possible.  Torch operations are preferred in training-time functions so
that future aged cbar calculations can remain differentiable with respect to
aging parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import json
import math

import numpy as np

try:  # pandas is needed for CSV I/O but not for pure tensor helpers.
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

NumberLike = Union[int, float, np.number]
PathLike = Union[str, Path]


@dataclass
class CycleTableBundle:
    """Container returned by :func:`load_cycle_table`.

    Attributes are stored as numpy arrays so the object is serializable and
    easy to inspect.  Use :meth:`as_torch` inside training code.
    """

    csv_path: str
    frame: "pd.DataFrame"
    cycle_id: np.ndarray
    t_start_s: np.ndarray
    t_end_s: np.ndarray
    q_net_cycle_C: np.ndarray
    throughput_cycle_C: np.ndarray
    q_net_start_C: np.ndarray
    q_net_end_C: np.ndarray
    throughput_start_C: np.ndarray
    throughput_end_C: np.ndarray
    split: np.ndarray
    feature_columns: Tuple[str, ...]
    features_np: np.ndarray

    def as_torch(self, device=None, dtype=None) -> Dict[str, "torch.Tensor"]:
        if torch is None:
            raise RuntimeError("PyTorch is required for CycleTableBundle.as_torch().")
        dtype = dtype or torch.float64
        device = device or torch.device("cpu")
        out = {
            "cycle_id": torch.as_tensor(self.cycle_id, dtype=torch.long, device=device),
            "t_start_s": torch.as_tensor(self.t_start_s, dtype=dtype, device=device),
            "t_end_s": torch.as_tensor(self.t_end_s, dtype=dtype, device=device),
            "q_net_cycle_C": torch.as_tensor(self.q_net_cycle_C, dtype=dtype, device=device),
            "throughput_cycle_C": torch.as_tensor(self.throughput_cycle_C, dtype=dtype, device=device),
            "q_net_start_C": torch.as_tensor(self.q_net_start_C, dtype=dtype, device=device),
            "q_net_end_C": torch.as_tensor(self.q_net_end_C, dtype=dtype, device=device),
            "throughput_start_C": torch.as_tensor(self.throughput_start_C, dtype=dtype, device=device),
            "throughput_end_C": torch.as_tensor(self.throughput_end_C, dtype=dtype, device=device),
            "features": torch.as_tensor(self.features_np, dtype=dtype, device=device),
        }
        # Keep masks as tensors too; strings remain in the DataFrame.
        split_lower = np.asarray([str(s).lower() for s in self.split])
        out["train_mask"] = torch.as_tensor(split_lower == "train", dtype=torch.bool, device=device)
        out["val_mask"] = torch.as_tensor(split_lower == "val", dtype=torch.bool, device=device)
        out["test_mask"] = torch.as_tensor(split_lower == "test", dtype=torch.bool, device=device)
        return out


def _require_pandas() -> None:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for ASSB cycle-table CSV utilities.")


def _safe_numeric(frame: "pd.DataFrame", col: str, default: float = 0.0) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float, copy=True)
    vals = np.asarray(vals, dtype=float, copy=True)
    vals[~np.isfinite(vals)] = float(default)
    return vals


def _normalize(x: np.ndarray, *, default: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, default, dtype=float)
    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    if hi - lo <= 1.0e-15:
        return np.full_like(x, default, dtype=float)
    y = (x - lo) / (hi - lo)
    y[~np.isfinite(y)] = default
    return y.astype(float, copy=False)


def _ensure_required_columns(frame: "pd.DataFrame") -> "pd.DataFrame":
    missing = [c for c in ("cycle_id", "t_start_s", "t_end_s") if c not in frame.columns]
    if missing:
        raise KeyError(f"Cycle table is missing required columns: {missing}")
    out = frame.copy()
    out["cycle_id"] = pd.to_numeric(out["cycle_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["cycle_id"]).copy()
    out["cycle_id"] = out["cycle_id"].astype(int)
    out = out.sort_values("cycle_id").reset_index(drop=True)
    if len(out) == 0:
        raise RuntimeError("Cycle table is empty after cleaning cycle_id.")
    for col in ("t_start_s", "t_end_s", "q_net_cycle_C", "throughput_cycle_C"):
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    # Cumulative columns are optional in CSV and can be reconstructed.
    if "q_net_start_C" not in out.columns:
        q = out["q_net_cycle_C"].to_numpy(dtype=float)
        out["q_net_start_C"] = np.concatenate([[0.0], np.cumsum(q[:-1])])
    if "q_net_end_C" not in out.columns:
        out["q_net_end_C"] = out["q_net_start_C"] + out["q_net_cycle_C"]
    if "throughput_start_C" not in out.columns:
        q_abs = out["throughput_cycle_C"].to_numpy(dtype=float)
        out["throughput_start_C"] = np.concatenate([[0.0], np.cumsum(q_abs[:-1])])
    if "throughput_end_C" not in out.columns:
        out["throughput_end_C"] = out["throughput_start_C"] + out["throughput_cycle_C"]
    if "split" not in out.columns:
        out["split"] = "train"
    return out


def make_cycle_features(
    frame: "pd.DataFrame",
    *,
    feature_columns: Optional[Sequence[str]] = None,
) -> Tuple["pd.DataFrame", Tuple[str, ...]]:
    """Ensure default cycle-level aging features exist.

    The default features do not include Q_obs/SOH_obs.  They are protocol and
    cycle-history features so that the aging head cannot receive the target
    capacity directly.
    """
    _require_pandas()
    out = frame.copy()
    cycle = _safe_numeric(out, "cycle_id")
    t0 = _safe_numeric(out, "t_start_s")
    t1 = _safe_numeric(out, "t_end_s")
    duration = np.maximum(t1 - t0, 0.0)
    throughput = _safe_numeric(out, "throughput_cycle_C")
    throughput_end = _safe_numeric(out, "throughput_end_C")
    q_net = _safe_numeric(out, "q_net_cycle_C")
    charge = _safe_numeric(out, "charge_C")
    discharge = _safe_numeric(out, "discharge_C")
    rest_time = _safe_numeric(out, "rest_time_s")

    if "cycle_norm" not in out.columns:
        out["cycle_norm"] = _normalize(cycle)
    if "duration_norm" not in out.columns:
        out["duration_norm"] = _normalize(duration)
    if "throughput_cycle_norm" not in out.columns:
        out["throughput_cycle_norm"] = _normalize(throughput)
    if "throughput_cum_norm" not in out.columns:
        out["throughput_cum_norm"] = _normalize(throughput_end)
    if "q_net_cycle_norm" not in out.columns:
        out["q_net_cycle_norm"] = _normalize(q_net)
    if "charge_fraction" not in out.columns:
        denom = np.maximum(charge + discharge, 1.0e-30)
        out["charge_fraction"] = np.clip(charge / denom, 0.0, 1.0)
    if "rest_fraction" not in out.columns:
        out["rest_fraction"] = np.clip(rest_time / np.maximum(duration, 1.0e-30), 0.0, 1.0)

    default_cols = (
        "cycle_norm",
        "duration_norm",
        "throughput_cycle_norm",
        "throughput_cum_norm",
        "q_net_cycle_norm",
        "charge_fraction",
        "rest_fraction",
    )
    cols = tuple(feature_columns) if feature_columns else default_cols
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing requested aging feature columns: {missing}")
    return out, cols


def load_cycle_table(
    csv_path: PathLike,
    *,
    cycle_from: Optional[int] = None,
    cycle_to: Optional[int] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> CycleTableBundle:
    """Load a ModelFin_109 cycle table from CSV."""
    _require_pandas()
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Cycle table CSV not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = _ensure_required_columns(frame)
    if cycle_from is not None:
        frame = frame[frame["cycle_id"] >= int(cycle_from)]
    if cycle_to is not None:
        frame = frame[frame["cycle_id"] <= int(cycle_to)]
    frame = frame.reset_index(drop=True)
    if len(frame) == 0:
        raise RuntimeError("No cycle rows remain after cycle filtering.")
    frame, cols = make_cycle_features(frame, feature_columns=feature_columns)
    features = frame.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return CycleTableBundle(
        csv_path=str(path),
        frame=frame,
        cycle_id=frame["cycle_id"].to_numpy(dtype=int),
        t_start_s=_safe_numeric(frame, "t_start_s"),
        t_end_s=_safe_numeric(frame, "t_end_s"),
        q_net_cycle_C=_safe_numeric(frame, "q_net_cycle_C"),
        throughput_cycle_C=_safe_numeric(frame, "throughput_cycle_C"),
        q_net_start_C=_safe_numeric(frame, "q_net_start_C"),
        q_net_end_C=_safe_numeric(frame, "q_net_end_C"),
        throughput_start_C=_safe_numeric(frame, "throughput_start_C"),
        throughput_end_C=_safe_numeric(frame, "throughput_end_C"),
        split=frame["split"].astype(str).to_numpy(),
        feature_columns=cols,
        features_np=features,
    )


def summarize_cycle_table(bundle: CycleTableBundle) -> Dict[str, object]:
    split_values, split_counts = np.unique(bundle.split.astype(str), return_counts=True)
    return {
        "csv_path": bundle.csv_path,
        "n_cycles": int(len(bundle.cycle_id)),
        "cycle_min": int(np.min(bundle.cycle_id)),
        "cycle_max": int(np.max(bundle.cycle_id)),
        "t_start_s": float(np.min(bundle.t_start_s)),
        "t_end_s": float(np.max(bundle.t_end_s)),
        "throughput_total_C": float(np.max(bundle.throughput_end_C)),
        "q_net_total_C": float(np.sum(bundle.q_net_cycle_C)),
        "feature_columns": list(bundle.feature_columns),
        "split_counts": {str(k): int(v) for k, v in zip(split_values, split_counts)},
    }


def _get_cycle_tensors(params: Dict[str, object], device=None, dtype=None) -> Dict[str, "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("PyTorch is required for tensor cycle-table helpers.")
    dtype = dtype or torch.float64
    if "cycle_table_torch" in params and isinstance(params["cycle_table_torch"], dict):
        table = params["cycle_table_torch"]
        if device is not None:
            return {k: (v.to(device=device) if hasattr(v, "to") else v) for k, v in table.items()}
        return table
    if "cycle_table_bundle" in params:
        table = params["cycle_table_bundle"].as_torch(device=device, dtype=dtype)
        params["cycle_table_torch"] = table
        return table
    csv_path = params.get("ASSB_AGING_CYCLE_TABLE") or params.get("aging_cycle_table_csv") or params.get("cycle_table_csv")
    if csv_path is None:
        raise KeyError("params must contain ASSB_AGING_CYCLE_TABLE or cycle_table_bundle for aging helpers.")
    bundle = load_cycle_table(str(csv_path))
    params["cycle_table_bundle"] = bundle
    table = bundle.as_torch(device=device, dtype=dtype)
    params["cycle_table_torch"] = table
    return table


def cycle_at_t(params: Dict[str, object], t) -> "torch.Tensor":
    """Return zero-based cycle-table row indices for times ``t``.

    ``t`` is expected to be in seconds on the same continuous timeline as the
    cycle table.  The returned values index rows in the cycle table, not the
    experimental cycle_id itself.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for cycle_at_t().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    table = _get_cycle_tensors(params, device=t.device, dtype=t.dtype)
    t_start = table["t_start_s"].to(device=t.device, dtype=t.dtype)
    # searchsorted returns insertion index; subtract 1 gives the active row.
    idx = torch.searchsorted(t_start, t.contiguous(), right=True) - 1
    return torch.clamp(idx, 0, t_start.numel() - 1).long()


def cycle_id_at_t(params: Dict[str, object], t) -> "torch.Tensor":
    table = _get_cycle_tensors(params, device=t.device if torch.is_tensor(t) else None)
    idx = cycle_at_t(params, t)
    return table["cycle_id"].to(device=idx.device)[idx]


def qnet_within_cycle_at_t(params: Dict[str, object], t) -> "torch.Tensor":
    """Approximate signed charge passed within the active cycle at time ``t``.

    First-package implementation uses a linear fraction of q_net_cycle_C between
    t_start_s and t_end_s.  The modified training files may replace this with a
    higher-resolution current interpolation from solution.npz.  The function is
    kept differentiable with respect to aging variables and stable for rest-heavy
    cycles.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for qnet_within_cycle_at_t().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    table = _get_cycle_tensors(params, device=t.device, dtype=t.dtype)
    idx = cycle_at_t(params, t)
    t0 = table["t_start_s"].to(device=t.device, dtype=t.dtype)[idx]
    t1 = table["t_end_s"].to(device=t.device, dtype=t.dtype)[idx]
    q_cycle = table["q_net_cycle_C"].to(device=t.device, dtype=t.dtype)[idx]
    frac = torch.clamp((t - t0) / torch.clamp(t1 - t0, min=torch.as_tensor(1.0e-12, dtype=t.dtype, device=t.device)), 0.0, 1.0)
    return frac * q_cycle


def throughput_within_cycle_at_t(params: Dict[str, object], t) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for throughput_within_cycle_at_t().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    table = _get_cycle_tensors(params, device=t.device, dtype=t.dtype)
    idx = cycle_at_t(params, t)
    t0 = table["t_start_s"].to(device=t.device, dtype=t.dtype)[idx]
    t1 = table["t_end_s"].to(device=t.device, dtype=t.dtype)[idx]
    q_cycle = table["throughput_cycle_C"].to(device=t.device, dtype=t.dtype)[idx]
    frac = torch.clamp((t - t0) / torch.clamp(t1 - t0, min=torch.as_tensor(1.0e-12, dtype=t.dtype, device=t.device)), 0.0, 1.0)
    return frac * q_cycle


def cycle_features_from_params(params: Dict[str, object], device=None, dtype=None) -> "torch.Tensor":
    table = _get_cycle_tensors(params, device=device, dtype=dtype)
    return table["features"]


def attach_cycle_table_to_params(params: Dict[str, object], csv_path: Optional[PathLike] = None) -> Dict[str, object]:
    """Load and attach the cycle table to an existing params dictionary."""
    path = csv_path or params.get("ASSB_AGING_CYCLE_TABLE") or params.get("aging_cycle_table_csv")
    if path is None:
        raise KeyError("No cycle table path was provided.")
    bundle = load_cycle_table(path)
    params["ASSB_AGING_CYCLE_TABLE"] = str(path)
    params["cycle_table_bundle"] = bundle
    return params


def dump_cycle_table_summary(bundle: CycleTableBundle, json_path: PathLike) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summarize_cycle_table(bundle), f, ensure_ascii=False, indent=2)
