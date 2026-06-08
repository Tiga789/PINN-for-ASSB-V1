# -*- coding: utf-8 -*-
"""Metrics for D14-P5/P5A XJTU P2Dlite soft-label NN smoke.

D14-P5A fixes a bug in the original `aggregate_metrics` implementation:
it attempted to convert non-numeric metadata columns such as `batch` and
`protocol` to float. The fixed version aggregates only numeric metric columns
and preserves clean split-level summaries.
"""

from __future__ import annotations

from typing import Dict, Any, List

import math
import numpy as np


NON_NUMERIC_KEYS = {
    "cell_uid",
    "split",
    "batch",
    "protocol",
    "softlabel_npz",
    "source_profile_npz",
    "profile_npz",
}


def mae(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.nanmean(np.abs(a - b)))


def rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def corr(a, b) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def state_metrics(prefix: str, pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mae": mae(pred, true),
        f"{prefix}_rmse": rmse(pred, true),
        f"{prefix}_corr": corr(pred, true),
    }


def compact_profile_metrics(cell_uid: str, split: str, pred: Dict[str, np.ndarray], true: Dict[str, np.ndarray]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"cell_uid": cell_uid, "split": split, "n_points": int(len(true["phis_c"]))}
    row.update(state_metrics("theta_a", pred["theta_a"], true["theta_a"]))
    row.update(state_metrics("theta_c", pred["theta_c"], true["theta_c"]))
    row.update(state_metrics("phie", pred["phie"], true["phie"]))
    row.update(state_metrics("phis_c", pred["phis_c"], true["phis_c"]))
    row["theta_mean_mae"] = float(0.5 * (row["theta_a_mae"] + row["theta_c_mae"]))
    return row


def _to_float_or_nan(value) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _is_numeric_metric_key(key: str, rows: List[dict]) -> bool:
    if key in NON_NUMERIC_KEYS:
        return False
    if key.endswith("_path") or key.endswith("_json") or key.endswith("_npz"):
        return False
    vals = [_to_float_or_nan(r.get(key, "")) for r in rows if key in r]
    return bool(vals) and any(np.isfinite(v) for v in vals)


def aggregate_metrics(rows: List[dict]) -> List[dict]:
    """Aggregate metrics by split, using only numeric columns.

    Returns rows like:
      split, profile_count, n_points, mean_theta_a_mae, ...
    """
    if not rows:
        return []
    out = []
    splits = sorted(set(str(r.get("split", "")) for r in rows if str(r.get("split", ""))))
    for split in splits:
        subset = [r for r in rows if str(r.get("split", "")) == split]
        agg: Dict[str, Any] = {
            "split": split,
            "profile_count": int(len(subset)),
            "n_points": int(sum(int(float(r.get("n_points", 0) or 0)) for r in subset)),
        }
        keys = []
        for r in subset:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        for key in keys:
            if not _is_numeric_metric_key(key, subset):
                continue
            vals = np.asarray([_to_float_or_nan(r.get(key, "")) for r in subset], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                agg[f"mean_{key}"] = float(np.nanmean(vals))
                agg[f"max_{key}"] = float(np.nanmax(vals))
        out.append(agg)
    return out


def global_compact_summary(rows: List[dict]) -> Dict[str, Any]:
    by_split = {r["split"]: r for r in aggregate_metrics(rows)}
    return {
        "split_count": len(by_split),
        "splits": sorted(by_split.keys()),
        "by_split": by_split,
    }
