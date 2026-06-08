# -*- coding: utf-8 -*-
"""Metrics for D14-P5B closed-set precision benchmark."""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np


def mae(a, b) -> float:
    return float(np.nanmean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def rmse(a, b) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.nanmean(d * d)))


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


def profile_metrics(cell_uid: str, batch: str, protocol: str, pred: Dict[str, np.ndarray], true: Dict[str, np.ndarray]) -> Dict[str, Any]:
    row = {"cell_uid": cell_uid, "batch": batch, "protocol": protocol, "n_points": int(len(true["phis_c"]))}
    for key in ["theta_a", "theta_c", "phie", "phis_c", "cs_a", "cs_c"]:
        row[f"{key}_mae"] = mae(pred[key], true[key])
        row[f"{key}_rmse"] = rmse(pred[key], true[key])
        row[f"{key}_corr"] = corr(pred[key], true[key])
    row["theta_mean_mae"] = 0.5 * (row["theta_a_mae"] + row["theta_c_mae"])
    row["cs_mean_mae"] = 0.5 * (row["cs_a_mae"] + row["cs_c_mae"])
    return row


def aggregate(rows: List[dict]) -> Dict[str, Any]:
    numeric_keys = []
    for row in rows:
        for k, v in row.items():
            if k in {"cell_uid", "batch", "protocol"}:
                continue
            try:
                float(v)
                if k not in numeric_keys:
                    numeric_keys.append(k)
            except Exception:
                pass
    out: Dict[str, Any] = {"profile_count": len(rows)}
    for k in numeric_keys:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except Exception:
                pass
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"max_{k}"] = float(np.max(vals))
            out[f"min_{k}"] = float(np.min(vals))
    return out


def by_group(rows: List[dict], key: str) -> List[dict]:
    groups = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "")), []).append(r)
    out = []
    for g, subset in sorted(groups.items()):
        agg = aggregate(subset)
        agg[key] = g
        out.append(agg)
    return out
