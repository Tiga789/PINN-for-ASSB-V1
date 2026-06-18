from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.nanmean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 2:
        return float("nan")
    yt = yt[mask]
    yp = yp[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if ss_tot <= 1e-30:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def corr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 3:
        return float("nan")
    yt = yt[mask]
    yp = yp[mask]
    sy = float(np.std(yt))
    sp = float(np.std(yp))
    if sy <= 1e-30 or sp <= 1e-30:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_slices: Mapping[str, Tuple[int, int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, (a, b) in target_slices.items():
        yt = y_true[:, a:b]
        yp = y_pred[:, a:b]
        out[key] = {
            "mae": mae(yt, yp),
            "rmse": rmse(yt, yp),
            "r2": r2_score(yt, yp),
            "corr": corr_score(yt, yp),
        }
    vals = [v["r2"] for v in out.values() if np.isfinite(v.get("r2", float("nan")))]
    out["__aggregate__"] = {
        "r2_mean": float(np.mean(vals)) if vals else float("nan"),
        "r2_min": float(np.min(vals)) if vals else float("nan"),
        "target_count": float(len(vals)),
    }
    return out


def profile_metrics(profiles: Sequence[Any], pred_arrays: Sequence[np.ndarray]) -> Dict[str, Any]:
    rows = []
    for prof, pred in zip(profiles, pred_arrays):
        gm = group_metrics(prof.targets, pred, prof.target_slices)
        row: Dict[str, Any] = {
            "split": prof.split,
            "canonical_cell_uid": prof.canonical_cell_uid,
            "protocol": prof.protocol,
            "semantic_branch": prof.branch,
            "n_points": int(prof.targets.shape[0]),
            "r2_mean": gm["__aggregate__"]["r2_mean"],
            "r2_min": gm["__aggregate__"]["r2_min"],
        }
        for k, stats in gm.items():
            if k == "__aggregate__":
                continue
            for sk, sv in stats.items():
                row[f"{k}_{sk}"] = sv
        rows.append(row)
    return {"rows": rows}


def aggregate_profile_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys() if k.endswith("_r2") or k.endswith("_mae") or k in {"r2_mean", "r2_min"}})
    out: Dict[str, float] = {}
    for k in keys:
        vals = []
        for r in rows:
            try:
                v = float(r.get(k))
                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_min"] = float(np.min(vals))
            out[f"{k}_max"] = float(np.max(vals))
    return out
