from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np


def _slice_pair(slices: Mapping[str, Tuple[int, int]], name: str) -> Tuple[int, int]:
    if name not in slices:
        raise KeyError(f"target_slices missing {name!r}; available={sorted(slices.keys())}")
    s, e = slices[name]
    return int(s), int(e)


def apply_theta_projection(
    y_pred: np.ndarray,
    target_slices: Mapping[str, Tuple[int, int]],
    theta_min: float = 1e-4,
    theta_max: float = 1.0 - 1e-4,
    apply_to: tuple[str, ...] = ("theta_a", "theta_c"),
) -> np.ndarray:
    """Clip only theta_a/theta_c output channels and leave phie/phis_c unchanged."""
    yp = np.asarray(y_pred, dtype=np.float32).copy()
    lo = float(theta_min)
    hi = float(theta_max)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo < 0 or hi > 1 or lo >= hi:
        raise ValueError(f"Invalid theta projection range: [{theta_min}, {theta_max}]")
    for key in apply_to:
        s, e = _slice_pair(target_slices, key)
        yp[:, s:e] = np.clip(yp[:, s:e], lo, hi)
    return yp


def theta_outside_counts(
    y_pred: np.ndarray,
    target_slices: Mapping[str, Tuple[int, int]],
    eps: float = 1e-5,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    total = 0
    outside = 0
    boundary = 0
    min_v = float("inf")
    max_v = float("-inf")
    for key in ("theta_a", "theta_c"):
        s, e = _slice_pair(target_slices, key)
        arr = np.asarray(y_pred[:, s:e], dtype=float).reshape(-1)
        m = np.isfinite(arr)
        arr = arr[m]
        n = int(arr.size)
        if n == 0:
            out[f"{key}_outside_fraction"] = float("nan")
            out[f"{key}_boundary_hit_fraction"] = float("nan")
            continue
        o = int(np.sum((arr < -eps) | (arr > 1.0 + eps)))
        b = int(np.sum((arr <= eps) | (arr >= 1.0 - eps)))
        out[f"{key}_outside_count"] = o
        out[f"{key}_outside_fraction"] = float(o / n)
        out[f"{key}_boundary_hit_count"] = b
        out[f"{key}_boundary_hit_fraction"] = float(b / n)
        out[f"{key}_min"] = float(np.min(arr))
        out[f"{key}_max"] = float(np.max(arr))
        total += n
        outside += o
        boundary += b
        min_v = min(min_v, float(np.min(arr)))
        max_v = max(max_v, float(np.max(arr)))
    out["theta_total_count"] = int(total)
    out["theta_outside_count"] = int(outside)
    out["theta_outside_fraction"] = float(outside / total) if total else float("nan")
    out["theta_boundary_hit_count"] = int(boundary)
    out["theta_boundary_hit_fraction"] = float(boundary / total) if total else float("nan")
    out["theta_min"] = min_v if np.isfinite(min_v) else float("nan")
    out["theta_max"] = max_v if np.isfinite(max_v) else float("nan")
    return out


def top_theta_outside_points(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_slices: Mapping[str, Tuple[int, int]],
    profile_id: str,
    t_global_s: np.ndarray | None = None,
    top_k: int = 50,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for key in ("theta_a", "theta_c"):
        s, e = _slice_pair(target_slices, key)
        pred = np.asarray(y_pred[:, s:e], dtype=float)
        true = np.asarray(y_true[:, s:e], dtype=float)
        # distance outside [0, 1]
        dist = np.maximum(-pred, pred - 1.0)
        idx = np.argwhere(dist > 0)
        for ti, ri in idx:
            rows.append({
                "profile_id": profile_id,
                "electrode": key,
                "time_index": int(ti),
                "radial_index": int(ri),
                "t_global_s": float(t_global_s[ti]) if t_global_s is not None and ti < len(t_global_s) else None,
                "theta_true": float(true[ti, ri]),
                "theta_pred_raw": float(pred[ti, ri]),
                "outside_distance": float(dist[ti, ri]),
                "abs_error_raw": float(abs(pred[ti, ri] - true[ti, ri])),
            })
    rows.sort(key=lambda r: float(r["outside_distance"]), reverse=True)
    return rows[:max(0, int(top_k))]


def compare_mae_nonregression(raw: Mapping[str, Any], projected: Mapping[str, Any], thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    """Check projection did not repair boundary by destroying average precision."""
    rel = float(thresholds.get("allowed_theta_mae_relative_worsening", 0.15))
    grad_rel = float(thresholds.get("allowed_gradient_mae_relative_worsening", 0.20))
    abs_extra = float(thresholds.get("allowed_absolute_mae_worsening", 0.002))
    checks = []

    def _check(metric: str, rel_allowed: float):
        r = float(raw.get(metric, float("nan")))
        p = float(projected.get(metric, float("nan")))
        limit = r * (1.0 + rel_allowed) + abs_extra
        ok = np.isfinite(r) and np.isfinite(p) and p <= limit
        checks.append({
            "metric": metric,
            "raw": r if np.isfinite(r) else None,
            "projected": p if np.isfinite(p) else None,
            "limit": float(limit) if np.isfinite(limit) else None,
            "status": "PASS" if ok else "REVIEW",
        })

    for m in ["theta_a_mae", "theta_c_mae", "theta_a_mean_mae", "theta_c_mean_mae"]:
        _check(m, rel)
    for m in ["grad_a_surface_center_mae", "grad_c_surface_center_mae"]:
        _check(m, grad_rel)
    review_count = sum(1 for c in checks if c["status"] != "PASS")
    return {"overall_status": "PASS" if review_count == 0 else "REVIEW", "review_count": review_count, "checks": checks}
