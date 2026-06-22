from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    mae: float
    rmse: float
    bias: float
    r2: float
    corr: float
    nmae_range: float
    nrmse_range: float
    std_ratio: float
    range_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "r2": self.r2,
            "corr": self.corr,
            "nmae_range": self.nmae_range,
            "nrmse_range": self.nrmse_range,
            "std_ratio": self.std_ratio,
            "range_ratio": self.range_ratio,
        }


def finite_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(yt.size, yp.size)
    yt = yt[:n]
    yp = yp[:n]
    good = np.isfinite(yt) & np.isfinite(yp)
    return yt[good], yp[good]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> RegressionMetrics:
    yt, yp = finite_pairs(y_true, y_pred)
    n = int(yt.size)
    if n == 0:
        nan = float("nan")
        return RegressionMetrics(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(np.mean(err))
    true_mean = float(np.mean(yt))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - true_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else (1.0 if ss_res <= eps else float("nan"))
    yt_std = float(np.std(yt))
    yp_std = float(np.std(yp))
    if yt_std > eps and yp_std > eps:
        corr = float(np.corrcoef(yt, yp)[0, 1])
    else:
        corr = float("nan")
    true_range = float(np.max(yt) - np.min(yt))
    pred_range = float(np.max(yp) - np.min(yp))
    nmae = mae / true_range if true_range > eps else float("nan")
    nrmse = rmse / true_range if true_range > eps else float("nan")
    std_ratio = yp_std / yt_std if yt_std > eps else float("nan")
    range_ratio = pred_range / true_range if true_range > eps else float("nan")
    return RegressionMetrics(n, mae, rmse, bias, r2, corr, nmae, nrmse, std_ratio, range_ratio)


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray]:
    yt, yp = finite_pairs(y_true, y_pred)
    if yt.size < 2 or np.nanstd(yp) < 1e-12:
        offset = float(np.nanmean(yt - yp)) if yt.size else 0.0
        return 1.0, offset, np.asarray(y_pred, dtype=np.float64) + offset
    design = np.column_stack([yp, np.ones_like(yp)])
    coeff, *_ = np.linalg.lstsq(design, yt, rcond=None)
    slope, intercept = float(coeff[0]), float(coeff[1])
    corrected = slope * np.asarray(y_pred, dtype=np.float64) + intercept
    return slope, intercept, corrected


def constant_shift_correct(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    yt, yp = finite_pairs(y_true, y_pred)
    shift = float(np.mean(yt - yp)) if yt.size else 0.0
    return shift, np.asarray(y_pred, dtype=np.float64) + shift


def shift_time_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = min(yt.shape[0], yp.shape[0])
    yt = yt[:n]
    yp = yp[:n]
    if lag > 0:
        # Prediction is delayed: compare true[lag:] against pred[:-lag].
        return yt[lag:], yp[:-lag]
    if lag < 0:
        k = -lag
        return yt[:-k], yp[k:]
    return yt, yp


def best_integer_lag(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_lag: int = 64,
    min_points: int = 32,
) -> dict[str, float | int]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim > 1:
        yt = np.nanmean(yt, axis=tuple(range(1, yt.ndim)))
    if yp.ndim > 1:
        yp = np.nanmean(yp, axis=tuple(range(1, yp.ndim)))
    yt = yt.reshape(-1)
    yp = yp.reshape(-1)
    n = min(yt.size, yp.size)
    if n < min_points:
        return {"best_lag_samples": 0, "best_lag_r2": float("nan"), "zero_lag_r2": regression_metrics(yt, yp).r2}
    max_lag = max(0, min(int(max_lag), max(0, n // 4)))
    zero_r2 = regression_metrics(yt, yp).r2
    best_lag = 0
    best_r2 = zero_r2
    for lag in range(-max_lag, max_lag + 1):
        a, b = shift_time_arrays(yt, yp, lag)
        if a.size < min_points:
            continue
        score = regression_metrics(a, b).r2
        if np.isfinite(score) and (not np.isfinite(best_r2) or score > best_r2):
            best_lag = lag
            best_r2 = score
    return {
        "best_lag_samples": int(best_lag),
        "best_lag_r2": float(best_r2),
        "zero_lag_r2": float(zero_r2),
        "lag_r2_gain": float(best_r2 - zero_r2) if np.isfinite(best_r2) and np.isfinite(zero_r2) else float("nan"),
    }


def shell_volume_weights(radial_grid: np.ndarray | None, n_r: int) -> np.ndarray:
    if radial_grid is None:
        rho = np.linspace(0.0, 1.0, n_r, dtype=np.float64)
    else:
        rho = np.asarray(radial_grid, dtype=np.float64).reshape(-1)
        if rho.size != n_r or not np.all(np.isfinite(rho)):
            rho = np.linspace(0.0, 1.0, n_r, dtype=np.float64)
        else:
            lo, hi = float(np.min(rho)), float(np.max(rho))
            span = hi - lo
            rho = (rho - lo) / span if span > 0 else np.linspace(0.0, 1.0, n_r, dtype=np.float64)
    if n_r == 1:
        return np.ones(1, dtype=np.float64)
    edges = np.empty(n_r + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (rho[:-1] + rho[1:])
    edges = np.clip(np.maximum.accumulate(edges), 0.0, 1.0)
    weights = np.diff(edges**3)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        return np.full(n_r, 1.0 / n_r, dtype=np.float64)
    return weights / total


def volume_mean(field: np.ndarray, radial_grid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 1:
        return arr.copy()
    if arr.ndim != 2:
        raise ValueError(f"volume_mean expects (time, radial), got {arr.shape}")
    weights = shell_volume_weights(radial_grid, arr.shape[1])
    return np.sum(arr * weights[None, :], axis=1)


def radial_deviation(field: np.ndarray, radial_grid: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"radial_deviation expects (time, radial), got {arr.shape}")
    return arr - volume_mean(arr, radial_grid)[:, None]


def residual_svd_summary(residual: np.ndarray, energy_thresholds: tuple[float, ...] = (0.9, 0.95, 0.99)) -> dict[str, Any]:
    arr = np.asarray(residual, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    good_rows = np.all(np.isfinite(arr), axis=1)
    arr = arr[good_rows]
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        return {
            "rank_max": 0,
            "singular_values": [],
            "energy_cumulative": [],
            "rank_at_90": 0,
            "rank_at_95": 0,
            "rank_at_99": 0,
            "energy_rank1": float("nan"),
            "energy_rank2": float("nan"),
            "energy_rank4": float("nan"),
            "energy_rank8": float("nan"),
        }
    centered = arr - np.mean(arr, axis=0, keepdims=True)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    energy = singular**2
    total = float(np.sum(energy))
    cumulative = np.cumsum(energy) / total if total > 0 else np.zeros_like(energy)

    def rank_at(threshold: float) -> int:
        if cumulative.size == 0:
            return 0
        return int(np.searchsorted(cumulative, threshold, side="left") + 1)

    def energy_at(rank: int) -> float:
        if cumulative.size == 0:
            return float("nan")
        return float(cumulative[min(rank, cumulative.size) - 1])

    return {
        "rank_max": int(singular.size),
        "singular_values": singular[:16].tolist(),
        "energy_cumulative": cumulative[:16].tolist(),
        "rank_at_90": rank_at(energy_thresholds[0]),
        "rank_at_95": rank_at(energy_thresholds[1]),
        "rank_at_99": rank_at(energy_thresholds[2]),
        "energy_rank1": energy_at(1),
        "energy_rank2": energy_at(2),
        "energy_rank4": energy_at(4),
        "energy_rank8": energy_at(8),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    a, b = finite_pairs(x, y)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def linear_trend(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    a, b = finite_pairs(x, y)
    if a.size < 2 or np.std(a) < 1e-12:
        return {"slope": float("nan"), "intercept": float("nan"), "corr": float("nan")}
    design = np.column_stack([a, np.ones_like(a)])
    coeff, *_ = np.linalg.lstsq(design, b, rcond=None)
    return {"slope": float(coeff[0]), "intercept": float(coeff[1]), "corr": safe_corr(a, b)}
