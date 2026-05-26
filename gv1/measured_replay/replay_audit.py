from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from .profile_builder import ReplayProfile, build_replay_profile
except Exception:  # pragma: no cover
    ReplayProfile = Any  # type: ignore
    build_replay_profile = None  # type: ignore


@dataclass
class ReplayAuditResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_stats(name: str, arr: np.ndarray, metrics: dict[str, Any], errors: list[str]) -> None:
    arr = np.asarray(arr)
    finite = np.isfinite(arr.astype(float)) if arr.dtype.kind in 'biufc' else np.ones(len(arr), dtype=bool)
    metrics[f'{name}_finite_fraction'] = float(finite.mean()) if len(arr) else 0.0
    if len(arr) and not finite.all():
        errors.append(f'{name} contains non-finite values')
    if len(arr) and arr.dtype.kind in 'biufc':
        metrics[f'{name}_min'] = float(np.nanmin(arr.astype(float)))
        metrics[f'{name}_max'] = float(np.nanmax(arr.astype(float)))


def audit_replay_profile(profile: ReplayProfile, *, current_threshold_A: float = 1e-9) -> ReplayAuditResult:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    t = np.asarray(profile.t_s, dtype=float)
    i = np.asarray(profile.current_A, dtype=float)
    metrics['n_points'] = int(len(t))
    if len(t) < 2:
        errors.append('profile has fewer than two time points')
    _finite_stats('time_s', t, metrics, errors)
    _finite_stats('current_A', i, metrics, errors)
    if len(t) >= 2:
        dt = np.diff(t)
        metrics['dt_min_s'] = float(np.nanmin(dt))
        metrics['dt_median_s'] = float(np.nanmedian(dt))
        metrics['dt_max_s'] = float(np.nanmax(dt))
        if np.nanmin(dt) < 0:
            errors.append('time_s is not monotonic increasing')
        if np.nanmin(dt) == 0:
            warnings.append('time_s contains duplicate adjacent timestamps')
    metrics['has_charge'] = bool(np.nanmax(i) > current_threshold_A) if len(i) else False
    metrics['has_discharge'] = bool(np.nanmin(i) < -current_threshold_A) if len(i) else False
    metrics['has_rest'] = bool(np.any(np.abs(i) <= current_threshold_A)) if len(i) else False
    metrics['current_abs_max_A'] = float(np.nanmax(np.abs(i))) if len(i) else None
    if not metrics['has_charge'] and not metrics['has_discharge']:
        errors.append('current profile contains neither charge nor discharge above threshold')
    if profile.voltage_V is not None:
        v = np.asarray(profile.voltage_V, dtype=float)
        _finite_stats('voltage_V', v, metrics, errors)
        if len(v):
            if np.nanmin(v) < 0 or np.nanmax(v) > 10:
                warnings.append('voltage range is outside a typical single-cell Li-ion range')
            p = v * i
            metrics['power_W_mean'] = float(np.nanmean(p))
            metrics['power_W_std'] = float(np.nanstd(p))
            denom = max(abs(metrics['power_W_mean']), 1e-12)
            metrics['power_cv_abs'] = float(metrics['power_W_std'] / denom)
    if profile.cycle_id is not None:
        cyc = pd.to_numeric(pd.Series(profile.cycle_id), errors='coerce').dropna()
        metrics['cycle_count'] = int(cyc.nunique()) if not cyc.empty else 0
    ok = len(errors) == 0
    return ReplayAuditResult(ok=ok, errors=errors, warnings=warnings, metrics=metrics)


def audit_standard_table(df: pd.DataFrame, *, current_threshold_A: float = 1e-9) -> ReplayAuditResult:
    if build_replay_profile is None:
        raise ImportError('gv1.measured_replay.profile_builder is required')
    try:
        profile = build_replay_profile(df)
        return audit_replay_profile(profile, current_threshold_A=current_threshold_A)
    except Exception as exc:
        return ReplayAuditResult(ok=False, errors=[f'{type(exc).__name__}: {exc}'], warnings=[], metrics={'n_rows': int(len(df))})
