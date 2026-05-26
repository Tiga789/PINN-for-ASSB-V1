from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


InterpMode = Literal['linear', 'previous']


@dataclass
class CurrentInterpolator:
    """Callable I(t) interpolator for measured-current replay.

    ``linear`` is good for already-sampled smooth profiles. ``previous`` keeps
    piecewise-constant steps exactly, which is useful for CC/rest data.
    """

    t_s: np.ndarray
    current_A: np.ndarray
    mode: InterpMode = 'linear'
    fill_value: str = 'edge'

    def __post_init__(self) -> None:
        t = np.asarray(self.t_s, dtype=float)
        i = np.asarray(self.current_A, dtype=float)
        mask = np.isfinite(t) & np.isfinite(i)
        if mask.sum() < 2:
            raise ValueError('At least two finite t/current points are required')
        t = t[mask]
        i = i[mask]
        order = np.argsort(t)
        t = t[order]
        i = i[order]
        # Collapse duplicate timestamps by taking the last observed value.
        uniq, last_idx = np.unique(t, return_index=False, return_inverse=False, return_counts=False), None
        if len(uniq) != len(t):
            vals = []
            for u in uniq:
                vals.append(i[t == u][-1])
            t, i = uniq, np.asarray(vals, dtype=float)
        self.t_s = t
        self.current_A = i

    @property
    def t_min(self) -> float:
        return float(self.t_s[0])

    @property
    def t_max(self) -> float:
        return float(self.t_s[-1])

    def __call__(self, t_query_s: np.ndarray | float) -> np.ndarray:
        q = np.asarray(t_query_s, dtype=float)
        if self.mode == 'previous':
            idx = np.searchsorted(self.t_s, q, side='right') - 1
            idx = np.clip(idx, 0, len(self.current_A) - 1)
            out = self.current_A[idx]
        else:
            out = np.interp(q, self.t_s, self.current_A, left=self.current_A[0], right=self.current_A[-1])
        return out

    def derivative(self, t_query_s: np.ndarray | float) -> np.ndarray:
        """Return dI/dt estimated from the measured profile."""
        if len(self.t_s) < 3:
            return np.zeros_like(np.asarray(t_query_s, dtype=float))
        d = np.gradient(self.current_A, self.t_s, edge_order=1)
        return np.interp(np.asarray(t_query_s, dtype=float), self.t_s, d, left=d[0], right=d[-1])

    def features_at(self, t_query_s: np.ndarray | float, *, rest_threshold_A: float = 1e-9) -> dict[str, np.ndarray]:
        i = self(t_query_s)
        di = self.derivative(t_query_s)
        return {
            'I_A': i,
            'abs_I_A': np.abs(i),
            'dI_dt_A_per_s': di,
            'rest_flag': (np.abs(i) <= abs(rest_threshold_A)).astype(float),
        }


def build_current_interpolator(t_s: np.ndarray, current_A: np.ndarray, *, mode: InterpMode = 'linear') -> CurrentInterpolator:
    return CurrentInterpolator(np.asarray(t_s, dtype=float), np.asarray(current_A, dtype=float), mode=mode)
