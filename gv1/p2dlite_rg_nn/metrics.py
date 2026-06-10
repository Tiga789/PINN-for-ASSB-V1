from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    x = x[m]
    y = y[m]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(t) & np.isfinite(p)
    if m.sum() == 0:
        return {f'{prefix}_count': 0, f'{prefix}_mae': float('nan'), f'{prefix}_rmse': float('nan'), f'{prefix}_max_abs': float('nan'), f'{prefix}_corr': float('nan')}
    e = p[m] - t[m]
    return {
        f'{prefix}_count': int(m.sum()),
        f'{prefix}_mae': float(np.mean(np.abs(e))),
        f'{prefix}_rmse': float(np.sqrt(np.mean(e ** 2))),
        f'{prefix}_max_abs': float(np.max(np.abs(e))),
        f'{prefix}_bias': float(np.mean(e)),
        f'{prefix}_corr': safe_corr(t[m], p[m]),
    }


def volume_weights(nr: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, nr + 1)
    w = edges[1:] ** 3 - edges[:-1] ** 3
    return w / w.sum()


def unpack_targets(Y: np.ndarray, slices: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    return {name: Y[:, s:e] for name, (s, e) in slices.items()}


def compute_rg_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, slices: Dict[str, Tuple[int, int]], eps: float = 1e-5) -> Dict[str, Any]:
    t = unpack_targets(Y_true, slices)
    p = unpack_targets(Y_pred, slices)
    out: Dict[str, Any] = {}
    th_a_t = t['theta_a']
    th_a_p = p['theta_a']
    th_c_t = t['theta_c']
    th_c_p = p['theta_c']
    phie_t = t['phie'].reshape(-1)
    phie_p = p['phie'].reshape(-1)
    phis_t = t['phis_c'].reshape(-1)
    phis_p = p['phis_c'].reshape(-1)
    out.update(basic_metrics(phis_t, phis_p, 'phis_c'))
    out.update(basic_metrics(phie_t, phie_p, 'phie'))
    out.update(basic_metrics(th_a_t, th_a_p, 'theta_a'))
    out.update(basic_metrics(th_c_t, th_c_p, 'theta_c'))
    wa = volume_weights(th_a_t.shape[1])
    wc = volume_weights(th_c_t.shape[1])
    mean_a_t = np.sum(th_a_t * wa[None, :], axis=1)
    mean_a_p = np.sum(th_a_p * wa[None, :], axis=1)
    mean_c_t = np.sum(th_c_t * wc[None, :], axis=1)
    mean_c_p = np.sum(th_c_p * wc[None, :], axis=1)
    out.update(basic_metrics(mean_a_t, mean_a_p, 'theta_a_mean'))
    out.update(basic_metrics(mean_c_t, mean_c_p, 'theta_c_mean'))
    grad_a_t = th_a_t[:, -1] - th_a_t[:, 0]
    grad_a_p = th_a_p[:, -1] - th_a_p[:, 0]
    grad_c_t = th_c_t[:, -1] - th_c_t[:, 0]
    grad_c_p = th_c_p[:, -1] - th_c_p[:, 0]
    out.update(basic_metrics(grad_a_t, grad_a_p, 'grad_a_surface_center'))
    out.update(basic_metrics(grad_c_t, grad_c_p, 'grad_c_surface_center'))
    all_theta_pred = np.concatenate([th_a_p.reshape(-1), th_c_p.reshape(-1)])
    all_theta_true = np.concatenate([th_a_t.reshape(-1), th_c_t.reshape(-1)])
    out['true_theta_boundary_hit_fraction'] = float(np.mean((all_theta_true <= eps) | (all_theta_true >= 1.0 - eps)))
    out['pred_theta_boundary_hit_fraction'] = float(np.mean((all_theta_pred <= eps) | (all_theta_pred >= 1.0 - eps)))
    out['pred_theta_outside_fraction'] = float(np.mean((all_theta_pred < -eps) | (all_theta_pred > 1.0 + eps)))
    out['pred_theta_min'] = float(np.nanmin(all_theta_pred))
    out['pred_theta_max'] = float(np.nanmax(all_theta_pred))
    out['true_theta_min'] = float(np.nanmin(all_theta_true))
    out['true_theta_max'] = float(np.nanmax(all_theta_true))
    return out


def thresholds_status(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, Any]:
    checks = []
    def chk(name: str, value_key: str, op: str, threshold_key: str):
        val = metrics.get(value_key, float('nan'))
        thr = thresholds.get(threshold_key, None)
        if thr is None:
            return
        if not np.isfinite(float(val)):
            ok = False
        elif op == '<=':
            ok = float(val) <= float(thr)
        elif op == '>=':
            ok = float(val) >= float(thr)
        else:
            raise ValueError(op)
        checks.append({'name': name, 'metric': value_key, 'value': float(val) if np.isfinite(float(val)) else None, 'op': op, 'threshold': float(thr), 'status': 'PASS' if ok else 'FAIL'})
    chk('phis_c MAE', 'phis_c_mae', '<=', 'smoke_global_phis_c_mae_max_v')
    chk('phie MAE', 'phie_mae', '<=', 'smoke_global_phie_mae_max_v')
    chk('theta_a MAE', 'theta_a_mae', '<=', 'smoke_global_theta_a_mae_max')
    chk('theta_c MAE', 'theta_c_mae', '<=', 'smoke_global_theta_c_mae_max')
    chk('theta_a_mean MAE', 'theta_a_mean_mae', '<=', 'smoke_global_theta_a_mean_mae_max')
    chk('theta_c_mean MAE', 'theta_c_mean_mae', '<=', 'smoke_global_theta_c_mean_mae_max')
    chk('grad_a MAE', 'grad_a_surface_center_mae', '<=', 'smoke_global_grad_a_mae_max')
    chk('grad_c MAE', 'grad_c_surface_center_mae', '<=', 'smoke_global_grad_c_mae_max')
    min_corr_keys = ['phis_c_corr', 'phie_corr', 'theta_a_corr', 'theta_c_corr', 'theta_a_mean_corr', 'theta_c_mean_corr', 'grad_a_surface_center_corr', 'grad_c_surface_center_corr']
    corr_vals = [float(metrics.get(k, float('nan'))) for k in min_corr_keys]
    finite_corr = [v for v in corr_vals if np.isfinite(v)]
    metrics['min_selected_corr'] = float(np.min(finite_corr)) if finite_corr else float('nan')
    chk('min selected corr', 'min_selected_corr', '>=', 'smoke_global_min_corr')
    chk('pred theta outside fraction', 'pred_theta_outside_fraction', '<=', 'pred_theta_outside_fraction_max')
    chk('pred theta boundary hit fraction', 'pred_theta_boundary_hit_fraction', '<=', 'pred_theta_boundary_hit_fraction_max')
    fail_count = sum(1 for c in checks if c['status'] != 'PASS')
    return {'overall_status': 'PASS' if fail_count == 0 else 'REVIEW', 'fail_count': fail_count, 'checks': checks}
