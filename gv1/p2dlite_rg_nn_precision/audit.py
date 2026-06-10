from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return str(x)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def write_csv(rows: List[Mapping[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with p.open('w', encoding='utf-8', newline='') as f:
            f.write('')
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def read_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float('nan')
    x = x[m]
    y = y[m]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def r2_score(a: np.ndarray, b: np.ndarray) -> float:
    t = np.asarray(a, dtype=float).reshape(-1)
    p = np.asarray(b, dtype=float).reshape(-1)
    m = np.isfinite(t) & np.isfinite(p)
    if int(m.sum()) < 3:
        return float('nan')
    t = t[m]
    p = p[m]
    denom = float(np.sum((t - np.mean(t)) ** 2))
    if denom < 1e-18:
        return float('nan')
    return float(1.0 - np.sum((p - t) ** 2) / denom)


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    t = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.isfinite(t) & np.isfinite(p)
    if int(m.sum()) == 0:
        return {
            f'{prefix}_count': 0,
            f'{prefix}_mae': float('nan'),
            f'{prefix}_rmse': float('nan'),
            f'{prefix}_nrmse_range': float('nan'),
            f'{prefix}_max_abs': float('nan'),
            f'{prefix}_bias': float('nan'),
            f'{prefix}_corr': float('nan'),
            f'{prefix}_r2': float('nan'),
        }
    t = t[m]
    p = p[m]
    e = p - t
    rng = float(np.max(t) - np.min(t))
    if not np.isfinite(rng) or rng < 1e-12:
        std = float(np.std(t))
        rng = std if std > 1e-12 else 1.0
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return {
        f'{prefix}_count': int(m.sum()),
        f'{prefix}_mae': float(np.mean(np.abs(e))),
        f'{prefix}_rmse': rmse,
        f'{prefix}_nrmse_range': float(rmse / rng),
        f'{prefix}_max_abs': float(np.max(np.abs(e))),
        f'{prefix}_bias': float(np.mean(e)),
        f'{prefix}_corr': safe_corr(t, p),
        f'{prefix}_r2': r2_score(t, p),
    }


def target_slices_from_names(target_names: Iterable[Any]) -> Dict[str, Tuple[int, int]]:
    names = [str(x) for x in list(target_names)]
    def starts(prefix: str) -> List[int]:
        return [i for i, n in enumerate(names) if n.startswith(prefix)]
    a = starts('theta_a_r')
    c = starts('theta_c_r')
    if not a or not c:
        raise ValueError('target_names does not include theta_a_r*/theta_c_r*')
    phie = names.index('phie') if 'phie' in names else None
    phis = names.index('phis_c') if 'phis_c' in names else None
    if phie is None or phis is None:
        raise ValueError('target_names must include phie and phis_c')
    return {
        'theta_a': (min(a), max(a) + 1),
        'theta_c': (min(c), max(c) + 1),
        'phie': (phie, phie + 1),
        'phis_c': (phis, phis + 1),
    }


def volume_weights(nr: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, nr + 1)
    w = edges[1:] ** 3 - edges[:-1] ** 3
    return w / w.sum()


def unpack(Y: np.ndarray, slices: Mapping[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    return {k: Y[:, s:e] for k, (s, e) in slices.items()}


def computed_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, slices: Mapping[str, Tuple[int, int]], eps: float = 1e-5) -> Dict[str, Any]:
    t = unpack(Y_true, slices)
    p = unpack(Y_pred, slices)
    out: Dict[str, Any] = {}
    th_a_t = t['theta_a']
    th_a_p = p['theta_a']
    th_c_t = t['theta_c']
    th_c_p = p['theta_c']
    out.update(basic_metrics(t['phis_c'].reshape(-1), p['phis_c'].reshape(-1), 'phis_c'))
    out.update(basic_metrics(t['phie'].reshape(-1), p['phie'].reshape(-1), 'phie'))
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
    for elec, arr_t, arr_p in [('a', th_a_t, th_a_p), ('c', th_c_t, th_c_p)]:
        pred = arr_p.reshape(-1)
        true = arr_t.reshape(-1)
        out[f'theta_{elec}_true_boundary_hit_fraction'] = float(np.mean((true <= eps) | (true >= 1.0 - eps)))
        out[f'theta_{elec}_pred_boundary_hit_fraction'] = float(np.mean((pred <= eps) | (pred >= 1.0 - eps)))
        out[f'theta_{elec}_pred_outside_fraction'] = float(np.mean((pred < -eps) | (pred > 1.0 + eps)))
        out[f'theta_{elec}_pred_min'] = float(np.nanmin(pred))
        out[f'theta_{elec}_pred_max'] = float(np.nanmax(pred))
        out[f'theta_{elec}_true_min'] = float(np.nanmin(true))
        out[f'theta_{elec}_true_max'] = float(np.nanmax(true))
    all_pred = np.concatenate([th_a_p.reshape(-1), th_c_p.reshape(-1)])
    all_true = np.concatenate([th_a_t.reshape(-1), th_c_t.reshape(-1)])
    out['pred_theta_boundary_hit_fraction'] = float(np.mean((all_pred <= eps) | (all_pred >= 1.0 - eps)))
    out['pred_theta_outside_fraction'] = float(np.mean((all_pred < -eps) | (all_pred > 1.0 + eps)))
    out['true_theta_boundary_hit_fraction'] = float(np.mean((all_true <= eps) | (all_true >= 1.0 - eps)))
    out['pred_theta_min'] = float(np.nanmin(all_pred))
    out['pred_theta_max'] = float(np.nanmax(all_pred))
    out['true_theta_min'] = float(np.nanmin(all_true))
    out['true_theta_max'] = float(np.nanmax(all_true))
    corr_keys = [
        'phis_c_corr', 'phie_corr', 'theta_a_corr', 'theta_c_corr',
        'theta_a_mean_corr', 'theta_c_mean_corr', 'grad_a_surface_center_corr', 'grad_c_surface_center_corr'
    ]
    corrs = [float(out.get(k, float('nan'))) for k in corr_keys]
    corrs = [v for v in corrs if np.isfinite(v)]
    out['min_selected_corr'] = float(np.min(corrs)) if corrs else float('nan')
    r2_keys = [k for k in out.keys() if k.endswith('_r2')]
    r2s = [float(out.get(k, float('nan'))) for k in r2_keys]
    r2s = [v for v in r2s if np.isfinite(v)]
    out['min_selected_r2'] = float(np.min(r2s)) if r2s else float('nan')
    return out


def _scalar_string(x: Any, default: str = '') -> str:
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return str(arr.reshape(-1)[0])
    except Exception:
        return default


def load_source_profile(softlabel_dir: Path, profile_id: str, filename: str = 'solution_softlabels.npz') -> Optional[Dict[str, Any]]:
    p = softlabel_dir / profile_id / filename
    if not p.exists():
        # Some profile ids use forward slashes in npz and Windows backslashes in folder trees.
        p = softlabel_dir / Path(profile_id) / filename
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    return d


def _extract_current_and_cycle(src: Optional[Mapping[str, Any]], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    I = np.zeros(n, dtype=np.float64)
    cycle = np.full(n, -1, dtype=np.int64)
    step_code = np.zeros(n, dtype=np.int64)
    if not src:
        return I, cycle, step_code
    for key in ['I_profile', 'current_A', 'I_A', 'current', 'I']:
        if key in src:
            arr = np.asarray(src[key]).reshape(-1)
            if arr.size == n and arr.dtype.kind not in {'U', 'S', 'O'}:
                I = arr.astype(np.float64)
            break
    for key in ['cycle_id', 'cycle', 'cycle_index']:
        if key in src:
            arr = np.asarray(src[key]).reshape(-1)
            if arr.size == n and arr.dtype.kind not in {'U', 'S', 'O'}:
                cycle = arr.astype(np.int64)
            break
    if 'step_type' in src:
        arr = np.asarray(src['step_type']).reshape(-1)
        if arr.size == n:
            for i, v in enumerate(arr):
                s = str(v).lower()
                if 'rest' in s or '静' in s or '搁' in s:
                    step_code[i] = 0
                elif 'dis' in s or '放' in s:
                    step_code[i] = -1
                elif 'cha' in s or '充' in s:
                    step_code[i] = 1
    else:
        eps = max(1e-12, 0.02 * float(np.nanmax(np.abs(I)) + 1e-12))
        step_code[I > eps] = 1
        step_code[I < -eps] = -1
    return I, cycle, step_code


def transition_mask_from_current(I: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    I = np.asarray(I, dtype=np.float64).reshape(-1)
    if I.size == 0:
        return np.zeros(0, dtype=bool)
    frac = float(cfg.get('transition_abs_dI_fraction', 0.20))
    win = int(cfg.get('transition_window_points', 20))
    max_i = float(np.nanmax(np.abs(I)))
    if not np.isfinite(max_i) or max_i <= 1e-12:
        return np.zeros_like(I, dtype=bool)
    dI = np.abs(np.diff(I, prepend=I[0]))
    idx = np.where(dI >= frac * max_i)[0]
    m = np.zeros_like(I, dtype=bool)
    for i in idx:
        a = max(0, int(i) - win)
        b = min(I.size, int(i) + win + 1)
        m[a:b] = True
    return m


def rest_mask_from_current(I: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    I = np.asarray(I, dtype=np.float64).reshape(-1)
    if I.size == 0:
        return np.zeros(0, dtype=bool)
    frac = float(cfg.get('rest_current_fraction', 0.02))
    max_i = float(np.nanmax(np.abs(I)))
    eps = max(1e-12, frac * max_i)
    return np.abs(I) <= eps


def _var_arrays(Y: np.ndarray, slices: Mapping[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    u = unpack(Y, slices)
    th_a = u['theta_a']
    th_c = u['theta_c']
    wa = volume_weights(th_a.shape[1])
    wc = volume_weights(th_c.shape[1])
    return {
        'phis_c': u['phis_c'].reshape(-1),
        'phie': u['phie'].reshape(-1),
        'theta_a': th_a.reshape(th_a.shape[0], -1),
        'theta_c': th_c.reshape(th_c.shape[0], -1),
        'theta_a_mean': np.sum(th_a * wa[None, :], axis=1),
        'theta_c_mean': np.sum(th_c * wc[None, :], axis=1),
        'grad_a_surface_center': th_a[:, -1] - th_a[:, 0],
        'grad_c_surface_center': th_c[:, -1] - th_c[:, 0],
    }


def topk_errors(Y_true: np.ndarray, Y_pred: np.ndarray, slices: Mapping[str, Tuple[int, int]], t: np.ndarray, cycle: np.ndarray, profile_id: str, k: int = 50) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    vt = _var_arrays(Y_true, slices)
    vp = _var_arrays(Y_pred, slices)
    for name in ['phis_c', 'phie', 'theta_a_mean', 'theta_c_mean', 'grad_a_surface_center', 'grad_c_surface_center']:
        true = np.asarray(vt[name]).reshape(-1)
        pred = np.asarray(vp[name]).reshape(-1)
        err = pred - true
        kk = min(k, err.size)
        if kk <= 0:
            continue
        idx = np.argpartition(np.abs(err), -kk)[-kk:]
        idx = idx[np.argsort(-np.abs(err[idx]))]
        for rank, i in enumerate(idx, 1):
            rows.append({
                'profile_id': profile_id,
                'variable': name,
                'rank_within_variable': rank,
                'time_index': int(i),
                't_global_s': float(t[i]) if i < len(t) else None,
                'cycle_id': int(cycle[i]) if i < len(cycle) else None,
                'true': float(true[i]),
                'pred': float(pred[i]),
                'error': float(err[i]),
                'abs_error': float(abs(err[i])),
            })
    # radial point top-k; flatten and map row/radial index.
    for name, elec_key in [('theta_a', 'theta_a'), ('theta_c', 'theta_c')]:
        true2 = np.asarray(vt[name])
        pred2 = np.asarray(vp[name])
        err2 = pred2 - true2
        flat = np.abs(err2).reshape(-1)
        kk = min(k, flat.size)
        if kk <= 0:
            continue
        idx = np.argpartition(flat, -kk)[-kk:]
        idx = idx[np.argsort(-flat[idx])]
        nr = true2.shape[1]
        for rank, flat_i in enumerate(idx, 1):
            ti = int(flat_i // nr)
            ri = int(flat_i % nr)
            rows.append({
                'profile_id': profile_id,
                'variable': elec_key,
                'rank_within_variable': rank,
                'time_index': ti,
                'radial_index': ri,
                't_global_s': float(t[ti]) if ti < len(t) else None,
                'cycle_id': int(cycle[ti]) if ti < len(cycle) else None,
                'true': float(true2[ti, ri]),
                'pred': float(pred2[ti, ri]),
                'error': float(err2[ti, ri]),
                'abs_error': float(abs(err2[ti, ri])),
            })
    rows.sort(key=lambda r: float(r.get('abs_error', 0.0)), reverse=True)
    return rows[: max(1, k)]


def audit_prediction_file(pred_npz: Path, softlabel_dir: Path, cfg: Mapping[str, Any], filename: str = 'solution_softlabels.npz') -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with np.load(pred_npz, allow_pickle=True) as z:
        Y_true = np.asarray(z['y_true'], dtype=np.float64)
        Y_pred = np.asarray(z['y_pred'], dtype=np.float64)
        target_names = [str(x) for x in np.asarray(z['target_names']).reshape(-1)]
        t = np.asarray(z['t_global_s'], dtype=np.float64).reshape(-1)
        profile_id = _scalar_string(z['profile_id'], pred_npz.stem)
    slices = target_slices_from_names(target_names)
    eps = float(cfg.get('theta_outside_eps', 1e-5))
    metrics = computed_metrics(Y_true, Y_pred, slices, eps=eps)
    src = load_source_profile(softlabel_dir, profile_id, filename=filename)
    I, cycle, step_code = _extract_current_and_cycle(src, Y_true.shape[0])
    trans = transition_mask_from_current(I, cfg)
    rest = rest_mask_from_current(I, cfg)
    active = ~rest
    for mask_name, mask in [('transition', trans), ('non_transition', ~trans), ('rest', rest), ('active', active)]:
        if mask.size == Y_true.shape[0] and int(mask.sum()) >= 3:
            mm = computed_metrics(Y_true[mask], Y_pred[mask], slices, eps=eps)
            for k, v in mm.items():
                if k.endswith('_count') or k.endswith('_mae') or k.endswith('_rmse') or k.endswith('_nrmse_range') or k.endswith('_max_abs') or k.endswith('_corr') or k.endswith('_r2') or k in {'pred_theta_outside_fraction', 'pred_theta_boundary_hit_fraction'}:
                    metrics[f'{mask_name}_{k}'] = v
            metrics[f'{mask_name}_point_count'] = int(mask.sum())
        else:
            metrics[f'{mask_name}_point_count'] = int(mask.sum()) if mask.size else 0
    metrics.update({
        'profile_id': profile_id,
        'prediction_npz': str(pred_npz),
        'n_time': int(Y_true.shape[0]),
        'cycle_min': int(np.min(cycle[cycle >= 0])) if np.any(cycle >= 0) else None,
        'cycle_max': int(np.max(cycle[cycle >= 0])) if np.any(cycle >= 0) else None,
        'transition_point_fraction': float(np.mean(trans)) if trans.size else 0.0,
        'rest_point_fraction': float(np.mean(rest)) if rest.size else 0.0,
    })
    top = topk_errors(Y_true, Y_pred, slices, t, cycle, profile_id, k=int(cfg.get('topk_errors', 50)))
    # cycle-level compact metrics for easier diagnosis.
    cycle_rows: List[Dict[str, Any]] = []
    if np.any(cycle >= 0):
        for c in sorted(np.unique(cycle[cycle >= 0]).tolist()):
            m = cycle == c
            if int(m.sum()) < 3:
                continue
            cm = computed_metrics(Y_true[m], Y_pred[m], slices, eps=eps)
            cycle_rows.append({
                'profile_id': profile_id,
                'cycle_id': int(c),
                'point_count': int(m.sum()),
                'phis_c_mae': cm.get('phis_c_mae'),
                'phie_mae': cm.get('phie_mae'),
                'theta_a_mae': cm.get('theta_a_mae'),
                'theta_c_mae': cm.get('theta_c_mae'),
                'grad_a_surface_center_mae': cm.get('grad_a_surface_center_mae'),
                'grad_c_surface_center_mae': cm.get('grad_c_surface_center_mae'),
                'min_selected_corr': cm.get('min_selected_corr'),
                'pred_theta_outside_fraction': cm.get('pred_theta_outside_fraction'),
            })
    return metrics, top, cycle_rows


def aggregate_rows(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {'profile_count': 0}
    out: Dict[str, Any] = {'profile_count': len(rows)}
    keys = sorted({k for r in rows for k in r.keys() if isinstance(r.get(k), (int, float, np.integer, np.floating))})
    for k in keys:
        vals = np.asarray([float(r[k]) for r in rows if r.get(k) is not None and np.isfinite(float(r[k]))], dtype=float)
        if vals.size:
            out[f'{k}_mean'] = float(np.mean(vals))
            out[f'{k}_max'] = float(np.max(vals))
            out[f'{k}_min'] = float(np.min(vals))
    return out


def precision_status(rows: List[Mapping[str, Any]], summary_metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    ppt = cfg.get('per_profile_thresholds', {}) if isinstance(cfg.get('per_profile_thresholds', {}), Mapping) else {}
    tt = cfg.get('transition_thresholds', {}) if isinstance(cfg.get('transition_thresholds', {}), Mapping) else {}
    mt = cfg.get('max_abs_review_thresholds', {}) if isinstance(cfg.get('max_abs_review_thresholds', {}), Mapping) else {}
    checks: List[Dict[str, Any]] = []
    def add(name: str, value: Any, op: str, thr: Any, severity: str = 'fail') -> None:
        try:
            v = float(value)
            t = float(thr)
            ok = np.isfinite(v) and ((v <= t) if op == '<=' else (v >= t))
        except Exception:
            v = None
            t = float(thr) if thr is not None else None
            ok = False
        checks.append({'name': name, 'value': v, 'op': op, 'threshold': t, 'status': 'PASS' if ok else 'FAIL', 'severity': severity})
    if rows:
        add('worst per-profile phis_c_mae', max(float(r.get('phis_c_mae', float('nan'))) for r in rows), '<=', ppt.get('phis_c_mae_max_v', 0.012))
        add('worst per-profile phie_mae', max(float(r.get('phie_mae', float('nan'))) for r in rows), '<=', ppt.get('phie_mae_max_v', 0.015))
        add('worst per-profile theta_a_mae', max(float(r.get('theta_a_mae', float('nan'))) for r in rows), '<=', ppt.get('theta_a_mae_max', 0.009))
        add('worst per-profile theta_c_mae', max(float(r.get('theta_c_mae', float('nan'))) for r in rows), '<=', ppt.get('theta_c_mae_max', 0.008))
        add('worst per-profile grad_a_mae', max(float(r.get('grad_a_surface_center_mae', float('nan'))) for r in rows), '<=', ppt.get('grad_a_mae_max', 0.012))
        add('worst per-profile grad_c_mae', max(float(r.get('grad_c_surface_center_mae', float('nan'))) for r in rows), '<=', ppt.get('grad_c_mae_max', 0.010))
        add('min per-profile selected corr', min(float(r.get('min_selected_corr', float('nan'))) for r in rows), '>=', ppt.get('min_corr', 0.992))
        add('worst per-profile theta outside fraction', max(float(r.get('pred_theta_outside_fraction', float('nan'))) for r in rows), '<=', ppt.get('theta_outside_fraction_max', 0.006))
        # Transition hard/review checks. Transition can legitimately be harder, so use fail thresholds only if enough points were present.
        trans_rows = [r for r in rows if int(r.get('transition_point_count', 0) or 0) >= 3]
        if trans_rows:
            add('worst transition phis_c_mae', max(float(r.get('transition_phis_c_mae', float('nan'))) for r in trans_rows), '<=', tt.get('phis_c_transition_mae_max_v', 0.040), severity='review')
            add('worst transition theta_a_mae', max(float(r.get('transition_theta_a_mae', float('nan'))) for r in trans_rows), '<=', tt.get('theta_transition_mae_max', 0.035), severity='review')
            add('worst transition theta_c_mae', max(float(r.get('transition_theta_c_mae', float('nan'))) for r in trans_rows), '<=', tt.get('theta_transition_mae_max', 0.035), severity='review')
        add('worst phis_c max abs review', max(float(r.get('phis_c_max_abs', float('nan'))) for r in rows), '<=', mt.get('phis_c_max_abs_review_v', 0.250), severity='review')
        add('worst phie max abs review', max(float(r.get('phie_max_abs', float('nan'))) for r in rows), '<=', mt.get('phie_max_abs_review_v', 0.300), severity='review')
        add('worst theta_a max abs review', max(float(r.get('theta_a_max_abs', float('nan'))) for r in rows), '<=', mt.get('theta_a_max_abs_review', 0.120), severity='review')
        add('worst theta_c max abs review', max(float(r.get('theta_c_max_abs', float('nan'))) for r in rows), '<=', mt.get('theta_c_max_abs_review', 0.120), severity='review')
    hard_fails = [c for c in checks if c['status'] != 'PASS' and c.get('severity') == 'fail']
    review_fails = [c for c in checks if c['status'] != 'PASS' and c.get('severity') == 'review']
    status = 'PASS' if not hard_fails and not review_fails else ('REVIEW' if not hard_fails else 'FAIL')
    return {'overall_status': status, 'hard_fail_count': len(hard_fails), 'review_fail_count': len(review_fails), 'checks': checks}
