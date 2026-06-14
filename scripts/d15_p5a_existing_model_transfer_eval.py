from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def discover_npz(root: Path, filename: str) -> List[Path]:
    return sorted(root.rglob(filename), key=lambda p: str(p).lower()) if root.exists() else []


def canonical_cell_id(name: str) -> str:
    s = str(name).replace('\\', '/').split('/')[-1]
    m = re.search(r'Batch-([1-6]).*battery-(\d+)', s)
    if m:
        return f'Batch-{m.group(1)}_battery-{int(m.group(2))}'
    m = re.search(r'^\d+_battery-(\d+)_2C_battery-\d+$', s)
    if m:
        return f'Batch-1_battery-{int(m.group(1))}'
    m = re.search(r'^\d+_battery-(\d+)_R2\.5_battery-\d+$', s)
    if m:
        return f'Batch-3_battery-{int(m.group(1))}'
    m = re.search(r'^\d+_battery-(\d+)_R3_battery-\d+$', s)
    if m:
        return f'Batch-4_battery-{int(m.group(1))}'
    raise ValueError(f'Cannot canonicalize {name!r}')


def batch_of(cell_id: str) -> str:
    m = re.search(r'Batch-(\d+)', str(cell_id))
    return f'Batch-{m.group(1)}' if m else 'unknown'


def model_file(model_dir: Path) -> Optional[Path]:
    for c in [model_dir / 'model' / 'best_with_state.pt', model_dir / 'best_with_state.pt']:
        if c.exists():
            return c
    return None


def first_key(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def _as_float_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        raise TypeError(f'{name} is not numeric')
    return arr.astype(np.float32).reshape(-1)


def _orient_time_radial(x: Any, n_time: int, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        raise TypeError(f'{name} is not numeric')
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        if arr.size != n_time:
            raise ValueError(f'{name}: length {arr.size} != n_time {n_time}')
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f'{name}: expected 2D time/radial, got {arr.shape}')
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return arr.T
    raise ValueError(f'{name}: cannot orient shape {arr.shape} with n_time={n_time}')


def _get_time(d: Mapping[str, Any]) -> np.ndarray:
    k = first_key(d, ['t_global_s', 'time_s', 't_s', 't', 'time'])
    if not k:
        raise KeyError('Missing time key')
    return _as_float_1d(d[k], k)


def _get_current(d: Mapping[str, Any], n: int) -> np.ndarray:
    k = first_key(d, ['I_profile', 'current_A', 'I_A', 'current', 'I'])
    if not k:
        return np.zeros(n, dtype=np.float32)
    arr = _as_float_1d(d[k], k)
    if arr.size != n:
        raise ValueError(f'{k} length {arr.size} != n_time={n}')
    return arr


def _get_optional_1d(d: Mapping[str, Any], keys: Sequence[str], n: int, fill: float) -> np.ndarray:
    k = first_key(d, keys)
    if not k:
        return np.full(n, fill, dtype=np.float32)
    try:
        arr = _as_float_1d(d[k], k)
        if arr.size == n:
            return arr
    except Exception:
        pass
    return np.full(n, fill, dtype=np.float32)


def _get_theta(d: Mapping[str, Any], electrode: str, n: int) -> np.ndarray:
    if electrode == 'a':
        tkeys = ['theta_a', 'theta_n', 'theta_negative']
        cskeys = ['cs_a', 'cs_n', 'cs_negative']
        cmax_keys = ['csmax_a', 'csmax_n']
    else:
        tkeys = ['theta_c', 'theta_p', 'theta_positive']
        cskeys = ['cs_c', 'cs_p', 'cs_positive']
        cmax_keys = ['csmax_c', 'csmax_p']
    k = first_key(d, tkeys)
    if k:
        return _orient_time_radial(d[k], n, k)
    k = first_key(d, cskeys)
    if not k:
        raise KeyError(f'Missing theta/cs for electrode {electrode}')
    cs = _orient_time_radial(d[k], n, k)
    cmax = None
    for ck in cmax_keys:
        if ck in d:
            try:
                cmax = float(np.asarray(d[ck]).reshape(-1)[0])
                break
            except Exception:
                pass
    if cmax is None or not np.isfinite(cmax) or cmax <= 0:
        raise KeyError(f'Missing cmax to convert {k} for electrode {electrode}')
    return (cs / cmax).astype(np.float32)


def _get_target_scalar(d: Mapping[str, Any], n: int, keys: Sequence[str], name: str) -> np.ndarray:
    k = first_key(d, keys)
    if not k:
        raise KeyError(f'Missing target scalar {name}')
    arr = _as_float_1d(d[k], k)
    if arr.size != n:
        raise ValueError(f'{k} length {arr.size} != n_time={n}')
    return arr


def _step_features(I: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    eps = max(1e-9, 0.001 * float(np.nanmax(np.abs(I)) + 1e-12))
    charge = (I > eps).astype(np.float32)
    discharge = (I < -eps).astype(np.float32)
    rest = (np.abs(I) <= eps).astype(np.float32)
    return np.stack([charge, rest, discharge], axis=1), ['is_charge', 'is_rest', 'is_discharge']


def _cumtrapz_charge(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    q = np.cumsum(I.astype(np.float32) * dt) / 3600.0
    scale = float(np.nanmax(np.abs(q)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(q, dtype=np.float32)
    return (q / scale).astype(np.float32)


def build_features_sampled(d: Mapping[str, Any], idx: np.ndarray, profile_onehot_index: int, onehot_count: int, include_onehot: bool) -> Tuple[np.ndarray, List[str]]:
    t_full = _get_time(d)
    n = int(t_full.size)
    I_full = _get_current(d, n)
    voltage_full = _get_optional_1d(d, ['voltage_exp', 'voltage_V', 'V_exp', 'V'], n, fill=0.0)
    temp_full = _get_optional_1d(d, ['temperature_C', 'temperature_K', 'temp_C', 'T_C', 'T'], n, fill=25.0)
    t = t_full[idx]
    I = I_full[idx]
    voltage = voltage_full[idx]
    temp = temp_full[idx]
    span = float(t_full[-1] - t_full[0]) if n > 1 else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    tn = ((t - t_full[0]) / span).astype(np.float32)
    I_scale = float(np.nanpercentile(np.abs(I_full), 99.5))
    if not np.isfinite(I_scale) or I_scale <= 1e-12:
        I_scale = 1.0
    In = I / I_scale
    # dI is computed on full array then sampled to mimic previous code's transition feature.
    dI_full = np.diff(I_full, prepend=I_full[0]) / I_scale
    dI = dI_full[idx]
    qn_full = _cumtrapz_charge(t_full, I_full)
    qn = qn_full[idx]
    vmean = float(np.nanmean(voltage_full)) if np.isfinite(np.nanmean(voltage_full)) else 0.0
    vstd = float(np.nanstd(voltage_full))
    if not np.isfinite(vstd) or vstd <= 1e-9:
        vstd = 1.0
    vn = (voltage - vmean) / vstd
    tmean = float(np.nanmean(temp_full)) if np.isfinite(np.nanmean(temp_full)) else 25.0
    tstd = float(np.nanstd(temp_full))
    if not np.isfinite(tstd) or tstd <= 1e-9:
        tstd = 1.0
    Tn = (temp - tmean) / tstd
    base = [tn, tn ** 2, np.sin(2 * np.pi * tn).astype(np.float32), np.cos(2 * np.pi * tn).astype(np.float32), In.astype(np.float32), np.abs(In).astype(np.float32), dI.astype(np.float32), qn.astype(np.float32), vn.astype(np.float32), Tn.astype(np.float32)]
    names = ['t_norm', 't_norm2', 'sin_t', 'cos_t', 'I_norm', 'absI_norm', 'dI_norm', 'q_norm', 'voltage_exp_norm_local', 'temperature_norm_local']
    X = np.stack(base, axis=1)
    step, step_names = _step_features(I)
    X = np.concatenate([X, step], axis=1)
    names.extend(step_names)
    if include_onehot:
        oh = np.zeros((len(idx), onehot_count), dtype=np.float32)
        if 0 <= profile_onehot_index < onehot_count:
            oh[:, profile_onehot_index] = 1.0
        X = np.concatenate([X, oh], axis=1)
        names.extend([f'profile_onehot_{i:02d}' for i in range(onehot_count)])
    return X.astype(np.float32), names


def build_targets_sampled(d: Mapping[str, Any], idx: np.ndarray) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    t = _get_time(d)
    n = int(t.size)
    th_a = _get_theta(d, 'a', n)[idx]
    th_c = _get_theta(d, 'c', n)[idx]
    phie = _get_target_scalar(d, n, ['phie', 'phi_e', 'phi_e_eff'], 'phie')[idx].reshape(-1, 1)
    phis = _get_target_scalar(d, n, ['phis_c_soft', 'phis_c', 'voltage_soft', 'V_soft', 'V_pred'], 'phis_c')[idx].reshape(-1, 1)
    nra = th_a.shape[1]
    nrc = th_c.shape[1]
    slices = {'theta_a': (0, nra), 'theta_c': (nra, nra+nrc), 'phie': (nra+nrc, nra+nrc+1), 'phis_c': (nra+nrc+1, nra+nrc+2)}
    Y = np.concatenate([th_a, th_c, phie, phis], axis=1).astype(np.float32)
    return Y, slices


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
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
        return {f'{prefix}_count': 0, f'{prefix}_mae': float('nan'), f'{prefix}_rmse': float('nan'), f'{prefix}_max_abs': float('nan'), f'{prefix}_bias': float('nan'), f'{prefix}_corr': float('nan')}
    e = p[m] - t[m]
    return {f'{prefix}_count': int(m.sum()), f'{prefix}_mae': float(np.mean(np.abs(e))), f'{prefix}_rmse': float(np.sqrt(np.mean(e**2))), f'{prefix}_max_abs': float(np.max(np.abs(e))), f'{prefix}_bias': float(np.mean(e)), f'{prefix}_corr': safe_corr(t[m], p[m])}


def volume_weights(nr: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, nr + 1)
    w = edges[1:] ** 3 - edges[:-1] ** 3
    return w / w.sum()


def unpack_targets(Y: np.ndarray, slices: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    return {name: Y[:, s:e] for name, (s, e) in slices.items()}


def project_theta(Y: np.ndarray, slices: Dict[str, Tuple[int, int]], theta_min: float, theta_max: float) -> np.ndarray:
    out = np.array(Y, copy=True)
    for key in ['theta_a', 'theta_c']:
        s, e = slices[key]
        out[:, s:e] = np.clip(out[:, s:e], theta_min, theta_max)
    return out


def compute_rg_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, slices: Dict[str, Tuple[int, int]], eps: float = 1e-5) -> Dict[str, Any]:
    t = unpack_targets(Y_true, slices)
    p = unpack_targets(Y_pred, slices)
    out: Dict[str, Any] = {}
    th_a_t = t['theta_a']; th_a_p = p['theta_a']
    th_c_t = t['theta_c']; th_c_p = p['theta_c']
    phie_t = t['phie'].reshape(-1); phie_p = p['phie'].reshape(-1)
    phis_t = t['phis_c'].reshape(-1); phis_p = p['phis_c'].reshape(-1)
    out.update(basic_metrics(phis_t, phis_p, 'phis_c'))
    out.update(basic_metrics(phie_t, phie_p, 'phie'))
    out.update(basic_metrics(th_a_t, th_a_p, 'theta_a'))
    out.update(basic_metrics(th_c_t, th_c_p, 'theta_c'))
    wa = volume_weights(th_a_t.shape[1]); wc = volume_weights(th_c_t.shape[1])
    mean_a_t = np.sum(th_a_t * wa[None, :], axis=1); mean_a_p = np.sum(th_a_p * wa[None, :], axis=1)
    mean_c_t = np.sum(th_c_t * wc[None, :], axis=1); mean_c_p = np.sum(th_c_p * wc[None, :], axis=1)
    out.update(basic_metrics(mean_a_t, mean_a_p, 'theta_a_mean'))
    out.update(basic_metrics(mean_c_t, mean_c_p, 'theta_c_mean'))
    grad_a_t = th_a_t[:, -1] - th_a_t[:, 0]; grad_a_p = th_a_p[:, -1] - th_a_p[:, 0]
    grad_c_t = th_c_t[:, -1] - th_c_t[:, 0]; grad_c_p = th_c_p[:, -1] - th_c_p[:, 0]
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
    corr_keys = ['phis_c_corr','phie_corr','theta_a_corr','theta_c_corr','theta_a_mean_corr','theta_c_mean_corr','grad_a_surface_center_corr','grad_c_surface_center_corr']
    corr_vals = [float(out.get(k, float('nan'))) for k in corr_keys]
    finite = [v for v in corr_vals if np.isfinite(v)]
    out['min_selected_corr'] = float(min(finite)) if finite else float('nan')
    return out


class ResidualMLPBlockTorch:
    pass


def build_torch_model(input_dim: int, output_dim: int, cfg: Dict[str, Any]):
    import torch
    from torch import nn
    def activation(name: str):
        n = str(name).lower()
        if n in {'silu', 'swish'}:
            return nn.SiLU()
        if n == 'gelu':
            return nn.GELU()
        if n == 'relu':
            return nn.ReLU()
        if n == 'tanh':
            return nn.Tanh()
        raise ValueError(f'Unsupported activation: {name}')
    class ResidualMLPBlock(nn.Module):
        def __init__(self, dim: int, activation_name: str, dropout: float):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, dim), activation(activation_name), nn.Dropout(dropout) if dropout > 0 else nn.Identity(), nn.Linear(dim, dim))
            self.out_act = activation(activation_name)
        def forward(self, x):
            return self.out_act(x + self.net(x))
    class ClosedSetRGMLP(nn.Module):
        def __init__(self):
            super().__init__()
            hidden_dim = int(cfg.get('hidden_dim', 256))
            num_hidden_layers = int(cfg.get('num_hidden_layers', 4))
            act = str(cfg.get('activation', 'silu'))
            dropout = float(cfg.get('dropout', 0.0))
            residual_blocks = bool(cfg.get('residual_blocks', True))
            layers = [nn.Linear(input_dim, hidden_dim), activation(act)]
            if residual_blocks:
                for _ in range(max(1, num_hidden_layers)):
                    layers.append(ResidualMLPBlock(hidden_dim, act, dropout))
            else:
                for _ in range(max(1, num_hidden_layers)):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), activation(act)])
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)
    return ClosedSetRGMLP()


def predict_numpy(model, X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, device, batch_size: int) -> np.ndarray:
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = ((X[i:i+batch_size] - x_mean[None, :]) / x_std[None, :]).astype(np.float32)
            yp = model(torch.from_numpy(xb).to(device)).detach().cpu().numpy()
            outs.append((yp * y_std[None, :] + y_mean[None, :]).astype(np.float32))
    return np.concatenate(outs, axis=0)


def device_from_name(name: str):
    import torch
    if name == 'auto' or not name:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def determine_onehot(state: Dict[str, Any]) -> Tuple[bool, int]:
    input_dim = int(state['input_dim'])
    include = bool(state.get('include_profile_onehot', True))
    if not include:
        return False, 0
    # D15-P1 feature schema uses 10 base + 3 step = 13 non-onehot features.
    count = input_dim - 13
    if count < 0:
        raise ValueError(f'Cannot infer onehot_count from input_dim={input_dim}')
    return True, count


def resolve_profile_index(cell_id: str, train_profile_ids: List[str], onehot_count: int, policy: str) -> Tuple[int, bool, str]:
    train_canon = []
    for pid in train_profile_ids:
        try:
            train_canon.append(canonical_cell_id(pid))
        except Exception:
            train_canon.append(str(pid))
    if cell_id in train_canon:
        idx = train_canon.index(cell_id)
        return (idx if idx < onehot_count else -1), True, 'exact_seen'
    if policy == 'same_batch_first':
        b = batch_of(cell_id)
        for i, cid in enumerate(train_canon):
            if batch_of(cid) == b and i < onehot_count:
                return i, False, f'same_batch_fallback:{cid}'
    return -1, False, 'unseen_zero_onehot'


def quality_status(metrics: Dict[str, Any], thresholds: Dict[str, Any], prefix: str) -> Tuple[str, List[Dict[str, Any]]]:
    checks = []
    def chk(metric_key: str, op: str, threshold_key: str, label: str):
        thr = thresholds.get(threshold_key, None)
        if thr is None:
            return
        val = metrics.get(metric_key, float('nan'))
        try:
            fv = float(val)
        except Exception:
            fv = float('nan')
        ok = np.isfinite(fv) and ((fv <= float(thr)) if op == '<=' else (fv >= float(thr)))
        checks.append({'name': label, 'metric': metric_key, 'value': fv if np.isfinite(fv) else None, 'op': op, 'threshold': float(thr), 'status': 'PASS' if ok else 'REVIEW'})
    chk(f'{prefix}_phis_c_mae', '<=', f'{prefix}_phis_c_mae_max_v', 'phis_c MAE')
    chk(f'{prefix}_phie_mae', '<=', f'{prefix}_phie_mae_max', 'phie MAE')
    chk(f'{prefix}_theta_a_mae', '<=', f'{prefix}_theta_a_mae_max', 'theta_a MAE')
    chk(f'{prefix}_theta_c_mae', '<=', f'{prefix}_theta_c_mae_max', 'theta_c MAE')
    chk(f'{prefix}_grad_a_surface_center_mae', '<=', f'{prefix}_grad_a_mae_max', 'grad_a MAE')
    chk(f'{prefix}_grad_c_surface_center_mae', '<=', f'{prefix}_grad_c_mae_max', 'grad_c MAE')
    chk(f'{prefix}_pred_theta_outside_fraction', '<=', f'{prefix}_pred_theta_outside_fraction_max', 'theta outside')
    chk(f'{prefix}_min_selected_corr', '>=', f'{prefix}_min_selected_corr_min', 'min corr')
    return ('PASS' if all(c['status'] == 'PASS' for c in checks) else 'REVIEW'), checks


def parse_args():
    p = argparse.ArgumentParser(description='D15-P5A evaluate existing D15 models on ALL55 P2Dlite-RG soft labels.')
    p.add_argument('--config', default='configs/d15_p5a_all55_existing_model_transfer_config.json')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--eval-stride', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--onehot-unseen-policy', choices=['zero', 'same_batch_first'], default=None)
    p.add_argument('--allow-overwrite', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    out_dir = Path(args.out_dir or cfg['output_dir'])
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {out_dir}. Use --allow-overwrite for deliberate rerun.')
    out_dir.mkdir(parents=True, exist_ok=True)
    soft_root = Path(cfg['all55_softlabel_dir'])
    filename = cfg.get('filename', 'solution_softlabels.npz')
    files = discover_npz(soft_root, filename)
    if not files:
        raise FileNotFoundError(f'No {filename} found under {soft_root}')
    eval_stride = int(args.eval_stride or cfg.get('eval_stride', 64))
    batch_size = int(args.batch_size or cfg.get('batch_size', 262144))
    policy = args.onehot_unseen_policy or cfg.get('onehot_unseen_policy', 'zero')
    proj_cfg = cfg.get('projection', {})
    theta_min = float(proj_cfg.get('theta_min', 0.0001))
    theta_max = float(proj_cfg.get('theta_max', 0.9999))
    device = device_from_name(args.device or cfg.get('device', 'auto'))
    import torch
    print(f'[D15-P5A eval] soft_root={soft_root}; profiles={len(files)}; stride={eval_stride}; device={device}; policy={policy}', flush=True)
    by_profile_rows: List[Dict[str, Any]] = []
    by_batch_rows: List[Dict[str, Any]] = []
    by_seen_rows: List[Dict[str, Any]] = []
    by_model_rows: List[Dict[str, Any]] = []
    scorecard_models: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for model_cfg in cfg.get('models', []):
        if not model_cfg.get('enabled', True):
            continue
        model_id = model_cfg.get('model_id', Path(model_cfg['model_dir']).name)
        md = Path(model_cfg['model_dir'])
        mf = model_file(md)
        if mf is None:
            errors.append({'model_id': model_id, 'error': 'missing_model_file', 'model_dir': str(md)})
            scorecard_models.append({'model_id': model_id, 'evaluation_status': 'MISSING', 'transfer_quality_status': 'MISSING'})
            print(f'[D15-P5A eval] SKIP missing model: {model_id} {md}', flush=True)
            continue
        t0 = time.time()
        ck = torch.load(mf, map_location=device, weights_only=False)
        state = ck['state']
        model = build_torch_model(int(state['input_dim']), int(state['output_dim']), state['model_config']).to(device)
        model.load_state_dict(ck['model_state_dict'])
        train_profile_ids = [str(x) for x in state.get('profile_ids', [])]
        include_oh, onehot_count = determine_onehot(state)
        x_mean = np.asarray(state['x_mean'], dtype=np.float32)
        x_std = np.asarray(state['x_std'], dtype=np.float32)
        y_mean = np.asarray(state['y_mean'], dtype=np.float32)
        y_std = np.asarray(state['y_std'], dtype=np.float32)
        slices = {k: tuple(v) for k, v in state['target_slices'].items()}
        group_true_raw: Dict[str, List[np.ndarray]] = {'global': []}
        group_pred_raw: Dict[str, List[np.ndarray]] = {'global': []}
        group_pred_proj: Dict[str, List[np.ndarray]] = {'global': []}
        # For simplicity and reliable corr, aggregate strided arrays by group; stride keeps memory bounded.
        for n_i, npz_path in enumerate(files, start=1):
            cell_id = canonical_cell_id(npz_path.parent.name)
            batch = batch_of(cell_id)
            profile_idx, seen, map_policy = resolve_profile_index(cell_id, train_profile_ids, onehot_count, policy)
            with np.load(npz_path, allow_pickle=True) as z:
                d = {k: z[k] for k in z.files}
            n_time = int(_get_time(d).size)
            idx = np.arange(0, n_time, max(1, eval_stride), dtype=np.int64)
            X, feature_names = build_features_sampled(d, idx, profile_idx, onehot_count, include_oh)
            if X.shape[1] != int(state['input_dim']):
                raise ValueError(f'{model_id}/{cell_id}: feature_dim {X.shape[1]} != model input_dim {state["input_dim"]}. include_oh={include_oh}, onehot_count={onehot_count}')
            Y_true, _slices = build_targets_sampled(d, idx)
            Y_pred = predict_numpy(model, X, x_mean, x_std, y_mean, y_std, device, batch_size=batch_size)
            Y_proj = project_theta(Y_pred, slices, theta_min, theta_max)
            raw_m = compute_rg_metrics(Y_true, Y_pred, slices)
            proj_m = compute_rg_metrics(Y_true, Y_proj, slices)
            row: Dict[str, Any] = {
                'model_id': model_id,
                'cell_id': cell_id,
                'batch': batch,
                'profile_folder': npz_path.parent.name,
                'npz_path': str(npz_path),
                'n_time': n_time,
                'n_eval': int(len(idx)),
                'eval_stride': eval_stride,
                'seen_in_model': bool(seen),
                'onehot_policy': map_policy,
            }
            for k, v in raw_m.items():
                row[f'raw_{k}'] = v
            for k, v in proj_m.items():
                row[f'projected_{k}'] = v
            by_profile_rows.append(row)
            for group in ['global', f'batch::{batch}', f'seen::{"seen" if seen else "unseen"}']:
                group_true_raw.setdefault(group, []).append(Y_true)
                group_pred_raw.setdefault(group, []).append(Y_pred)
                group_pred_proj.setdefault(group, []).append(Y_proj)
            print(f'[D15-P5A eval] {model_id} {n_i:02d}/{len(files)} {cell_id} seen={seen} raw_phis_mae={raw_m["phis_c_mae"]:.5g} proj_theta_out={proj_m["pred_theta_outside_fraction"]:.3g}', flush=True)
            # Explicitly release large arrays.
            del d, X, Y_true, Y_pred, Y_proj
        # Aggregate metrics by group.
        group_rows = []
        for group, true_parts in group_true_raw.items():
            YT = np.concatenate(true_parts, axis=0)
            YR = np.concatenate(group_pred_raw[group], axis=0)
            YP = np.concatenate(group_pred_proj[group], axis=0)
            raw_g = compute_rg_metrics(YT, YR, slices)
            proj_g = compute_rg_metrics(YT, YP, slices)
            row = {'model_id': model_id, 'group': group, 'n_eval': int(YT.shape[0])}
            for k, v in raw_g.items():
                row[f'raw_{k}'] = v
            for k, v in proj_g.items():
                row[f'projected_{k}'] = v
            group_rows.append(row)
            if group == 'global':
                by_model_rows.append(row)
            elif group.startswith('batch::'):
                rr = dict(row); rr['batch'] = group.split('::',1)[1]; by_batch_rows.append(rr)
            elif group.startswith('seen::'):
                rr = dict(row); rr['seen_group'] = group.split('::',1)[1]; by_seen_rows.append(rr)
        global_row = [r for r in group_rows if r['group'] == 'global'][0]
        status, checks = quality_status(global_row, cfg.get('transfer_quality_thresholds', {}), 'projected')
        elapsed = time.time() - t0
        scorecard_models.append({
            'model_id': model_id,
            'model_file': str(mf),
            'description': model_cfg.get('description', ''),
            'evaluation_status': 'PASS',
            'transfer_quality_status': status,
            'elapsed_s': round(elapsed, 3),
            'eval_stride': eval_stride,
            'onehot_unseen_policy': policy,
            'onehot_count': onehot_count,
            'include_onehot': include_oh,
            'trained_profile_count': len(train_profile_ids),
            'quality_checks': checks,
            'global_projected_metrics': {k: v for k, v in global_row.items() if k.startswith('projected_')},
            'global_raw_metrics': {k: v for k, v in global_row.items() if k.startswith('raw_')},
        })
    write_csv(by_profile_rows, out_dir / 'D15_P5A_METRICS_BY_MODEL_PROFILE.csv')
    write_csv(by_batch_rows, out_dir / 'D15_P5A_METRICS_BY_MODEL_BATCH.csv')
    write_csv(by_seen_rows, out_dir / 'D15_P5A_METRICS_BY_MODEL_SEEN_UNSEEN.csv')
    write_csv(by_model_rows, out_dir / 'D15_P5A_METRICS_BY_MODEL_GLOBAL.csv')
    write_json(scorecard_models, out_dir / 'D15_P5A_MODEL_SCORECARDS.json')
    write_json(errors, out_dir / 'D15_P5A_EVAL_ERRORS.json')
    completed = [m for m in scorecard_models if m.get('evaluation_status') == 'PASS']
    transfer_pass = [m for m in completed if m.get('transfer_quality_status') == 'PASS']
    recommendation = 'train_all55_unified_or_batch_protocol_aware_model' if not transfer_pass else 'inspect_transfer_pass_model_and_consider_finetune_before_new_training'
    summary = {
        'stage': 'D15-P5A ALL55 existing-model transfer evaluation',
        'all55_softlabel_dir': str(soft_root),
        'out_dir': str(out_dir),
        'model_count_configured': len([m for m in cfg.get('models', []) if m.get('enabled', True)]),
        'model_count_evaluated': len(completed),
        'model_count_missing_or_error': len(errors) + len([m for m in scorecard_models if m.get('evaluation_status') not in {'PASS', 'MISSING'}]),
        'transfer_quality_pass_count': len(transfer_pass),
        'transfer_quality_review_count': len([m for m in completed if m.get('transfer_quality_status') != 'PASS']),
        'eval_stride': eval_stride,
        'batch_size': batch_size,
        'device': str(device),
        'onehot_unseen_policy': policy,
        'evaluation_status': 'PASS' if completed else 'FAIL',
        'final_status': 'PASS' if completed else 'FAIL',
        'recommendation': recommendation,
        'notes': cfg.get('notes', []),
        'models': scorecard_models,
    }
    write_json(summary, out_dir / 'D15_P5A_FINAL_SCORECARD.json')
    print('[D15-P5A eval] evaluated models:', len(completed))
    print('[D15-P5A eval] transfer_quality_pass_count:', len(transfer_pass))
    print('[D15-P5A eval] final_status:', summary['final_status'])
    print('[D15-P5A eval] recommendation:', recommendation)
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
