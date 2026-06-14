from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

FEATURE_NAMES = [
    't_norm', 't_norm2', 'sin_t', 'cos_t',
    'I_norm', 'absI_norm', 'dI_norm', 'q_norm',
    'voltage_exp_norm_local', 'dV_norm',
    'is_charge', 'is_rest', 'is_discharge'
]
METRICS = [
    'phis_c', 'phie', 'theta_a', 'theta_c',
    'theta_a_mean', 'theta_c_mean',
    'grad_a_surface_center', 'grad_c_surface_center'
]
PROTOCOL_MAP = {
    'Batch-1': '2C',
    'Batch-2': '3C',
    'Batch-3': 'R2.5',
    'Batch-4': 'R3',
    'Batch-5': 'random_walk',
    'Batch-6': 'GEO',
}
TRAIN6 = {
    'profiles/Batch-1_battery-3',
    'profiles/Batch-2_battery-8',
    'profiles/Batch-3_battery-6',
    'profiles/Batch-4_battery-7',
    'profiles/Batch-5_battery-7',
    'profiles/Batch-6_battery-3',
}


def jdump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ObsPhysicsMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 5, output_dim: int = 6):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(1, int(num_layers))):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def transform_outputs(raw: torch.Tensor, grad_clip: float) -> Dict[str, torch.Tensor]:
    return {
        'theta_a_mean': torch.sigmoid(raw[:, 0]),
        'theta_c_mean': torch.sigmoid(raw[:, 1]),
        'grad_a': grad_clip * torch.tanh(raw[:, 2]),
        'grad_c': grad_clip * torch.tanh(raw[:, 3]),
        'phie_norm': raw[:, 4],
        'phis_c_norm': raw[:, 5],
    }


def apply_p5g_gap_refinement(y: Dict[str, torch.Tensor], xb_std: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # This reproduces the P5G eval-side correction. It is intentionally applied only for P5G.
    p = cfg.get('p5g_outlier_protocol_balanced_gauge', {}) if isinstance(cfg, dict) else {}
    if not bool(p.get('enable_gap_refinement', True)):
        return y
    v_z = xb_std[:, 8]
    q_z = xb_std[:, 7]
    abs_i = torch.clamp(torch.abs(xb_std[:, 5]), 0.0, 6.0)
    d_i = torch.clamp(torch.abs(xb_std[:, 6]), 0.0, 6.0)
    stress = torch.clamp(0.55 * torch.tanh(abs_i) + 0.45 * torch.tanh(3.0 * d_i), 0.0, 1.0)
    base = float(p.get('gap_base', -0.02))
    v_amp = float(p.get('gap_voltage_amplitude', 0.30))
    q_amp = float(p.get('gap_q_amplitude', 0.10))
    slope = float(p.get('gap_voltage_slope', 0.85))
    stress_reduction = float(p.get('gap_stress_reduction', 0.08))
    target_gap = base + v_amp * torch.tanh(slope * v_z) + q_amp * torch.tanh(0.90 * q_z) - stress_reduction * stress
    target_gap = torch.clamp(target_gap, float(p.get('gap_min', -0.42)), float(p.get('gap_max', 0.42)))
    gap = y['theta_a_mean'] - y['theta_c_mean']
    gain = float(p.get('gap_projection_gain', 0.58))
    delta = 0.5 * gain * (gap - target_gap)
    yy = dict(y)
    yy['theta_a_mean'] = torch.clamp(y['theta_a_mean'] - delta, float(p.get('theta_floor', 0.02)), float(p.get('theta_ceiling', 0.98)))
    yy['theta_c_mean'] = torch.clamp(y['theta_c_mean'] + delta, float(p.get('theta_floor', 0.02)), float(p.get('theta_ceiling', 0.98)))
    return yy


@dataclass
class Accum:
    n: int = 0
    sum_abs: float = 0.0
    sum_sq: float = 0.0
    sum_err: float = 0.0
    max_abs: float = 0.0
    sum_t: float = 0.0
    sum_p: float = 0.0
    sum_t2: float = 0.0
    sum_p2: float = 0.0
    sum_tp: float = 0.0

    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        t = np.asarray(true, dtype=np.float64).reshape(-1)
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        mask = np.isfinite(t) & np.isfinite(p)
        if not np.any(mask):
            return
        t = t[mask]
        p = p[mask]
        e = p - t
        ae = np.abs(e)
        self.n += int(t.size)
        self.sum_abs += float(np.sum(ae))
        self.sum_sq += float(np.sum(e * e))
        self.sum_err += float(np.sum(e))
        self.max_abs = max(self.max_abs, float(np.max(ae)))
        self.sum_t += float(np.sum(t))
        self.sum_p += float(np.sum(p))
        self.sum_t2 += float(np.sum(t * t))
        self.sum_p2 += float(np.sum(p * p))
        self.sum_tp += float(np.sum(t * p))

    def merge(self, other: 'Accum') -> None:
        self.n += other.n
        self.sum_abs += other.sum_abs
        self.sum_sq += other.sum_sq
        self.sum_err += other.sum_err
        self.max_abs = max(self.max_abs, other.max_abs)
        self.sum_t += other.sum_t
        self.sum_p += other.sum_p
        self.sum_t2 += other.sum_t2
        self.sum_p2 += other.sum_p2
        self.sum_tp += other.sum_tp

    def row(self, prefix: str) -> Dict[str, Any]:
        if self.n <= 0:
            return {
                f'{prefix}_count': 0,
                f'{prefix}_mae': float('nan'),
                f'{prefix}_rmse': float('nan'),
                f'{prefix}_bias': float('nan'),
                f'{prefix}_max_abs': float('nan'),
                f'{prefix}_corr': float('nan'),
                f'{prefix}_r2': float('nan'),
            }
        n = float(self.n)
        cov = self.sum_tp - self.sum_t * self.sum_p / n
        vt = self.sum_t2 - self.sum_t * self.sum_t / n
        vp = self.sum_p2 - self.sum_p * self.sum_p / n
        corr = cov / math.sqrt(vt * vp) if vt > 1e-20 and vp > 1e-20 else float('nan')
        r2 = 1.0 - self.sum_sq / vt if vt > 1e-20 else float('nan')
        return {
            f'{prefix}_count': int(self.n),
            f'{prefix}_mae': self.sum_abs / n,
            f'{prefix}_rmse': math.sqrt(self.sum_sq / n),
            f'{prefix}_bias': self.sum_err / n,
            f'{prefix}_max_abs': self.max_abs,
            f'{prefix}_corr': corr,
            f'{prefix}_r2': r2,
            f'{prefix}_sse': self.sum_sq,
            f'{prefix}_sst': vt,
        }


def ensure_group(container: Dict[str, Dict[str, Accum]], group: str) -> Dict[str, Accum]:
    if group not in container:
        container[group] = {m: Accum() for m in METRICS}
    return container[group]


@dataclass
class ModelBundle:
    name: str
    model_dir: Path
    ckpt_path: Path
    model: nn.Module
    x_mean: np.ndarray
    x_std: np.ndarray
    grad_clip: float
    cfg: Dict[str, Any]
    apply_gap_refinement: bool = False
    global_acc: Dict[str, Accum] = field(default_factory=lambda: {m: Accum() for m in METRICS})
    split_acc: Dict[str, Dict[str, Accum]] = field(default_factory=dict)
    batch_acc: Dict[str, Dict[str, Accum]] = field(default_factory=dict)
    protocol_acc: Dict[str, Dict[str, Accum]] = field(default_factory=dict)
    profile_rows: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, str]] = field(default_factory=list)


def checkpoint_path(model_dir: Path) -> Path:
    a = model_dir / 'model' / 'best_with_state.pt'
    b = model_dir / 'best_with_state.pt'
    if a.exists():
        return a
    if b.exists():
        return b
    raise FileNotFoundError(f'No best_with_state.pt under {model_dir}')


def load_model_bundle(name: str, model_dir: Path, device: torch.device) -> ModelBundle:
    ckpt = checkpoint_path(model_dir)
    obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    model_cfg = obj.get('model_config', {}) if isinstance(obj, dict) else {}
    hidden_dim = int(model_cfg.get('hidden_dim', 256))
    num_layers = int(model_cfg.get('num_layers', 5))
    model = ObsPhysicsMLP(input_dim=len(FEATURE_NAMES), hidden_dim=hidden_dim, num_layers=num_layers, output_dim=6)
    model.load_state_dict(obj['state'])
    model.to(device).eval()
    x_mean = np.asarray(obj['x_mean'], dtype=np.float32)
    x_std = np.asarray(obj['x_std'], dtype=np.float32)
    cfg = obj.get('config', {}) if isinstance(obj, dict) else {}
    grad_clip = float(cfg.get('model', {}).get('gradient_clip', 0.25)) if isinstance(cfg, dict) else 0.25
    apply_gap = name.upper() == 'P5G' or ('p5g_outlier_protocol_balanced_gauge' in cfg if isinstance(cfg, dict) else False)
    return ModelBundle(
        name=name.upper(), model_dir=model_dir, ckpt_path=ckpt, model=model,
        x_mean=x_mean, x_std=x_std, grad_clip=grad_clip, cfg=cfg,
        apply_gap_refinement=apply_gap,
    )


def safe_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '__', text.strip())[:180]


def pick_member(members: set[str], keys: List[str], required: bool = True) -> Optional[str]:
    for k in keys:
        if k in members:
            return k
    if required:
        raise KeyError(f'Missing any of {keys}')
    return None


def extract_member(npz_path: Path, key: str, out_dir: Path) -> Path:
    out = out_dir / (key + '.npy')
    if out.exists() and out.stat().st_size > 0:
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.npy.tmp')
    with zipfile.ZipFile(npz_path, 'r') as zf:
        member = key + '.npy'
        if member not in zf.namelist():
            raise KeyError(f'{npz_path}: missing {member}')
        with zf.open(member, 'r') as src, tmp.open('wb') as dst:
            while True:
                block = src.read(16 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
    tmp.replace(out)
    return out


def load_profile_arrays(npz_path: Path, cache_dir: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(npz_path, 'r') as zf:
        members = {Path(n).stem for n in zf.namelist() if n.endswith('.npy')}
    mapping = {
        't': pick_member(members, ['t_global_s', 'time_s', 't_s', 'time', 't']),
        'I': pick_member(members, ['I_profile', 'current_A', 'I_A', 'current', 'I']),
        'V': pick_member(members, ['voltage_exp', 'voltage_V', 'V_exp', 'V']),
        'theta_a': pick_member(members, ['theta_a', 'theta_n', 'theta_negative']),
        'theta_c': pick_member(members, ['theta_c', 'theta_p', 'theta_positive']),
        'phie': pick_member(members, ['phie', 'phi_e', 'phi_e_eff']),
        'phis_c': pick_member(members, ['phis_c_soft', 'phis_c', 'voltage_soft', 'V_soft', 'V_pred']),
    }
    arrs: Dict[str, Any] = {'_keys': mapping}
    for alias, key in mapping.items():
        p = extract_member(npz_path, str(key), cache_dir)
        arrs[alias] = np.load(p, mmap_mode='r')
    return arrs


def build_q_norm(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[~np.isfinite(dt)] = 0.0
    if dt.size > 10:
        cap = float(np.nanpercentile(dt, 99.9)) * 10.0
        if np.isfinite(cap) and cap > 0:
            dt = np.clip(dt, 0.0, cap)
    else:
        dt = np.clip(dt, 0.0, np.inf)
    q = np.cumsum(I.astype(np.float32) * dt) / 3600.0
    scale = float(np.nanmax(np.abs(q))) if q.size else 1.0
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (q / scale).astype(np.float32)


def feature_chunk(t: np.ndarray, I: np.ndarray, V: np.ndarray, start: int, stop: int, stats: Dict[str, float], qn: np.ndarray) -> np.ndarray:
    idx = slice(start, stop)
    tn = ((t[idx].astype(np.float32) - float(t[0])) / stats['t_span']).astype(np.float32)
    In = (I[idx].astype(np.float32) / stats['I_scale']).astype(np.float32)
    dI = np.diff(In, prepend=In[0]).astype(np.float32)
    vn = ((V[idx].astype(np.float32) - stats['v_mean']) / stats['v_std']).astype(np.float32)
    dV = np.diff(vn, prepend=vn[0]).astype(np.float32)
    eps = max(1e-9, 0.001 * float(stats['I_abs_max'] + 1e-12))
    charge = (I[idx] > eps).astype(np.float32)
    discharge = (I[idx] < -eps).astype(np.float32)
    rest = (np.abs(I[idx]) <= eps).astype(np.float32)
    return np.stack([
        tn, tn ** 2,
        np.sin(2 * np.pi * tn).astype(np.float32),
        np.cos(2 * np.pi * tn).astype(np.float32),
        In, np.abs(In).astype(np.float32), dI,
        qn[idx].astype(np.float32), vn, dV,
        charge, rest, discharge,
    ], axis=1).astype(np.float32)


def orient_time_radial(a: Any, n: int, start: int, stop: int, name: str) -> np.ndarray:
    shape = getattr(a, 'shape', None)
    if shape is None:
        x = np.asarray(a, dtype=np.float32)
        if x.ndim == 1:
            return x[start:stop, None]
        return x[start:stop] if x.shape[0] == n else x[:, start:stop].T
    if len(shape) == 1:
        return np.asarray(a[start:stop], dtype=np.float32).reshape(-1, 1)
    if len(shape) != 2:
        raise ValueError(f'{name}: expected 1D/2D, got shape={shape}')
    if shape[0] == n:
        return np.asarray(a[start:stop], dtype=np.float32)
    if shape[1] == n:
        return np.asarray(a[:, start:stop], dtype=np.float32).T
    raise ValueError(f'{name}: cannot orient shape={shape}, n={n}')


def infer_bundle(bundle: ModelBundle, X: np.ndarray, stats: Dict[str, float], nr_a: int, nr_c: int, device: torch.device, batch_size: int) -> Dict[str, np.ndarray]:
    radial_a = np.linspace(-0.5, 0.5, nr_a, dtype=np.float32)
    radial_c = np.linspace(-0.5, 0.5, nr_c, dtype=np.float32)
    outs = {k: [] for k in METRICS}
    Xs = ((X - bundle.x_mean) / bundle.x_std).astype(np.float32)
    with torch.no_grad():
        for i in range(0, Xs.shape[0], batch_size):
            xb = torch.from_numpy(Xs[i:i+batch_size]).to(device)
            raw = bundle.model(xb)
            y = transform_outputs(raw, bundle.grad_clip)
            if bundle.apply_gap_refinement:
                y = apply_p5g_gap_refinement(y, xb, bundle.cfg)
            ta_m = y['theta_a_mean'].cpu().numpy().astype(np.float32)
            tc_m = y['theta_c_mean'].cpu().numpy().astype(np.float32)
            ga = y['grad_a'].cpu().numpy().astype(np.float32)
            gc = y['grad_c'].cpu().numpy().astype(np.float32)
            # Correct gauge convention: phis_c is voltage-like; phie is effective ionic potential around zero gauge.
            phie = (y['phie_norm'].cpu().numpy().astype(np.float32) * stats['v_std'])
            phis = (y['phis_c_norm'].cpu().numpy().astype(np.float32) * stats['v_std'] + stats['v_mean'])
            theta_a = np.clip(ta_m[:, None] + ga[:, None] * radial_a[None, :], 0.0, 1.0).astype(np.float32)
            theta_c = np.clip(tc_m[:, None] + gc[:, None] * radial_c[None, :], 0.0, 1.0).astype(np.float32)
            outs['theta_a'].append(theta_a)
            outs['theta_c'].append(theta_c)
            outs['theta_a_mean'].append(ta_m)
            outs['theta_c_mean'].append(tc_m)
            outs['grad_a_surface_center'].append((theta_a[:, -1] - theta_a[:, 0]).astype(np.float32))
            outs['grad_c_surface_center'].append((theta_c[:, -1] - theta_c[:, 0]).astype(np.float32))
            outs['phie'].append(phie)
            outs['phis_c'].append(phis)
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def update_model_accs(bundle: ModelBundle, meta: Dict[str, str], true: Dict[str, np.ndarray], pred: Dict[str, np.ndarray], profile_accs: Dict[str, Accum]) -> None:
    split = meta['split']
    batch = meta['batch']
    protocol = meta['protocol']
    split_acc = ensure_group(bundle.split_acc, split)
    batch_acc = ensure_group(bundle.batch_acc, batch)
    protocol_acc = ensure_group(bundle.protocol_acc, protocol)
    for m in METRICS:
        profile_accs[m].update(true[m], pred[m])
        bundle.global_acc[m].update(true[m], pred[m])
        split_acc[m].update(true[m], pred[m])
        batch_acc[m].update(true[m], pred[m])
        protocol_acc[m].update(true[m], pred[m])


def row_from_accs(accs: Dict[str, Accum], prefix_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(prefix_extra or {})
    for m in METRICS:
        row.update(accs[m].row(m))
    return row


def default_model_dirs(cache_root: Path) -> Dict[str, Path]:
    return {
        'P5B': cache_root / 'xjtu_d16_p5b_train6_eval49_observation_physics_FAST' / 'model_train6_observation_physics',
        'P5D': cache_root / 'xjtu_d16_p5d_train6_eval49_delta_gauge_FAST' / 'model_train6_delta_gauge_observation_physics',
        'P5E': cache_root / 'xjtu_d16_p5e_train6_eval49_cathode_gauge_FAST' / 'model_train6_cathode_gauge_observation_physics',
        'P5F': cache_root / 'xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST' / 'model_train6_balanced_gauge_observation_physics',
        'P5G': cache_root / 'xjtu_d16_p5g_train6_eval49_outlier_protocol_balanced_gauge_FAST' / 'model_train6_outlier_protocol_balanced_gauge_observation_physics',
    }


def default_manifest(cache_root: Path) -> Path:
    candidates = [
        cache_root / 'xjtu_d16_p5g_train6_eval49_outlier_protocol_balanced_gauge_FAST' / 'D16_P5G_TRAIN6_EVAL49_MANIFEST.csv',
        cache_root / 'xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST' / 'D16_P5F_TRAIN6_EVAL49_MANIFEST.csv',
        cache_root / 'xjtu_d16_p5e_train6_eval49_cathode_gauge_FAST' / 'D16_P5E_TRAIN6_EVAL49_MANIFEST.csv',
        cache_root / 'xjtu_d16_p5d_train6_eval49_delta_gauge_FAST' / 'D16_P5D_TRAIN6_EVAL49_MANIFEST.csv',
        cache_root / 'xjtu_d16_p5b_train6_eval49_observation_physics_FAST' / 'D16_P5B_TRAIN6_EVAL49_MANIFEST.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError('No P5B/P5D/P5E/P5F/P5G manifest found under cache root')


def parse_model_arg(values: List[str], cache_root: Path) -> Dict[str, Path]:
    defaults = default_model_dirs(cache_root)
    if not values:
        return {k: v for k, v in defaults.items() if v.exists()}
    selected: Dict[str, Path] = {}
    for item in values:
        for piece in item.split(','):
            piece = piece.strip()
            if not piece:
                continue
            if '=' in piece:
                name, path = piece.split('=', 1)
                selected[name.strip().upper()] = Path(path.strip().strip('"'))
            else:
                name = piece.upper()
                if name not in defaults:
                    raise KeyError(f'Unknown model short name: {piece}')
                selected[name] = defaults[name]
    return selected


def fmt(x: Any, nd: int = 6) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return 'nan'
        if abs(v) >= 1000 or (abs(v) < 1e-4 and v != 0):
            return f'{v:.3e}'
        return f'{v:.{nd}f}'
    except Exception:
        return str(x)


def metric_line(row: Dict[str, Any], m: str) -> str:
    return f"MAE={fmt(row.get(m+'_mae'))}, RMSE={fmt(row.get(m+'_rmse'))}, Bias={fmt(row.get(m+'_bias'))}, R2={fmt(row.get(m+'_r2'))}, Corr={fmt(row.get(m+'_corr', row.get(m+'_corr_mean', float('nan'))))}"


def summarize_group(accs: Dict[str, Accum], group: str) -> Dict[str, Any]:
    return row_from_accs(accs, {'group': group, 'profile_count': None})


def make_report(bundles: List[ModelBundle], meta_info: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append('# D16-P5H Exact-R2 Audit Report')
    lines.append('')
    lines.append('This is a no-training audit. It recomputes exact R² from streaming full-profile soft-label arrays and model predictions.')
    lines.append('')
    lines.append('## Run metadata')
    for k, v in meta_info.items():
        lines.append(f'- {k}: `{v}`')
    lines.append('')
    lines.append('## Model ranking summary, eval49 split')
    lines.append('')
    lines.append('| model | profiles | phis_c_mae | phis_c_r2 | phie_mae | phie_r2 | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | theta_a_bias | theta_c_bias |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    eval_rows: List[Tuple[str, Dict[str, Any]]] = []
    for b in bundles:
        if 'eval' in b.split_acc:
            row = row_from_accs(b.split_acc['eval'], {'group': 'eval'})
            eval_rows.append((b.name, row))
    def score_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, float]:
        r = item[1]
        return (float(r.get('theta_a_mean_mae', float('inf'))) + float(r.get('theta_c_mean_mae', float('inf'))), float(r.get('phis_c_mae', float('inf'))))
    for name, row in sorted(eval_rows, key=score_key):
        lines.append(
            f"| {name} | {int(row.get('theta_a_mean_count', 0))} | {fmt(row.get('phis_c_mae'))} | {fmt(row.get('phis_c_r2'))} | "
            f"{fmt(row.get('phie_mae'))} | {fmt(row.get('phie_r2'))} | {fmt(row.get('theta_a_mean_mae'))} | {fmt(row.get('theta_a_mean_r2'))} | "
            f"{fmt(row.get('theta_c_mean_mae'))} | {fmt(row.get('theta_c_mean_r2'))} | {fmt(row.get('theta_a_mean_bias'))} | {fmt(row.get('theta_c_mean_bias'))} |"
        )
    lines.append('')
    lines.append('## Promotion gate interpretation')
    lines.append('')
    lines.append('- Operational PASS requires 55 evaluated profiles and zero model/profile failures.')
    lines.append('- Four-state high-precision candidate gate used here: eval49 `theta_a_mean_mae < 0.15`, `theta_c_mean_mae < 0.15`, `theta_a_mean_r2 > 0.85`, `theta_c_mean_r2 > 0.85`, `phis_c_r2 > 0.99`.')
    lines.append('- If theta R² is negative, absolute gauge is worse than predicting the eval-set mean, even when correlation looks high.')
    lines.append('')
    lines.append('## Per-model full details')
    for b in bundles:
        lines.append('')
        lines.append(f'### {b.name}')
        lines.append('')
        lines.append(f'- model_dir: `{b.model_dir}`')
        lines.append(f'- checkpoint: `{b.ckpt_path}`')
        lines.append(f'- apply_gap_refinement: `{b.apply_gap_refinement}`')
        if b.failures:
            lines.append(f'- failures: `{len(b.failures)}`')
            for fail in b.failures[:5]:
                lines.append(f"  - {fail}")
        else:
            lines.append('- failures: `0`')
        lines.append('')
        global_row = row_from_accs(b.global_acc, {'group': 'ALL'})
        lines.append('**ALL55 exact metrics**')
        for m in ['phis_c', 'phie', 'theta_a_mean', 'theta_c_mean', 'theta_a', 'theta_c', 'grad_a_surface_center', 'grad_c_surface_center']:
            lines.append(f'- {m}: {metric_line(global_row, m)}')
        if b.split_acc:
            lines.append('')
            lines.append('**Split exact metrics**')
            for split in sorted(b.split_acc):
                r = row_from_accs(b.split_acc[split], {'group': split})
                lines.append(f'- {split}: phis_c({metric_line(r, "phis_c")}); theta_a_mean({metric_line(r, "theta_a_mean")}); theta_c_mean({metric_line(r, "theta_c_mean")})')
        lines.append('')
        lines.append('**Worst 10 eval profiles by theta mean MAE sum**')
        eval_profiles = [r for r in b.profile_rows if r.get('split') == 'eval']
        eval_profiles.sort(key=lambda r: float(r.get('theta_a_mean_mae', 0)) + float(r.get('theta_c_mean_mae', 0)), reverse=True)
        lines.append('| rank | profile_id | batch | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | phis_c_mae | phis_c_r2 |')
        lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|')
        for i, r in enumerate(eval_profiles[:10], 1):
            lines.append(f"| {i} | {r.get('profile_id')} | {r.get('batch')} | {fmt(r.get('theta_a_mean_mae'))} | {fmt(r.get('theta_a_mean_r2'))} | {fmt(r.get('theta_c_mean_mae'))} | {fmt(r.get('theta_c_mean_r2'))} | {fmt(r.get('phis_c_mae'))} | {fmt(r.get('phis_c_r2'))} |")
        if b.batch_acc:
            lines.append('')
            lines.append('**Batch summary**')
            lines.append('| batch | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | phis_c_mae | phis_c_r2 |')
            lines.append('|---|---:|---:|---:|---:|---:|---:|')
            rows = [(k, row_from_accs(v, {'group': k})) for k, v in b.batch_acc.items()]
            rows.sort(key=lambda kv: float(kv[1].get('theta_a_mean_mae', 0)) + float(kv[1].get('theta_c_mean_mae', 0)), reverse=True)
            for k, r in rows:
                lines.append(f"| {k} | {fmt(r.get('theta_a_mean_mae'))} | {fmt(r.get('theta_a_mean_r2'))} | {fmt(r.get('theta_c_mean_mae'))} | {fmt(r.get('theta_c_mean_r2'))} | {fmt(r.get('phis_c_mae'))} | {fmt(r.get('phis_c_r2'))} |")
    lines.append('')
    lines.append('## Machine-readable compact JSON')
    compact = {
        'models': {
            b.name: {
                'all': row_from_accs(b.global_acc, {'group': 'ALL'}),
                'eval': row_from_accs(b.split_acc.get('eval', {m: Accum() for m in METRICS}), {'group': 'eval'}),
                'train': row_from_accs(b.split_acc.get('train', {m: Accum() for m in METRICS}), {'group': 'train'}),
                'failures': b.failures,
            } for b in bundles
        }
    }
    lines.append('```json')
    lines.append(json.dumps(compact, indent=2, ensure_ascii=False))
    lines.append('```')
    lines.append('')
    return '\n'.join(lines)


def normalize_manifest_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        rr = dict(r)
        profile_id = rr.get('profile_id') or rr.get('cell_uid') or rr.get('profile') or ''
        if not profile_id:
            # infer from path parent if possible
            p = Path(rr.get('softlabel_npz', ''))
            if p.parent.name:
                profile_id = 'profiles/' + p.parent.name
        if not profile_id.startswith('profiles/'):
            profile_id = 'profiles/' + profile_id
        rr['profile_id'] = profile_id
        if not rr.get('split'):
            rr['split'] = 'train' if profile_id in TRAIN6 else 'eval'
        if not rr.get('batch'):
            m = re.search(r'(Batch-\d+)', profile_id)
            rr['batch'] = m.group(1) if m else ''
        if not rr.get('battery'):
            m = re.search(r'(battery-\d+)', profile_id)
            rr['battery'] = m.group(1) if m else ''
        rr['protocol'] = rr.get('protocol') or PROTOCOL_MAP.get(rr.get('batch', ''), rr.get('batch', 'unknown'))
        if not rr.get('softlabel_npz'):
            raise KeyError(f'Manifest row missing softlabel_npz: {rr}')
        out.append(rr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5H exact-R2 audit across P5B/P5D/P5E/P5F/P5G. No training. One report file output.')
    ap.add_argument('--cache-root', default=r'E:\XJTU battery dataset\_gv1_cache')
    ap.add_argument('--manifest', default='')
    ap.add_argument('--model', action='append', default=[], help='Model selector: P5F or P5F=path. Can be repeated or comma-separated.')
    ap.add_argument('--output-file', default='')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=65536)
    ap.add_argument('--chunk-size', type=int, default=200000)
    ap.add_argument('--limit-profiles', type=int, default=0)
    ap.add_argument('--keep-cache', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    cache_root = Path(args.cache_root)
    manifest_path = Path(args.manifest) if args.manifest else default_manifest(cache_root)
    output_file = Path(args.output_file) if args.output_file else cache_root / 'xjtu_d16_p5h_exact_r2_audit' / 'D16_P5H_EXACT_R2_AUDIT_REPORT.md'
    work_cache = output_file.parent / 'p5h_profile_mmap_cache'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = normalize_manifest_rows(read_csv(manifest_path))
    if args.limit_profiles and args.limit_profiles > 0:
        rows = rows[:int(args.limit_profiles)]

    model_dirs = parse_model_arg(args.model, cache_root)
    if not model_dirs:
        raise FileNotFoundError('No model directories found. Provide --model P5F=...')
    device = torch.device(args.device if args.device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(f'[P5H] manifest={manifest_path} rows={len(rows)}', flush=True)
    print(f'[P5H] output_file={output_file}', flush=True)
    print(f'[P5H] device={device} models={list(model_dirs)}', flush=True)

    bundles: List[ModelBundle] = []
    for name, mdir in model_dirs.items():
        try:
            b = load_model_bundle(name, mdir, device)
            bundles.append(b)
            print(f'[P5H] loaded {b.name}: {b.ckpt_path}', flush=True)
        except Exception as exc:
            print(f'[P5H] SKIP {name}: {exc!r}', flush=True)

    if not bundles:
        raise RuntimeError('No loadable models')

    for pi, row in enumerate(rows, 1):
        profile_id = row['profile_id']
        npz_path = Path(row['softlabel_npz'])
        profile_cache = work_cache / safe_name(profile_id)
        print(f'[P5H] profile {pi}/{len(rows)} {profile_id}', flush=True)
        try:
            arr = load_profile_arrays(npz_path, profile_cache)
            t = np.asarray(arr['t'], dtype=np.float32).reshape(-1)
            I = np.asarray(arr['I'], dtype=np.float32).reshape(-1)
            V = np.asarray(arr['V'], dtype=np.float32).reshape(-1)
            n = len(t)
            stats = {
                't_span': float(t[-1] - t[0]) if n > 1 else 1.0,
                'I_scale': float(np.nanpercentile(np.abs(I), 99.5)) if n else 1.0,
                'I_abs_max': float(np.nanmax(np.abs(I))) if n else 0.0,
                'v_mean': float(np.nanmean(V)) if n else 0.0,
                'v_std': float(np.nanstd(V)) if n else 1.0,
            }
            if not np.isfinite(stats['t_span']) or stats['t_span'] <= 0:
                stats['t_span'] = 1.0
            if not np.isfinite(stats['I_scale']) or stats['I_scale'] < 1e-12:
                stats['I_scale'] = 1.0
            if not np.isfinite(stats['v_std']) or stats['v_std'] < 1e-8:
                stats['v_std'] = 1.0
            qn = build_q_norm(t, I)
            ta_shape = arr['theta_a'].shape
            tc_shape = arr['theta_c'].shape
            nr_a = int(ta_shape[1] if len(ta_shape) == 2 and ta_shape[0] == n else ta_shape[0] if len(ta_shape) == 2 else 1)
            nr_c = int(tc_shape[1] if len(tc_shape) == 2 and tc_shape[0] == n else tc_shape[0] if len(tc_shape) == 2 else 1)
            profile_accs = {b.name: {m: Accum() for m in METRICS} for b in bundles}
            for s in range(0, n, int(args.chunk_size)):
                e = min(n, s + int(args.chunk_size))
                X = feature_chunk(t, I, V, s, e, stats, qn)
                true_ta = orient_time_radial(arr['theta_a'], n, s, e, 'theta_a')
                true_tc = orient_time_radial(arr['theta_c'], n, s, e, 'theta_c')
                true = {
                    'phis_c': np.asarray(arr['phis_c'][s:e], dtype=np.float32).reshape(-1),
                    'phie': np.asarray(arr['phie'][s:e], dtype=np.float32).reshape(-1),
                    'theta_a': true_ta,
                    'theta_c': true_tc,
                    'theta_a_mean': np.mean(true_ta, axis=1).astype(np.float32),
                    'theta_c_mean': np.mean(true_tc, axis=1).astype(np.float32),
                    'grad_a_surface_center': (true_ta[:, -1] - true_ta[:, 0]).astype(np.float32),
                    'grad_c_surface_center': (true_tc[:, -1] - true_tc[:, 0]).astype(np.float32),
                }
                for b in bundles:
                    try:
                        pred = infer_bundle(b, X, stats, nr_a, nr_c, device, int(args.batch_size))
                        update_model_accs(b, row, true, pred, profile_accs[b.name])
                    except Exception as exc:
                        b.failures.append({'profile_id': profile_id, 'chunk': f'{s}:{e}', 'error': repr(exc)})
                        print(f'[P5H] FAIL model={b.name} profile={profile_id} chunk={s}:{e}: {exc!r}', flush=True)
                if s == 0 or e == n:
                    print(f'[P5H] {profile_id} chunk {s}:{e}/{n}', flush=True)
            for b in bundles:
                prow = row_from_accs(profile_accs[b.name], {
                    'model': b.name,
                    'profile_id': profile_id,
                    'batch': row.get('batch', ''),
                    'battery': row.get('battery', ''),
                    'split': row.get('split', ''),
                    'protocol': row.get('protocol', ''),
                    'n_time': n,
                })
                b.profile_rows.append(prow)
        except Exception as exc:
            for b in bundles:
                b.failures.append({'profile_id': profile_id, 'softlabel_npz': str(npz_path), 'error': repr(exc)})
            print(f'[P5H] FAIL profile load/eval {profile_id}: {exc!r}', flush=True)
        finally:
            if not args.keep_cache:
                shutil.rmtree(profile_cache, ignore_errors=True)

    # Write single Markdown report.
    meta = {
        'manifest': str(manifest_path),
        'output_file': str(output_file),
        'profile_count_requested': len(rows),
        'models': ','.join([b.name for b in bundles]),
        'device': str(device),
        'batch_size': args.batch_size,
        'chunk_size': args.chunk_size,
        'elapsed_seconds': f'{time.time() - t0:.1f}',
    }
    report = make_report(bundles, meta)
    output_file.write_text(report, encoding='utf-8')
    print(f'[P5H] wrote one report file: {output_file}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
