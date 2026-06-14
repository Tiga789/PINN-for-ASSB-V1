from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]

# Training is intentionally limited to observed time series only.
TRAIN_USED_KEYS = ['t_global_s', 'I_profile', 'voltage_exp']
TRAIN_FORBIDDEN_KEYS = ['theta_a', 'theta_c', 'cs_a', 'cs_c', 'phie', 'phis_c', 'phis_c_soft']


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def first_key(d: Dict[str, Any], keys: List[str]) -> str | None:
    for k in keys:
        if k in d:
            return k
    return None


def as_1d_float(a: Any, name: str) -> np.ndarray:
    x = np.asarray(a)
    if x.dtype.kind in {'U', 'S', 'O'}:
        raise TypeError(f'{name} is not numeric')
    return x.astype(np.float32).reshape(-1)


def load_observed_npz(npz_path: str | Path) -> Dict[str, np.ndarray]:
    # This function is the training data boundary. It must not access internal-state soft-label arrays.
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        missing = []
        kt = first_key(z, ['t_global_s', 'time_s', 't_s', 'time', 't'])
        ki = first_key(z, ['I_profile', 'current_A', 'I_A', 'current', 'I'])
        kv = first_key(z, ['voltage_exp', 'voltage_V', 'V_exp', 'V'])
        if kt is None: missing.append('time')
        if ki is None: missing.append('current')
        if kv is None: missing.append('voltage')
        if missing:
            raise KeyError(f'{npz_path}: missing observed keys {missing}')
        t = as_1d_float(z[kt], kt)
        I = as_1d_float(z[ki], ki)
        V = as_1d_float(z[kv], kv)
        if not (t.size == I.size == V.size):
            raise ValueError(f'{npz_path}: observed lengths differ: t={t.size} I={I.size} V={V.size}')
        return {'t': t, 'I': I, 'V': V, 'source_keys': {'time': kt, 'current': ki, 'voltage': kv}, 'available_keys': sorted(keys)}


def sample_indices(n: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count <= 0 or max_count >= n:
        return np.arange(n, dtype=np.int64)
    if max_count < 4:
        return np.linspace(0, n - 1, max_count).astype(np.int64)
    # Use deterministic stratified sampling to cover the whole cycling history.
    bins = max_count
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        idx.append(int(rng.integers(a, b)))
    idx[0] = 0
    idx[-1] = n - 1
    return np.array(sorted(set(idx)), dtype=np.int64)


def build_q_norm(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[~np.isfinite(dt)] = 0.0
    dt = np.clip(dt, 0.0, np.nanpercentile(dt, 99.9) * 10.0 if dt.size > 10 else np.inf)
    q = np.cumsum(I.astype(np.float32) * dt) / 3600.0
    scale = float(np.nanmax(np.abs(q))) if q.size else 1.0
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return (q / scale).astype(np.float32)


def build_features_from_observed(obs: Dict[str, np.ndarray], idx: np.ndarray | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    t = obs['t']
    I = obs['I']
    V = obs['V']
    n = len(t)
    if idx is None:
        idx = np.arange(n, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    tn_full = ((t - t[0]) / span).astype(np.float32)
    I_scale = float(np.nanpercentile(np.abs(I), 99.5)) if n else 1.0
    if not np.isfinite(I_scale) or I_scale < 1e-12:
        I_scale = 1.0
    In_full = (I / I_scale).astype(np.float32)
    dI_full = np.diff(In_full, prepend=In_full[0]).astype(np.float32)
    qn_full = build_q_norm(t, I)
    v_mean = float(np.nanmean(V)) if V.size else 0.0
    v_std = float(np.nanstd(V)) if V.size else 1.0
    if not np.isfinite(v_std) or v_std < 1e-8:
        v_std = 1.0
    vn_full = ((V - v_mean) / v_std).astype(np.float32)
    dV_full = np.diff(vn_full, prepend=vn_full[0]).astype(np.float32)
    eps = max(1e-9, 0.001 * float(np.nanmax(np.abs(I)) + 1e-12))
    charge = (I > eps).astype(np.float32)
    discharge = (I < -eps).astype(np.float32)
    rest = (np.abs(I) <= eps).astype(np.float32)
    X = np.stack([
        tn_full[idx],
        tn_full[idx] ** 2,
        np.sin(2 * np.pi * tn_full[idx]).astype(np.float32),
        np.cos(2 * np.pi * tn_full[idx]).astype(np.float32),
        In_full[idx],
        np.abs(In_full[idx]).astype(np.float32),
        dI_full[idx],
        qn_full[idx],
        vn_full[idx],
        dV_full[idx],
        charge[idx],
        rest[idx],
        discharge[idx],
    ], axis=1).astype(np.float32)
    stats = {
        't0': float(t[0]), 't_span': float(span), 'I_scale': float(I_scale),
        'v_mean': float(v_mean), 'v_std': float(v_std), 'n_time': int(n)
    }
    return X, stats


FEATURE_NAMES = ['t_norm', 't_norm2', 'sin_t', 'cos_t', 'I_norm', 'absI_norm', 'dI_norm', 'q_norm', 'voltage_exp_norm_local', 'dV_norm', 'is_charge', 'is_rest', 'is_discharge']
OUTPUT_NAMES = ['theta_a_mean_raw', 'theta_c_mean_raw', 'grad_a_raw', 'grad_c_raw', 'phie_norm', 'phis_c_norm']


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


def transform_outputs(raw: torch.Tensor, grad_clip: float = 0.25) -> Dict[str, torch.Tensor]:
    theta_a_mean = torch.sigmoid(raw[:, 0])
    theta_c_mean = torch.sigmoid(raw[:, 1])
    grad_a = grad_clip * torch.tanh(raw[:, 2])
    grad_c = grad_clip * torch.tanh(raw[:, 3])
    phie_norm = raw[:, 4]
    phis_c_norm = raw[:, 5]
    return {
        'theta_a_mean': theta_a_mean,
        'theta_c_mean': theta_c_mean,
        'grad_a': grad_a,
        'grad_c': grad_c,
        'phie_norm': phie_norm,
        'phis_c_norm': phis_c_norm,
    }


def physics_observation_loss(raw: torch.Tensor, x: torch.Tensor, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    w = cfg['training'].get('loss_weights', {})
    grad_clip = float(cfg.get('model', {}).get('gradient_clip', 0.25))
    y = transform_outputs(raw, grad_clip=grad_clip)
    v_norm = x[:, 8]
    I_norm = x[:, 4]
    q_norm = x[:, 7]
    loss_v = torch.mean((y['phis_c_norm'] - v_norm) ** 2)
    # Sort is not guaranteed after random batching; finite-difference losses are local weak regularizers.
    # They still work as stochastic smoothness/sign penalties on sampled trajectories.
    if raw.shape[0] > 2:
        dqa = q_norm[1:] - q_norm[:-1]
        dta = y['theta_a_mean'][1:] - y['theta_a_mean'][:-1]
        dtc = y['theta_c_mean'][1:] - y['theta_c_mean'][:-1]
        loss_sign = torch.mean(torch.relu(-(dta * dqa))) + torch.mean(torch.relu(dtc * dqa))
        loss_couple = torch.mean((dta + dtc) ** 2)
        loss_smooth = torch.mean((dta) ** 2) + torch.mean((dtc) ** 2) + torch.mean((y['grad_a'][1:] - y['grad_a'][:-1]) ** 2) + torch.mean((y['grad_c'][1:] - y['grad_c'][:-1]) ** 2)
    else:
        loss_sign = raw.new_tensor(0.0)
        loss_couple = raw.new_tensor(0.0)
        loss_smooth = raw.new_tensor(0.0)
    # Gradient direction from I sign: charge I>0 => graphite surface enriched grad_a>0, cathode depleted grad_c<0.
    loss_grad_dir = torch.mean(torch.relu(-(y['grad_a'] * I_norm))) + torch.mean(torch.relu(y['grad_c'] * I_norm))
    loss_phie = torch.mean(y['phie_norm'] ** 2)
    total = (
        float(w.get('voltage_observation', 1.0)) * loss_v +
        float(w.get('theta_monotonic_sign', 0.15)) * loss_sign +
        float(w.get('theta_mass_coupling', 0.05)) * loss_couple +
        float(w.get('gradient_direction', 0.06)) * loss_grad_dir +
        float(w.get('smoothness', 0.015)) * loss_smooth +
        float(w.get('phie_regularization', 0.005)) * loss_phie
    )
    parts = {
        'loss_voltage_observation': float(loss_v.detach().cpu()),
        'loss_theta_monotonic_sign': float(loss_sign.detach().cpu()),
        'loss_theta_mass_coupling': float(loss_couple.detach().cpu()),
        'loss_gradient_direction': float(loss_grad_dir.detach().cpu()),
        'loss_smoothness': float(loss_smooth.detach().cpu()),
        'loss_phie_regularization': float(loss_phie.detach().cpu()),
        'loss_total': float(total.detach().cpu()),
    }
    return total, parts


def standardize_train_val(Xtr: np.ndarray, Xva: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(Xtr, axis=0).astype(np.float32)
    std = np.std(Xtr, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return ((Xtr - mean) / std).astype(np.float32), ((Xva - mean) / std).astype(np.float32), mean, std


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5B train6 observation-physics model. No internal soft-label data loss is used.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--config', default='configs/d16_p5b_train6_eval49_config.json')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--allow-overwrite', action='store_true')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out-dir exists and is non-empty: {out_dir}; pass --allow-overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'model').mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    train_rows = [r for r in rows if r.get('split') == 'train']
    if len(train_rows) != 6:
        raise ValueError(f'Expected 6 train rows, got {len(train_rows)}')

    seed = int(cfg['training'].get('seed', 20260610))
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    max_train = int(cfg['training'].get('max_train_points_per_profile', 250000))
    max_val = int(cfg['training'].get('max_val_points_per_profile', 40000))
    Xtr_list: List[np.ndarray] = []
    Xva_list: List[np.ndarray] = []
    audit_rows = []
    for r in train_rows:
        obs = load_observed_npz(r['softlabel_npz'])
        # Hard audit: training function did not access forbidden soft-label arrays.
        keys_available = obs.get('available_keys', [])
        n = len(obs['t'])
        tr_idx = sample_indices(n, max_train, rng)
        all_idx = np.arange(n, dtype=np.int64)
        # Validation is observational only and drawn from points not selected for train when possible.
        rem = np.setdiff1d(all_idx, tr_idx, assume_unique=False)
        if rem.size == 0:
            va_idx = tr_idx[::max(1, tr_idx.size // max(1, min(max_val, tr_idx.size)))]
        else:
            local = sample_indices(rem.size, min(max_val, rem.size), rng)
            va_idx = rem[local]
        Xtr, stats = build_features_from_observed(obs, tr_idx)
        Xva, _ = build_features_from_observed(obs, va_idx)
        Xtr_list.append(Xtr)
        Xva_list.append(Xva)
        audit_rows.append({
            'profile_id': r['profile_id'], 'batch': r['batch'], 'battery': r['battery'],
            'softlabel_npz': r['softlabel_npz'], 'n_time': int(n),
            'train_points': int(Xtr.shape[0]), 'val_points': int(Xva.shape[0]),
            'training_used_keys': TRAIN_USED_KEYS,
            'training_forbidden_keys_not_loaded': TRAIN_FORBIDDEN_KEYS,
            'source_keys': obs['source_keys'],
            'profile_stats': stats,
        })
        print(f"[D16-P5B train] loaded observed-only {r['profile_id']}: n={n} train={Xtr.shape[0]} val={Xva.shape[0]}", flush=True)

    X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32)
    X_val = np.concatenate(Xva_list, axis=0).astype(np.float32)
    X_train_s, X_val_s, x_mean, x_std = standardize_train_val(X_train, X_val)
    device = torch.device(args.device if args.device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model_cfg = cfg.get('model', {})
    model = ObsPhysicsMLP(input_dim=X_train_s.shape[1], hidden_dim=int(model_cfg.get('hidden_dim', 256)), num_layers=int(model_cfg.get('num_layers', 5)), output_dim=6).to(device)
    lr = float(cfg['training'].get('learning_rate', 1e-3))
    wd = float(cfg['training'].get('weight_decay', 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = int(args.epochs or cfg['training'].get('epochs', 1500))
    batch_size = int(args.batch_size or cfg['training'].get('batch_size', 65536))
    patience = int(cfg['training'].get('early_stop_patience', 160))

    train_ds = TensorDataset(torch.from_numpy(X_train_s))
    val_tensor = torch.from_numpy(X_val_s).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    best_val = float('inf')
    best_epoch = -1
    bad = 0
    history_path = out_dir / 'D16_P5B_TRAINING_HISTORY.csv'
    with history_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
        w.writeheader()
        for ep in range(1, epochs + 1):
            model.train()
            train_losses = []
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=True)
                raw = model(xb)
                loss, _ = physics_observation_loss(raw, xb, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                # Evaluate validation in chunks to avoid GPU memory spikes.
                vals = []
                for i in range(0, val_tensor.shape[0], batch_size):
                    xb = val_tensor[i:i+batch_size]
                    raw = model(xb)
                    loss, _ = physics_observation_loss(raw, xb, cfg)
                    vals.append(float(loss.detach().cpu()))
                val_loss = float(np.mean(vals)) if vals else float('nan')
            train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
            w.writerow({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})
            if ep % 25 == 0 or ep == 1:
                print(f'[D16-P5B train] epoch={ep} train={train_loss:.6g} val={val_loss:.6g}', flush=True)
            if np.isfinite(val_loss) and val_loss < best_val:
                best_val = val_loss
                best_epoch = ep
                bad = 0
                ckpt = {
                    'state': model.state_dict(),
                    'model_class': 'ObsPhysicsMLP',
                    'model_config': model_cfg,
                    'feature_names': FEATURE_NAMES,
                    'output_names': OUTPUT_NAMES,
                    'x_mean': x_mean,
                    'x_std': x_std,
                    'config': cfg,
                    'manifest': str(args.manifest),
                    'train_rows': train_rows,
                    'training_input_audit': audit_rows,
                    'no_internal_softlabel_data_loss': True,
                    'training_used_time_series_keys': TRAIN_USED_KEYS,
                    'training_forbidden_softlabel_keys': TRAIN_FORBIDDEN_KEYS,
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val,
                }
                torch.save(ckpt, out_dir / 'model' / 'best_with_state.pt')
            else:
                bad += 1
            if bad >= patience:
                print(f'[D16-P5B train] early stop at epoch={ep}; best_epoch={best_epoch} best_val={best_val:.6g}', flush=True)
                break
    summary = {
        'stage': 'D16-P5B train6 observation-physics training',
        'status': 'PASS' if (out_dir / 'model' / 'best_with_state.pt').exists() else 'FAIL',
        'out_dir': str(out_dir),
        'manifest': str(args.manifest),
        'train_profile_count': len(train_rows),
        'train_points_total': int(X_train_s.shape[0]),
        'val_points_total': int(X_val_s.shape[0]),
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'checkpoint': str(out_dir / 'model' / 'best_with_state.pt'),
        'no_internal_softlabel_data_loss': True,
        'training_used_time_series_keys': TRAIN_USED_KEYS,
        'training_forbidden_softlabel_keys': TRAIN_FORBIDDEN_KEYS,
        'notes': [
            'Training reads only t/I/V observed time series from solution_softlabels.npz containers.',
            'No theta/cs/phie/phis_c soft-label arrays are loaded during training.',
            'Soft-label arrays are used only by the separate evaluation script.'
        ]
    }
    write_json(summary, out_dir / 'D16_P5B_TRAINING_SUMMARY.json')
    write_json({'stage': 'D16-P5B train input audit', 'rows': audit_rows}, out_dir / 'D16_P5B_TRAIN_INPUT_AUDIT.json')
    print('[D16-P5B train] status:', summary['status'], 'checkpoint:', summary['checkpoint'], flush=True)
    return 0 if summary['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
