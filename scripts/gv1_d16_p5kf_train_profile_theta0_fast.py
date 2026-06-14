from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

# D16-P5K-F: hard cbar / OCP-style residual model.
# Training boundary: observed time series only (t, I, V). Soft-label internal states are never loaded here.
TRAIN_USED_KEYS = ['t_global_s', 'I_profile', 'voltage_exp']
TRAIN_FORBIDDEN_KEYS = ['theta_a', 'theta_c', 'cs_a', 'cs_c', 'phie', 'phis_c', 'phis_c_soft']

FEATURE_NAMES = [
    't_norm', 't_norm2', 'sin_t', 'cos_t',
    'I_norm', 'absI_norm', 'dI_norm', 'q_norm',
    'q_cell_frac', 'q_cell_frac_abs',
    'voltage_exp_norm_local', 'dV_norm',
    'v_window_phase', 'v0_window_phase', 'v_mean_window_phase',
    'is_charge', 'is_rest', 'is_discharge',
]
OUTPUT_NAMES = ['res_a_raw', 'res_c_raw', 'grad_a_raw', 'grad_c_raw', 'phie_norm', 'phis_c_norm']


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


def first_key(z: Any, keys: List[str]) -> str | None:
    for k in keys:
        if k in z:
            return k
    return None


def as_1d_float(a: Any, name: str) -> np.ndarray:
    x = np.asarray(a)
    if x.dtype.kind in {'U', 'S', 'O'}:
        raise TypeError(f'{name} is not numeric')
    return x.astype(np.float32).reshape(-1)


def load_observed_npz(npz_path: str | Path) -> Dict[str, np.ndarray]:
    # IMPORTANT: this function is the training input boundary.
    # It does not load theta/cs/phie/phis target arrays.
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        kt = first_key(z, ['t_global_s', 'time_s', 't_s', 'time', 't'])
        ki = first_key(z, ['I_profile', 'current_A', 'I_A', 'current', 'I'])
        kv = first_key(z, ['voltage_exp', 'voltage_V', 'V_exp', 'V'])
        missing = []
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
        return {
            't': t, 'I': I, 'V': V,
            'source_keys': {'time': kt, 'current': ki, 'voltage': kv},
            'available_keys': sorted(keys),
        }


def sample_indices(n: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count <= 0 or max_count >= n:
        return np.arange(n, dtype=np.int64)
    if max_count < 4:
        return np.linspace(0, n - 1, max_count).astype(np.int64)
    edges = np.linspace(0, n, max_count + 1, dtype=np.int64)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        idx.append(int(rng.integers(a, b)))
    if idx:
        idx[0] = 0
        idx[-1] = n - 1
    return np.array(sorted(set(idx)), dtype=np.int64)



def _window_phase(V: np.ndarray, low: float = 2.5, high: float = 4.2) -> np.ndarray:
    span = max(1e-6, float(high - low))
    return np.clip((V.astype(np.float32) - float(low)) / span, 0.0, 1.0).astype(np.float32)


def build_q_features(t: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[~np.isfinite(dt)] = 0.0
    if dt.size > 10:
        p = np.nanpercentile(dt, 99.9)
        if np.isfinite(p) and p > 0:
            dt = np.clip(dt, 0.0, p * 10.0)
    q_ah = np.cumsum(I.astype(np.float32) * dt) / 3600.0
    q0 = q_ah - np.nanmean(q_ah)
    scale = float(np.nanpercentile(np.abs(q0), 99.5)) if q0.size else 1.0
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    q_norm = np.clip(q0 / scale, -1.5, 1.5).astype(np.float32)
    q_cell_frac = np.clip((q_ah - q_ah[0]) / 2.0, -1.5, 1.5).astype(np.float32)
    return q_norm, q_cell_frac, np.abs(q_cell_frac).astype(np.float32)


def build_q_norm(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    # Backward-compatible helper used by earlier code paths.
    return build_q_features(t, I)[0]


def build_features_from_observed(obs: Dict[str, np.ndarray], idx: np.ndarray | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    t = obs['t']; I = obs['I']; V = obs['V']
    n = len(t)
    if idx is None:
        idx = np.arange(n, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    if not np.isfinite(span) or span <= 0: span = 1.0
    tn_full = ((t - t[0]) / span).astype(np.float32)
    I_scale = float(np.nanpercentile(np.abs(I), 99.5)) if n else 1.0
    if not np.isfinite(I_scale) or I_scale < 1e-12: I_scale = 1.0
    In_full = (I / I_scale).astype(np.float32)
    dI_full = np.diff(In_full, prepend=In_full[0]).astype(np.float32)
    qn_full, qcell_full, qcell_abs_full = build_q_features(t, I)
    v_mean = float(np.nanmean(V)) if V.size else 0.0
    v_std = float(np.nanstd(V)) if V.size else 1.0
    if not np.isfinite(v_std) or v_std < 1e-8: v_std = 1.0
    vn_full = ((V - v_mean) / v_std).astype(np.float32)
    dV_full = np.diff(vn_full, prepend=vn_full[0]).astype(np.float32)
    v_phase_full = _window_phase(V)
    v0_phase_full = np.full_like(v_phase_full, float(v_phase_full[0]) if v_phase_full.size else 0.5, dtype=np.float32)
    vmean_phase_full = np.full_like(v_phase_full, float(_window_phase(np.array([v_mean], dtype=np.float32))[0]), dtype=np.float32)
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
        qcell_full[idx],
        qcell_abs_full[idx],
        vn_full[idx],
        dV_full[idx],
        v_phase_full[idx],
        v0_phase_full[idx],
        vmean_phase_full[idx],
        charge[idx],
        rest[idx],
        discharge[idx],
    ], axis=1).astype(np.float32)
    stats = {
        't0': float(t[0]), 't_span': float(span), 'I_scale': float(I_scale),
        'v_mean': float(v_mean), 'v_std': float(v_std), 'v0_phase': float(v0_phase_full[0]) if v0_phase_full.size else 0.5,
        'v_mean_phase': float(vmean_phase_full[0]) if vmean_phase_full.size else 0.5,
        'q_cell_frac_min': float(np.nanmin(qcell_full)) if qcell_full.size else 0.0,
        'q_cell_frac_max': float(np.nanmax(qcell_full)) if qcell_full.size else 0.0,
        'n_time': int(n)
    }
    return X, stats


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class HardCbarOCPResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 5, output_dim: int = 6):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(1, int(num_layers))):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _feature(x_raw: torch.Tensor, name: str) -> torch.Tensor:
    return x_raw[:, FEATURE_NAMES.index(name)]



def hard_baseline_from_observed(x_raw: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    h = cfg.get('hard_cbar_ocp_baseline', {})
    v_z = _feature(x_raw, 'voltage_exp_norm_local')
    q_z = _feature(x_raw, 'q_norm')
    q_cell = _feature(x_raw, 'q_cell_frac')
    v_phase = torch.clamp(_feature(x_raw, 'v_window_phase'), 0.0, 1.0)
    v0_phase = torch.clamp(_feature(x_raw, 'v0_window_phase'), 0.0, 1.0)
    vmean_phase = torch.clamp(_feature(x_raw, 'v_mean_window_phase'), 0.0, 1.0)

    # P5K-C legacy phase: local normalized voltage + centered Coulomb trajectory.
    soc_v_local = torch.sigmoid(float(h.get('voltage_sigmoid_gain', 1.15)) * v_z)
    soc_q_centered = 0.5 + 0.5 * torch.tanh(float(h.get('q_tanh_gain', 1.25)) * q_z)
    legacy_phase = torch.clamp(float(h.get('voltage_weight', 0.72))*soc_v_local + float(h.get('q_weight', 0.28))*soc_q_centered, 0.0, 1.0)

    # P5K-F profile-level theta0/OCP initializer. This is observed-only: absolute V0/Vmean plus measured-current Coulomb integral.
    theta0_phase = torch.clamp(float(h.get('v0_weight', 0.70))*v0_phase + float(h.get('vmean_weight', 0.30))*vmean_phase, 0.0, 1.0)
    coulomb_phase = torch.clamp(theta0_phase + float(h.get('q_cell_gain', 0.42))*q_cell, 0.0, 1.0)

    phase = torch.clamp(
        float(h.get('legacy_phase_weight', 0.62))*legacy_phase +
        float(h.get('profile_coulomb_phase_weight', 0.30))*coulomb_phase +
        float(h.get('absolute_voltage_phase_weight', 0.08))*v_phase,
        0.0, 1.0
    )
    centered = 2.0 * phase - 1.0
    a_mid = float(h.get('theta_a_mid', 0.405)); c_mid = float(h.get('theta_c_mid', 0.610))
    a_amp = float(h.get('theta_a_amplitude', 0.245)); c_amp = float(h.get('theta_c_amplitude', 0.245))
    a_min = float(h.get('theta_a_min', 0.02)); a_max = float(h.get('theta_a_max', 0.96))
    c_min = float(h.get('theta_c_min', 0.02)); c_max = float(h.get('theta_c_max', 0.96))
    base_a = torch.clamp(a_mid + a_amp * centered, a_min, a_max)
    base_c = torch.clamp(c_mid - c_amp * centered, c_min, c_max)
    return {
        'phase': phase,
        'legacy_phase': legacy_phase,
        'theta0_phase': theta0_phase,
        'coulomb_phase': coulomb_phase,
        'theta_a_base': base_a,
        'theta_c_base': base_c,
    }


def transform_outputs(raw: torch.Tensor, x_raw: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    model_cfg = cfg.get('model', {})
    grad_clip = float(model_cfg.get('gradient_clip', 0.25))
    rb_a = float(model_cfg.get('residual_bound_a', 0.055))
    rb_c = float(model_cfg.get('residual_bound_c', 0.055))
    b = hard_baseline_from_observed(x_raw, cfg)
    res_a = rb_a * torch.tanh(raw[:, 0])
    res_c = rb_c * torch.tanh(raw[:, 1])
    ta = torch.clamp(b['theta_a_base'] + res_a, 0.0, 1.0)
    tc = torch.clamp(b['theta_c_base'] + res_c, 0.0, 1.0)
    return {
        'theta_a_mean': ta,
        'theta_c_mean': tc,
        'theta_a_base': b['theta_a_base'],
        'theta_c_base': b['theta_c_base'],
        'theta_phase': b['phase'],
        'theta_a_residual': res_a,
        'theta_c_residual': res_c,
        'grad_a': grad_clip * torch.tanh(raw[:, 2]),
        'grad_c': grad_clip * torch.tanh(raw[:, 3]),
        'phie_norm': raw[:, 4],
        'phis_c_norm': raw[:, 5],
    }


def _safe_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a0 = a - torch.mean(a); b0 = b - torch.mean(b)
    denom = torch.sqrt(torch.mean(a0 * a0) * torch.mean(b0 * b0) + eps)
    return torch.mean(a0 * b0) / denom


def _corr_loss(a: torch.Tensor, b: torch.Tensor, target_sign: float = 1.0) -> torch.Tensor:
    if a.numel() < 8:
        return a.new_tensor(0.0)
    return 1.0 - float(target_sign) * _safe_corr(a, b)


def physics_observation_loss(raw: torch.Tensor, x_std: torch.Tensor, x_raw: torch.Tensor, cfg: Dict[str, Any], teacher_raw: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    w = cfg.get('training', {}).get('loss_weights', {})
    y = transform_outputs(raw, x_raw, cfg)
    v_z = _feature(x_raw, 'voltage_exp_norm_local')
    q_z = _feature(x_raw, 'q_norm')
    I_norm = _feature(x_raw, 'I_norm')
    absI_norm = _feature(x_raw, 'absI_norm')
    is_rest = _feature(x_raw, 'is_rest')
    ta = y['theta_a_mean']; tc = y['theta_c_mean']

    # Direct observed voltage branch. This uses V(t), which is explicitly allowed for inverse state estimation.
    loss_v = torch.mean((y['phis_c_norm'] - v_z) ** 2)
    loss_phie = torch.mean(y['phie_norm'] ** 2)

    # Hard baseline residual smallness: prevents the NN from becoming another free theta-mean predictor.
    loss_res_small = torch.mean(y['theta_a_residual'] ** 2 + y['theta_c_residual'] ** 2)
    loss_res_pair = torch.mean((y['theta_a_residual'] + y['theta_c_residual']) ** 2)
    loss_baseline_nonreg = torch.mean((ta - y['theta_a_base']) ** 2 + (tc - y['theta_c_base']) ** 2)

    # Only trend/cbar consistency, not label supervision.
    loss_q_corr = _corr_loss(ta, q_z, target_sign=+1.0) + _corr_loss(tc, q_z, target_sign=-1.0)
    loss_v_corr = _corr_loss(ta, v_z, target_sign=+1.0) + _corr_loss(tc, v_z, target_sign=-1.0)

    h = cfg.get('hard_cbar_ocp_baseline', {})
    pair_center = float(h.get('pair_sum_center', 1.015))
    pair_slack = float(h.get('pair_sum_slack', 0.10))
    pair_dev = torch.relu(torch.abs((ta + tc) - pair_center) - pair_slack)
    loss_pair_slack = torch.mean(pair_dev ** 2)
    # Keep learned residuals from drifting far from the profile-level OCP/Coulomb theta0 baseline.
    loss_profile_theta0_non_drift = torch.mean(torch.abs(y['theta_a_residual'])**1.5 + torch.abs(y['theta_c_residual'])**1.5)

    grad_target = 0.045 * torch.tanh(2.0 * absI_norm) * torch.sign(I_norm)
    loss_grad_anchor = torch.mean((y['grad_a'] - grad_target) ** 2) + torch.mean((y['grad_c'] + grad_target) ** 2)
    loss_grad_dir = torch.mean(torch.relu(-(y['grad_a'] * I_norm))) + torch.mean(torch.relu(y['grad_c'] * I_norm))
    loss_rest = torch.mean(is_rest * (y['grad_a'] ** 2 + y['grad_c'] ** 2))

    if raw.shape[0] > 2:
        loss_smooth = (
            torch.mean((ta[1:] - ta[:-1]) ** 2) + torch.mean((tc[1:] - tc[:-1]) ** 2) +
            torch.mean((y['grad_a'][1:] - y['grad_a'][:-1]) ** 2) + torch.mean((y['grad_c'][1:] - y['grad_c'][:-1]) ** 2) +
            torch.mean((y['theta_a_residual'][1:] - y['theta_a_residual'][:-1]) ** 2) +
            torch.mean((y['theta_c_residual'][1:] - y['theta_c_residual'][:-1]) ** 2)
        )
    else:
        loss_smooth = raw.new_tensor(0.0)

    loss_teacher_voltage = raw.new_tensor(0.0)
    loss_teacher_grad = raw.new_tensor(0.0)
    if teacher_raw is not None:
        # Preserve only voltage and gradient branches from older candidate. Do not preserve its direct theta gauge.
        # The older raw output can be mapped through the same P5K-F transform for branch compatibility.
        yt = transform_outputs(teacher_raw.detach(), x_raw, cfg)
        loss_teacher_voltage = torch.mean((y['phis_c_norm'] - yt['phis_c_norm']) ** 2)
        loss_teacher_grad = torch.mean((y['grad_a'] - yt['grad_a']) ** 2) + torch.mean((y['grad_c'] - yt['grad_c']) ** 2)

    total = (
        float(w.get('voltage_observation', 1.0)) * loss_v +
        float(w.get('hard_baseline_residual_small', 0.32)) * loss_res_small +
        float(w.get('residual_pair_balance', 0.14)) * loss_res_pair +
        float(w.get('baseline_non_regression', 0.18)) * loss_baseline_nonreg +
        float(w.get('theta_cbar_q_correlation', 0.16)) * loss_q_corr +
        float(w.get('theta_voltage_correlation', 0.03)) * loss_v_corr +
        float(w.get('theta_pair_slack', 0.10)) * loss_pair_slack +
        float(w.get('profile_theta0_non_drift', 0.030)) * loss_profile_theta0_non_drift +
        float(w.get('gradient_anchor', 0.070)) * loss_grad_anchor +
        float(w.get('gradient_direction', 0.040)) * loss_grad_dir +
        float(w.get('rest_relaxation', 0.050)) * loss_rest +
        float(w.get('smoothness', 0.006)) * loss_smooth +
        float(w.get('phie_regularization', 0.006)) * loss_phie +
        float(w.get('teacher_voltage_preservation', 0.015)) * loss_teacher_voltage +
        float(w.get('teacher_gradient_preservation', 0.006)) * loss_teacher_grad
    )
    parts = {
        'loss_total': float(total.detach().cpu()),
        'loss_voltage_observation': float(loss_v.detach().cpu()),
        'loss_hard_baseline_residual_small': float(loss_res_small.detach().cpu()),
        'loss_residual_pair_balance': float(loss_res_pair.detach().cpu()),
        'loss_baseline_non_regression': float(loss_baseline_nonreg.detach().cpu()),
        'loss_theta_cbar_q_correlation': float(loss_q_corr.detach().cpu()),
        'loss_theta_voltage_correlation': float(loss_v_corr.detach().cpu()),
        'loss_theta_pair_slack': float(loss_pair_slack.detach().cpu()),
        'loss_profile_theta0_non_drift': float(loss_profile_theta0_non_drift.detach().cpu()),
        'loss_gradient_anchor': float(loss_grad_anchor.detach().cpu()),
        'loss_gradient_direction': float(loss_grad_dir.detach().cpu()),
        'loss_rest_relaxation': float(loss_rest.detach().cpu()),
        'loss_smoothness': float(loss_smooth.detach().cpu()),
        'loss_phie_regularization': float(loss_phie.detach().cpu()),
        'loss_teacher_voltage_preservation': float(loss_teacher_voltage.detach().cpu()),
        'loss_teacher_gradient_preservation': float(loss_teacher_grad.detach().cpu()),
    }
    return total, parts


def standardize_train_val(Xtr: np.ndarray, Xva: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(Xtr, axis=0).astype(np.float32)
    std = np.std(Xtr, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-8)] = 1.0
    return ((Xtr - mean) / std).astype(np.float32), ((Xva - mean) / std).astype(np.float32), mean, std


def _find_checkpoint(model_dir: str | Path) -> Path | None:
    if not model_dir:
        return None
    p = Path(model_dir)
    for cand in [p / 'model' / 'best_with_state.pt', p / 'best_with_state.pt']:
        if cand.exists():
            return cand
    return None


def load_warm_start(model: nn.Module, warm_model_dir: str, device: torch.device) -> Dict[str, Any]:
    ckpt_path = _find_checkpoint(warm_model_dir)
    if ckpt_path is None:
        return {'loaded': False, 'reason': 'warm_start_checkpoint_not_found', 'warm_start_model_dir': str(warm_model_dir)}
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('state', ckpt)
        own = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
        missing = [k for k in own.keys() if k not in compatible]
        model.load_state_dict({**own, **compatible})
        model.to(device)
        return {'loaded': True, 'checkpoint': str(ckpt_path), 'compatible_param_count': len(compatible), 'missing_param_count': len(missing)}
    except Exception as exc:
        return {'loaded': False, 'checkpoint': str(ckpt_path), 'error': repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-F FAST train hard-cbar/OCP residual model. No internal-state soft-label data loss is used.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--config', default='configs/d16_p5kf_profile_theta0_hard_cbar_config.json')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--allow-overwrite', action='store_true')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--val-every', type=int, default=10)
    ap.add_argument('--steps-per-epoch', type=int, default=0)
    ap.add_argument('--warm-start-model-dir', default='')
    ap.add_argument('--no-warm-start', action='store_true')
    args = ap.parse_args()

    cfg = load_json(args.config)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out-dir exists and is non-empty: {out_dir}; pass --allow-overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'model').mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    train_rows = [r for r in rows if r.get('split') in ('core_train', 'hard_probe', 'train')]
    if len(train_rows) not in (10, 12):
        raise ValueError(f'Expected 10/12 train rows for P5K-F, got {len(train_rows)}')

    seed = int(cfg.get('training', {}).get('seed', 20260612))
    rng = np.random.default_rng(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    max_train = int(cfg.get('training', {}).get('max_train_points_per_profile', 260000))
    max_train_hard = int(cfg.get('training', {}).get('max_train_points_per_hard_probe', max_train))
    max_val = int(cfg.get('training', {}).get('max_val_points_per_profile', 50000))
    Xtr_list: List[np.ndarray] = []
    Xva_list: List[np.ndarray] = []
    audit_rows = []
    for r in train_rows:
        obs = load_observed_npz(r['softlabel_npz'])
        n = len(obs['t'])
        role_max_train = max_train_hard if r.get('split') == 'hard_probe' else max_train
        tr_idx = sample_indices(n, role_max_train, rng)
        rem = np.setdiff1d(np.arange(n, dtype=np.int64), tr_idx, assume_unique=False)
        if rem.size == 0:
            va_idx = tr_idx[::max(1, tr_idx.size // max(1, min(max_val, tr_idx.size)))]
        else:
            local = sample_indices(rem.size, min(max_val, rem.size), rng)
            va_idx = rem[local]
        Xtr, stats = build_features_from_observed(obs, tr_idx)
        Xva, _ = build_features_from_observed(obs, va_idx)
        Xtr_list.append(Xtr); Xva_list.append(Xva)
        audit_rows.append({
            'profile_id': r['profile_id'], 'batch': r['batch'], 'battery': r['battery'], 'split': r.get('split', 'core_train'),
            'reason': r.get('reason', ''), 'softlabel_npz': r['softlabel_npz'], 'n_time': int(n),
            'train_points': int(Xtr.shape[0]), 'val_points': int(Xva.shape[0]),
            'training_used_keys': TRAIN_USED_KEYS,
            'training_forbidden_keys_not_loaded': TRAIN_FORBIDDEN_KEYS,
            'source_keys': obs['source_keys'], 'profile_stats': stats,
        })
        print(f"[D16-P5K-F train] loaded observed-only {r['profile_id']}: n={n} train={Xtr.shape[0]} val={Xva.shape[0]}", flush=True)

    X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32)
    X_val = np.concatenate(Xva_list, axis=0).astype(np.float32)
    X_train_s, X_val_s, x_mean, x_std = standardize_train_val(X_train, X_val)

    device = torch.device(args.device if args.device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model_cfg = cfg.get('model', {})
    model = HardCbarOCPResidualMLP(input_dim=X_train_s.shape[1], hidden_dim=int(model_cfg.get('hidden_dim', 256)), num_layers=int(model_cfg.get('num_layers', 5)), output_dim=6).to(device)

    warm_info = {'loaded': False, 'reason': 'disabled'}
    if not args.no_warm_start:
        warm_dir = args.warm_start_model_dir or cfg.get('warm_start', {}).get('preferred_model_dir', '')
        warm_info = load_warm_start(model, warm_dir, device)
        if not warm_info.get('loaded'):
            fallback = cfg.get('warm_start', {}).get('fallback_model_dir', '')
            if fallback and fallback != warm_dir:
                warm_info = {'first_attempt': warm_info, 'fallback_attempt': load_warm_start(model, fallback, device)}
                if warm_info['fallback_attempt'].get('loaded'):
                    warm_info['loaded'] = True
    print('[D16-P5K-F train] warm_start:', warm_info, flush=True)

    lr = float(cfg.get('training', {}).get('lr', 0.0012))
    wd = float(cfg.get('training', {}).get('weight_decay', 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = int(args.epochs if args.epochs is not None else cfg.get('training', {}).get('epochs', 1200))
    batch_size = int(args.batch_size if args.batch_size is not None else 131072)
    val_every = max(1, int(args.val_every))
    steps_per_epoch = int(args.steps_per_epoch)

    Xtr_s_t = torch.from_numpy(X_train_s).to(device)
    Xtr_raw_t = torch.from_numpy(X_train).to(device)
    Xva_s_t = torch.from_numpy(X_val_s).to(device)
    Xva_raw_t = torch.from_numpy(X_val).to(device)

    best_val = float('inf')
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    def eval_loss() -> Tuple[float, Dict[str, float]]:
        model.eval()
        losses = []
        last_parts: Dict[str, float] = {}
        with torch.no_grad():
            for i in range(0, Xva_s_t.shape[0], batch_size):
                xs = Xva_s_t[i:i+batch_size]
                xr = Xva_raw_t[i:i+batch_size]
                raw = model(xs)
                loss, parts = physics_observation_loss(raw, xs, xr, cfg)
                losses.append(float(loss.detach().cpu()))
                last_parts = parts
        return float(np.mean(losses)) if losses else float('inf'), last_parts

    for ep in range(1, epochs + 1):
        model.train()
        if steps_per_epoch > 0:
            step_losses = []
            last_train_parts: Dict[str, float] = {}
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, Xtr_s_t.shape[0], (min(batch_size, Xtr_s_t.shape[0]),), device=device, generator=gen)
                xs = Xtr_s_t[idx]
                xr = Xtr_raw_t[idx]
                opt.zero_grad(set_to_none=True)
                raw = model(xs)
                loss, parts = physics_observation_loss(raw, xs, xr, cfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                step_losses.append(float(loss.detach().cpu()))
                last_train_parts = parts
            train_loss = float(np.mean(step_losses))
            train_parts = last_train_parts
        else:
            perm = torch.randperm(Xtr_s_t.shape[0], device=device, generator=gen)
            losses = []
            train_parts: Dict[str, float] = {}
            for i in range(0, perm.numel(), batch_size):
                idx = perm[i:i+batch_size]
                xs = Xtr_s_t[idx]
                xr = Xtr_raw_t[idx]
                opt.zero_grad(set_to_none=True)
                raw = model(xs)
                loss, parts = physics_observation_loss(raw, xs, xr, cfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
                train_parts = parts
            train_loss = float(np.mean(losses))

        if ep == 1 or ep % val_every == 0 or ep == epochs:
            val_loss, val_parts = eval_loss()
            row = {'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss, **{f'train_{k}': v for k, v in train_parts.items()}, **{f'val_{k}': v for k, v in val_parts.items()}}
            history.append(row)
            print(f"[D16-P5K-F train] epoch={ep} train={train_loss:.6g} val={val_loss:.6g} best={best_val:.6g}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = ep
                ckpt = {
                    'stage': 'D16-P5K-F hard-cbar OCP residual model',
                    'state': copy.deepcopy(model.state_dict()),
                    'model_class': 'HardCbarOCPResidualMLP',
                    'model_config': model_cfg,
                    'config': cfg,
                    'feature_names': FEATURE_NAMES,
                    'output_names': OUTPUT_NAMES,
                    'x_mean': x_mean,
                    'x_std': x_std,
                    'best_epoch': best_epoch,
                    'best_val_loss': best_val,
                    'warm_start': warm_info,
                    'train_count': len(train_rows),
                    'training_boundary': {
                        'used_keys': TRAIN_USED_KEYS,
                        'forbidden_keys_not_loaded': TRAIN_FORBIDDEN_KEYS,
                        'no_softlabel_data_loss': True,
                    },
                }
                torch.save(ckpt, out_dir / 'model' / 'best_with_state.pt')

    # Write history CSV.
    if history:
        keys = sorted(set().union(*(h.keys() for h in history)))
        with (out_dir / 'training_history.csv').open('w', newline='', encoding='utf-8') as f:
            wcsv = csv.DictWriter(f, fieldnames=keys)
            wcsv.writeheader(); wcsv.writerows(history)

    write_json({'stage': 'D16-P5K-F train input audit', 'rows': audit_rows}, out_dir / 'D16_P5KF_TRAIN_INPUT_AUDIT.json')
    summary = {
        'stage': 'D16-P5K-F hard-cbar OCP residual training summary',
        'status': 'PASS' if (out_dir / 'model' / 'best_with_state.pt').exists() else 'FAIL',
        'manifest': str(args.manifest),
        'out_dir': str(out_dir),
        'train_profile_count': len(train_rows),
        'epochs_requested': epochs,
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'checkpoint': str(out_dir / 'model' / 'best_with_state.pt'),
        'warm_start': warm_info,
        'training_boundary': {
            'used_keys': TRAIN_USED_KEYS,
            'forbidden_keys_not_loaded': TRAIN_FORBIDDEN_KEYS,
            'no_softlabel_data_loss': True,
            'softlabels_evaluation_only': True,
        },
        'interpretation': 'P5K-F restores ASSB-style hard cbar/OCP baseline with bounded residual. Internal soft labels are not used during training.',
    }
    write_json(summary, out_dir / 'D16_P5KF_TRAINING_SUMMARY.json')
    print('[D16-P5K-F train] wrote:', out_dir / 'D16_P5KF_TRAINING_SUMMARY.json', flush=True)
    return 0 if summary['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
