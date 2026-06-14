from __future__ import annotations

import argparse
import copy
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




def apply_p5g_gap_refinement(y: Dict[str, torch.Tensor], x: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Observed-only theta gauge refinement used in P5G.

    It does not read soft labels. The target gap is a smooth function of measured
    voltage feature, integrated charge feature, current magnitude and transition
    intensity. If the model predicts an excessive anode-cathode gap, the transform
    lowers theta_a_mean and raises theta_c_mean by the same amount, preserving the
    pair-sum inventory while correcting the common P5F failure mode.
    """
    p = cfg.get('p5g_outlier_protocol_balanced_gauge', {})
    if not bool(p.get('enable_gap_refinement', True)):
        return y
    v_z = x[:, 8]
    q_z = x[:, 7]
    abs_i = torch.clamp(torch.abs(x[:, 5]), 0.0, 6.0)
    d_i = torch.clamp(torch.abs(x[:, 6]), 0.0, 6.0)
    # Stress is higher for random-walk / transition-like segments and high current zones.
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
    ta = torch.clamp(y['theta_a_mean'] - delta, float(p.get('theta_floor', 0.02)), float(p.get('theta_ceiling', 0.98)))
    tc = torch.clamp(y['theta_c_mean'] + delta, float(p.get('theta_floor', 0.02)), float(p.get('theta_ceiling', 0.98)))
    yy = dict(y)
    yy['theta_a_mean'] = ta
    yy['theta_c_mean'] = tc
    yy['p5g_target_gap'] = target_gap
    yy['p5g_gap_delta'] = delta
    yy['p5g_stress'] = stress
    return yy

def _safe_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a0 = a - torch.mean(a)
    b0 = b - torch.mean(b)
    denom = torch.sqrt(torch.mean(a0 * a0) * torch.mean(b0 * b0) + eps)
    return torch.mean(a0 * b0) / denom


def _corr_loss(a: torch.Tensor, b: torch.Tensor, target_sign: float = 1.0) -> torch.Tensor:
    # If the batch is almost constant, the correlation term becomes uninformative; return 0 safely.
    if a.numel() < 8:
        return a.new_tensor(0.0)
    return 1.0 - float(target_sign) * _safe_corr(a, b)


def physics_observation_loss(raw: torch.Tensor, x: torch.Tensor, cfg: Dict[str, Any], teacher_raw: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """D16-P5G outlier/protocol-balanced gauge observation-physics loss.

    Training boundary is unchanged:
      * only t_global_s, I_profile, voltage_exp-derived features are used;
      * no theta/cs/phie/phis_c soft-label data loss is used;
      * soft-label arrays remain evaluation-only.

    P5G objective:
      * start from P5F/P5D-style strong voltage and trend behavior;
      * add an observed-only theta gap refinement to reduce the paired +theta_a/-theta_c offset;
      * upweight stress/outlier-like regimes through I/V-derived stress terms;
      * keep physical projection and avoid P5C/P5E hard absolute theta anchors.
    """
    w = cfg['training'].get('loss_weights', {})
    p5g = cfg.get('p5g_outlier_protocol_balanced_gauge', {})
    grad_clip = float(cfg.get('model', {}).get('gradient_clip', 0.25))
    y = apply_p5g_gap_refinement(transform_outputs(raw, grad_clip=grad_clip), x, cfg)

    v_z = x[:, 8]        # standardized local voltage feature; observed V(t), not soft-label.
    I_norm = x[:, 4]
    absI_norm = x[:, 5]
    q_norm = x[:, 7]
    is_rest = x[:, 11]

    ta = y['theta_a_mean']
    tc = y['theta_c_mean']

    # Voltage observation anchor for phis_c; V(t) is an allowed observed signal.
    loss_v = torch.mean((y['phis_c_norm'] - v_z) ** 2)

    # Relative theta constraints: q/V trend only, no soft-label target.
    loss_q_corr = _corr_loss(ta, q_norm, target_sign=+1.0) + _corr_loss(tc, q_norm, target_sign=-1.0)
    loss_v_corr = _corr_loss(ta, v_z, target_sign=+1.0) + _corr_loss(tc, v_z, target_sign=-1.0)

    # Centered two-electrode coupling. Suppresses paired variation drift without forcing a hard sum.
    ta_c = ta - torch.mean(ta)
    tc_c = tc - torch.mean(tc)
    loss_centered_couple = torch.mean((ta_c + tc_c) ** 2)

    # Weak midrange guard prevents saturation only.
    mid_margin = float(p5g.get('midrange_margin', 0.43))
    loss_midrange = torch.mean(torch.relu(torch.abs(ta - 0.5) - mid_margin) ** 2) + torch.mean(torch.relu(torch.abs(tc - 0.5) - mid_margin) ** 2)

    # P5G balanced mean-level gauge guards.
    # These use fixed prior ranges only. They are not soft-label supervision.
    # Mean-level guards are less aggressive than P5E pointwise cathode guards.
    anode_mean_ceiling = float(p5g.get('anode_mean_ceiling', 0.70))
    cathode_mean_floor = float(p5g.get('cathode_mean_floor', 0.31))
    anode_point_ceiling = float(p5g.get('anode_point_ceiling', 0.86))
    cathode_point_floor = float(p5g.get('cathode_point_floor', 0.18))
    loss_mean_gauge = torch.relu(torch.mean(ta) - anode_mean_ceiling) ** 2 + torch.relu(cathode_mean_floor - torch.mean(tc)) ** 2
    loss_point_guard = torch.mean(torch.relu(ta - anode_point_ceiling) ** 2) + torch.mean(torch.relu(cathode_point_floor - tc) ** 2)

    # Pair-sum soft slack discourages large inventory offsets but permits profile-dependent offset.
    pair_center = float(p5g.get('pair_sum_center', 1.02))
    pair_slack = float(p5g.get('pair_sum_slack', 0.18))
    pair_dev = torch.relu(torch.abs((ta + tc) - pair_center) - pair_slack)
    loss_pair_slack = torch.mean(pair_dev ** 2)

    # Very weak voltage-shaped center prior, smaller than P5E. It avoids hard absolute theta anchor.
    amp = float(p5g.get('center_prior_amplitude', 0.10))
    slope = float(p5g.get('center_prior_slope', 0.60))
    base_a = float(p5g.get('center_prior_base_a', 0.46))
    base_c = float(p5g.get('center_prior_base_c', 0.56))
    vshape = torch.tanh(slope * v_z)
    target_a_center = torch.clamp(base_a + amp * vshape, 0.05, 0.95)
    target_c_center = torch.clamp(base_c - amp * vshape, 0.05, 0.95)
    loss_weak_center_prior = torch.mean((ta - target_a_center.detach()) ** 2) + torch.mean((tc - target_c_center.detach()) ** 2)

    # P5G observed-only gap target. This is the central P5G anti-bias term.
    target_gap = y.get('p5g_target_gap', torch.zeros_like(ta))
    gap = ta - tc
    stress = y.get('p5g_stress', torch.zeros_like(ta))
    gap_err = gap - target_gap.detach()
    loss_gap_target = torch.mean((1.0 + float(p5g.get('stress_gap_weight', 1.0)) * stress.detach()) * gap_err ** 2)

    # Additional high-stress balanced guard prevents random_walk/GEO-like segments from drifting too far.
    stress_guard_margin = float(p5g.get('stress_pair_gap_margin', 0.32))
    loss_stress_guard = torch.mean(stress.detach() * torch.relu(torch.abs(gap) - stress_guard_margin) ** 2)

    # Optional warm-start teacher preservation. This is NOT soft-label supervision; teacher is P5D/P5E model output from I/V.
    loss_teacher_anode = raw.new_tensor(0.0)
    loss_teacher_voltage = raw.new_tensor(0.0)
    loss_teacher_grad = raw.new_tensor(0.0)
    if teacher_raw is not None:
        yt = transform_outputs(teacher_raw.detach(), grad_clip=grad_clip)
        # Preserve P5D's better anode gauge more strongly than cathode.
        loss_teacher_anode = torch.mean((ta - yt['theta_a_mean']) ** 2)
        loss_teacher_voltage = torch.mean((y['phis_c_norm'] - yt['phis_c_norm']) ** 2)
        loss_teacher_grad = torch.mean((y['grad_a'] - yt['grad_a']) ** 2) + torch.mean((y['grad_c'] - yt['grad_c']) ** 2)

    # Current-driven radial gradient direction and rest relaxation.
    grad_target_scale = float(p5g.get('gradient_target_scale', 0.05))
    grad_target = grad_target_scale * torch.tanh(2.0 * absI_norm) * torch.sign(I_norm)
    loss_grad_anchor = torch.mean((y['grad_a'] - grad_target) ** 2) + torch.mean((y['grad_c'] + grad_target) ** 2)
    loss_rest_relax = torch.mean(is_rest * (y['grad_a'] ** 2 + y['grad_c'] ** 2))
    loss_grad_dir = torch.mean(torch.relu(-(y['grad_a'] * I_norm))) + torch.mean(torch.relu(y['grad_c'] * I_norm))

    # Smoothness on stochastic batch outputs.
    if raw.shape[0] > 2:
        loss_smooth = (
            torch.mean((ta[1:] - ta[:-1]) ** 2) +
            torch.mean((tc[1:] - tc[:-1]) ** 2) +
            torch.mean((y['grad_a'][1:] - y['grad_a'][:-1]) ** 2) +
            torch.mean((y['grad_c'][1:] - y['grad_c'][:-1]) ** 2)
        )
    else:
        loss_smooth = raw.new_tensor(0.0)

    loss_phie = torch.mean(y['phie_norm'] ** 2)

    total = (
        float(w.get('voltage_observation', 1.0)) * loss_v +
        float(w.get('theta_q_correlation', 0.20)) * loss_q_corr +
        float(w.get('theta_voltage_correlation', 0.03)) * loss_v_corr +
        float(w.get('theta_centered_mass_coupling', 0.18)) * loss_centered_couple +
        float(w.get('theta_midrange_guard', 0.003)) * loss_midrange +
        float(w.get('balanced_mean_gauge', 0.10)) * loss_mean_gauge +
        float(w.get('balanced_point_guard', 0.020)) * loss_point_guard +
        float(w.get('theta_pair_slack', 0.10)) * loss_pair_slack +
        float(w.get('weak_center_prior', 0.010)) * loss_weak_center_prior +
        float(w.get('p5g_gap_target', 0.20)) * loss_gap_target +
        float(w.get('p5g_stress_gap_guard', 0.08)) * loss_stress_guard +
        float(w.get('teacher_anode_preservation', 0.020)) * loss_teacher_anode +
        float(w.get('teacher_voltage_preservation', 0.010)) * loss_teacher_voltage +
        float(w.get('teacher_gradient_preservation', 0.010)) * loss_teacher_grad +
        float(w.get('gradient_anchor', 0.075)) * loss_grad_anchor +
        float(w.get('rest_relaxation', 0.050)) * loss_rest_relax +
        float(w.get('gradient_direction', 0.040)) * loss_grad_dir +
        float(w.get('smoothness', 0.008)) * loss_smooth +
        float(w.get('phie_regularization', 0.006)) * loss_phie
    )
    parts = {
        'loss_voltage_observation': float(loss_v.detach().cpu()),
        'loss_theta_q_correlation': float(loss_q_corr.detach().cpu()),
        'loss_theta_voltage_correlation': float(loss_v_corr.detach().cpu()),
        'loss_theta_centered_mass_coupling': float(loss_centered_couple.detach().cpu()),
        'loss_theta_midrange_guard': float(loss_midrange.detach().cpu()),
        'loss_balanced_mean_gauge': float(loss_mean_gauge.detach().cpu()),
        'loss_balanced_point_guard': float(loss_point_guard.detach().cpu()),
        'loss_theta_pair_slack': float(loss_pair_slack.detach().cpu()),
        'loss_weak_center_prior': float(loss_weak_center_prior.detach().cpu()),
        'loss_p5g_gap_target': float(loss_gap_target.detach().cpu()),
        'loss_p5g_stress_gap_guard': float(loss_stress_guard.detach().cpu()),
        'loss_teacher_anode_preservation': float(loss_teacher_anode.detach().cpu()),
        'loss_teacher_voltage_preservation': float(loss_teacher_voltage.detach().cpu()),
        'loss_teacher_gradient_preservation': float(loss_teacher_grad.detach().cpu()),
        'loss_gradient_anchor': float(loss_grad_anchor.detach().cpu()),
        'loss_rest_relaxation': float(loss_rest_relax.detach().cpu()),
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
    ap = argparse.ArgumentParser(description='D16-P5G FAST train6 outlier-protocol-balanced-gauge physics model. No internal soft-label data loss is used.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--config', default='configs/d16_p5g_outlier_protocol_balanced_gauge_config.json')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--allow-overwrite', action='store_true')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--val-every', type=int, default=10, help='Validate every N epochs; epoch 1 is always validated.')
    ap.add_argument('--steps-per-epoch', type=int, default=0, help='If >0, use random GPU-resident batches instead of one full pass per epoch.')
    ap.add_argument('--warm-start-model-dir', default='', help='Optional P5D/P5B model dir containing model/best_with_state.pt for weight warm start; default runner uses P5D-800 candidate.')
    ap.add_argument('--no-warm-start', action='store_true', help='Disable warm start even if warm-start-model-dir is provided.')
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    max_train = int(cfg['training'].get('max_train_points_per_profile', 250000))
    max_val = int(cfg['training'].get('max_val_points_per_profile', 40000))
    Xtr_list: List[np.ndarray] = []
    Xva_list: List[np.ndarray] = []
    audit_rows = []
    for r in train_rows:
        obs = load_observed_npz(r['softlabel_npz'])
        n = len(obs['t'])
        tr_idx = sample_indices(n, max_train, rng)
        all_idx = np.arange(n, dtype=np.int64)
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
        print(f"[D16-P5G fast train] loaded observed-only {r['profile_id']}: n={n} train={Xtr.shape[0]} val={Xva.shape[0]}", flush=True)

    X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32)
    X_val = np.concatenate(Xva_list, axis=0).astype(np.float32)
    X_train_s, X_val_s, x_mean, x_std = standardize_train_val(X_train, X_val)

    device = torch.device(args.device if args.device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    model_cfg = cfg.get('model', {})
    model = ObsPhysicsMLP(
        input_dim=X_train_s.shape[1],
        hidden_dim=int(model_cfg.get('hidden_dim', 256)),
        num_layers=int(model_cfg.get('num_layers', 5)),
        output_dim=6,
    ).to(device)

    warm_start_used = False
    warm_start_checkpoint = ''
    teacher_model = None
    if args.warm_start_model_dir and not args.no_warm_start:
        wm = Path(args.warm_start_model_dir)
        ck = wm / 'model' / 'best_with_state.pt'
        if not ck.exists():
            ck = wm / 'best_with_state.pt'
        if ck.exists():
            try:
                old = torch.load(ck, map_location='cpu', weights_only=False)
                state = old.get('state', old)
                missing, unexpected = model.load_state_dict(state, strict=False)
                warm_start_used = True
                warm_start_checkpoint = str(ck)
                print(f"[D16-P5G train] warm-start loaded: {ck}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            except Exception as exc:
                print(f"[D16-P5G train] WARNING: warm-start failed from {ck}: {exc}", flush=True)
        else:
            print(f"[D16-P5G train] warm-start requested but checkpoint not found under {wm}", flush=True)

    if warm_start_used:
        teacher_model = copy.deepcopy(model).to(device).eval()
        for _p in teacher_model.parameters():
            _p.requires_grad_(False)
        print('[D16-P5G train] teacher preservation model frozen from warm-start checkpoint', flush=True)

    # Main speed fix: keep all sampled training/validation tensors resident on GPU.
    X_train_t = torch.from_numpy(X_train_s).to(device, non_blocking=True)
    X_val_t = torch.from_numpy(X_val_s).to(device, non_blocking=True)
    n_train = int(X_train_t.shape[0])
    n_val = int(X_val_t.shape[0])

    lr = float(cfg['training'].get('learning_rate', 1e-3))
    wd = float(cfg['training'].get('weight_decay', 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    epochs = int(args.epochs or cfg['training'].get('epochs', 1500))
    batch_size = int(args.batch_size or cfg['training'].get('batch_size', 65536))
    val_every = max(1, int(args.val_every))
    patience = int(cfg['training'].get('early_stop_patience', 160))
    steps_per_epoch = int(args.steps_per_epoch or 0)
    if steps_per_epoch <= 0:
        steps_per_epoch = int(math.ceil(n_train / max(1, batch_size)))
        sampling_mode = 'full_pass_gpu_resident'
    else:
        sampling_mode = 'random_gpu_resident'

    best_val = float('inf')
    best_epoch = -1
    bad_validations = 0
    history_path = out_dir / 'D16_P5G_TRAINING_HISTORY.csv'
    with history_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'steps_per_epoch', 'sampling_mode'])
        w.writeheader()
        for ep in range(1, epochs + 1):
            model.train()
            train_losses = []
            if sampling_mode == 'full_pass_gpu_resident':
                perm = torch.randperm(n_train, device=device)
                for start in range(0, n_train, batch_size):
                    idx = perm[start:start + batch_size]
                    xb = X_train_t.index_select(0, idx)
                    raw = model(xb)
                    teacher_raw = None
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_raw = teacher_model(xb)
                    loss, _ = physics_observation_loss(raw, xb, cfg, teacher_raw=teacher_raw)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    train_losses.append(float(loss.detach().cpu()))
            else:
                for _ in range(steps_per_epoch):
                    idx = torch.randint(0, n_train, (batch_size,), device=device)
                    xb = X_train_t.index_select(0, idx)
                    raw = model(xb)
                    teacher_raw = None
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_raw = teacher_model(xb)
                    loss, _ = physics_observation_loss(raw, xb, cfg, teacher_raw=teacher_raw)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    train_losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
            do_val = (ep == 1) or (ep % val_every == 0) or (ep == epochs)
            val_loss = float('nan')
            if do_val:
                model.eval()
                vals = []
                with torch.no_grad():
                    for i in range(0, n_val, batch_size):
                        xb = X_val_t[i:i + batch_size]
                        raw = model(xb)
                        teacher_raw = None
                        if teacher_model is not None:
                            teacher_raw = teacher_model(xb)
                        loss, _ = physics_observation_loss(raw, xb, cfg, teacher_raw=teacher_raw)
                        vals.append(float(loss.detach().cpu()))
                val_loss = float(np.mean(vals)) if vals else float('nan')
                if np.isfinite(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    best_epoch = ep
                    bad_validations = 0
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
                        'fast_trainer': True,
                        'gpu_resident_tensors': True,
                        'outlier_protocol_balanced_gauge_physics': True,
                        'warm_start_used': warm_start_used,
                        'warm_start_checkpoint': warm_start_checkpoint,
                        'no_internal_softlabel_data_loss': True,
                        'val_every': val_every,
                        'steps_per_epoch': steps_per_epoch,
                        'sampling_mode': sampling_mode,
                    }
                    torch.save(ckpt, out_dir / 'model' / 'best_with_state.pt')
                else:
                    bad_validations += 1
            w.writerow({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss, 'steps_per_epoch': steps_per_epoch, 'sampling_mode': sampling_mode})
            if ep % max(1, val_every) == 0 or ep == 1:
                msg = f'[D16-P5G fast train] epoch={ep} train={train_loss:.6g}'
                if np.isfinite(val_loss):
                    msg += f' val={val_loss:.6g} best_epoch={best_epoch} best_val={best_val:.6g}'
                print(msg, flush=True)
            if bad_validations >= patience:
                print(f'[D16-P5G fast train] early stop at epoch={ep}; best_epoch={best_epoch} best_val={best_val:.6g}', flush=True)
                break

    summary = {
        'stage': 'D16-P5G FAST train6 outlier-protocol-balanced-gauge physics training',
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
        'fast_trainer': True,
        'gpu_resident_tensors': True,
        'outlier_protocol_balanced_gauge_physics': True,
        'warm_start_used': warm_start_used,
        'warm_start_checkpoint': warm_start_checkpoint,
        'val_every': val_every,
        'steps_per_epoch': steps_per_epoch,
        'sampling_mode': sampling_mode,
        'notes': [
            'Training reads only t/I/V observed time series from solution_softlabels.npz containers.',
            'No theta/cs/phie/phis_c soft-label arrays are loaded during training.',
            'Soft-label arrays are used only by the separate evaluation script.',
            'This P5G fast trainer keeps sampled X_train/X_val tensors on GPU and validates every val_every epochs.',
            'P5G adds offset-safe balanced-floor/anode-ceiling, centered pair coupling and q/V trend constraints derived only from I/V and fixed priors.'
        ]
    }
    write_json(summary, out_dir / 'D16_P5G_TRAINING_SUMMARY.json')
    write_json({'stage': 'D16-P5G train input audit', 'rows': audit_rows}, out_dir / 'D16_P5G_TRAIN_INPUT_AUDIT.json')
    print('[D16-P5G fast train] status:', summary['status'], 'checkpoint:', summary['checkpoint'], flush=True)
    return 0 if summary['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
