from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - local env should have torch, but provenance still works without it.
    torch = None

# D16-P5K-E2: provenance + baseline-only audit.
# No training. No checkpoint modification. No model promotion.
# Purpose:
#   1) Fix the P5K-E path/provenance bug by resolving all known model-dir aliases.
#   2) Compare P5K-C and P5K-D final scorecards with hard-baseline-only predictions.
#   3) Determine whether regression comes from hard baseline/initial gauge or residual network.

FEATURE_NAMES_C = [
    't_norm', 't_norm2', 'sin_t', 'cos_t',
    'I_norm', 'absI_norm', 'dI_norm', 'q_norm',
    'voltage_exp_norm_local', 'dV_norm',
    'is_charge', 'is_rest', 'is_discharge',
]

FEATURE_NAMES_D = [
    't_norm', 't_norm2', 'sin_t', 'cos_t',
    'I_norm', 'absI_norm', 'dI_norm', 'q_norm',
    'q_cell_frac', 'voltage_abs_soc',
    'voltage_exp_norm_local', 'dV_norm',
    'p2dlite_phase', 'v0_abs_soc', 'current_stress',
    'is_charge', 'is_rest', 'is_discharge',
]


def norm_path(p: str | Path) -> Path:
    return Path(str(p).replace('\\', os.sep)) if os.sep == '/' and ':' not in str(p)[:3] else Path(p)


def sha1_file(path: Path, block_size: int = 2**20) -> str:
    h = hashlib.sha1()
    with path.open('rb') as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8', errors='replace')).hexdigest()


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def safe_read_json(path: Path) -> Any:
    try:
        if path.exists():
            return read_json(path)
    except Exception as e:
        return {'__read_error__': repr(e)}
    return None


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    cols: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def file_info(path: Path) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        'path': str(path),
        'exists': path.exists(),
    }
    if path.exists() and path.is_file():
        d.update({
            'size_bytes': path.stat().st_size,
            'size_kb': round(path.stat().st_size / 1024.0, 3),
            'sha1': sha1_file(path),
        })
    return d


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


@dataclass
class ModelSpec:
    key: str
    label: str
    stage_run_dir: Path
    manifest_candidates: List[Path]
    model_dir_candidates: List[Path]
    scorecard_candidates: List[Path]
    split_metrics_candidates: List[Path]
    profile_metrics_candidates: List[Path]
    batch_metrics_candidates: List[Path]
    protocol_metrics_candidates: List[Path]
    config_candidates: List[Path]
    variant: str


def build_model_specs(cache_root: Path, project_root: Path) -> Dict[str, ModelSpec]:
    p5kc = cache_root / 'xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST' / 'C_train10'
    p5kd = cache_root / 'xjtu_d16_p5kd_train10_generator_aligned_hard_cbar_ocp_FAST' / 'D_train10_prior_balanced'
    return {
        'P5K-C': ModelSpec(
            key='P5K-C', label='P5K-C train10 hard-cbar/OCP residual', stage_run_dir=p5kc,
            manifest_candidates=[p5kc / 'D16_P5K_C_train10_MANIFEST.csv'],
            model_dir_candidates=[p5kc / 'model_hard_cbar_ocp_residual'],
            scorecard_candidates=[p5kc / 'eval_all55_vs_softlabels' / 'D16_P5K_FINAL_SCORECARD.json'],
            split_metrics_candidates=[p5kc / 'eval_all55_vs_softlabels' / 'D16_P5K_SPLIT_METRICS.csv'],
            profile_metrics_candidates=[p5kc / 'eval_all55_vs_softlabels' / 'D16_P5K_METRICS_BY_PROFILE.csv'],
            batch_metrics_candidates=[p5kc / 'eval_all55_vs_softlabels' / 'D16_P5K_BATCH_METRICS.csv'],
            protocol_metrics_candidates=[p5kc / 'eval_all55_vs_softlabels' / 'D16_P5K_PROTOCOL_METRICS.csv'],
            config_candidates=[project_root / 'configs' / 'd16_p5k_hard_cbar_ocp_residual_config.json'],
            variant='C',
        ),
        'P5K-D': ModelSpec(
            key='P5K-D', label='P5K-D generator-aligned hard-cbar/OCP residual', stage_run_dir=p5kd,
            manifest_candidates=[p5kd / 'D16_P5KD_D_train10_prior_balanced_MANIFEST.csv', p5kd / 'D16_P5K_D_train10_prior_balanced_MANIFEST.csv', p5kd / 'D16_P5KD_TRAIN10_MANIFEST.csv', p5kd / 'D16_P5KD_MANIFEST.csv'],
            model_dir_candidates=[
                p5kd / 'model_generator_aligned_hard_cbar_ocp_residual',
                p5kd / 'model_generator_aligned_hard_cbar_ocp',  # old wrong name kept only as fallback/provenance check
            ],
            scorecard_candidates=[p5kd / 'eval_all55_vs_softlabels' / 'D16_P5KD_FINAL_SCORECARD.json'],
            split_metrics_candidates=[p5kd / 'eval_all55_vs_softlabels' / 'D16_P5KD_SPLIT_METRICS.csv'],
            profile_metrics_candidates=[p5kd / 'eval_all55_vs_softlabels' / 'D16_P5KD_METRICS_BY_PROFILE.csv'],
            batch_metrics_candidates=[p5kd / 'eval_all55_vs_softlabels' / 'D16_P5KD_BATCH_METRICS.csv'],
            protocol_metrics_candidates=[p5kd / 'eval_all55_vs_softlabels' / 'D16_P5KD_PROTOCOL_METRICS.csv'],
            config_candidates=[project_root / 'configs' / 'd16_p5kd_generator_aligned_hard_cbar_ocp_config.json'],
            variant='D',
        ),
    }


def locate_train_summary(model_dir: Optional[Path], key: str) -> Optional[Path]:
    if model_dir is None:
        return None
    names = []
    if key == 'P5K-C':
        names = ['D16_P5K_TRAINING_SUMMARY.json']
    elif key == 'P5K-D':
        names = ['D16_P5KD_TRAINING_SUMMARY.json', 'D16_P5K_D_TRAINING_SUMMARY.json']
    return first_existing([model_dir / n for n in names])


def locate_train_audit(model_dir: Optional[Path], key: str) -> Optional[Path]:
    if model_dir is None:
        return None
    names = []
    if key == 'P5K-C':
        names = ['D16_P5K_TRAIN_INPUT_AUDIT.json']
    elif key == 'P5K-D':
        names = ['D16_P5KD_TRAIN_INPUT_AUDIT.json', 'D16_P5K_D_TRAIN_INPUT_AUDIT.json']
    return first_existing([model_dir / n for n in names])


def locate_checkpoint(model_dir: Optional[Path]) -> Optional[Path]:
    if model_dir is None:
        return None
    return first_existing([model_dir / 'model' / 'best_with_state.pt', model_dir / 'best_with_state.pt'])


def load_checkpoint_config(ckpt_path: Optional[Path], config_path: Optional[Path]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    cfg: Dict[str, Any] = {}
    if ckpt_path is not None and ckpt_path.exists() and torch is not None:
        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
            meta['checkpoint_load_ok'] = True
            meta['checkpoint_top_keys'] = sorted(list(ckpt.keys())) if isinstance(ckpt, dict) else []
            if isinstance(ckpt, dict):
                cfg = ckpt.get('config') or {}
                meta['model_config'] = ckpt.get('model_config', {})
                state = ckpt.get('state')
                if isinstance(state, dict):
                    schema = '\n'.join([f'{k}:{tuple(v.shape) if hasattr(v, "shape") else "?"}' for k, v in state.items()])
                    meta['state_key_count'] = len(state)
                    meta['state_schema_sha1_16'] = sha1_text(schema)[:16]
                    meta['first_state_keys'] = list(state.keys())[:12]
                if 'x_mean' in ckpt:
                    try: meta['x_mean_shape'] = list(np.asarray(ckpt['x_mean']).shape)
                    except Exception: pass
                if 'x_std' in ckpt:
                    try: meta['x_std_shape'] = list(np.asarray(ckpt['x_std']).shape)
                    except Exception: pass
        except Exception as e:
            meta['checkpoint_load_ok'] = False
            meta['checkpoint_load_error'] = repr(e)
    elif ckpt_path is not None and ckpt_path.exists() and torch is None:
        meta['checkpoint_load_ok'] = False
        meta['checkpoint_load_error'] = 'torch import failed/unavailable'
    if not cfg and config_path is not None and config_path.exists():
        try:
            cfg = read_json(config_path)
            meta['config_loaded_from'] = str(config_path)
        except Exception as e:
            meta['config_load_error'] = repr(e)
    else:
        meta['config_loaded_from'] = 'checkpoint.config' if cfg else None
    return cfg, meta


def parse_csv_first_by_group(path: Optional[Path], group_name: str = 'eval') -> Optional[Dict[str, str]]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open('r', newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if str(r.get('group', '')).lower() == group_name.lower():
                return r
        return rows[0] if rows else None
    except Exception:
        return None


def metric_float(row: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not row:
        return None
    try:
        v = row.get(key, None)
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None


class Accum:
    __slots__ = ('n', 'sum_abs', 'sum_sq', 'sum_err', 'max_abs', 'sum_t', 'sum_p', 'sum_t2', 'sum_p2', 'sum_tp')
    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_err = 0.0
        self.max_abs = 0.0
        self.sum_t = 0.0
        self.sum_p = 0.0
        self.sum_t2 = 0.0
        self.sum_p2 = 0.0
        self.sum_tp = 0.0
    def update(self, true: np.ndarray, pred: np.ndarray) -> None:
        t = np.asarray(true, dtype=np.float64).reshape(-1)
        p = np.asarray(pred, dtype=np.float64).reshape(-1)
        mask = np.isfinite(t) & np.isfinite(p)
        if not np.any(mask):
            return
        t = t[mask]; p = p[mask]
        e = p - t
        ae = np.abs(e)
        self.n += int(t.size)
        self.sum_abs += float(np.sum(ae))
        self.sum_sq += float(np.sum(e*e))
        self.sum_err += float(np.sum(e))
        self.max_abs = max(self.max_abs, float(np.max(ae))) if ae.size else self.max_abs
        self.sum_t += float(np.sum(t))
        self.sum_p += float(np.sum(p))
        self.sum_t2 += float(np.sum(t*t))
        self.sum_p2 += float(np.sum(p*p))
        self.sum_tp += float(np.sum(t*p))
    def row(self, prefix: str) -> Dict[str, Any]:
        if self.n <= 0:
            return {f'{prefix}_count': 0, f'{prefix}_mae': float('nan'), f'{prefix}_rmse': float('nan'), f'{prefix}_bias': float('nan'), f'{prefix}_max_abs': float('nan'), f'{prefix}_corr': float('nan'), f'{prefix}_r2': float('nan')}
        n = float(self.n)
        vt = self.sum_t2 - self.sum_t*self.sum_t/n
        vp = self.sum_p2 - self.sum_p*self.sum_p/n
        cov = self.sum_tp - self.sum_t*self.sum_p/n
        corr = cov / math.sqrt(vt*vp) if vt > 1e-20 and vp > 1e-20 else float('nan')
        r2 = 1.0 - self.sum_sq/vt if vt > 1e-20 else float('nan')
        return {
            f'{prefix}_count': int(self.n),
            f'{prefix}_mae': self.sum_abs/n,
            f'{prefix}_rmse': math.sqrt(self.sum_sq/n),
            f'{prefix}_bias': self.sum_err/n,
            f'{prefix}_max_abs': self.max_abs,
            f'{prefix}_corr': corr,
            f'{prefix}_r2': r2,
            f'{prefix}_sum_true': self.sum_t,
            f'{prefix}_sum_true_sq': self.sum_t2,
            f'{prefix}_sum_pred': self.sum_p,
            f'{prefix}_sum_pred_sq': self.sum_p2,
            f'{prefix}_sum_err_sq': self.sum_sq,
        }


def first_key(keys: Iterable[str], candidates: List[str]) -> Optional[str]:
    s = set(keys)
    for c in candidates:
        if c in s:
            return c
    return None


def extract_npy_member(npz_path: Path, key: str, cache_root: Path) -> Path:
    # Short cache path: avoids Windows long-path + colon issues.
    h = hashlib.sha1((str(npz_path.resolve()) + '::' + key).encode('utf-8', errors='replace')).hexdigest()[:20]
    out_dir = cache_root / h
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f'{key}.npy'
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    with zipfile.ZipFile(npz_path, 'r') as zf:
        members = [n for n in zf.namelist() if n.endswith('.npy') and Path(n).stem == key]
        if not members:
            raise KeyError(f'{npz_path}: member {key}.npy not found')
        tmp = dst.with_suffix('.npy.tmp')
        with zf.open(members[0]) as src, tmp.open('wb') as f:
            shutil.copyfileobj(src, f)
        tmp.replace(dst)
    return dst


def load_mmap_arrays(npz_path: Path, cache_root: Path) -> Dict[str, np.ndarray]:
    with zipfile.ZipFile(npz_path, 'r') as zf:
        keys = {Path(n).stem for n in zf.namelist() if n.endswith('.npy')}
    kt = first_key(keys, ['t_global_s', 'time_s', 't_s', 'time', 't'])
    ki = first_key(keys, ['I_profile', 'current_A', 'I_A', 'current', 'I'])
    kv = first_key(keys, ['voltage_exp', 'voltage_V', 'V_exp', 'V'])
    kth_a = first_key(keys, ['theta_a', 'theta_n', 'theta_negative'])
    kth_c = first_key(keys, ['theta_c', 'theta_p', 'theta_positive'])
    kcs_a = first_key(keys, ['cs_a', 'c_s_a', 'cs_n', 'cs_negative'])
    kcs_c = first_key(keys, ['cs_c', 'c_s_c', 'cs_p', 'cs_positive'])
    missing = [name for name, k in [('t', kt), ('I', ki), ('V', kv), ('theta_a', kth_a), ('theta_c', kth_c)] if k is None]
    if missing:
        raise KeyError(f'{npz_path}: missing arrays {missing}; available={sorted(keys)}')
    mapping = {'t': kt, 'I': ki, 'V': kv, 'theta_a': kth_a, 'theta_c': kth_c}
    if kcs_a: mapping['cs_a'] = kcs_a
    if kcs_c: mapping['cs_c'] = kcs_c
    out: Dict[str, np.ndarray] = {}
    for alias, key in mapping.items():
        p = extract_npy_member(npz_path, key, cache_root)
        out[alias] = np.load(p, mmap_mode='r')
    return out


def orient2d(arr: np.ndarray, n: int, s: int, e: int) -> np.ndarray:
    if len(arr.shape) == 1:
        return np.asarray(arr[s:e], dtype=np.float32).reshape(-1, 1)
    if arr.shape[0] == n:
        return np.asarray(arr[s:e], dtype=np.float32)
    if len(arr.shape) == 2 and arr.shape[1] == n:
        return np.asarray(arr[:, s:e], dtype=np.float32).T
    raise ValueError(f'cannot orient shape={arr.shape} for n={n}')


def as_1d(arr: np.ndarray, s: int, e: int) -> np.ndarray:
    return np.asarray(arr[s:e], dtype=np.float32).reshape(-1)


def compute_global_features(t: np.ndarray, I: np.ndarray, V: np.ndarray, variant: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    I = np.asarray(I, dtype=np.float32).reshape(-1)
    V = np.asarray(V, dtype=np.float32).reshape(-1)
    n = len(t)
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    if not np.isfinite(span) or span <= 0: span = 1.0
    t_norm = ((t - t[0]) / span).astype(np.float32)
    I_scale = float(np.nanpercentile(np.abs(I), 99.5)) if n else 1.0
    if not np.isfinite(I_scale) or I_scale < 1e-12: I_scale = 1.0
    I_norm = (I / I_scale).astype(np.float32)
    dI_norm = np.diff(I_norm, prepend=I_norm[0]).astype(np.float32)
    v_mean = float(np.nanmean(V)) if n else 0.0
    v_std = float(np.nanstd(V)) if n else 1.0
    if not np.isfinite(v_std) or v_std < 1e-8: v_std = 1.0
    v_norm = ((V - v_mean) / v_std).astype(np.float32)
    dV_norm = np.diff(v_norm, prepend=v_norm[0]).astype(np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[~np.isfinite(dt)] = 0.0
    if dt.size > 10:
        p = np.nanpercentile(dt, 99.9)
        if np.isfinite(p) and p > 0:
            dt = np.clip(dt, 0.0, p * 10.0)
    q_Ah = np.cumsum(I * dt) / 3600.0
    q0 = q_Ah - np.nanmean(q_Ah)
    q_scale = float(np.nanpercentile(np.abs(q0), 99.5)) if q0.size else 1.0
    if not np.isfinite(q_scale) or q_scale < 1e-12: q_scale = 1.0
    q_norm = np.clip(q0 / q_scale, -1.5, 1.5).astype(np.float32)
    eps = max(1e-9, 0.001 * float(np.nanmax(np.abs(I)) + 1e-12))
    base = {
        't': t, 'I': I, 'V': V,
        't_norm': t_norm, 'I_norm': I_norm, 'dI_norm': dI_norm,
        'v_norm': v_norm, 'dV_norm': dV_norm, 'q_norm': q_norm,
        'is_charge': (I > eps).astype(np.float32),
        'is_rest': (np.abs(I) <= eps).astype(np.float32),
        'is_discharge': (I < -eps).astype(np.float32),
        'n': n, 't_span': span, 'I_scale': I_scale, 'v_mean': v_mean, 'v_std': v_std,
        'q_Ah': q_Ah,
    }
    if variant == 'D':
        p = cfg.get('p2dlite_rg_prior', {})
        g = cfg.get('generator_aligned_baseline', {})
        cap = max(1e-6, float(p.get('nominal_capacity_Ah', 2.0)))
        vmin = float(p.get('voltage_min', 2.5)); vmax = float(p.get('voltage_max', 4.2))
        vr = max(1e-6, vmax - vmin)
        q_cell_frac = np.clip((q_Ah - q_Ah[0]) / cap, -1.5, 1.5).astype(np.float32)
        v_abs_soc = np.clip((V - vmin) / vr, 0.0, 1.0).astype(np.float32)
        v0_abs_soc = float(np.clip((V[0] - vmin) / vr, 0.0, 1.0)) if n else 0.5
        q_soc = np.clip(v0_abs_soc + q_cell_frac, 0.0, 1.0).astype(np.float32)
        phase = np.clip(float(g.get('phase_voltage_weight', 0.82))*v_abs_soc + float(g.get('phase_coulomb_weight', 0.18))*q_soc, 0.0, 1.0).astype(np.float32)
        I_1C = max(1e-6, float(p.get('I_1C_A', 2.0)))
        current_stress = np.clip(np.abs(I) / I_1C, 0.0, 5.0).astype(np.float32)
        base.update({'q_cell_frac': q_cell_frac, 'voltage_abs_soc': v_abs_soc, 'v0_abs_soc': v0_abs_soc, 'p2dlite_phase': phase, 'current_stress': current_stress})
    return base


def baseline_chunk_C(feats: Dict[str, Any], s: int, e: int, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    h = cfg.get('hard_cbar_ocp_baseline', {})
    v_z = feats['v_norm'][s:e]
    q_z = feats['q_norm'][s:e]
    vg = float(h.get('voltage_sigmoid_gain', 1.15))
    qg = float(h.get('q_tanh_gain', 1.25))
    wv = float(h.get('voltage_weight', 0.72))
    wq = float(h.get('q_weight', 0.28))
    soc_v = 1.0 / (1.0 + np.exp(-vg * v_z))
    soc_q = 0.5 + 0.5 * np.tanh(qg * q_z)
    phase = np.clip(wv * soc_v + wq * soc_q, 0.0, 1.0)
    centered = 2.0 * phase - 1.0
    a_mid = float(h.get('theta_a_mid', 0.405)); c_mid = float(h.get('theta_c_mid', 0.610))
    a_amp = float(h.get('theta_a_amplitude', 0.245)); c_amp = float(h.get('theta_c_amplitude', 0.245))
    a_min = float(h.get('theta_a_min', 0.02)); a_max = float(h.get('theta_a_max', 0.96))
    c_min = float(h.get('theta_c_min', 0.02)); c_max = float(h.get('theta_c_max', 0.96))
    return np.clip(a_mid + a_amp * centered, a_min, a_max).astype(np.float32), np.clip(c_mid - c_amp * centered, c_min, c_max).astype(np.float32)


def baseline_chunk_D(feats: Dict[str, Any], s: int, e: int, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    p = cfg.get('p2dlite_rg_prior', {})
    g = cfg.get('generator_aligned_baseline', {})
    phase = np.clip(feats['p2dlite_phase'][s:e], 0.0, 1.0)
    v_soc = np.clip(feats['voltage_abs_soc'][s:e], 0.0, 1.0)
    q_frac = feats['q_cell_frac'][s:e]
    blend = float(g.get('voltage_phase_backstop', 0.10))
    phase = np.clip((1.0 - blend) * phase + blend * v_soc, 0.0, 1.0)
    a_min = float(p.get('theta_a_min', 0.0079)); a_max = float(p.get('theta_a_max', 0.8544))
    c_min = float(p.get('theta_c_min', 0.2535)); c_max = float(p.get('theta_c_max', 0.9149))
    base_a = np.clip(a_min + (a_max - a_min) * phase, 0.0, 1.0)
    base_c = np.clip(c_max - (c_max - c_min) * phase, 0.0, 1.0)
    base_a = np.clip(base_a + float(g.get('coulomb_theta_gain_a', 0.045)) * np.tanh(q_frac), 0.0, 1.0)
    base_c = np.clip(base_c - float(g.get('coulomb_theta_gain_c', 0.045)) * np.tanh(q_frac), 0.0, 1.0)
    return base_a.astype(np.float32), base_c.astype(np.float32)


def estimate_csmax(cs_arr: Optional[np.ndarray], theta_arr: np.ndarray, n: int) -> Optional[float]:
    if cs_arr is None:
        return None
    try:
        m = min(n, 20000)
        cs = orient2d(cs_arr, n, 0, m).reshape(-1)
        th = orient2d(theta_arr, n, 0, m).reshape(-1)
        mask = np.isfinite(cs) & np.isfinite(th) & (np.abs(th) > 1e-5)
        if not np.any(mask):
            return None
        ratio = cs[mask] / th[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size < 10:
            return None
        val = float(np.nanmedian(ratio))
        if np.isfinite(val) and val > 0:
            return val
    except Exception:
        return None
    return None


def resolve_npz(row: Dict[str, str], softlabel_root: Path) -> Path:
    raw = row.get('softlabel_npz') or row.get('softlabel_path') or row.get('npz_path') or ''
    if raw:
        p = Path(raw)
        if p.exists():
            return p
    pid = row.get('profile_id') or row.get('cell_uid') or ''
    if pid:
        p2 = softlabel_root / pid / 'solution_softlabels.npz'
        if p2.exists():
            return p2
        # Some manifests may have profiles__Batch-x naming; fallback by suffix match.
        tail = pid.replace('profiles/', '').replace('profiles__', '')
        matches = list(softlabel_root.rglob(f'*{tail}*/solution_softlabels.npz'))
        if matches:
            return matches[0]
    if raw:
        return Path(raw)
    raise FileNotFoundError(f'Cannot resolve softlabel npz for row={row}')


def baseline_audit_model(model_key: str, spec: ModelSpec, manifest_path: Path, cfg: Dict[str, Any], softlabel_root: Path, cache_root: Path, chunk_size: int, limit_profiles: int = 0, sample_stride: int = 1) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Accum]], List[Dict[str, Any]]]:
    rows = read_manifest(manifest_path)
    if limit_profiles and limit_profiles > 0:
        rows = rows[:limit_profiles]
    by_profile: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    group_acc: Dict[str, Dict[str, Accum]] = {}

    def add_group(group: str, metric: str, true: np.ndarray, pred: np.ndarray):
        group_acc.setdefault(group, {}).setdefault(metric, Accum()).update(true, pred)

    for i, row in enumerate(rows):
        meta = {
            'model': model_key,
            'profile_id': row.get('profile_id', ''),
            'batch': row.get('batch', ''),
            'battery': row.get('battery', ''),
            'split': row.get('split', 'eval'),
            'reason': row.get('reason', ''),
            'protocol': row.get('protocol', ''),
        }
        try:
            npz = resolve_npz(row, softlabel_root)
            arr = load_mmap_arrays(npz, cache_root)
            t = np.asarray(arr['t'], dtype=np.float32).reshape(-1)
            I = np.asarray(arr['I'], dtype=np.float32).reshape(-1)
            V = np.asarray(arr['V'], dtype=np.float32).reshape(-1)
            n = len(t)
            feats = compute_global_features(t, I, V, spec.variant, cfg)
            th_a_shape = arr['theta_a'].shape
            th_c_shape = arr['theta_c'].shape
            csmax_a = estimate_csmax(arr.get('cs_a'), arr['theta_a'], n)
            csmax_c = estimate_csmax(arr.get('cs_c'), arr['theta_c'], n)
            accs = {m: Accum() for m in ['theta_a_mean_base', 'theta_c_mean_base']}
            if csmax_a is not None:
                accs['cs_a_mean_base'] = Accum()
            if csmax_c is not None:
                accs['cs_c_mean_base'] = Accum()
            for s in range(0, n, int(chunk_size)):
                e = min(n, s + int(chunk_size))
                if spec.variant == 'D':
                    ba, bc = baseline_chunk_D(feats, s, e, cfg)
                else:
                    ba, bc = baseline_chunk_C(feats, s, e, cfg)
                true_ta = orient2d(arr['theta_a'], n, s, e)
                true_tc = orient2d(arr['theta_c'], n, s, e)
                true_ta_m = np.mean(true_ta, axis=1)
                true_tc_m = np.mean(true_tc, axis=1)
                if sample_stride > 1:
                    sl = slice(None, None, int(sample_stride))
                    true_ta_m = true_ta_m[sl]; true_tc_m = true_tc_m[sl]; ba = ba[sl]; bc = bc[sl]
                accs['theta_a_mean_base'].update(true_ta_m, ba)
                accs['theta_c_mean_base'].update(true_tc_m, bc)
                add_group('ALL', 'theta_a_mean_base', true_ta_m, ba)
                add_group('ALL', 'theta_c_mean_base', true_tc_m, bc)
                add_group(str(meta['split']), 'theta_a_mean_base', true_ta_m, ba)
                add_group(str(meta['split']), 'theta_c_mean_base', true_tc_m, bc)
                add_group(f"batch:{meta['batch']}", 'theta_a_mean_base', true_ta_m, ba)
                add_group(f"batch:{meta['batch']}", 'theta_c_mean_base', true_tc_m, bc)
                if meta.get('protocol'):
                    add_group(f"protocol:{meta['protocol']}", 'theta_a_mean_base', true_ta_m, ba)
                    add_group(f"protocol:{meta['protocol']}", 'theta_c_mean_base', true_tc_m, bc)
                if csmax_a is not None:
                    accs['cs_a_mean_base'].update(true_ta_m * csmax_a, ba * csmax_a)
                    add_group('ALL', 'cs_a_mean_base', true_ta_m * csmax_a, ba * csmax_a)
                    add_group(str(meta['split']), 'cs_a_mean_base', true_ta_m * csmax_a, ba * csmax_a)
                if csmax_c is not None:
                    accs['cs_c_mean_base'].update(true_tc_m * csmax_c, bc * csmax_c)
                    add_group('ALL', 'cs_c_mean_base', true_tc_m * csmax_c, bc * csmax_c)
                    add_group(str(meta['split']), 'cs_c_mean_base', true_tc_m * csmax_c, bc * csmax_c)
            out = dict(meta)
            out['n_time'] = n
            out['csmax_a_est'] = csmax_a if csmax_a is not None else ''
            out['csmax_c_est'] = csmax_c if csmax_c is not None else ''
            for name, ac in accs.items():
                out.update(ac.row(name))
            by_profile.append(out)
            if (i + 1) % 5 == 0 or i == 0:
                print(f'[P5K-E2] {model_key}: baseline-only audited {i+1}/{len(rows)} profiles', flush=True)
        except Exception as e:
            failures.append({**meta, 'error': repr(e), 'traceback': traceback.format_exc(limit=3)})
            print(f'[P5K-E2] FAIL {model_key} {meta.get("profile_id")}: {repr(e)}', flush=True)
    return by_profile, group_acc, failures


def group_rows(group_acc: Dict[str, Dict[str, Accum]], model_key: str) -> List[Dict[str, Any]]:
    rows = []
    for group, metrics in sorted(group_acc.items()):
        r: Dict[str, Any] = {'model': model_key, 'group': group}
        for m, ac in sorted(metrics.items()):
            r.update(ac.row(m))
        rows.append(r)
    return rows


def table(rows: List[List[Any]]) -> str:
    if not rows:
        return ''
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for ri, row in enumerate(rows):
        line = '| ' + ' | '.join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + ' |'
        lines.append(line)
        if ri == 0:
            lines.append('| ' + ' | '.join('-' * widths[i] for i in range(len(row))) + ' |')
    return '\n'.join(lines)


def fmt(x: Any) -> str:
    if x is None or x == '':
        return ''
    try:
        f = float(x)
        if not np.isfinite(f): return 'nan'
        if abs(f) >= 10000 or (abs(f) < 1e-4 and f != 0): return f'{f:.6e}'
        return f'{f:.6f}'
    except Exception:
        return str(x)


def collect_provenance(specs: Dict[str, ModelSpec]) -> Dict[str, Any]:
    prov: Dict[str, Any] = {}
    for key, spec in specs.items():
        model_dir = first_existing(spec.model_dir_candidates)
        manifest = first_existing(spec.manifest_candidates)
        scorecard = first_existing(spec.scorecard_candidates)
        split = first_existing(spec.split_metrics_candidates)
        profile = first_existing(spec.profile_metrics_candidates)
        batch = first_existing(spec.batch_metrics_candidates)
        protocol = first_existing(spec.protocol_metrics_candidates)
        config_path = first_existing(spec.config_candidates)
        train_summary = locate_train_summary(model_dir, key)
        train_audit = locate_train_audit(model_dir, key)
        ckpt = locate_checkpoint(model_dir)
        cfg, ckpt_meta = load_checkpoint_config(ckpt, config_path)
        audit_obj = safe_read_json(train_audit) if train_audit else None
        train_used_counts: Dict[str, int] = {}
        forbidden_counts: Dict[str, int] = {}
        sidecar_counts: Dict[str, int] = {}
        if isinstance(audit_obj, dict):
            for r in audit_obj.get('rows', []) if isinstance(audit_obj.get('rows', []), list) else []:
                for k in r.get('training_used_keys', []) or []:
                    train_used_counts[k] = train_used_counts.get(k, 0) + 1
                for k in r.get('training_forbidden_keys_not_loaded', []) or []:
                    forbidden_counts[k] = forbidden_counts.get(k, 0) + 1
                if isinstance(r.get('generator_sidecar_audit'), dict):
                    for sk, sv in r['generator_sidecar_audit'].items():
                        if sv not in (None, '', False, [], {}):
                            sidecar_counts[sk] = sidecar_counts.get(sk, 0) + 1
        prov[key] = {
            'label': spec.label,
            'stage_run_dir': str(spec.stage_run_dir),
            'resolved_paths': {
                'model_dir': str(model_dir) if model_dir else None,
                'manifest': str(manifest) if manifest else None,
                'train_summary': str(train_summary) if train_summary else None,
                'train_audit': str(train_audit) if train_audit else None,
                'checkpoint': str(ckpt) if ckpt else None,
                'scorecard': str(scorecard) if scorecard else None,
                'split_metrics': str(split) if split else None,
                'profile_metrics': str(profile) if profile else None,
                'batch_metrics': str(batch) if batch else None,
                'protocol_metrics': str(protocol) if protocol else None,
                'config': str(config_path) if config_path else None,
            },
            'file_info': {
                'manifest': file_info(manifest) if manifest else {'exists': False},
                'train_summary': file_info(train_summary) if train_summary else {'exists': False},
                'train_audit': file_info(train_audit) if train_audit else {'exists': False},
                'checkpoint': file_info(ckpt) if ckpt else {'exists': False},
                'scorecard': file_info(scorecard) if scorecard else {'exists': False},
                'split_metrics': file_info(split) if split else {'exists': False},
                'profile_metrics': file_info(profile) if profile else {'exists': False},
                'config': file_info(config_path) if config_path else {'exists': False},
            },
            'checkpoint_meta': ckpt_meta,
            'config_fingerprint': sha1_text(json.dumps(cfg, sort_keys=True, ensure_ascii=False))[:16] if cfg else None,
            'config_top_keys': sorted(list(cfg.keys())) if isinstance(cfg, dict) else [],
            'training_input_audit_counts': {
                'training_used_keys_counts': train_used_counts,
                'forbidden_keys_counts': forbidden_counts,
                'generator_sidecar_related_counts': sidecar_counts,
            },
        }
    return prov


def load_final_metric_context(specs: Dict[str, ModelSpec]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    for key, spec in specs.items():
        split = first_existing(spec.split_metrics_candidates)
        score = first_existing(spec.scorecard_candidates)
        eval_row = parse_csv_first_by_group(split, 'eval')
        train_row = parse_csv_first_by_group(split, 'train')
        score_obj = safe_read_json(score) if score else None
        ctx[key] = {
            'eval': eval_row,
            'train': train_row,
            'scorecard_global': score_obj.get('global_metrics_weighted') if isinstance(score_obj, dict) else None,
            'operational_status': score_obj.get('operational_status') if isinstance(score_obj, dict) else None,
            'profile_count_evaluated': score_obj.get('profile_count_evaluated') or score_obj.get('profile_count_requested') if isinstance(score_obj, dict) else None,
            'failure_count': score_obj.get('failure_count') if isinstance(score_obj, dict) else None,
        }
    return ctx


def report_markdown(out_report: Path, args: argparse.Namespace, provenance: Dict[str, Any], final_ctx: Dict[str, Any], baseline_group_rows: List[Dict[str, Any]], by_profile_path: Path, failures: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append('# D16-P5K-E2 Provenance + Baseline-Only Audit Report')
    lines.append('')
    lines.append('This is a **no-training** audit. It does not modify checkpoints. It checks P5K-C/P5K-D provenance and recomputes hard-baseline-only exact metrics against P2Dlite-RG soft labels.')
    lines.append('')
    lines.append('## 0. Run metadata')
    lines.append(f'- output_file: `{out_report}`')
    lines.append(f'- cache_root: `{args.cache_root}`')
    lines.append(f'- project_root: `{args.project_root}`')
    lines.append(f'- softlabel_root: `{args.softlabel_root}`')
    lines.append(f'- models: `{args.models}`')
    lines.append(f'- limit_profiles: `{args.limit_profiles}`')
    lines.append(f'- chunk_size: `{args.chunk_size}`')
    lines.append(f'- sample_stride: `{args.sample_stride}`')
    lines.append(f'- by_profile_csv: `{by_profile_path}`')
    lines.append('')

    lines.append('## 1. Provenance resolution')
    p_rows = [['item', 'P5K-C', 'P5K-D']]
    for item in ['model_dir', 'manifest', 'train_summary', 'train_audit', 'checkpoint', 'scorecard', 'split_metrics', 'profile_metrics', 'config']:
        row = [item]
        for key in ['P5K-C', 'P5K-D']:
            rp = provenance.get(key, {}).get('resolved_paths', {}).get(item)
            exists = False
            if rp:
                exists = Path(rp).exists()
            row.append(('OK ' if exists else 'MISS ') + (rp or ''))
        p_rows.append(row)
    lines.append(table(p_rows))
    lines.append('')
    lines.append('### Checkpoint/schema fingerprints')
    fp_rows = [['model', 'checkpoint_exists', 'checkpoint_sha1_16', 'state_key_count', 'state_schema_sha1_16', 'config_fingerprint', 'training_used_keys', 'forbidden_keys']] 
    for key in ['P5K-C', 'P5K-D']:
        fi = provenance[key]['file_info']['checkpoint']
        cm = provenance[key].get('checkpoint_meta', {})
        audit = provenance[key].get('training_input_audit_counts', {})
        fp_rows.append([
            key,
            str(fi.get('exists')),
            str(fi.get('sha1', ''))[:16],
            cm.get('state_key_count', ''),
            cm.get('state_schema_sha1_16', ''),
            provenance[key].get('config_fingerprint', ''),
            audit.get('training_used_keys_counts', {}),
            audit.get('forbidden_keys_counts', {}),
        ])
    lines.append(table(fp_rows))
    lines.append('')
    lines.append('**Interpretation:** P5K-E had a path bug if it reports P5K-D checkpoint/audit missing while this table resolves `model_generator_aligned_hard_cbar_ocp_residual`. This report is the provenance correction.')
    lines.append('')

    lines.append('## 2. Final scorecard context from existing evaluations')
    sc_rows = [['model', 'split', 'phis_c_mae', 'phis_c_r2', 'theta_a_mean_mae', 'theta_a_mean_bias', 'theta_a_mean_r2', 'theta_c_mean_mae', 'theta_c_mean_bias', 'theta_c_mean_r2']]
    for key in ['P5K-C', 'P5K-D']:
        for split in ['eval', 'train']:
            r = final_ctx.get(key, {}).get(split)
            sc_rows.append([key, split, fmt(metric_float(r, 'phis_c_mae')), fmt(metric_float(r, 'phis_c_r2')), fmt(metric_float(r, 'theta_a_mean_mae')), fmt(metric_float(r, 'theta_a_mean_bias')), fmt(metric_float(r, 'theta_a_mean_r2')), fmt(metric_float(r, 'theta_c_mean_mae')), fmt(metric_float(r, 'theta_c_mean_bias')), fmt(metric_float(r, 'theta_c_mean_r2'))])
    lines.append(table(sc_rows))
    lines.append('')

    lines.append('## 3. Hard-baseline-only exact metrics')
    # Only report ALL/eval/train groups, not every batch/protocol.
    bg_by = { (r['model'], r['group']): r for r in baseline_group_rows }
    b_rows = [['model', 'group', 'theta_a_base_mae', 'theta_a_base_bias', 'theta_a_base_r2', 'theta_c_base_mae', 'theta_c_base_bias', 'theta_c_base_r2', 'cs_a_base_r2', 'cs_c_base_r2']]
    for key in ['P5K-C', 'P5K-D']:
        for group in ['ALL', 'eval', 'train']:
            r = bg_by.get((key, group), {})
            b_rows.append([
                key, group,
                fmt(r.get('theta_a_mean_base_mae')), fmt(r.get('theta_a_mean_base_bias')), fmt(r.get('theta_a_mean_base_r2')),
                fmt(r.get('theta_c_mean_base_mae')), fmt(r.get('theta_c_mean_base_bias')), fmt(r.get('theta_c_mean_base_r2')),
                fmt(r.get('cs_a_mean_base_r2')), fmt(r.get('cs_c_mean_base_r2')),
            ])
    lines.append(table(b_rows))
    lines.append('')

    lines.append('## 4. Final-vs-baseline diagnosis, eval split')
    d_rows = [['model', 'final_theta_a_mae', 'base_theta_a_mae', 'final-base', 'final_theta_a_r2', 'base_theta_a_r2', 'final_theta_c_mae', 'base_theta_c_mae', 'final-base', 'final_theta_c_r2', 'base_theta_c_r2']]
    verdict_notes = []
    for key in ['P5K-C', 'P5K-D']:
        fr = final_ctx.get(key, {}).get('eval')
        br = bg_by.get((key, 'eval'), {})
        fa = metric_float(fr, 'theta_a_mean_mae'); fc = metric_float(fr, 'theta_c_mean_mae')
        ba = br.get('theta_a_mean_base_mae'); bc = br.get('theta_c_mean_base_mae')
        fra = metric_float(fr, 'theta_a_mean_r2'); frc = metric_float(fr, 'theta_c_mean_r2')
        bra = br.get('theta_a_mean_base_r2'); brc = br.get('theta_c_mean_base_r2')
        d_rows.append([key, fmt(fa), fmt(ba), fmt((fa - ba) if fa is not None and ba is not None else None), fmt(fra), fmt(bra), fmt(fc), fmt(bc), fmt((fc - bc) if fc is not None and bc is not None else None), fmt(frc), fmt(brc)])
        try:
            if ba is not None and bc is not None and fa is not None and fc is not None:
                if ba > 0.25 or bc > 0.25 or (bra is not None and bra < 0) or (brc is not None and brc < 0):
                    verdict_notes.append(f'- {key}: hard baseline itself is poor on eval split; prioritize theta0/OCP/coulomb scale initialization rather than more residual training.')
                elif fa > ba + 0.03 or fc > bc + 0.03:
                    verdict_notes.append(f'- {key}: residual network worsens a relatively better baseline; inspect residual bounds/loss weights.')
                else:
                    verdict_notes.append(f'- {key}: baseline and final are close; residual cannot overcome baseline limitations.')
        except Exception:
            pass
    lines.append(table(d_rows))
    lines.append('')
    lines.append('### Automatic diagnosis notes')
    if verdict_notes:
        lines.extend(verdict_notes)
    else:
        lines.append('- No automatic diagnosis available; inspect failures and by-profile CSV.')
    lines.append('')

    lines.append('## 5. Worst baseline-only profiles')
    try:
        # Read generated by-profile rows and rank by baseline theta mean MAE sum.
        with by_profile_path.open('r', newline='', encoding='utf-8') as f:
            prof_rows = list(csv.DictReader(f))
        ranked = sorted(prof_rows, key=lambda r: float(r.get('theta_a_mean_base_mae') or 0) + float(r.get('theta_c_mean_base_mae') or 0), reverse=True)[:15]
        wr = [['model', 'profile_id', 'split', 'theta_a_base_mae', 'theta_a_base_r2', 'theta_c_base_mae', 'theta_c_base_r2']]
        for r in ranked:
            wr.append([r.get('model',''), r.get('profile_id',''), r.get('split',''), fmt(r.get('theta_a_mean_base_mae')), fmt(r.get('theta_a_mean_base_r2')), fmt(r.get('theta_c_mean_base_mae')), fmt(r.get('theta_c_mean_base_r2'))])
        lines.append(table(wr))
    except Exception as e:
        lines.append(f'Could not render worst profiles: `{repr(e)}`')
    lines.append('')

    lines.append('## 6. Failure log')
    lines.append(f'- baseline_audit_failure_count: `{len(failures)}`')
    if failures:
        f_rows = [['model', 'profile_id', 'error']]
        for f in failures[:20]:
            f_rows.append([f.get('model',''), f.get('profile_id',''), f.get('error','')])
        lines.append(table(f_rows))
    lines.append('')

    lines.append('## 7. Recommended next action')
    lines.append('- If P5K-D provenance is now resolved but its baseline-only metrics are already poor, do **not** continue P5K-D training. Treat generator-aligned strong baseline as failed.')
    lines.append('- If P5K-C baseline-only is much better than its final train-hard-probe metrics, inspect residual behavior on hard probes; otherwise target profile-level theta0/OCP initialization.')
    lines.append('- Generator/prior information should be used as weak audit/initializer/no-regression guard, not as a strong absolute output anchor.')
    lines.append('')
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-E2 provenance + baseline-only audit. No training.')
    ap.add_argument('--project-root', default=r'C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1')
    ap.add_argument('--cache-root', default=r'E:\XJTU battery dataset\_gv1_cache')
    ap.add_argument('--softlabel-root', default=r'E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL')
    ap.add_argument('--out-dir', default=r'E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke2_provenance_baseline_only_audit')
    ap.add_argument('--models', default='P5K-C,P5K-D')
    ap.add_argument('--limit-profiles', type=int, default=0, help='0 means all profiles. For smoke use 2 or 4.')
    ap.add_argument('--chunk-size', type=int, default=200000)
    ap.add_argument('--sample-stride', type=int, default=1, help='Use >1 only for fast approximate smoke. Full audit should use 1.')
    args = ap.parse_args()

    project_root = Path(args.project_root)
    cache_root = Path(args.cache_root)
    softlabel_root = Path(args.softlabel_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mmap_cache = out_dir / 'mmap_cache_short'
    mmap_cache.mkdir(parents=True, exist_ok=True)

    model_specs_all = build_model_specs(cache_root, project_root)
    selected_keys = [m.strip() for m in args.models.split(',') if m.strip()]
    specs = {k: model_specs_all[k] for k in selected_keys if k in model_specs_all}
    if not specs:
        raise ValueError(f'No valid models selected from {args.models}; available={sorted(model_specs_all)}')

    provenance = collect_provenance(specs)
    final_ctx = load_final_metric_context(specs)
    all_profile_rows: List[Dict[str, Any]] = []
    all_group_rows: List[Dict[str, Any]] = []
    all_failures: List[Dict[str, Any]] = []

    for key, spec in specs.items():
        manifest_path_str = provenance[key]['resolved_paths']['manifest']
        if not manifest_path_str:
            all_failures.append({'model': key, 'profile_id': '__manifest__', 'error': 'manifest not found'})
            continue
        cfg = {}
        cfg_path_str = provenance[key]['resolved_paths'].get('config')
        ckpt_path_str = provenance[key]['resolved_paths'].get('checkpoint')
        ckpt_path = Path(ckpt_path_str) if ckpt_path_str else None
        cfg_path = Path(cfg_path_str) if cfg_path_str else None
        cfg, _ = load_checkpoint_config(ckpt_path, cfg_path)
        if not cfg and cfg_path and cfg_path.exists():
            cfg = read_json(cfg_path)
        print(f'[P5K-E2] baseline-only audit start: {key}', flush=True)
        prof_rows, groups, fails = baseline_audit_model(key, spec, Path(manifest_path_str), cfg, softlabel_root, mmap_cache / key.replace('-', '_'), int(args.chunk_size), int(args.limit_profiles), max(1, int(args.sample_stride)))
        all_profile_rows.extend(prof_rows)
        all_group_rows.extend(group_rows(groups, key))
        all_failures.extend(fails)

    by_profile_path = out_dir / 'D16_P5K_E2_BASELINE_ONLY_BY_PROFILE.csv'
    by_group_path = out_dir / 'D16_P5K_E2_BASELINE_ONLY_BY_GROUP.csv'
    failures_path = out_dir / 'D16_P5K_E2_FAILURES.json'
    prov_path = out_dir / 'D16_P5K_E2_PROVENANCE_SUMMARY.json'
    report_path = out_dir / 'D16_P5K_E2_PROVENANCE_BASELINE_AUDIT_REPORT.md'

    write_csv(all_profile_rows, by_profile_path)
    write_csv(all_group_rows, by_group_path)
    write_json(all_failures, failures_path)
    write_json({'provenance': provenance, 'final_context': final_ctx}, prov_path)
    report_markdown(report_path, args, provenance, final_ctx, all_group_rows, by_profile_path, all_failures)

    print('[P5K-E2] wrote:', report_path, flush=True)
    print('[P5K-E2] by_profile:', by_profile_path, flush=True)
    print('[P5K-E2] by_group:', by_group_path, flush=True)
    print('[P5K-E2] failures:', failures_path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
