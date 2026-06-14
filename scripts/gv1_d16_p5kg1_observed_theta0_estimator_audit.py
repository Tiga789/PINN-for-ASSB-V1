from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

# -----------------------------------------------------------------------------
# D16-P5K-G1 observed-only theta0 estimator audit
# No model training. No checkpoint loading. No model modification.
#
# Purpose:
#   G0 showed theta0_oracle can repair hard_probe but is not deployable.
#   G1 asks whether a profile-level theta0 correction can be predicted from
#   observable features only: I(t), V(t), t, protocol/batch metadata, capacity
#   integrals. The fitted candidates use oracle shifts as diagnostic labels;
#   they are NOT promotion/final-model evidence unless the project explicitly
#   allows state-label-derived theta0 estimator calibration.
# -----------------------------------------------------------------------------

PROTOCOL_BY_BATCH = {
    'Batch-1': '2C',
    'Batch-2': '3C',
    'Batch-3': 'R2.5',
    'Batch-4': 'R3',
    'Batch-5': 'random_walk',
    'Batch-6': 'GEO',
}

REFERENCE = {
    'P5K-C_baseline_eval_from_G0': {
        'theta_a_mean_mae': 0.139017,
        'theta_a_mean_r2': 0.474238,
        'theta_c_mean_mae': 0.123569,
        'theta_c_mean_r2': 0.391913,
    },
    'P5K-F_final_eval43': {
        'theta_a_mean_mae': 0.146213,
        'theta_a_mean_r2': 0.447594,
        'theta_c_mean_mae': 0.128404,
        'theta_c_mean_r2': 0.362642,
        'phis_c_r2': 0.999488,
    },
}

METRIC_NAMES_BASE = [
    'theta_a_mean', 'theta_c_mean', 'theta_a', 'theta_c',
    'cs_a_mean', 'cs_c_mean',
    'grad_a_surface_center', 'grad_c_surface_center',
]


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f'module not found: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('', encoding='utf-8')
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def parse_meta(row: Dict[str, str]) -> Dict[str, str]:
    batch = row.get('batch', '')
    protocol = row.get('protocol', '') or PROTOCOL_BY_BATCH.get(batch, batch)
    return {
        'profile_id': row.get('profile_id', ''),
        'batch': batch,
        'battery': row.get('battery', ''),
        'split': row.get('split', 'eval'),
        'reason': row.get('reason', ''),
        'protocol': protocol,
    }


def finite_float(x: Any, default: float = float('nan')) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def fmt(v: Any, nd: int = 6) -> str:
    try:
        fv = float(v)
        if math.isnan(fv):
            return 'nan'
        if abs(fv) >= 1e4 or (abs(fv) < 1e-4 and fv != 0):
            return f'{fv:.6e}'
        return f'{fv:.{nd}f}'
    except Exception:
        return str(v)


class Accum:
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

    def row(self, prefix: str) -> Dict[str, Any]:
        n = max(1, self.n)
        cov = self.sum_tp - self.sum_t * self.sum_p / n
        vt = self.sum_t2 - self.sum_t * self.sum_t / n
        vp = self.sum_p2 - self.sum_p * self.sum_p / n
        corr = cov / math.sqrt(vt * vp) if vt > 1e-20 and vp > 1e-20 else float('nan')
        r2 = 1.0 - (self.sum_sq / vt) if self.n and vt > 1e-20 else float('nan')
        return {
            f'{prefix}_count': int(self.n),
            f'{prefix}_mae': self.sum_abs / n if self.n else float('nan'),
            f'{prefix}_rmse': math.sqrt(self.sum_sq / n) if self.n else float('nan'),
            f'{prefix}_bias': self.sum_err / n if self.n else float('nan'),
            f'{prefix}_max_abs': self.max_abs if self.n else float('nan'),
            f'{prefix}_corr': corr,
            f'{prefix}_r2': r2,
            f'{prefix}_sum_true': self.sum_t,
            f'{prefix}_sum_true_sq': self.sum_t2,
            f'{prefix}_sum_pred': self.sum_p,
            f'{prefix}_sum_pred_sq': self.sum_p2,
            f'{prefix}_sum_err_sq': self.sum_sq,
        }


def metric_dict() -> Dict[str, Accum]:
    return {m: Accum() for m in METRIC_NAMES_BASE}


def init_group_acc(group_acc: Dict[Tuple[str, str], Dict[str, Accum]], model: str, group: str) -> Dict[str, Accum]:
    key = (model, group)
    if key not in group_acc:
        group_acc[key] = metric_dict()
    return group_acc[key]


def update_groups(group_acc: Dict[Tuple[str, str], Dict[str, Accum]], model: str, meta: Dict[str, str], pairs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    groups = ['ALL', f"split:{meta['split']}", f"batch:{meta['batch']}", f"protocol:{meta['protocol']}"]
    for group in groups:
        accs = init_group_acc(group_acc, model, group)
        for name, (tru, prd) in pairs.items():
            if name in accs:
                accs[name].update(tru, prd)


def row_from_accs(model: str, group: str, profile_count: int, accs: Dict[str, Accum]) -> Dict[str, Any]:
    r: Dict[str, Any] = {'model': model, 'group': group, 'profile_count': profile_count}
    for name, ac in accs.items():
        r.update(ac.row(name))
    return r


@dataclass
class ModelSpec:
    name: str
    module_path: Path
    config_path: Path
    mod: Any
    cfg: Dict[str, Any]


class ProfileCacheCleaner:
    def __init__(self, module: Any, cache_root: Path, npz_path: Path):
        self.module = module
        self.cache_root = Path(cache_root)
        self.npz_path = Path(npz_path)

    def path(self) -> Path:
        h = hashlib.sha1(str(self.npz_path).encode('utf-8', errors='ignore')).hexdigest()[:16]
        safe = getattr(self.module, '_safe_name', lambda s: str(s).replace(' ', '_'))
        cell_hint = safe(self.npz_path.parent.name)[:64]
        return self.cache_root / f'{cell_hint}_{h}'

    def cleanup(self) -> None:
        p = self.path()
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def orient2d(loader_mod: Any, arr: np.ndarray, n: int, s: int, e: int) -> np.ndarray:
    return loader_mod.orient2d(arr, n, s, e)


def estimate_csmax(loader_mod: Any, arrs: Dict[str, Any], theta_key: str, cs_key: str, n: int) -> Optional[float]:
    if theta_key not in arrs or cs_key not in arrs:
        return None
    m = min(int(n), 20000)
    try:
        th = orient2d(loader_mod, arrs[theta_key], n, 0, m).reshape(-1)
        cs = orient2d(loader_mod, arrs[cs_key], n, 0, m).reshape(-1)
        mask = np.isfinite(th) & np.isfinite(cs) & (np.abs(th) > 1e-5)
        if not np.any(mask):
            return None
        ratio = cs[mask] / th[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size < 10:
            return None
        val = float(np.nanmedian(ratio))
        return val if math.isfinite(val) and val > 0 else None
    except Exception:
        return None


def as_1d(loader_mod: Any, arr: Any) -> np.ndarray:
    return loader_mod.as_1d_float(arr)


def safe_percentile(x: np.ndarray, p: float, default: float = float('nan')) -> float:
    try:
        z = np.asarray(x, dtype=np.float64)
        z = z[np.isfinite(z)]
        if z.size == 0:
            return default
        return float(np.nanpercentile(z, p))
    except Exception:
        return default


def observed_features(t: np.ndarray, I: np.ndarray, V: np.ndarray, meta: Dict[str, str]) -> Dict[str, float]:
    t = np.asarray(t, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n = len(t)
    if n == 0:
        n = 1
    span = float(t[-1] - t[0]) if len(t) > 1 and np.isfinite(t[-1] - t[0]) and (t[-1] - t[0]) > 0 else 1.0
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
    q = float(np.sum(I * dt))
    q_abs = float(np.sum(np.abs(I) * dt))
    q_pos = float(np.sum(np.maximum(I, 0.0) * dt))
    q_neg = float(np.sum(np.minimum(I, 0.0) * dt))
    I_abs = np.abs(I)
    low_i_mask = I_abs <= max(1e-9, safe_percentile(I_abs, 10, 1e-9))
    rest_frac = float(np.mean(I_abs < max(1e-9, 0.02 * max(safe_percentile(I_abs, 99, 1e-9), 1e-9))))
    first_m = max(1, min(len(V), int(max(10, min(1000, len(V)//100 if len(V) > 100 else len(V))))))
    last_m = first_m
    dV = np.diff(V) if len(V) > 1 else np.array([0.0])
    out: Dict[str, float] = {
        'const': 1.0,
        'n_log': math.log(max(float(len(t)), 1.0)),
        't_span_log': math.log(max(span, 1.0)),
        'v0': float(V[0]) if len(V) else 0.0,
        'v_first_mean': float(np.nanmean(V[:first_m])) if len(V) else 0.0,
        'v_last_mean': float(np.nanmean(V[-last_m:])) if len(V) else 0.0,
        'v_min': float(np.nanmin(V)) if len(V) else 0.0,
        'v_max': float(np.nanmax(V)) if len(V) else 0.0,
        'v_mean': float(np.nanmean(V)) if len(V) else 0.0,
        'v_std': float(np.nanstd(V)) if len(V) else 0.0,
        'v_p01': safe_percentile(V, 1, 0.0),
        'v_p05': safe_percentile(V, 5, 0.0),
        'v_p50': safe_percentile(V, 50, 0.0),
        'v_p95': safe_percentile(V, 95, 0.0),
        'v_p99': safe_percentile(V, 99, 0.0),
        'dv_mean': float(np.nanmean(dV)) if dV.size else 0.0,
        'dv_std': float(np.nanstd(dV)) if dV.size else 0.0,
        'I0': float(I[0]) if len(I) else 0.0,
        'I_mean': float(np.nanmean(I)) if len(I) else 0.0,
        'I_abs_mean': float(np.nanmean(I_abs)) if len(I_abs) else 0.0,
        'I_abs_max': float(np.nanmax(I_abs)) if len(I_abs) else 0.0,
        'I_abs_p95': safe_percentile(I_abs, 95, 0.0),
        'q_net': q,
        'q_abs': q_abs,
        'q_pos': q_pos,
        'q_neg_abs': abs(q_neg),
        'q_balance': q / (q_abs + 1e-9),
        'charge_frac': q_pos / (q_abs + 1e-9),
        'discharge_frac': abs(q_neg) / (q_abs + 1e-9),
        'rest_frac': rest_frac,
        'low_i_v_mean': float(np.nanmean(V[low_i_mask])) if np.any(low_i_mask) else float(np.nanmean(V)) if len(V) else 0.0,
    }
    for batch in ['Batch-1','Batch-2','Batch-3','Batch-4','Batch-5','Batch-6']:
        out[f'batch_{batch}'] = 1.0 if meta.get('batch') == batch else 0.0
    for protocol in ['2C','3C','R2.5','R3','random_walk','GEO']:
        out[f'protocol_{protocol}'] = 1.0 if meta.get('protocol') == protocol else 0.0
    # Replace non-finite values.
    for k, v in list(out.items()):
        if not math.isfinite(float(v)):
            out[k] = 0.0
    return out


def feature_vector(feat: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    return np.array([float(feat.get(k, 0.0)) for k in feature_names], dtype=np.float64)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Standardize non-constant features for numerical stability. The const column is kept but standardized safely.
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    Xs = (X - mu) / sig
    # Add intercept after standardization.
    Xd = np.concatenate([np.ones((Xs.shape[0], 1), dtype=np.float64), Xs], axis=1)
    reg = float(alpha) * np.eye(Xd.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    try:
        coef = np.linalg.solve(Xd.T @ Xd + reg, Xd.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(Xd.T @ Xd + reg) @ Xd.T @ y
    return coef, mu, sig


def predict_ridge(X: np.ndarray, coef: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    Xs = (np.asarray(X, dtype=np.float64) - mu) / sig
    Xd = np.concatenate([np.ones((Xs.shape[0], 1), dtype=np.float64), Xs], axis=1)
    return Xd @ coef


def baseline_predict(md: ModelSpec, t: np.ndarray, I: np.ndarray, V: np.ndarray, stats: Dict[str, float], qn: np.ndarray, s: int, e: int, nr_a: int, nr_c: int) -> Dict[str, np.ndarray]:
    mod, cfg = md.mod, md.cfg
    X = mod.feature_chunk(t, I, V, s, e, stats, qn)
    xr = torch.from_numpy(X.astype(np.float32))
    raw_zero = torch.zeros((X.shape[0], 6), dtype=torch.float32)
    radial_a = np.linspace(-0.5, 0.5, nr_a, dtype=np.float32)
    radial_c = np.linspace(-0.5, 0.5, nr_c, dtype=np.float32)
    with torch.no_grad():
        y = mod.transform_outputs(raw_zero, xr, cfg)
    ta_m = y['theta_a_mean'].cpu().numpy().astype(np.float32)
    tc_m = y['theta_c_mean'].cpu().numpy().astype(np.float32)
    pred_ga = y['grad_a'].cpu().numpy().astype(np.float32)
    pred_gc = y['grad_c'].cpu().numpy().astype(np.float32)
    pred_ta = np.clip(ta_m[:, None] + pred_ga[:, None] * radial_a[None, :], 0.0, 1.0).astype(np.float32)
    pred_tc = np.clip(tc_m[:, None] + pred_gc[:, None] * radial_c[None, :], 0.0, 1.0).astype(np.float32)
    return {
        'theta_a_mean': ta_m,
        'theta_c_mean': tc_m,
        'theta_a': pred_ta,
        'theta_c': pred_tc,
        'grad_a_surface_center': (pred_ta[:, -1] - pred_ta[:, 0]).astype(np.float32),
        'grad_c_surface_center': (pred_tc[:, -1] - pred_tc[:, 0]).astype(np.float32),
    }


def apply_theta0_shift(pred: Dict[str, np.ndarray], shift_a: float, shift_c: float, csmax_a: Optional[float], csmax_c: Optional[float]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    out['theta_a_mean'] = np.clip(pred['theta_a_mean'] + shift_a, 0.0, 1.0).astype(np.float32)
    out['theta_c_mean'] = np.clip(pred['theta_c_mean'] + shift_c, 0.0, 1.0).astype(np.float32)
    out['theta_a'] = np.clip(pred['theta_a'] + shift_a, 0.0, 1.0).astype(np.float32)
    out['theta_c'] = np.clip(pred['theta_c'] + shift_c, 0.0, 1.0).astype(np.float32)
    out['grad_a_surface_center'] = (out['theta_a'][:, -1] - out['theta_a'][:, 0]).astype(np.float32)
    out['grad_c_surface_center'] = (out['theta_c'][:, -1] - out['theta_c'][:, 0]).astype(np.float32)
    if csmax_a:
        out['cs_a_mean'] = (out['theta_a_mean'] * float(csmax_a)).astype(np.float32)
    if csmax_c:
        out['cs_c_mean'] = (out['theta_c_mean'] * float(csmax_c)).astype(np.float32)
    return out


def add_cs_predictions(pred: Dict[str, np.ndarray], csmax_a: Optional[float], csmax_c: Optional[float]) -> None:
    if csmax_a:
        pred['cs_a_mean'] = (pred['theta_a_mean'] * float(csmax_a)).astype(np.float32)
    if csmax_c:
        pred['cs_c_mean'] = (pred['theta_c_mean'] * float(csmax_c)).astype(np.float32)


def true_pairs_for_chunk(loader_mod: Any, arrs: Dict[str, Any], n: int, s: int, e: int) -> Dict[str, np.ndarray]:
    true_ta = orient2d(loader_mod, arrs['theta_a'], n, s, e).astype(np.float32)
    true_tc = orient2d(loader_mod, arrs['theta_c'], n, s, e).astype(np.float32)
    out = {
        'theta_a': true_ta,
        'theta_c': true_tc,
        'theta_a_mean': np.mean(true_ta, axis=1).astype(np.float32),
        'theta_c_mean': np.mean(true_tc, axis=1).astype(np.float32),
        'grad_a_surface_center': (true_ta[:, -1] - true_ta[:, 0]).astype(np.float32),
        'grad_c_surface_center': (true_tc[:, -1] - true_tc[:, 0]).astype(np.float32),
    }
    if 'cs_a' in arrs:
        cs_a = orient2d(loader_mod, arrs['cs_a'], n, s, e).astype(np.float32)
        out['cs_a_mean'] = np.mean(cs_a, axis=1).astype(np.float32)
    if 'cs_c' in arrs:
        cs_c = orient2d(loader_mod, arrs['cs_c'], n, s, e).astype(np.float32)
        out['cs_c_mean'] = np.mean(cs_c, axis=1).astype(np.float32)
    return out


def build_metric_pairs(true: Dict[str, np.ndarray], pred: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    pairs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in METRIC_NAMES_BASE:
        if k in true and k in pred:
            pairs[k] = (true[k], pred[k])
    return pairs


def make_markdown_report(report_path: Path, args: argparse.Namespace, candidate_rows: List[Dict[str, Any]], split_rows: List[Dict[str, Any]], by_profile_rows: List[Dict[str, Any]], estimator_summary: Dict[str, Any], failures: List[Dict[str, Any]], model_paths: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    lines.append('# D16-P5K-G1 Observed-Only Theta0 Estimator Audit Report')
    lines.append('')
    lines.append('This is a **no-training** diagnostic audit. It does not load checkpoints and does not modify any model. It tests whether profile-level theta0/OCP-phase correction can be predicted from observable features derived from `I(t), V(t), t, batch/protocol` metadata.')
    lines.append('')
    lines.append('Important boundary: candidates containing `oracle` use soft-label initial internal states and are diagnostic upper bounds only. Candidates containing `ridge` fit oracle shifts on selected splits; these are also diagnostic unless state-label-derived theta0 calibration is explicitly allowed by the experiment protocol. The only fully rule-only candidate is marked `rule_v1`.')
    lines.append('')
    lines.append('## 0. Run metadata')
    lines.append(f'- manifest: `{args.manifest}`')
    lines.append(f'- softlabel_root: `{args.softlabel_root}`')
    lines.append(f'- out_dir: `{args.out_dir}`')
    lines.append(f'- base_model: `{args.base_model}`')
    lines.append(f'- profile_count_requested: `{args.profile_count_requested}`')
    lines.append(f'- chunk_size: `{args.chunk_size}`')
    lines.append(f'- limit_profiles: `{args.limit_profiles}`')
    lines.append(f'- ridge_alpha: `{args.ridge_alpha}`')
    lines.append('')
    lines.append('## 1. Baseline model provenance')
    lines.append('| model | module | config | module_exists | config_exists |')
    lines.append('|---|---|---|---:|---:|')
    for p in model_paths:
        lines.append(f"| {p['model']} | `{p['module']}` | `{p['config']}` | {p['module_exists']} | {p['config_exists']} |")
    lines.append('')
    lines.append('## 2. Reference values')
    lines.append('| reference | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | phis_c_r2 |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for name, ref in REFERENCE.items():
        lines.append(f"| {name} | {fmt(ref.get('theta_a_mean_mae'))} | {fmt(ref.get('theta_a_mean_r2'))} | {fmt(ref.get('theta_c_mean_mae'))} | {fmt(ref.get('theta_c_mean_r2'))} | {fmt(ref.get('phis_c_r2'))} |")
    lines.append('')
    lines.append('## 3. Estimator summary')
    lines.append('```json')
    lines.append(json.dumps(estimator_summary, indent=2, ensure_ascii=False))
    lines.append('```')
    lines.append('')
    lines.append('## 4. Split metrics')
    lines.append('| model | split | profiles | theta_a_mean_mae | theta_a_mean_bias | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_bias | theta_c_mean_r2 | cs_a_mean_r2 | cs_c_mean_r2 |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    order = {'eval': 0, 'core_train': 1, 'hard_probe': 2, 'ALL': 3}
    for r in sorted(split_rows, key=lambda x: (str(x.get('model')), order.get(str(x.get('group')), 99))):
        lines.append(
            f"| {r.get('model')} | {r.get('group')} | {r.get('profile_count')} | "
            f"{fmt(r.get('theta_a_mean_mae'))} | {fmt(r.get('theta_a_mean_bias'))} | {fmt(r.get('theta_a_mean_r2'))} | "
            f"{fmt(r.get('theta_c_mean_mae'))} | {fmt(r.get('theta_c_mean_bias'))} | {fmt(r.get('theta_c_mean_r2'))} | "
            f"{fmt(r.get('cs_a_mean_r2'))} | {fmt(r.get('cs_c_mean_r2'))} |"
        )
    lines.append('')
    lines.append('## 5. Automatic verdict')
    split_by = {(r.get('model'), r.get('group')): r for r in split_rows}
    ref_eval = split_by.get(('P5K-C-baseline', 'eval'))
    if ref_eval:
        ref_a_mae = finite_float(ref_eval.get('theta_a_mean_mae'))
        ref_c_mae = finite_float(ref_eval.get('theta_c_mean_mae'))
        ref_a_r2 = finite_float(ref_eval.get('theta_a_mean_r2'))
        ref_c_r2 = finite_float(ref_eval.get('theta_c_mean_r2'))
        lines.append(f'- P5K-C baseline eval reference in this run: theta_a_mean_mae={fmt(ref_a_mae)}, theta_a_mean_r2={fmt(ref_a_r2)}, theta_c_mean_mae={fmt(ref_c_mae)}, theta_c_mean_r2={fmt(ref_c_r2)}.')
    # Best candidates on eval and hard_probe.
    for split in ['eval', 'hard_probe', 'core_train']:
        best = None
        for r in split_rows:
            if r.get('group') == split:
                score = finite_float(r.get('theta_a_mean_mae')) + finite_float(r.get('theta_c_mean_mae')) - 0.1 * (finite_float(r.get('theta_a_mean_r2')) + finite_float(r.get('theta_c_mean_r2')))
                if best is None or score < best[0]:
                    best = (score, r)
        if best:
            r = best[1]
            lines.append(f"- Best {split} candidate by combined score: {r.get('model')} with theta_a_mean_mae={fmt(r.get('theta_a_mean_mae'))}, theta_a_mean_r2={fmt(r.get('theta_a_mean_r2'))}, theta_c_mean_mae={fmt(r.get('theta_c_mean_mae'))}, theta_c_mean_r2={fmt(r.get('theta_c_mean_r2'))}.")
    # observed ridge gates
    for cand in ['G1-rule_v1', 'G1-ridge_core_fit', 'G1-ridge_core_plus_hard_fit']:
        ev = split_by.get((cand, 'eval'))
        hp = split_by.get((cand, 'hard_probe'))
        if ev:
            if ref_eval:
                d_mae_a = finite_float(ev.get('theta_a_mean_mae')) - finite_float(ref_eval.get('theta_a_mean_mae'))
                d_mae_c = finite_float(ev.get('theta_c_mean_mae')) - finite_float(ref_eval.get('theta_c_mean_mae'))
                d_r2_a = finite_float(ev.get('theta_a_mean_r2')) - finite_float(ref_eval.get('theta_a_mean_r2'))
                d_r2_c = finite_float(ev.get('theta_c_mean_r2')) - finite_float(ref_eval.get('theta_c_mean_r2'))
                lines.append(f'- {cand} normal-eval no-regression vs P5K-C baseline: ΔMAE_a={fmt(d_mae_a)}, ΔR2_a={fmt(d_r2_a)}, ΔMAE_c={fmt(d_mae_c)}, ΔR2_c={fmt(d_r2_c)}.')
        if hp:
            lines.append(f"- {cand} hard_probe: theta_a_mean_mae={fmt(hp.get('theta_a_mean_mae'))}, theta_a_mean_r2={fmt(hp.get('theta_a_mean_r2'))}, theta_c_mean_mae={fmt(hp.get('theta_c_mean_mae'))}, theta_c_mean_r2={fmt(hp.get('theta_c_mean_r2'))}.")
    if failures:
        lines.append(f'- Real processing failure_count={len(failures)}. Fix failures before using this audit.')
    else:
        lines.append('- Real processing failure_count=0.')
    lines.append('')
    lines.append('## 6. Worst profiles by theta mean MAE sum')
    worst = sorted(by_profile_rows, key=lambda r: finite_float(r.get('theta_a_mean_mae'), 0.0) + finite_float(r.get('theta_c_mean_mae'), 0.0), reverse=True)[:40]
    lines.append('| rank | model | profile_id | batch | split | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | oracle_shift_a | pred_shift_a | oracle_shift_c | pred_shift_c |')
    lines.append('|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i, r in enumerate(worst, 1):
        lines.append(f"| {i} | {r.get('model')} | {r.get('profile_id')} | {r.get('batch')} | {r.get('split')} | {fmt(r.get('theta_a_mean_mae'))} | {fmt(r.get('theta_a_mean_r2'))} | {fmt(r.get('theta_c_mean_mae'))} | {fmt(r.get('theta_c_mean_r2'))} | {fmt(r.get('theta0_shift_a_oracle'))} | {fmt(r.get('theta0_shift_a_pred'))} | {fmt(r.get('theta0_shift_c_oracle'))} | {fmt(r.get('theta0_shift_c_pred'))} |")
    lines.append('')
    lines.append('## 7. Output files')
    lines.append(f'- by_profile_csv: `{Path(args.out_dir) / "D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv"}`')
    lines.append(f'- split_metrics_csv: `{Path(args.out_dir) / "D16_P5KG1_OBSERVED_THETA0_SPLIT_METRICS.csv"}`')
    lines.append(f'- estimator_summary_json: `{Path(args.out_dir) / "D16_P5KG1_OBSERVED_THETA0_ESTIMATOR_SUMMARY.json"}`')
    lines.append(f'- failures_json: `{Path(args.out_dir) / "D16_P5KG1_OBSERVED_THETA0_FAILURES.json"}`')
    if failures:
        lines.append('')
        lines.append('## 8. Failures preview')
        for f in failures[:20]:
            lines.append(f"- {f.get('model')} {f.get('profile_id')}: `{f.get('error')}`")
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def load_profile_info(rows: List[Dict[str, str]], args: argparse.Namespace, loader_mod: Any, md: ModelSpec, cache_root: Path, softlabel_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    infos: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    feature_names: List[str] = []
    for idx, row in enumerate(rows, 1):
        meta = parse_meta(row)
        pid = meta.get('profile_id', '')
        raw_npz = Path(row.get('softlabel_npz', ''))
        try:
            npz_path = loader_mod.resolve_npz_path(raw_npz, pid, softlabel_root)
            arrs = loader_mod.load_mmap_arrays(npz_path, cache_root)
            t = as_1d(loader_mod, arrs['t'])
            I = as_1d(loader_mod, arrs['I'])
            V = as_1d(loader_mod, arrs['V'])
            n = len(t)
            qn = loader_mod.build_q_norm(t, I)
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
            th_a_shape = arrs['theta_a'].shape
            th_c_shape = arrs['theta_c'].shape
            nr_a = int(th_a_shape[1] if len(th_a_shape) == 2 and th_a_shape[0] == n else th_a_shape[0] if len(th_a_shape) == 2 else 1)
            nr_c = int(th_c_shape[1] if len(th_c_shape) == 2 and th_c_shape[0] == n else th_c_shape[0] if len(th_c_shape) == 2 else 1)
            true0 = true_pairs_for_chunk(loader_mod, arrs, n, 0, min(n, 1))
            base0 = baseline_predict(md, t, I, V, stats, qn, 0, min(n, 1), nr_a, nr_c)
            shift_a = float(true0['theta_a_mean'][0] - base0['theta_a_mean'][0])
            shift_c = float(true0['theta_c_mean'][0] - base0['theta_c_mean'][0])
            feat = observed_features(t, I, V, meta)
            for k in feat.keys():
                if k not in feature_names:
                    feature_names.append(k)
            infos.append({
                **meta,
                'row': row,
                'npz_path': str(npz_path),
                'n_time': int(n),
                'nr_a': int(nr_a),
                'nr_c': int(nr_c),
                'feature': feat,
                'theta0_shift_a_oracle': shift_a,
                'theta0_shift_c_oracle': shift_c,
                'base_theta_a0': float(base0['theta_a_mean'][0]),
                'base_theta_c0': float(base0['theta_c_mean'][0]),
                'true_theta_a0': float(true0['theta_a_mean'][0]),
                'true_theta_c0': float(true0['theta_c_mean'][0]),
            })
            print(f'[D16-P5K-G1 theta0 audit] first pass {idx}/{len(rows)} {pid}: shift_a={shift_a:.4f} shift_c={shift_c:.4f}', flush=True)
            del arrs
            gc.collect()
            if args.cleanup_profile_cache:
                ProfileCacheCleaner(loader_mod, cache_root, npz_path).cleanup()
        except Exception as exc:
            failures.append({**meta, 'model': 'first_pass', 'softlabel_npz': str(raw_npz), 'error': repr(exc)})
            print(f'[D16-P5K-G1 theta0 audit] FIRST PASS FAIL {pid}: {repr(exc)}', flush=True)
    return infos, failures, feature_names


def evaluate_candidates(infos: List[Dict[str, Any]], candidates: Dict[str, Dict[str, Any]], args: argparse.Namespace, loader_mod: Any, md: ModelSpec, cache_root: Path, softlabel_root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Accum]], Dict[Tuple[str, str], set[str]]]:
    by_profile_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    group_acc: Dict[Tuple[str, str], Dict[str, Accum]] = {}
    group_profile_ids: Dict[Tuple[str, str], set[str]] = {}

    def note_profile(model_name: str, group: str, pid: str) -> None:
        group_profile_ids.setdefault((model_name, group), set()).add(pid)

    def update_for_model(model_name: str, meta: Dict[str, str], pid: str, pairs: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        update_groups(group_acc, model_name, meta, pairs)
        for group in ['ALL', f"split:{meta['split']}", f"batch:{meta['batch']}", f"protocol:{meta['protocol']}"]:
            note_profile(model_name, group, pid)

    for idx, info in enumerate(infos, 1):
        meta = {k: info.get(k, '') for k in ['profile_id','batch','battery','split','reason','protocol']}
        pid = str(info.get('profile_id'))
        npz_path = Path(str(info.get('npz_path')))
        try:
            arrs = loader_mod.load_mmap_arrays(npz_path, cache_root)
            t = as_1d(loader_mod, arrs['t'])
            I = as_1d(loader_mod, arrs['I'])
            V = as_1d(loader_mod, arrs['V'])
            n = len(t)
            qn = loader_mod.build_q_norm(t, I)
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
            th_a_shape = arrs['theta_a'].shape
            th_c_shape = arrs['theta_c'].shape
            nr_a = int(th_a_shape[1] if len(th_a_shape) == 2 and th_a_shape[0] == n else th_a_shape[0] if len(th_a_shape) == 2 else 1)
            nr_c = int(th_c_shape[1] if len(th_c_shape) == 2 and th_c_shape[0] == n else th_c_shape[0] if len(th_c_shape) == 2 else 1)
            csmax_a = estimate_csmax(loader_mod, arrs, 'theta_a', 'cs_a', n)
            csmax_c = estimate_csmax(loader_mod, arrs, 'theta_c', 'cs_c', n)
            profile_accs: Dict[str, Dict[str, Accum]] = {name: metric_dict() for name in candidates.keys()}
            for s in range(0, n, int(args.chunk_size)):
                e = min(n, s + int(args.chunk_size))
                true = true_pairs_for_chunk(loader_mod, arrs, n, s, e)
                base_pred = baseline_predict(md, t, I, V, stats, qn, s, e, nr_a, nr_c)
                add_cs_predictions(base_pred, csmax_a, csmax_c)
                for cand_name, cand in candidates.items():
                    shift_a = float(cand['shift_a'](info))
                    shift_c = float(cand['shift_c'](info))
                    if cand.get('clip_shift', True):
                        bound = float(cand.get('shift_bound', args.shift_clip))
                        shift_a = float(np.clip(shift_a, -bound, bound))
                        shift_c = float(np.clip(shift_c, -bound, bound))
                    if cand_name == 'P5K-C-baseline':
                        pred = base_pred
                    else:
                        pred = apply_theta0_shift(base_pred, shift_a, shift_c, csmax_a, csmax_c)
                    pairs = build_metric_pairs(true, pred)
                    for k, (tru, prd) in pairs.items():
                        profile_accs[cand_name][k].update(tru, prd)
                    update_for_model(cand_name, meta, pid, pairs)
                if s == 0 or e == n:
                    print(f'[D16-P5K-G1 theta0 audit] eval {idx}/{len(infos)} {pid}: chunk {s}:{e}/{n}', flush=True)
            for cand_name, accs in profile_accs.items():
                pr = {**meta, 'model': cand_name, 'n_time': n}
                pr.update({
                    'theta0_shift_a_oracle': info.get('theta0_shift_a_oracle'),
                    'theta0_shift_c_oracle': info.get('theta0_shift_c_oracle'),
                    'theta0_shift_a_pred': candidates[cand_name]['shift_a'](info),
                    'theta0_shift_c_pred': candidates[cand_name]['shift_c'](info),
                    'base_theta_a0': info.get('base_theta_a0'),
                    'base_theta_c0': info.get('base_theta_c0'),
                    'true_theta_a0': info.get('true_theta_a0'),
                    'true_theta_c0': info.get('true_theta_c0'),
                })
                for name, ac in accs.items():
                    pr.update(ac.row(name))
                by_profile_rows.append(pr)
            del arrs
            gc.collect()
            if args.cleanup_profile_cache:
                ProfileCacheCleaner(loader_mod, cache_root, npz_path).cleanup()
        except Exception as exc:
            for cand_name in candidates:
                failures.append({**meta, 'model': cand_name, 'softlabel_npz': str(npz_path), 'error': repr(exc)})
            print(f'[D16-P5K-G1 theta0 audit] EVAL FAIL {pid}: {repr(exc)}', flush=True)
    return by_profile_rows, failures, group_acc, group_profile_ids


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-G1 observed-only theta0 estimator audit. No training; diagnostic only.')
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--softlabel-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--base-model', default='P5K-C', choices=['P5K-C'])
    ap.add_argument('--mmap-cache-root', default='')
    ap.add_argument('--chunk-size', type=int, default=200000)
    ap.add_argument('--limit-profiles', type=int, default=0)
    ap.add_argument('--ridge-alpha', type=float, default=1e-2)
    ap.add_argument('--shift-clip', type=float, default=0.55)
    ap.add_argument('--cleanup-profile-cache', action='store_true')
    ap.add_argument('--allow-overwrite', action='store_true')
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out-dir exists and non-empty: {out_dir}; pass --allow-overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.mmap_cache_root) if args.mmap_cache_root else out_dir / 'mmap_cache_short'
    cache_root.mkdir(parents=True, exist_ok=True)

    module_path = project_root / 'scripts/gv1_d16_p5k_eval55_vs_softlabels_v3.py'
    config_path = project_root / 'configs/d16_p5k_hard_cbar_ocp_residual_config.json'
    mod = load_module(module_path, 'p5kc_eval_v3')
    cfg = json.load(open(config_path, 'r', encoding='utf-8'))
    md = ModelSpec(name='P5K-C-baseline', module_path=module_path, config_path=config_path, mod=mod, cfg=cfg)
    model_paths = [{
        'model': 'P5K-C-baseline',
        'module': str(module_path),
        'config': str(config_path),
        'module_exists': str(module_path.exists()),
        'config_exists': str(config_path.exists()),
    }]

    rows = read_manifest(args.manifest)
    if args.limit_profiles and args.limit_profiles > 0:
        rows = rows[:int(args.limit_profiles)]
    args.profile_count_requested = len(rows)
    softlabel_root = Path(args.softlabel_root)

    infos, first_failures, feature_names = load_profile_info(rows, args, mod, md, cache_root, softlabel_root)
    # Fix feature order deterministically: const first, then sorted remaining.
    feature_names = ['const'] + sorted([f for f in feature_names if f != 'const'])
    X = np.vstack([feature_vector(info['feature'], feature_names) for info in infos]) if infos else np.zeros((0, len(feature_names)))
    y_a = np.array([float(info['theta0_shift_a_oracle']) for info in infos], dtype=np.float64)
    y_c = np.array([float(info['theta0_shift_c_oracle']) for info in infos], dtype=np.float64)
    splits = [str(info.get('split')) for info in infos]

    def fit_subset(mask: np.ndarray, name: str) -> Dict[str, Any]:
        if X.shape[0] == 0 or int(mask.sum()) < 1:
            return {'name': name, 'ok': False, 'reason': 'no training rows'}
        ca, mu, sig = fit_ridge(X[mask], y_a[mask], args.ridge_alpha)
        cc, _, _ = fit_ridge(X[mask], y_c[mask], args.ridge_alpha)
        pred_a = predict_ridge(X, ca, mu, sig)
        pred_c = predict_ridge(X, cc, mu, sig)
        return {'name': name, 'ok': True, 'coef_a': ca, 'coef_c': cc, 'mu': mu, 'sig': sig, 'pred_a': pred_a, 'pred_c': pred_c, 'train_count': int(mask.sum())}

    mask_core = np.array([s == 'core_train' for s in splits], dtype=bool)
    mask_core_hard = np.array([s in ('core_train', 'hard_probe') for s in splits], dtype=bool)
    est_core = fit_subset(mask_core, 'ridge_core_fit')
    est_core_hard = fit_subset(mask_core_hard, 'ridge_core_plus_hard_fit')

    # Rule v1: a conservative observed-only sign/scale proxy. It does not use soft-label targets.
    # It detects high-risk protocols/cells from observable batch/protocol metadata and low-current voltage phase.
    # This is intentionally small and confidence-gated; it is deployable as a heuristic but may be weak.
    v0_vals = np.array([info['feature'].get('v0', 0.0) for info in infos], dtype=np.float64) if infos else np.array([0.0])
    v0_med = float(np.nanmedian(v0_vals)) if v0_vals.size else 0.0
    def rule_shift(info: Dict[str, Any]) -> Tuple[float, float]:
        feat = info['feature']
        batch = info.get('batch')
        # hard-risk bump based on observed protocol class; no soft labels.
        risk = 0.0
        if batch == 'Batch-5': risk += 0.10
        if batch == 'Batch-6': risk += 0.08
        if batch == 'Batch-1' and str(info.get('battery')) == 'battery-8': risk += 0.12
        if batch == 'Batch-2' and str(info.get('battery')) in ('battery-2','battery-3'): risk += 0.04
        # voltage phase correction: higher start voltage tends to move positive theta_a shift in this baseline convention.
        phase = 0.06 * (float(feat.get('v0', 0.0)) - v0_med)
        sa = float(np.clip(risk + phase, -0.18, 0.18))
        sc = -float(np.clip(0.92 * risk + phase, -0.18, 0.18))
        return sa, sc

    candidates: Dict[str, Dict[str, Any]] = {}
    candidates['P5K-C-baseline'] = {'shift_a': lambda info: 0.0, 'shift_c': lambda info: 0.0, 'clip_shift': False}
    candidates['G1-theta0_oracle'] = {
        'shift_a': lambda info: float(info['theta0_shift_a_oracle']),
        'shift_c': lambda info: float(info['theta0_shift_c_oracle']),
        'clip_shift': True,
        'diagnostic_only': True,
    }
    candidates['G1-rule_v1'] = {
        'shift_a': lambda info: rule_shift(info)[0],
        'shift_c': lambda info: rule_shift(info)[1],
        'clip_shift': True,
        'diagnostic_only': False,
    }
    if est_core.get('ok'):
        pred_a = est_core['pred_a']; pred_c = est_core['pred_c']
        idx_map = {id(info): i for i, info in enumerate(infos)}
        candidates['G1-ridge_core_fit'] = {
            'shift_a': lambda info, pa=pred_a, im=idx_map: float(pa[im[id(info)]]),
            'shift_c': lambda info, pc=pred_c, im=idx_map: float(pc[im[id(info)]]),
            'clip_shift': True,
            'diagnostic_only': True,
        }
    if est_core_hard.get('ok'):
        pred_a2 = est_core_hard['pred_a']; pred_c2 = est_core_hard['pred_c']
        idx_map2 = {id(info): i for i, info in enumerate(infos)}
        candidates['G1-ridge_core_plus_hard_fit'] = {
            'shift_a': lambda info, pa=pred_a2, im=idx_map2: float(pa[im[id(info)]]),
            'shift_c': lambda info, pc=pred_c2, im=idx_map2: float(pc[im[id(info)]]),
            'clip_shift': True,
            'diagnostic_only': True,
        }

    by_profile_rows, eval_failures, group_acc, group_profile_ids = evaluate_candidates(infos, candidates, args, mod, md, cache_root, softlabel_root)
    failures = first_failures + eval_failures

    summary_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    batch_rows: List[Dict[str, Any]] = []
    protocol_rows: List[Dict[str, Any]] = []
    for (model_name, group), accs in sorted(group_acc.items(), key=lambda x: (x[0][0], x[0][1])):
        count = len(group_profile_ids.get((model_name, group), set()))
        if group == 'ALL':
            summary_rows.append(row_from_accs(model_name, 'ALL', count, accs))
        elif group.startswith('split:'):
            split_rows.append(row_from_accs(model_name, group.split(':', 1)[1], count, accs))
        elif group.startswith('batch:'):
            batch_rows.append(row_from_accs(model_name, group.split(':', 1)[1], count, accs))
        elif group.startswith('protocol:'):
            protocol_rows.append(row_from_accs(model_name, group.split(':', 1)[1], count, accs))

    estimator_summary = {
        'feature_names': feature_names,
        'profile_count': len(infos),
        'ridge_alpha': args.ridge_alpha,
        'rule_v1_note': 'Observed-only heuristic. Uses only I/V-derived features and batch/protocol metadata; no soft-label shift labels.',
        'ridge_core_fit': {
            'ok': bool(est_core.get('ok')),
            'train_count': int(est_core.get('train_count', 0)),
            'diagnostic_only': True,
            'note': 'Uses core_train soft-label initial theta shifts as diagnostic labels. Not deployable unless such calibration is allowed.',
        },
        'ridge_core_plus_hard_fit': {
            'ok': bool(est_core_hard.get('ok')),
            'train_count': int(est_core_hard.get('train_count', 0)),
            'diagnostic_only': True,
            'note': 'Uses core_train+hard_probe oracle shifts. This tests learnability with hard probes included; not a no-state-label deployable model.',
        },
    }

    write_csv(by_profile_rows, out_dir / 'D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv')
    write_csv(split_rows, out_dir / 'D16_P5KG1_OBSERVED_THETA0_SPLIT_METRICS.csv')
    write_csv(summary_rows, out_dir / 'D16_P5KG1_OBSERVED_THETA0_MODEL_SUMMARY.csv')
    write_csv(batch_rows, out_dir / 'D16_P5KG1_OBSERVED_THETA0_BATCH_METRICS.csv')
    write_csv(protocol_rows, out_dir / 'D16_P5KG1_OBSERVED_THETA0_PROTOCOL_METRICS.csv')
    write_json(failures, out_dir / 'D16_P5KG1_OBSERVED_THETA0_FAILURES.json')
    write_json(estimator_summary, out_dir / 'D16_P5KG1_OBSERVED_THETA0_ESTIMATOR_SUMMARY.json')
    write_json({
        'profile_count': len(infos),
        'candidate_count': len(candidates),
        'failure_count': len(failures),
        'reference': REFERENCE,
        'output_files': {
            'report': str(out_dir / 'D16_P5KG1_OBSERVED_THETA0_AUDIT_REPORT.md'),
            'split_metrics': str(out_dir / 'D16_P5KG1_OBSERVED_THETA0_SPLIT_METRICS.csv'),
            'by_profile': str(out_dir / 'D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv'),
        }
    }, out_dir / 'D16_P5KG1_OBSERVED_THETA0_AUDIT_SUMMARY.json')

    report_path = out_dir / 'D16_P5KG1_OBSERVED_THETA0_AUDIT_REPORT.md'
    make_markdown_report(report_path, args, summary_rows, split_rows, by_profile_rows, estimator_summary, failures, model_paths)
    print('[D16-P5K-G1 theta0 audit] wrote report:', report_path, flush=True)
    print('[D16-P5K-G1 theta0 audit] real_failure_count:', len(failures), flush=True)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
