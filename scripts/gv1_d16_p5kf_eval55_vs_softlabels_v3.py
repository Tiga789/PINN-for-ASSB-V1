from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

FEATURE_NAMES = [
    't_norm', 't_norm2', 'sin_t', 'cos_t',
    'I_norm', 'absI_norm', 'dI_norm', 'q_norm',
    'q_cell_frac', 'q_cell_frac_abs',
    'voltage_exp_norm_local', 'dV_norm',
    'v_window_phase', 'v0_window_phase', 'v_mean_window_phase',
    'is_charge', 'is_rest', 'is_discharge',
]


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('', encoding='utf-8'); return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys: keys.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def first_key(keys: set[str], aliases: List[str]) -> str | None:
    for k in aliases:
        if k in keys:
            return k
    return None


def _safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '__', s)


def extract_npy_member(npz_path: Path, key: str, cache_root: Path) -> Path:
    # .npz is a zip of .npy members. Extract once so large arrays can be memory-mapped.
    # v2 fix: do NOT use the full Windows source path as cache directory name.
    # The old implementation created paths longer than MAX_PATH on Windows and all 55 profiles
    # failed with FileNotFoundError before any metric could be computed.
    cache_root.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(str(npz_path).encode('utf-8', errors='ignore')).hexdigest()[:16]
    cell_hint = _safe_name(npz_path.parent.name)[:64]
    dst_dir = cache_root / f'{cell_hint}_{h}'
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f'{_safe_name(key)}.npy'
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    member = key if key.endswith('.npy') else f'{key}.npy'
    try:
        with zipfile.ZipFile(npz_path, 'r') as zf:
            if member not in zf.namelist():
                raise KeyError(f'{npz_path}: member {member} not found')
            tmp = dst.with_suffix(dst.suffix + '.tmp')
            with zf.open(member, 'r') as src, tmp.open('wb') as out:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
            tmp.replace(dst)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'P5K-F eval cache/source file missing or path too long. npz={npz_path}; key={key}; cache_root={cache_root}; dst={dst}') from exc
    return dst




def resolve_npz_path(raw_path: Path, profile_id: str = '', softlabel_root: Path | None = None) -> Path:
    """Resolve a soft-label npz path robustly on Windows.

    This v3 evaluator does not trust a stale manifest path blindly. It tries the
    manifest path first, then reconstructs the path from profile_id under the
    current softlabel_root, then falls back to a narrow recursive search.
    """
    raw = Path(raw_path)
    if raw.exists():
        return raw
    candidates: list[Path] = []
    root = Path(softlabel_root) if softlabel_root else None
    pid = str(profile_id or '').replace('\\', '/').strip('/')
    if root is not None and root.exists() and pid:
        candidates.append(root / pid / 'solution_softlabels.npz')
        if pid.startswith('profiles/'):
            short = pid.replace('profiles/', '', 1)
            candidates.append(root / short / 'solution_softlabels.npz')
            candidates.append(root / 'profiles' / short / 'solution_softlabels.npz')
    text = (pid + '/' + str(raw)).replace('\\', '/')
    bm = re.search(r'(Batch-\d+)', text, flags=re.I)
    batm = re.search(r'battery[-_ ]?(\d+)', text, flags=re.I)
    if root is not None and root.exists() and bm and batm:
        batch = bm.group(1)
        battery = f'battery-{int(batm.group(1))}'
        for base in [root / 'profiles', root]:
            candidates.append(base / f'{batch}_{battery}' / 'solution_softlabels.npz')
            candidates.append(base / f'{batch}-{battery}' / 'solution_softlabels.npz')
        try:
            hits = []
            for f in root.rglob('solution_softlabels.npz'):
                fs = str(f).replace('\\', '/')
                if batch in fs and battery in fs:
                    hits.append(f)
            candidates.extend(sorted(hits)[:5])
        except Exception:
            pass
    for c in candidates:
        if c.exists():
            return c
    cand_str = [str(c) for c in candidates[:10]]
    raise FileNotFoundError(
        f'softlabel npz not found. manifest_path={raw}; profile_id={profile_id}; '
        f'softlabel_root={softlabel_root}; tried={cand_str}'
    )

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
    kphie = first_key(keys, ['phie', 'phi_e', 'phi_e_eff'])
    kphis = first_key(keys, ['phis_c_soft', 'phis_c', 'voltage_soft', 'V_soft', 'V_pred'])
    required = {'t': kt, 'I': ki, 'V': kv, 'theta_a': kth_a, 'theta_c': kth_c, 'phie': kphie, 'phis_c': kphis}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise KeyError(f'{npz_path}: missing required arrays {missing}; available={sorted(keys)}')
    mapping = dict(required)
    if kcs_a is not None: mapping['cs_a'] = kcs_a
    if kcs_c is not None: mapping['cs_c'] = kcs_c
    arrs: Dict[str, np.ndarray] = {}
    for alias, key in mapping.items():
        p = extract_npy_member(npz_path, key, cache_root)
        arrs[alias] = np.load(p, mmap_mode='r')
    return arrs


def as_1d_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)



def build_q_features(t: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[~np.isfinite(dt)] = 0.0
    if dt.size > 10:
        p = np.nanpercentile(dt, 99.9)
        if np.isfinite(p) and p > 0: dt = np.clip(dt, 0.0, p * 10.0)
    q_ah = np.cumsum(I.astype(np.float32) * dt) / 3600.0
    q0 = q_ah - np.nanmean(q_ah)
    scale = float(np.nanpercentile(np.abs(q0), 99.5)) if q0.size else 1.0
    if not np.isfinite(scale) or scale < 1e-12: scale = 1.0
    q_norm = np.clip(q0 / scale, -1.5, 1.5).astype(np.float32)
    q_cell = np.clip((q_ah - q_ah[0]) / 2.0, -1.5, 1.5).astype(np.float32)
    return q_norm, q_cell, np.abs(q_cell).astype(np.float32)


def build_q_norm(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    return build_q_features(t, I)[0]


def _window_phase(V: np.ndarray, low: float = 2.5, high: float = 4.2) -> np.ndarray:
    return np.clip((np.asarray(V, dtype=np.float32) - float(low)) / max(1e-6, float(high-low)), 0.0, 1.0).astype(np.float32)


def feature_chunk(t: np.ndarray, I: np.ndarray, V: np.ndarray, s: int, e: int, stats: Dict[str, float], qn_full: np.ndarray) -> np.ndarray:
    span = max(1e-12, float(stats['t_span']))
    tn = ((t[s:e] - t[0]) / span).astype(np.float32)
    I_scale = max(1e-12, float(stats['I_scale']))
    In = (I[s:e] / I_scale).astype(np.float32)
    I_prev = I[s-1:e-1] if s > 0 else np.concatenate([[I[0]], I[s:e-1]])
    dI = ((I[s:e] - I_prev) / I_scale).astype(np.float32)
    v_mean = float(stats['v_mean']); v_std = max(1e-8, float(stats['v_std']))
    vn = ((V[s:e] - v_mean) / v_std).astype(np.float32)
    V_prev = V[s-1:e-1] if s > 0 else np.concatenate([[V[0]], V[s:e-1]])
    dV = ((V[s:e] - V_prev) / v_std).astype(np.float32)
    q_norm_full, q_cell_full, q_cell_abs_full = build_q_features(t, I)
    v_phase = _window_phase(V[s:e])
    v0_phase = np.full_like(v_phase, float(_window_phase(np.array([V[0]], dtype=np.float32))[0]), dtype=np.float32)
    vmean_phase = np.full_like(v_phase, float(_window_phase(np.array([v_mean], dtype=np.float32))[0]), dtype=np.float32)
    eps = max(1e-9, 0.001 * float(stats.get('I_abs_max', 1.0) + 1e-12))
    charge = (I[s:e] > eps).astype(np.float32)
    discharge = (I[s:e] < -eps).astype(np.float32)
    rest = (np.abs(I[s:e]) <= eps).astype(np.float32)
    X = np.stack([
        tn, tn**2, np.sin(2*np.pi*tn).astype(np.float32), np.cos(2*np.pi*tn).astype(np.float32),
        In, np.abs(In).astype(np.float32), dI, q_norm_full[s:e].astype(np.float32),
        q_cell_full[s:e].astype(np.float32), q_cell_abs_full[s:e].astype(np.float32),
        vn, dV, v_phase, v0_phase, vmean_phase,
        charge, rest, discharge,
    ], axis=1).astype(np.float32)
    return X


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
        for _ in range(max(1, int(num_layers))): layers.append(ResidualBlock(hidden_dim))
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
    b = hard_baseline_from_observed(x_raw, cfg)
    res_a = float(model_cfg.get('residual_bound_a', 0.055)) * torch.tanh(raw[:, 0])
    res_c = float(model_cfg.get('residual_bound_c', 0.055)) * torch.tanh(raw[:, 1])
    ta = torch.clamp(b['theta_a_base'] + res_a, 0.0, 1.0)
    tc = torch.clamp(b['theta_c_base'] + res_c, 0.0, 1.0)
    grad_clip = float(model_cfg.get('gradient_clip', 0.25))
    return {
        'theta_a_mean': ta,
        'theta_c_mean': tc,
        'theta_a_base': b['theta_a_base'],
        'theta_c_base': b['theta_c_base'],
        'theta_a_residual': res_a,
        'theta_c_residual': res_c,
        'grad_a': grad_clip * torch.tanh(raw[:, 2]),
        'grad_c': grad_clip * torch.tanh(raw[:, 3]),
        'phie_norm': raw[:, 4],
        'phis_c_norm': raw[:, 5],
    }


class Accum:
    def __init__(self):
        self.n = 0; self.sum_abs = 0.0; self.sum_sq = 0.0; self.sum_err = 0.0; self.max_abs = 0.0
        self.sum_t = 0.0; self.sum_p = 0.0; self.sum_t2 = 0.0; self.sum_p2 = 0.0; self.sum_tp = 0.0
    def update(self, true: np.ndarray, pred: np.ndarray):
        t = np.asarray(true, dtype=np.float64).reshape(-1); p = np.asarray(pred, dtype=np.float64).reshape(-1)
        mask = np.isfinite(t) & np.isfinite(p)
        if not np.any(mask): return
        t = t[mask]; p = p[mask]; e = p - t; ae = np.abs(e)
        self.n += int(t.size); self.sum_abs += float(np.sum(ae)); self.sum_sq += float(np.sum(e*e)); self.sum_err += float(np.sum(e)); self.max_abs = max(self.max_abs, float(np.max(ae)))
        self.sum_t += float(np.sum(t)); self.sum_p += float(np.sum(p)); self.sum_t2 += float(np.sum(t*t)); self.sum_p2 += float(np.sum(p*p)); self.sum_tp += float(np.sum(t*p))
    def row(self, prefix: str) -> Dict[str, Any]:
        n = max(1, self.n)
        cov = self.sum_tp - self.sum_t*self.sum_p/n
        vt = self.sum_t2 - self.sum_t*self.sum_t/n
        vp = self.sum_p2 - self.sum_p*self.sum_p/n
        corr = cov / math.sqrt(vt*vp) if vt > 1e-20 and vp > 1e-20 else float('nan')
        r2 = 1.0 - (self.sum_sq / vt) if self.n and vt > 1e-20 else float('nan')
        return {f'{prefix}_count': int(self.n), f'{prefix}_mae': self.sum_abs/n if self.n else float('nan'), f'{prefix}_rmse': math.sqrt(self.sum_sq/n) if self.n else float('nan'), f'{prefix}_bias': self.sum_err/n if self.n else float('nan'), f'{prefix}_max_abs': self.max_abs if self.n else float('nan'), f'{prefix}_corr': corr, f'{prefix}_r2': r2, f'{prefix}_sum_true': self.sum_t, f'{prefix}_sum_true_sq': self.sum_t2, f'{prefix}_sum_pred': self.sum_p, f'{prefix}_sum_pred_sq': self.sum_p2, f'{prefix}_sum_err_sq': self.sum_sq}


def orient2d(arr: np.ndarray, n: int, s: int, e: int) -> np.ndarray:
    if len(arr.shape) == 1:
        return np.asarray(arr[s:e], dtype=np.float32).reshape(-1, 1)
    if arr.shape[0] == n:
        return np.asarray(arr[s:e], dtype=np.float32)
    if arr.shape[1] == n:
        return np.asarray(arr[:, s:e], dtype=np.float32).T
    raise ValueError(f'Cannot orient array shape={arr.shape} for n={n}')


def estimate_csmax(cs_arr: np.ndarray | None, th_arr: np.ndarray, n: int) -> float | None:
    if cs_arr is None: return None
    m = min(n, 20000)
    try:
        cs = orient2d(cs_arr, n, 0, m).reshape(-1)
        th = orient2d(th_arr, n, 0, m).reshape(-1)
        mask = np.isfinite(cs) & np.isfinite(th) & (np.abs(th) > 1e-5)
        if not np.any(mask): return None
        ratio = cs[mask] / th[mask]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size < 10: return None
        val = float(np.nanmedian(ratio))
        if not np.isfinite(val) or val <= 0: return None
        return val
    except Exception:
        return None


def infer_chunk(model: nn.Module, Xraw: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, device: torch.device, batch_size: int, cfg: Dict[str, Any], v_mean: float, v_std: float, nr_a: int, nr_c: int) -> Dict[str, np.ndarray]:
    radial_a = np.linspace(-0.5, 0.5, nr_a, dtype=np.float32)
    radial_c = np.linspace(-0.5, 0.5, nr_c, dtype=np.float32)
    outs = {k: [] for k in ['theta_a', 'theta_c', 'theta_a_mean', 'theta_c_mean', 'grad_a_surface_center', 'grad_c_surface_center', 'phie', 'phis_c']}
    Xs = ((Xraw - x_mean) / x_std).astype(np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, Xs.shape[0], batch_size):
            xs = torch.from_numpy(Xs[i:i+batch_size]).to(device)
            xr = torch.from_numpy(Xraw[i:i+batch_size].astype(np.float32)).to(device)
            raw = model(xs)
            y = transform_outputs(raw, xr, cfg)
            ta_m = y['theta_a_mean'].cpu().numpy().astype(np.float32)
            tc_m = y['theta_c_mean'].cpu().numpy().astype(np.float32)
            ga = y['grad_a'].cpu().numpy().astype(np.float32)
            gc = y['grad_c'].cpu().numpy().astype(np.float32)
            phie = (y['phie_norm'].cpu().numpy().astype(np.float32) * v_std)
            phis = (y['phis_c_norm'].cpu().numpy().astype(np.float32) * v_std + v_mean)
            theta_a = np.clip(ta_m[:, None] + ga[:, None] * radial_a[None, :], 0.0, 1.0).astype(np.float32)
            theta_c = np.clip(tc_m[:, None] + gc[:, None] * radial_c[None, :], 0.0, 1.0).astype(np.float32)
            outs['theta_a'].append(theta_a); outs['theta_c'].append(theta_c)
            outs['theta_a_mean'].append(ta_m); outs['theta_c_mean'].append(tc_m)
            outs['grad_a_surface_center'].append((theta_a[:, -1] - theta_a[:, 0]).astype(np.float32))
            outs['grad_c_surface_center'].append((theta_c[:, -1] - theta_c[:, 0]).astype(np.float32))
            outs['phie'].append(phie); outs['phis_c'].append(phis)
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def parse_meta(row: Dict[str, str]) -> Dict[str, str]:
    batch = row.get('batch', '')
    protocol = {'Batch-1':'2C', 'Batch-2':'3C', 'Batch-3':'R2.5', 'Batch-4':'R3', 'Batch-5':'random_walk', 'Batch-6':'GEO'}.get(batch, batch)
    return {'profile_id': row.get('profile_id', ''), 'batch': batch, 'battery': row.get('battery', ''), 'split': row.get('split', 'eval'), 'reason': row.get('reason', ''), 'protocol': row.get('protocol', protocol)}


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-F v3 evaluate hard-cbar/OCP residual model on all55 against P2Dlite-RG soft labels. Soft labels are evaluation-only.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--config', default='configs/d16_p5kf_profile_theta0_hard_cbar_config.json')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=65536)
    ap.add_argument('--chunk-size', type=int, default=200000)
    ap.add_argument('--mmap-cache-root', default='')
    ap.add_argument('--softlabel-root', default='')
    ap.add_argument('--limit-profiles', type=int, default=0)
    ap.add_argument('--allow-overwrite', action='store_true')
    args = ap.parse_args()

    cfg = json.load(open(args.config, 'r', encoding='utf-8'))
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out-dir exists and non-empty: {out_dir}; pass --allow-overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.mmap_cache_root) if args.mmap_cache_root else out_dir / 'mmap_cache_short'
    print(f'[D16-P5K-F eval v3] mmap_cache_root={cache_root}', flush=True)
    ckpt_path = Path(args.model_dir) / 'model' / 'best_with_state.pt'
    if not ckpt_path.exists(): ckpt_path = Path(args.model_dir) / 'best_with_state.pt'
    if not ckpt_path.exists(): raise FileNotFoundError(f'missing checkpoint best_with_state.pt under {args.model_dir}')
    device = torch.device(args.device if args.device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_cfg = ckpt.get('model_config', cfg.get('model', {}))
    cfg = ckpt.get('config', cfg)
    model = HardCbarOCPResidualMLP(input_dim=len(FEATURE_NAMES), hidden_dim=int(model_cfg.get('hidden_dim', 256)), num_layers=int(model_cfg.get('num_layers', 5)), output_dim=6)
    model.load_state_dict(ckpt['state'])
    model.to(device).eval()
    x_mean = np.asarray(ckpt['x_mean'], dtype=np.float32)
    x_std = np.asarray(ckpt['x_std'], dtype=np.float32)

    rows = read_manifest(args.manifest)
    if args.limit_profiles and args.limit_profiles > 0:
        rows = rows[:int(args.limit_profiles)]
    softlabel_root = Path(args.softlabel_root) if str(args.softlabel_root).strip() else None
    metrics_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    group_acc: Dict[str, Dict[str, Accum]] = {}
    def add_group(group: str, metric: str, true: np.ndarray, pred: np.ndarray):
        group_acc.setdefault(group, {}).setdefault(metric, Accum()).update(true, pred)

    for row in rows:
        meta = parse_meta(row); npz_path = Path(row['softlabel_npz'])
        try:
            npz_path = resolve_npz_path(npz_path, meta.get('profile_id',''), softlabel_root)
            arr = load_mmap_arrays(npz_path, cache_root)
            t = as_1d_float(arr['t']); I = as_1d_float(arr['I']); V = as_1d_float(arr['V']); n = len(t)
            stats = {'t_span': float(t[-1]-t[0]) if n > 1 else 1.0, 'I_scale': float(np.nanpercentile(np.abs(I), 99.5)) if n else 1.0, 'I_abs_max': float(np.nanmax(np.abs(I))) if n else 0.0, 'v_mean': float(np.nanmean(V)) if n else 0.0, 'v_std': float(np.nanstd(V)) if n else 1.0}
            if not np.isfinite(stats['t_span']) or stats['t_span'] <= 0: stats['t_span'] = 1.0
            if not np.isfinite(stats['I_scale']) or stats['I_scale'] < 1e-12: stats['I_scale'] = 1.0
            if not np.isfinite(stats['v_std']) or stats['v_std'] < 1e-8: stats['v_std'] = 1.0
            qn = build_q_norm(t, I)
            th_a_shape = arr['theta_a'].shape; th_c_shape = arr['theta_c'].shape
            nr_a = int(th_a_shape[1] if len(th_a_shape) == 2 and th_a_shape[0] == n else th_a_shape[0] if len(th_a_shape) == 2 else 1)
            nr_c = int(th_c_shape[1] if len(th_c_shape) == 2 and th_c_shape[0] == n else th_c_shape[0] if len(th_c_shape) == 2 else 1)
            csmax_a = estimate_csmax(arr.get('cs_a'), arr['theta_a'], n)
            csmax_c = estimate_csmax(arr.get('cs_c'), arr['theta_c'], n)
            metric_names = ['phis_c', 'phie', 'theta_a', 'theta_c', 'theta_a_mean', 'theta_c_mean', 'grad_a_surface_center', 'grad_c_surface_center']
            if csmax_a is not None and 'cs_a' in arr: metric_names += ['cs_a', 'cs_a_mean']
            if csmax_c is not None and 'cs_c' in arr: metric_names += ['cs_c', 'cs_c_mean']
            accs = {m: Accum() for m in metric_names}
            outside_count = 0; outside_total = 0
            for s in range(0, n, int(args.chunk_size)):
                e = min(n, s + int(args.chunk_size))
                X = feature_chunk(t, I, V, s, e, stats, qn)
                pred = infer_chunk(model, X, x_mean, x_std, device, int(args.batch_size), cfg, stats['v_mean'], stats['v_std'], nr_a, nr_c)
                true_phis = np.asarray(arr['phis_c'][s:e], dtype=np.float32).reshape(-1)
                true_phie = np.asarray(arr['phie'][s:e], dtype=np.float32).reshape(-1)
                true_ta = orient2d(arr['theta_a'], n, s, e)
                true_tc = orient2d(arr['theta_c'], n, s, e)
                true_ta_m = np.mean(true_ta, axis=1); true_tc_m = np.mean(true_tc, axis=1)
                true_ga = true_ta[:, -1] - true_ta[:, 0]; true_gc = true_tc[:, -1] - true_tc[:, 0]
                pairs = {
                    'phis_c': (true_phis, pred['phis_c']), 'phie': (true_phie, pred['phie']),
                    'theta_a': (true_ta, pred['theta_a']), 'theta_c': (true_tc, pred['theta_c']),
                    'theta_a_mean': (true_ta_m, pred['theta_a_mean']), 'theta_c_mean': (true_tc_m, pred['theta_c_mean']),
                    'grad_a_surface_center': (true_ga, pred['grad_a_surface_center']), 'grad_c_surface_center': (true_gc, pred['grad_c_surface_center']),
                }
                if csmax_a is not None and 'cs_a' in arr:
                    true_cs_a = orient2d(arr['cs_a'], n, s, e)
                    pred_cs_a = pred['theta_a'] * float(csmax_a)
                    pairs['cs_a'] = (true_cs_a, pred_cs_a); pairs['cs_a_mean'] = (np.mean(true_cs_a, axis=1), np.mean(pred_cs_a, axis=1))
                if csmax_c is not None and 'cs_c' in arr:
                    true_cs_c = orient2d(arr['cs_c'], n, s, e)
                    pred_cs_c = pred['theta_c'] * float(csmax_c)
                    pairs['cs_c'] = (true_cs_c, pred_cs_c); pairs['cs_c_mean'] = (np.mean(true_cs_c, axis=1), np.mean(pred_cs_c, axis=1))
                for name, (tru, prd) in pairs.items():
                    accs[name].update(tru, prd)
                    add_group('ALL', name, tru, prd)
                    add_group(f"split:{meta['split']}", name, tru, prd)
                    add_group(f"batch:{meta['batch']}", name, tru, prd)
                    add_group(f"protocol:{meta['protocol']}", name, tru, prd)
                outside_count += int(np.sum((pred['theta_a'] < -1e-6) | (pred['theta_a'] > 1 + 1e-6)) + np.sum((pred['theta_c'] < -1e-6) | (pred['theta_c'] > 1 + 1e-6)))
                outside_total += int(pred['theta_a'].size + pred['theta_c'].size)
                if s == 0 or e == n:
                    print(f"[D16-P5K-F eval v3] {meta['profile_id']}: chunk {s}:{e}/{n}", flush=True)
            r = dict(meta); r['n_time'] = n; r['csmax_a_est'] = csmax_a if csmax_a is not None else ''; r['csmax_c_est'] = csmax_c if csmax_c is not None else ''
            for name, ac in accs.items(): r.update(ac.row(name))
            r['pred_theta_outside_fraction'] = outside_count / max(1, outside_total)
            metrics_rows.append(r)
        except Exception as exc:
            failures.append({**meta, 'softlabel_npz': str(npz_path), 'error': repr(exc)})
            print(f"[D16-P5K-F eval v3] FAIL {meta.get('profile_id')}: {repr(exc)}", flush=True)

    def aggregate_rows(prefix: str) -> List[Dict[str, Any]]:
        out = []
        for group, accdict in sorted(group_acc.items()):
            if not group.startswith(prefix): continue
            name = group.split(':', 1)[1] if ':' in group else group
            row: Dict[str, Any] = {'group': name, 'profile_count': len({r['profile_id'] for r in metrics_rows if (prefix == 'split:' and r['split'] == name) or (prefix == 'batch:' and r['batch'] == name) or (prefix == 'protocol:' and r['protocol'] == name)})}
            for m, ac in accdict.items(): row.update(ac.row(m))
            out.append(row)
        return out

    all_row: Dict[str, Any] = {'group': 'ALL', 'profile_count': len(metrics_rows)}
    for m, ac in group_acc.get('ALL', {}).items(): all_row.update(ac.row(m))

    write_csv(metrics_rows, out_dir / 'D16_P5KF_METRICS_BY_PROFILE.csv')
    write_csv(aggregate_rows('split:'), out_dir / 'D16_P5KF_SPLIT_METRICS.csv')
    write_csv(aggregate_rows('batch:'), out_dir / 'D16_P5KF_BATCH_METRICS.csv')
    write_csv(aggregate_rows('protocol:'), out_dir / 'D16_P5KF_PROTOCOL_METRICS.csv')
    write_json(failures, out_dir / 'D16_P5KF_FAILURES.json')

    score = {
        'stage': 'D16-P5K-F hard-cbar/OCP residual evaluation',
        'manifest': str(args.manifest), 'model_dir': str(args.model_dir), 'checkpoint': str(ckpt_path), 'out_dir': str(out_dir),
        'operational_status': 'PASS' if len(metrics_rows) == len(rows) and not failures else 'REVIEW',
        'profile_count_requested': len(rows), 'profile_count_evaluated': len(metrics_rows), 'failure_count': len(failures),
        'global_metrics_weighted': all_row,
        'promotion_gate_hint': {
            'eval_theta_a_mean_r2_target': '> 0.85', 'eval_theta_c_mean_r2_target': '> 0.85',
            'eval_theta_a_mean_mae_target': '< 0.15', 'eval_theta_c_mean_mae_target': '< 0.15', 'phis_c_r2_target': '> 0.99'
        },
        'notes': [
            'Exact R2 is computed from streaming SSE/SST, not corr^2.',
            'Training used only observed t/I/V; soft-label internal states are used here only for evaluation.',
            'P5K-F uses hard cbar/OCP-style baseline plus bounded residual; it avoids P5G heuristic theta-gap targets.'
        ],
        'failures': failures,
    }
    write_json(score, out_dir / 'D16_P5KF_FINAL_SCORECARD.json')
    print('[D16-P5K-F eval v3] operational_status:', score['operational_status'], 'evaluated=', len(metrics_rows), 'failures=', len(failures), flush=True)
    print('[D16-P5K-F eval v3] wrote:', out_dir / 'D16_P5KF_FINAL_SCORECARD.json', flush=True)
    return 0 if score['operational_status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
