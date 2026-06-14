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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# -----------------------------------------------------------------------------
# D16-P5K-G baseline-only no-regression audit
# No training. No checkpoint loading required. No soft-label data loss.
# It audits whether candidate hard baselines regress relative to P5K-C/P5K-F.
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
    # Existing final model results, for context only. Exact values came from the project logs.
    'P5K-C_final_eval45': {
        'theta_a_mean_mae': 0.150602,
        'theta_a_mean_r2': 0.423888,
        'theta_c_mean_mae': 0.134245,
        'theta_c_mean_r2': 0.325024,
        'phis_c_r2': 0.999719,
    },
    'P5K-F_final_eval43': {
        'theta_a_mean_mae': 0.146213,
        'theta_a_mean_r2': 0.447594,
        'theta_c_mean_mae': 0.128404,
        'theta_c_mean_r2': 0.362642,
        'phis_c_r2': 0.999488,
    },
    # P5K-E2 baseline-only result for P5K-C; the script recomputes it, this is only a guard sanity reference.
    'P5K-C_baseline_eval45_from_E2': {
        'theta_a_mean_mae': 0.152671,
        'theta_a_mean_r2': 0.381064,
        'theta_c_mean_mae': 0.137611,
        'theta_c_mean_r2': 0.263564,
    },
}


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
        # P5K/P5KF eval modules both expose _safe_name.
        safe = getattr(self.module, '_safe_name', lambda s: str(s).replace(' ', '_'))
        cell_hint = safe(self.npz_path.parent.name)[:64]
        return self.cache_root / f'{cell_hint}_{h}'

    def cleanup(self) -> None:
        p = self.path()
        if p.exists():
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass


def init_group_acc(group_acc: Dict[Tuple[str, str], Dict[str, Accum]], model: str, group: str) -> Dict[str, Accum]:
    key = (model, group)
    if key not in group_acc:
        group_acc[key] = {
            'theta_a_mean': Accum(),
            'theta_c_mean': Accum(),
            'theta_a': Accum(),
            'theta_c': Accum(),
            'grad_a_surface_center': Accum(),
            'grad_c_surface_center': Accum(),
        }
    return group_acc[key]


def update_groups(group_acc: Dict[Tuple[str, str], Dict[str, Accum]], model: str, meta: Dict[str, str], pairs: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    groups = ['ALL', f"split:{meta['split']}", f"batch:{meta['batch']}", f"protocol:{meta['protocol']}"]
    for group in groups:
        accs = init_group_acc(group_acc, model, group)
        for name, (tru, prd) in pairs.items():
            accs[name].update(tru, prd)


def row_from_accs(model: str, group: str, profile_count: int, accs: Dict[str, Accum]) -> Dict[str, Any]:
    r: Dict[str, Any] = {'model': model, 'group': group, 'profile_count': profile_count}
    for name, ac in accs.items():
        r.update(ac.row(name))
    return r


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float('nan')


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


def make_markdown_report(
    report_path: Path,
    args: argparse.Namespace,
    model_summary: List[Dict[str, Any]],
    split_rows: List[Dict[str, Any]],
    by_profile_rows: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    model_paths: List[Dict[str, str]],
) -> None:
    lines: List[str] = []
    lines.append('# D16-P5K-G Baseline-Only No-Regression Audit Report')
    lines.append('')
    lines.append('This is a **no-training** audit. It does not load or modify checkpoints. It evaluates candidate hard baselines with raw residuals set to zero, against D15 ALL55 P2Dlite-RG soft labels.')
    lines.append('')
    lines.append('## 0. Run metadata')
    lines.append(f'- manifest: `{args.manifest}`')
    lines.append(f'- softlabel_root: `{args.softlabel_root}`')
    lines.append(f'- out_dir: `{args.out_dir}`')
    lines.append(f'- models: `{args.models}`')
    lines.append(f'- profile_count_requested: `{args.profile_count_requested}`')
    lines.append(f'- chunk_size: `{args.chunk_size}`')
    lines.append(f'- limit_profiles: `{args.limit_profiles}`')
    lines.append(f'- cleanup_profile_cache: `{args.cleanup_profile_cache}`')
    lines.append('')
    lines.append('## 1. Baseline model provenance')
    lines.append('| model | module | config | module_exists | config_exists |')
    lines.append('|---|---|---|---:|---:|')
    for p in model_paths:
        lines.append(f"| {p['model']} | `{p['module']}` | `{p['config']}` | {p['module_exists']} | {p['config_exists']} |")
    lines.append('')
    lines.append('## 2. No-regression gate context')
    lines.append('Existing final-model reference values used for interpretation:')
    lines.append('| reference | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | phis_c_r2 |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for name, ref in REFERENCE.items():
        lines.append(f"| {name} | {fmt(ref.get('theta_a_mean_mae',''))} | {fmt(ref.get('theta_a_mean_r2',''))} | {fmt(ref.get('theta_c_mean_mae',''))} | {fmt(ref.get('theta_c_mean_r2',''))} | {fmt(ref.get('phis_c_r2',''))} |")
    lines.append('')
    lines.append('## 3. Baseline-only split metrics')
    lines.append('| model | split | profiles | theta_a_mean_mae | theta_a_mean_bias | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_bias | theta_c_mean_r2 | grad_a_r2 | grad_c_r2 |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in split_rows:
        lines.append(
            f"| {r.get('model')} | {r.get('group')} | {r.get('profile_count')} | "
            f"{fmt(r.get('theta_a_mean_mae'))} | {fmt(r.get('theta_a_mean_bias'))} | {fmt(r.get('theta_a_mean_r2'))} | "
            f"{fmt(r.get('theta_c_mean_mae'))} | {fmt(r.get('theta_c_mean_bias'))} | {fmt(r.get('theta_c_mean_r2'))} | "
            f"{fmt(r.get('grad_a_surface_center_r2'))} | {fmt(r.get('grad_c_surface_center_r2'))} |"
        )
    lines.append('')
    lines.append('## 4. Automatic verdict')
    # Determine main model comparisons.
    split_by = {(r.get('model'), r.get('group')): r for r in split_rows}
    verdicts: List[str] = []
    pc = split_by.get(('P5K-C-baseline', 'eval'))
    pf = split_by.get(('P5K-F-baseline', 'eval'))
    ph = split_by.get(('P5K-F-baseline', 'hard_probe'))
    if pc and pf:
        da_mae = safe_float(pf.get('theta_a_mean_mae')) - safe_float(pc.get('theta_a_mean_mae'))
        dc_mae = safe_float(pf.get('theta_c_mean_mae')) - safe_float(pc.get('theta_c_mean_mae'))
        da_r2 = safe_float(pf.get('theta_a_mean_r2')) - safe_float(pc.get('theta_a_mean_r2'))
        dc_r2 = safe_float(pf.get('theta_c_mean_r2')) - safe_float(pc.get('theta_c_mean_r2'))
        verdicts.append(f'P5K-F baseline vs P5K-C baseline on eval: Δtheta_a_mean_mae={fmt(da_mae)}, Δtheta_a_mean_r2={fmt(da_r2)}, Δtheta_c_mean_mae={fmt(dc_mae)}, Δtheta_c_mean_r2={fmt(dc_r2)}.')
        if da_mae <= 0.005 and dc_mae <= 0.005 and da_r2 >= -0.03 and dc_r2 >= -0.03:
            verdicts.append('No-regression gate on normal eval baseline: PASS or near-PASS.')
        else:
            verdicts.append('No-regression gate on normal eval baseline: REVIEW/FAIL. Do not start a long P5K-G training until baseline initializer is improved.')
    if ph:
        hp_mae_a = safe_float(ph.get('theta_a_mean_mae'))
        hp_mae_c = safe_float(ph.get('theta_c_mean_mae'))
        hp_r2_a = safe_float(ph.get('theta_a_mean_r2'))
        hp_r2_c = safe_float(ph.get('theta_c_mean_r2'))
        verdicts.append(f'P5K-F baseline hard_probe: theta_a_mean_mae={fmt(hp_mae_a)}, theta_a_mean_r2={fmt(hp_r2_a)}, theta_c_mean_mae={fmt(hp_mae_c)}, theta_c_mean_r2={fmt(hp_r2_c)}.')
        if hp_mae_a > 0.25 or hp_mae_c > 0.25 or hp_r2_a < -1.0 or hp_r2_c < -1.0:
            verdicts.append('Hard-probe baseline gate: FAIL. The next training should not proceed until hard-profile theta0/OCP phase initialization is corrected or hard probes are isolated from normal training logic.')
        else:
            verdicts.append('Hard-probe baseline gate: not catastrophic; training may be considered if normal eval also passes.')
    if failures:
        verdicts.append(f'There are {len(failures)} profile/model failures. Fix those before interpreting metrics.')
    if not verdicts:
        verdicts.append('Insufficient split rows to determine gate. Check failure CSV and script/module paths.')
    for v in verdicts:
        lines.append(f'- {v}')
    lines.append('')
    lines.append('## 5. Worst baseline-only profiles by theta mean MAE sum')
    lines.append('| rank | model | profile_id | batch | split | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | theta_a_mean_bias | theta_c_mean_bias |')
    lines.append('|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|')
    sorted_prof = sorted(by_profile_rows, key=lambda r: safe_float(r.get('theta_a_mean_mae')) + safe_float(r.get('theta_c_mean_mae')), reverse=True)
    for i, r in enumerate(sorted_prof[:25], 1):
        lines.append(
            f"| {i} | {r.get('model')} | {r.get('profile_id')} | {r.get('batch')} | {r.get('split')} | "
            f"{fmt(r.get('theta_a_mean_mae'))} | {fmt(r.get('theta_a_mean_r2'))} | {fmt(r.get('theta_c_mean_mae'))} | {fmt(r.get('theta_c_mean_r2'))} | {fmt(r.get('theta_a_mean_bias'))} | {fmt(r.get('theta_c_mean_bias'))} |"
        )
    lines.append('')
    lines.append('## 6. Output files')
    lines.append(f'- by_profile_csv: `{Path(args.out_dir) / "D16_P5KG_BASELINE_NOREGRESSION_BY_PROFILE.csv"}`')
    lines.append(f'- split_metrics_csv: `{Path(args.out_dir) / "D16_P5KG_BASELINE_NOREGRESSION_SPLIT_METRICS.csv"}`')
    lines.append(f'- model_summary_csv: `{Path(args.out_dir) / "D16_P5KG_BASELINE_NOREGRESSION_MODEL_SUMMARY.csv"}`')
    lines.append(f'- failures_json: `{Path(args.out_dir) / "D16_P5KG_BASELINE_NOREGRESSION_FAILURES.json"}`')
    lines.append('')
    if failures:
        lines.append('## 7. Failures preview')
        for f in failures[:20]:
            lines.append(f"- {f.get('model')} {f.get('profile_id')}: `{f.get('error')}`")
        lines.append('')
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-G baseline-only no-regression audit. No training; raw residuals are zero.')
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--softlabel-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--models', default='P5K-C,P5K-F', help='Comma-separated subset of P5K-C,P5K-F')
    ap.add_argument('--mmap-cache-root', default='')
    ap.add_argument('--chunk-size', type=int, default=200000)
    ap.add_argument('--limit-profiles', type=int, default=0)
    ap.add_argument('--cleanup-profile-cache', action='store_true', help='Delete each profile mmap cache after it is processed. Recommended when disk space is tight.')
    ap.add_argument('--allow-overwrite', action='store_true')
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out-dir exists and non-empty: {out_dir}; pass --allow-overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.mmap_cache_root) if args.mmap_cache_root else out_dir / 'mmap_cache_short'
    cache_root.mkdir(parents=True, exist_ok=True)

    requested = [m.strip() for m in args.models.split(',') if m.strip()]
    model_defs: List[ModelSpec] = []
    model_paths: List[Dict[str, str]] = []

    def add_model(name: str, script_rel: str, cfg_rel: str):
        module_path = project_root / script_rel
        config_path = project_root / cfg_rel
        model_paths.append({'model': name, 'module': str(module_path), 'config': str(config_path), 'module_exists': str(module_path.exists()), 'config_exists': str(config_path.exists())})
        mod = load_module(module_path, name.replace('-', '_'))
        cfg = json.load(open(config_path, 'r', encoding='utf-8'))
        model_defs.append(ModelSpec(name=f'{name}-baseline', module_path=module_path, config_path=config_path, mod=mod, cfg=cfg))

    if 'P5K-C' in requested:
        add_model('P5K-C', 'scripts/gv1_d16_p5k_eval55_vs_softlabels_v3.py', 'configs/d16_p5k_hard_cbar_ocp_residual_config.json')
    if 'P5K-F' in requested:
        add_model('P5K-F', 'scripts/gv1_d16_p5kf_eval55_vs_softlabels_v3.py', 'configs/d16_p5kf_profile_theta0_hard_cbar_config.json')
    if not model_defs:
        raise ValueError('No valid models selected. Use --models P5K-C,P5K-F')

    rows = read_manifest(args.manifest)
    if args.limit_profiles and args.limit_profiles > 0:
        rows = rows[:int(args.limit_profiles)]
    args.profile_count_requested = len(rows)
    softlabel_root = Path(args.softlabel_root)

    # Use P5K-F module for robust soft-label path and mmap loading. It is the newest v3 implementation.
    loader_mod = None
    for md in model_defs:
        if md.name.startswith('P5K-F'):
            loader_mod = md.mod
            break
    if loader_mod is None:
        loader_mod = model_defs[0].mod

    by_profile_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    group_acc: Dict[Tuple[str, str], Dict[str, Accum]] = {}
    group_profile_ids: Dict[Tuple[str, str], set[str]] = {}

    def note_profile(model_name: str, group: str, pid: str):
        group_profile_ids.setdefault((model_name, group), set()).add(pid)

    for idx, row in enumerate(rows, 1):
        meta = parse_meta(row)
        pid = meta.get('profile_id', '')
        raw_npz = Path(row.get('softlabel_npz', ''))
        try:
            npz_path = loader_mod.resolve_npz_path(raw_npz, pid, softlabel_root)
            arr = loader_mod.load_mmap_arrays(npz_path, cache_root)
            t = loader_mod.as_1d_float(arr['t'])
            I = loader_mod.as_1d_float(arr['I'])
            V = loader_mod.as_1d_float(arr['V'])
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
            th_a_shape = arr['theta_a'].shape
            th_c_shape = arr['theta_c'].shape
            nr_a = int(th_a_shape[1] if len(th_a_shape) == 2 and th_a_shape[0] == n else th_a_shape[0] if len(th_a_shape) == 2 else 1)
            nr_c = int(th_c_shape[1] if len(th_c_shape) == 2 and th_c_shape[0] == n else th_c_shape[0] if len(th_c_shape) == 2 else 1)
            radial_a = np.linspace(-0.5, 0.5, nr_a, dtype=np.float32)
            radial_c = np.linspace(-0.5, 0.5, nr_c, dtype=np.float32)
            # Per-model accumulators for this profile.
            profile_accs = {md.name: {m: Accum() for m in ['theta_a_mean', 'theta_c_mean', 'theta_a', 'theta_c', 'grad_a_surface_center', 'grad_c_surface_center']} for md in model_defs}
            initial_rows: Dict[str, Dict[str, Any]] = {md.name: {} for md in model_defs}

            for s in range(0, n, int(args.chunk_size)):
                e = min(n, s + int(args.chunk_size))
                true_ta = loader_mod.orient2d(arr['theta_a'], n, s, e)
                true_tc = loader_mod.orient2d(arr['theta_c'], n, s, e)
                true_ta_m = np.mean(true_ta, axis=1).astype(np.float32)
                true_tc_m = np.mean(true_tc, axis=1).astype(np.float32)
                true_ga = (true_ta[:, -1] - true_ta[:, 0]).astype(np.float32)
                true_gc = (true_tc[:, -1] - true_tc[:, 0]).astype(np.float32)

                for md in model_defs:
                    mod = md.mod
                    cfg = md.cfg
                    try:
                        qn = mod.build_q_norm(t, I)
                        X = mod.feature_chunk(t, I, V, s, e, stats, qn)
                        xr = torch.from_numpy(X.astype(np.float32))
                        raw_zero = torch.zeros((X.shape[0], 6), dtype=torch.float32)
                        with torch.no_grad():
                            y = mod.transform_outputs(raw_zero, xr, cfg)
                        ta_m = y['theta_a_mean'].cpu().numpy().astype(np.float32)
                        tc_m = y['theta_c_mean'].cpu().numpy().astype(np.float32)
                        ga = y['grad_a'].cpu().numpy().astype(np.float32)
                        gc = y['grad_c'].cpu().numpy().astype(np.float32)
                        pred_ta = np.clip(ta_m[:, None] + ga[:, None] * radial_a[None, :], 0.0, 1.0).astype(np.float32)
                        pred_tc = np.clip(tc_m[:, None] + gc[:, None] * radial_c[None, :], 0.0, 1.0).astype(np.float32)
                        pairs = {
                            'theta_a_mean': (true_ta_m, ta_m),
                            'theta_c_mean': (true_tc_m, tc_m),
                            'theta_a': (true_ta, pred_ta),
                            'theta_c': (true_tc, pred_tc),
                            'grad_a_surface_center': (true_ga, pred_ta[:, -1] - pred_ta[:, 0]),
                            'grad_c_surface_center': (true_gc, pred_tc[:, -1] - pred_tc[:, 0]),
                        }
                        for name, (tru, prd) in pairs.items():
                            profile_accs[md.name][name].update(tru, prd)
                        update_groups(group_acc, md.name, meta, pairs)
                        for group in ['ALL', f"split:{meta['split']}", f"batch:{meta['batch']}", f"protocol:{meta['protocol']}"]:
                            note_profile(md.name, group, pid)
                        if s == 0:
                            initial_rows[md.name] = {
                                'true_theta_a0_mean': float(true_ta_m[0]) if true_ta_m.size else float('nan'),
                                'pred_theta_a0_mean': float(ta_m[0]) if ta_m.size else float('nan'),
                                'theta_a0_error': float(ta_m[0] - true_ta_m[0]) if ta_m.size and true_ta_m.size else float('nan'),
                                'true_theta_c0_mean': float(true_tc_m[0]) if true_tc_m.size else float('nan'),
                                'pred_theta_c0_mean': float(tc_m[0]) if tc_m.size else float('nan'),
                                'theta_c0_error': float(tc_m[0] - true_tc_m[0]) if tc_m.size and true_tc_m.size else float('nan'),
                            }
                    except Exception as inner_exc:
                        failures.append({**meta, 'model': md.name, 'softlabel_npz': str(npz_path), 'chunk': f'{s}:{e}', 'error': repr(inner_exc)})
                if s == 0 or e == n:
                    print(f'[D16-P5K-G baseline audit] {idx}/{len(rows)} {pid}: chunk {s}:{e}/{n}', flush=True)

            for md in model_defs:
                pr = {**meta, 'model': md.name, 'n_time': n}
                pr.update(initial_rows.get(md.name, {}))
                for name, ac in profile_accs[md.name].items():
                    pr.update(ac.row(name))
                by_profile_rows.append(pr)
            # Release mmap arrays before optional cleanup.
            del arr
            gc.collect()
            if args.cleanup_profile_cache:
                ProfileCacheCleaner(loader_mod, cache_root, npz_path).cleanup()
        except Exception as exc:
            for md in model_defs:
                failures.append({**meta, 'model': md.name, 'softlabel_npz': str(raw_npz), 'error': repr(exc)})
            print(f'[D16-P5K-G baseline audit] FAIL {pid}: {repr(exc)}', flush=True)

    # Aggregate rows.
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

    write_csv(by_profile_rows, out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_BY_PROFILE.csv')
    write_csv(split_rows, out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_SPLIT_METRICS.csv')
    write_csv(summary_rows, out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_MODEL_SUMMARY.csv')
    write_csv(batch_rows, out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_BATCH_METRICS.csv')
    write_csv(protocol_rows, out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_PROTOCOL_METRICS.csv')
    write_json(failures, out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_FAILURES.json')
    write_json({'models': model_paths, 'requested_profiles': len(rows), 'profile_rows': len(by_profile_rows), 'failure_count': len(failures), 'reference': REFERENCE}, out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_AUDIT_SUMMARY.json')

    report_path = out_dir / 'D16_P5KG_BASELINE_NOREGRESSION_AUDIT_REPORT.md'
    make_markdown_report(report_path, args, summary_rows, split_rows, by_profile_rows, failures, model_paths)
    print('[D16-P5K-G baseline audit] wrote report:', report_path, flush=True)
    print('[D16-P5K-G baseline audit] failure_count:', len(failures), flush=True)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
