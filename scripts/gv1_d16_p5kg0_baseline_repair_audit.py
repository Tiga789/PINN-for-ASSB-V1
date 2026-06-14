from __future__ import annotations

import argparse
import csv
import gc as gc_mod
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
# D16-P5K-G0 baseline-repair audit
# No training. No checkpoint loading. No model modification.
#
# Purpose:
#   1) Re-run baseline-only no-regression audit with the cleanup bug fixed.
#   2) Compare P5K-C/P5K-F hard baselines.
#   3) Add diagnostic oracle repair candidates that shift profile theta0 to the
#      soft-label initial mean. These candidates are DIAGNOSTIC ONLY and are not
#      allowed as training/eval promotion evidence because they use soft-label
#      internal states. They answer: "would a correct profile theta0 initializer
#      fix the baseline, or is q-integral/scale/radial also wrong?"
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
    'P5K-C_baseline_eval_from_G_audit': {
        'theta_a_mean_mae': 0.139017,
        'theta_a_mean_r2': 0.474238,
        'theta_c_mean_mae': 0.123569,
        'theta_c_mean_r2': 0.391913,
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


def make_markdown_report(report_path: Path, args: argparse.Namespace, summary_rows: List[Dict[str, Any]], split_rows: List[Dict[str, Any]], by_profile_rows: List[Dict[str, Any]], failures: List[Dict[str, Any]], model_paths: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    lines.append('# D16-P5K-G0 Baseline-Repair Audit Report')
    lines.append('')
    lines.append('This is a **no-training** audit. It does not load or modify checkpoints. It evaluates hard baselines and diagnostic repair candidates with raw residuals set to zero, against D15 ALL55 P2Dlite-RG soft labels.')
    lines.append('')
    lines.append('Important: candidates containing `theta0_oracle` use soft-label initial internal states for diagnosis only. They are not deployable training/evaluation baselines. They are used to answer whether a correct profile-level theta0 initializer could repair the observed baseline failure.')
    lines.append('')
    lines.append('## 0. Run metadata')
    lines.append(f'- manifest: `{args.manifest}`')
    lines.append(f'- softlabel_root: `{args.softlabel_root}`')
    lines.append(f'- out_dir: `{args.out_dir}`')
    lines.append(f'- models: `{args.models}`')
    lines.append(f'- include_theta0_oracle: `{args.include_theta0_oracle}`')
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
    lines.append('## 2. Reference values')
    lines.append('| reference | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | phis_c_r2 |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for name, ref in REFERENCE.items():
        lines.append(f"| {name} | {fmt(ref.get('theta_a_mean_mae'))} | {fmt(ref.get('theta_a_mean_r2'))} | {fmt(ref.get('theta_c_mean_mae'))} | {fmt(ref.get('theta_c_mean_r2'))} | {fmt(ref.get('phis_c_r2'))} |")
    lines.append('')
    lines.append('## 3. Baseline / repair split metrics')
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
    lines.append('## 4. Automatic verdict')
    split_by = {(r.get('model'), r.get('group')): r for r in split_rows}
    verdicts: List[str] = []
    pc = split_by.get(('P5K-C-baseline', 'eval'))
    pf = split_by.get(('P5K-F-baseline', 'eval'))
    if pc and pf:
        da_mae = finite_float(pf.get('theta_a_mean_mae')) - finite_float(pc.get('theta_a_mean_mae'))
        dc_mae = finite_float(pf.get('theta_c_mean_mae')) - finite_float(pc.get('theta_c_mean_mae'))
        da_r2 = finite_float(pf.get('theta_a_mean_r2')) - finite_float(pc.get('theta_a_mean_r2'))
        dc_r2 = finite_float(pf.get('theta_c_mean_r2')) - finite_float(pc.get('theta_c_mean_r2'))
        verdicts.append(f'P5K-F baseline vs P5K-C baseline on eval: Δtheta_a_mean_mae={fmt(da_mae)}, Δtheta_a_mean_r2={fmt(da_r2)}, Δtheta_c_mean_mae={fmt(dc_mae)}, Δtheta_c_mean_r2={fmt(dc_r2)}.')
        if da_mae <= 0.005 and dc_mae <= 0.005 and da_r2 >= -0.03 and dc_r2 >= -0.03:
            verdicts.append('Normal-eval no-regression gate for P5K-F baseline: PASS/near-PASS.')
        else:
            verdicts.append('Normal-eval no-regression gate for P5K-F baseline: FAIL. Do not start long training from the current P5K-F initializer.')
    best_eval = None
    for r in split_rows:
        if r.get('group') == 'eval':
            score = finite_float(r.get('theta_a_mean_mae')) + finite_float(r.get('theta_c_mean_mae')) - 0.1 * (finite_float(r.get('theta_a_mean_r2')) + finite_float(r.get('theta_c_mean_r2')))
            if best_eval is None or score < best_eval[0]:
                best_eval = (score, r)
    if best_eval:
        r = best_eval[1]
        verdicts.append(f"Best eval baseline/diagnostic candidate by combined score: {r.get('model')} with theta_a_mean_mae={fmt(r.get('theta_a_mean_mae'))}, theta_a_mean_r2={fmt(r.get('theta_a_mean_r2'))}, theta_c_mean_mae={fmt(r.get('theta_c_mean_mae'))}, theta_c_mean_r2={fmt(r.get('theta_c_mean_r2'))}.")
    if args.include_theta0_oracle:
        for base in ['P5K-C', 'P5K-F']:
            b = split_by.get((f'{base}-baseline', 'hard_probe'))
            o = split_by.get((f'{base}-theta0_oracle', 'hard_probe'))
            if b and o:
                verdicts.append(
                    f"{base} theta0_oracle hard_probe delta: "
                    f"theta_a_mean_mae {fmt(finite_float(o.get('theta_a_mean_mae')) - finite_float(b.get('theta_a_mean_mae')))}, "
                    f"theta_c_mean_mae {fmt(finite_float(o.get('theta_c_mean_mae')) - finite_float(b.get('theta_c_mean_mae')))}, "
                    f"theta_a_mean_r2 {fmt(finite_float(o.get('theta_a_mean_r2')) - finite_float(b.get('theta_a_mean_r2')))}, "
                    f"theta_c_mean_r2 {fmt(finite_float(o.get('theta_c_mean_r2')) - finite_float(b.get('theta_c_mean_r2')))}."
                )
    if failures:
        verdicts.append(f'There are {len(failures)} real processing failures. These must be fixed before interpreting audit results.')
    else:
        verdicts.append('Real processing failure_count=0. The previous `.collect()` cleanup bug is fixed in this G0 audit package.')
    for v in verdicts:
        lines.append(f'- {v}')
    lines.append('')
    lines.append('## 5. Worst profiles by theta mean MAE sum')
    worst = sorted(by_profile_rows, key=lambda r: finite_float(r.get('theta_a_mean_mae'), 0.0) + finite_float(r.get('theta_c_mean_mae'), 0.0), reverse=True)[:30]
    lines.append('| rank | model | profile_id | batch | split | theta_a_mean_mae | theta_a_mean_r2 | theta_c_mean_mae | theta_c_mean_r2 | theta_a0_error | theta_c0_error |')
    lines.append('|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|')
    for i, r in enumerate(worst, 1):
        lines.append(f"| {i} | {r.get('model')} | {r.get('profile_id')} | {r.get('batch')} | {r.get('split')} | {fmt(r.get('theta_a_mean_mae'))} | {fmt(r.get('theta_a_mean_r2'))} | {fmt(r.get('theta_c_mean_mae'))} | {fmt(r.get('theta_c_mean_r2'))} | {fmt(r.get('theta_a0_error'))} | {fmt(r.get('theta_c0_error'))} |")
    lines.append('')
    lines.append('## 6. Output files')
    lines.append(f'- by_profile_csv: `{Path(args.out_dir) / "D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv"}`')
    lines.append(f'- split_metrics_csv: `{Path(args.out_dir) / "D16_P5KG0_BASELINE_REPAIR_SPLIT_METRICS.csv"}`')
    lines.append(f'- model_summary_csv: `{Path(args.out_dir) / "D16_P5KG0_BASELINE_REPAIR_MODEL_SUMMARY.csv"}`')
    lines.append(f'- failures_json: `{Path(args.out_dir) / "D16_P5KG0_BASELINE_REPAIR_FAILURES.json"}`')
    if failures:
        lines.append('')
        lines.append('## 7. Failures preview')
        for f in failures[:20]:
            lines.append(f"- {f.get('model')} {f.get('profile_id')}: `{f.get('error')}`")
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-G0 baseline-repair audit. No training; raw residuals are zero.')
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--softlabel-root', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--models', default='P5K-C,P5K-F', help='Comma-separated subset of P5K-C,P5K-F')
    ap.add_argument('--mmap-cache-root', default='')
    ap.add_argument('--chunk-size', type=int, default=200000)
    ap.add_argument('--limit-profiles', type=int, default=0)
    ap.add_argument('--include-theta0-oracle', action='store_true', default=True, help='Include diagnostic soft-label theta0 oracle repair candidates. Diagnostic only, not deployable.')
    ap.add_argument('--no-theta0-oracle', dest='include_theta0_oracle', action='store_false')
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

    requested = [m.strip() for m in args.models.split(',') if m.strip()]
    model_defs: List[ModelSpec] = []
    model_paths: List[Dict[str, str]] = []

    def add_model(name: str, script_rel: str, cfg_rel: str) -> None:
        module_path = project_root / script_rel
        config_path = project_root / cfg_rel
        model_paths.append({
            'model': name,
            'module': str(module_path),
            'config': str(config_path),
            'module_exists': str(module_path.exists()),
            'config_exists': str(config_path.exists()),
        })
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

    # Use newest P5K-F module for path resolver / mmap utilities when available.
    loader_mod = next((md.mod for md in model_defs if md.name.startswith('P5K-F')), model_defs[0].mod)

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

    for idx, row in enumerate(rows, 1):
        meta = parse_meta(row)
        pid = meta.get('profile_id', '')
        raw_npz = Path(row.get('softlabel_npz', ''))
        try:
            npz_path = loader_mod.resolve_npz_path(raw_npz, pid, softlabel_root)
            arrs = loader_mod.load_mmap_arrays(npz_path, cache_root)
            t = loader_mod.as_1d_float(arrs['t'])
            I = loader_mod.as_1d_float(arrs['I'])
            V = loader_mod.as_1d_float(arrs['V'])
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
            th_a_shape = arrs['theta_a'].shape
            th_c_shape = arrs['theta_c'].shape
            nr_a = int(th_a_shape[1] if len(th_a_shape) == 2 and th_a_shape[0] == n else th_a_shape[0] if len(th_a_shape) == 2 else 1)
            nr_c = int(th_c_shape[1] if len(th_c_shape) == 2 and th_c_shape[0] == n else th_c_shape[0] if len(th_c_shape) == 2 else 1)
            qn = loader_mod.build_q_norm(t, I)
            csmax_a = estimate_csmax(loader_mod, arrs, 'theta_a', 'cs_a', n)
            csmax_c = estimate_csmax(loader_mod, arrs, 'theta_c', 'cs_c', n)

            candidate_names: List[str] = []
            for md in model_defs:
                candidate_names.append(md.name)
                if args.include_theta0_oracle:
                    candidate_names.append(md.name.replace('-baseline', '-theta0_oracle'))
            profile_accs: Dict[str, Dict[str, Accum]] = {name: metric_dict() for name in candidate_names}
            profile_meta_extra: Dict[str, Dict[str, Any]] = {name: {} for name in candidate_names}
            theta0_shifts: Dict[str, Tuple[float, float]] = {}

            for s in range(0, n, int(args.chunk_size)):
                e = min(n, s + int(args.chunk_size))
                true = true_pairs_for_chunk(loader_mod, arrs, n, s, e)
                for md in model_defs:
                    try:
                        base_pred = baseline_predict(md, t, I, V, stats, qn, s, e, nr_a, nr_c)
                        add_cs_predictions(base_pred, csmax_a, csmax_c)
                        base_pairs = build_metric_pairs(true, base_pred)
                        for k, (tru, prd) in base_pairs.items():
                            profile_accs[md.name][k].update(tru, prd)
                        update_for_model(md.name, meta, pid, base_pairs)
                        if s == 0:
                            shift_a = float(true['theta_a_mean'][0] - base_pred['theta_a_mean'][0])
                            shift_c = float(true['theta_c_mean'][0] - base_pred['theta_c_mean'][0])
                            theta0_shifts[md.name] = (shift_a, shift_c)
                            profile_meta_extra[md.name].update({
                                'true_theta_a0_mean': float(true['theta_a_mean'][0]),
                                'pred_theta_a0_mean': float(base_pred['theta_a_mean'][0]),
                                'theta_a0_error': float(base_pred['theta_a_mean'][0] - true['theta_a_mean'][0]),
                                'true_theta_c0_mean': float(true['theta_c_mean'][0]),
                                'pred_theta_c0_mean': float(base_pred['theta_c_mean'][0]),
                                'theta_c0_error': float(base_pred['theta_c_mean'][0] - true['theta_c_mean'][0]),
                                'theta0_shift_a_oracle': shift_a,
                                'theta0_shift_c_oracle': shift_c,
                            })
                        if args.include_theta0_oracle:
                            shift_a, shift_c = theta0_shifts.get(md.name, (0.0, 0.0))
                            repaired_name = md.name.replace('-baseline', '-theta0_oracle')
                            repaired = apply_theta0_shift(base_pred, shift_a, shift_c, csmax_a, csmax_c)
                            rep_pairs = build_metric_pairs(true, repaired)
                            for k, (tru, prd) in rep_pairs.items():
                                profile_accs[repaired_name][k].update(tru, prd)
                            update_for_model(repaired_name, meta, pid, rep_pairs)
                            if s == 0:
                                profile_meta_extra[repaired_name].update(profile_meta_extra[md.name])
                                profile_meta_extra[repaired_name]['diagnostic_oracle'] = 'softlabel_theta0_initial_shift_only'
                    except Exception as inner_exc:
                        failures.append({**meta, 'model': md.name, 'softlabel_npz': str(npz_path), 'chunk': f'{s}:{e}', 'error': repr(inner_exc)})
                if s == 0 or e == n:
                    print(f'[D16-P5K-G0 baseline repair audit] {idx}/{len(rows)} {pid}: chunk {s}:{e}/{n}', flush=True)

            for cand in candidate_names:
                pr = {**meta, 'model': cand, 'n_time': n}
                pr.update(profile_meta_extra.get(cand, {}))
                for name, ac in profile_accs[cand].items():
                    pr.update(ac.row(name))
                by_profile_rows.append(pr)
            del arrs
            gc_mod.collect()
            if args.cleanup_profile_cache:
                ProfileCacheCleaner(loader_mod, cache_root, npz_path).cleanup()
        except Exception as exc:
            for md in model_defs:
                failures.append({**meta, 'model': md.name, 'softlabel_npz': str(raw_npz), 'error': repr(exc)})
            print(f'[D16-P5K-G0 baseline repair audit] FAIL {pid}: {repr(exc)}', flush=True)

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

    write_csv(by_profile_rows, out_dir / 'D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv')
    write_csv(split_rows, out_dir / 'D16_P5KG0_BASELINE_REPAIR_SPLIT_METRICS.csv')
    write_csv(summary_rows, out_dir / 'D16_P5KG0_BASELINE_REPAIR_MODEL_SUMMARY.csv')
    write_csv(batch_rows, out_dir / 'D16_P5KG0_BASELINE_REPAIR_BATCH_METRICS.csv')
    write_csv(protocol_rows, out_dir / 'D16_P5KG0_BASELINE_REPAIR_PROTOCOL_METRICS.csv')
    write_json(failures, out_dir / 'D16_P5KG0_BASELINE_REPAIR_FAILURES.json')
    write_json({
        'models': model_paths,
        'requested_profiles': len(rows),
        'profile_rows': len(by_profile_rows),
        'failure_count': len(failures),
        'reference': REFERENCE,
        'include_theta0_oracle': bool(args.include_theta0_oracle),
    }, out_dir / 'D16_P5KG0_BASELINE_REPAIR_AUDIT_SUMMARY.json')

    report_path = out_dir / 'D16_P5KG0_BASELINE_REPAIR_AUDIT_REPORT.md'
    make_markdown_report(report_path, args, summary_rows, split_rows, by_profile_rows, failures, model_paths)
    print('[D16-P5K-G0 baseline repair audit] wrote report:', report_path, flush=True)
    print('[D16-P5K-G0 baseline repair audit] real_failure_count:', len(failures), flush=True)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
