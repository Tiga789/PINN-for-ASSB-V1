from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Import the P5K-F evaluator as a local module so baseline math and exact-R2 accumulation stay identical.
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p5kf_eval", HERE / "gv1_d16_p5kf_eval55_vs_softlabels_v3.py")
p5kf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(p5kf)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text('', encoding='utf-8'); return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys: keys.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description='D16-P5K-F baseline-only exact-R2 preflight. No training and no checkpoint loading.')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--config', default='configs/d16_p5kf_profile_theta0_hard_cbar_config.json')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--softlabel-root', default='')
    ap.add_argument('--mmap-cache-root', default='')
    ap.add_argument('--chunk-size', type=int, default=200000)
    ap.add_argument('--limit-profiles', type=int, default=0)
    ap.add_argument('--allow-overwrite', action='store_true')
    args = ap.parse_args()

    cfg = json.load(open(args.config, 'r', encoding='utf-8'))
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out-dir exists and non-empty: {out_dir}; pass --allow-overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.mmap_cache_root) if args.mmap_cache_root else out_dir / 'mmap_cache_short'
    rows = p5kf.read_manifest(args.manifest)
    if args.limit_profiles and args.limit_profiles > 0:
        rows = rows[:int(args.limit_profiles)]
    softlabel_root = Path(args.softlabel_root) if str(args.softlabel_root).strip() else None

    metrics_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    group_acc: Dict[str, Dict[str, Any]] = {}

    def add_group(group: str, metric: str, true: np.ndarray, pred: np.ndarray):
        group_acc.setdefault(group, {}).setdefault(metric, p5kf.Accum()).update(true, pred)

    for row in rows:
        meta = p5kf.parse_meta(row); npz_path = Path(row['softlabel_npz'])
        try:
            npz_path = p5kf.resolve_npz_path(npz_path, meta.get('profile_id',''), softlabel_root)
            arr = p5kf.load_mmap_arrays(npz_path, cache_root)
            t = p5kf.as_1d_float(arr['t']); I = p5kf.as_1d_float(arr['I']); V = p5kf.as_1d_float(arr['V']); n = len(t)
            stats = {'t_span': float(t[-1]-t[0]) if n > 1 else 1.0, 'I_scale': float(np.nanpercentile(np.abs(I), 99.5)) if n else 1.0, 'I_abs_max': float(np.nanmax(np.abs(I))) if n else 0.0, 'v_mean': float(np.nanmean(V)) if n else 0.0, 'v_std': float(np.nanstd(V)) if n else 1.0}
            if not np.isfinite(stats['t_span']) or stats['t_span'] <= 0: stats['t_span'] = 1.0
            if not np.isfinite(stats['I_scale']) or stats['I_scale'] < 1e-12: stats['I_scale'] = 1.0
            if not np.isfinite(stats['v_std']) or stats['v_std'] < 1e-8: stats['v_std'] = 1.0
            qn = p5kf.build_q_norm(t, I)
            th_a_shape = arr['theta_a'].shape; th_c_shape = arr['theta_c'].shape
            nr_a = int(th_a_shape[1] if len(th_a_shape) == 2 and th_a_shape[0] == n else th_a_shape[0] if len(th_a_shape) == 2 else 1)
            nr_c = int(th_c_shape[1] if len(th_c_shape) == 2 and th_c_shape[0] == n else th_c_shape[0] if len(th_c_shape) == 2 else 1)
            accs = {m: p5kf.Accum() for m in ['theta_a_mean', 'theta_c_mean', 'theta_a', 'theta_c']}
            radial_a = np.linspace(-0.5, 0.5, nr_a, dtype=np.float32); radial_c = np.linspace(-0.5, 0.5, nr_c, dtype=np.float32)
            for s in range(0, n, int(args.chunk_size)):
                e = min(n, s + int(args.chunk_size))
                X = p5kf.feature_chunk(t, I, V, s, e, stats, qn)
                xr = torch.from_numpy(X.astype(np.float32))
                raw_zero = torch.zeros((X.shape[0], 6), dtype=torch.float32)
                with torch.no_grad():
                    y = p5kf.transform_outputs(raw_zero, xr, cfg)
                ta_m = y['theta_a_mean'].cpu().numpy().astype(np.float32)
                tc_m = y['theta_c_mean'].cpu().numpy().astype(np.float32)
                ga = y['grad_a'].cpu().numpy().astype(np.float32)
                gc = y['grad_c'].cpu().numpy().astype(np.float32)
                pred_ta = np.clip(ta_m[:, None] + ga[:, None] * radial_a[None, :], 0.0, 1.0).astype(np.float32)
                pred_tc = np.clip(tc_m[:, None] + gc[:, None] * radial_c[None, :], 0.0, 1.0).astype(np.float32)
                true_ta = p5kf.orient2d(arr['theta_a'], n, s, e); true_tc = p5kf.orient2d(arr['theta_c'], n, s, e)
                true_ta_m = np.mean(true_ta, axis=1); true_tc_m = np.mean(true_tc, axis=1)
                pairs = {'theta_a_mean': (true_ta_m, ta_m), 'theta_c_mean': (true_tc_m, tc_m), 'theta_a': (true_ta, pred_ta), 'theta_c': (true_tc, pred_tc)}
                for name, (tru, prd) in pairs.items():
                    accs[name].update(tru, prd)
                    add_group('ALL', name, tru, prd)
                    add_group(f"split:{meta['split']}", name, tru, prd)
                if s == 0 or e == n:
                    print(f"[D16-P5K-F baseline audit] {meta['profile_id']}: chunk {s}:{e}/{n}", flush=True)
            r = dict(meta); r['n_time'] = n
            for name, ac in accs.items(): r.update(ac.row(name))
            metrics_rows.append(r)
        except Exception as exc:
            failures.append({**meta, 'softlabel_npz': str(npz_path), 'error': repr(exc)})
            print(f"[D16-P5K-F baseline audit] FAIL {meta.get('profile_id')}: {repr(exc)}", flush=True)

    def aggregate(prefix: str) -> List[Dict[str, Any]]:
        out = []
        for group, accdict in sorted(group_acc.items()):
            if not group.startswith(prefix): continue
            name = group.split(':', 1)[1]
            row: Dict[str, Any] = {'group': name, 'profile_count': len({r['profile_id'] for r in metrics_rows if r['split'] == name})}
            for m, ac in accdict.items(): row.update(ac.row(m))
            out.append(row)
        return out

    all_row: Dict[str, Any] = {'group': 'ALL', 'profile_count': len(metrics_rows)}
    for m, ac in group_acc.get('ALL', {}).items(): all_row.update(ac.row(m))
    write_csv(metrics_rows, out_dir / 'D16_P5KF_BASELINE_ONLY_BY_PROFILE.csv')
    write_csv(aggregate('split:'), out_dir / 'D16_P5KF_BASELINE_ONLY_SPLIT_METRICS.csv')
    write_json(failures, out_dir / 'D16_P5KF_BASELINE_ONLY_FAILURES.json')
    score = {'stage': 'D16-P5K-F baseline-only preflight', 'manifest': str(args.manifest), 'config': str(args.config), 'profile_count_requested': len(rows), 'profile_count_evaluated': len(metrics_rows), 'failure_count': len(failures), 'operational_status': 'PASS' if len(metrics_rows) == len(rows) and not failures else 'REVIEW', 'global_metrics_weighted': all_row, 'notes': ['No training, no checkpoint loading. This computes the hard baseline with raw residuals set to zero.', 'Use this before long training to detect P5K-D-style bad baseline drift.']}
    write_json(score, out_dir / 'D16_P5KF_BASELINE_ONLY_SCORECARD.json')
    print('[D16-P5K-F baseline audit] operational_status:', score['operational_status'], 'evaluated=', len(metrics_rows), 'failures=', len(failures), flush=True)
    print('[D16-P5K-F baseline audit] wrote:', out_dir / 'D16_P5KF_BASELINE_ONLY_SCORECARD.json', flush=True)
    return 0 if score['operational_status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
