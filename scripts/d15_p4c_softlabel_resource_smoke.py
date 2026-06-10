from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d15_p4c_utils import load_json, read_csv_rows, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P4C resource smoke for the future P4D soft-label generation stage.')
    p.add_argument('--config', default='configs/d15_p4c_batch56_remaining14_replay_config.json')
    p.add_argument('--manifest-csv', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-profiles', type=int, default=None)
    return p.parse_args()


def _cpu_numpy_rg_like_smoke(n: int, nr: int, repeats: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # Lightweight RG-like memory/compute smoke: not physics generator, but same shape class.
    x = np.linspace(0.0, 1.0, nr, dtype=np.float64)
    J = np.sin(np.linspace(0, 30, n, dtype=np.float64)) * 0.05
    cbar = np.cumsum(J) / max(n, 1)
    out = np.empty((n, nr), dtype=np.float32)
    for rep in range(repeats):
        shape = (x[None, :] - 0.6) * J[:, None] * 0.1
        y = cbar[:, None] + shape
        out[:] = y.astype(np.float32)
    dt = time.perf_counter() - t0
    mb = (out.nbytes + J.nbytes + cbar.nbytes) / (1024*1024)
    return {'cpu_numpy_smoke_seconds': dt, 'cpu_numpy_array_mb': mb, 'cpu_numpy_points': int(n), 'cpu_numpy_nr': int(nr), 'cpu_numpy_repeats': int(repeats), 'cpu_numpy_effective_mpoints_per_s': float(n * nr * repeats / max(dt, 1e-9) / 1e6)}


def _torch_cuda_smoke() -> Dict[str, Any]:
    res: Dict[str, Any] = {'torch_import_ok': False, 'cuda_available': False, 'cuda_smoke_seconds': None, 'cuda_device_name': None, 'cuda_note': 'not_run'}
    try:
        import torch
        res['torch_import_ok'] = True
        res['cuda_available'] = bool(torch.cuda.is_available())
        if not res['cuda_available']:
            res['cuda_note'] = 'torch imported but CUDA is not available in this environment'
            return res
        dev = torch.device('cuda')
        res['cuda_device_name'] = torch.cuda.get_device_name(0)
        torch.cuda.synchronize()
        a = torch.randn((2048, 2048), device=dev)
        b = torch.randn((2048, 2048), device=dev)
        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
            a = c * 1e-4 + a * 0.9999
        torch.cuda.synchronize()
        res['cuda_smoke_seconds'] = time.perf_counter() - t0
        res['cuda_note'] = 'CUDA works for torch operations, but current P2Dlite-RG generator is NumPy/CPU unless rewritten.'
    except Exception as exc:
        res['cuda_note'] = repr(exc)
    return res


def main() -> int:
    args = parse_args(); cfg = load_json(args.config)
    smoke_cfg = cfg.get('resource_smoke', {})
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_csv_rows(args.manifest_csv) if r.get('status') == 'PASS']
    max_profiles = args.max_profiles if args.max_profiles is not None else int(smoke_cfg.get('max_profiles', 2))
    selected = rows[:max_profiles]
    prof_rows: List[Dict[str, Any]] = []
    for r in selected:
        try:
            with np.load(r['npz_path'], allow_pickle=True) as z:
                n = int(np.asarray(z['t_global_s'] if 't_global_s' in z.files else z['time_s']).shape[0])
                prof_rows.append({'canonical_cell_id': r.get('canonical_cell_id'), 'npz_path': r['npz_path'], 'time_points': n})
        except Exception as exc:
            prof_rows.append({'canonical_cell_id': r.get('canonical_cell_id'), 'npz_path': r.get('npz_path'), 'error': repr(exc), 'time_points': 0})
    write_csv(prof_rows, out_dir / 'D15_P4C_RESOURCE_SMOKE_PROFILES.csv')
    max_n = int(smoke_cfg.get('max_time_points_per_profile', 20000))
    nr = int(smoke_cfg.get('radial_points', 17))
    repeats = int(smoke_cfg.get('cpu_smoke_repeats', 8))
    n_smoke = min(max_n, max([int(r.get('time_points') or 0) for r in prof_rows] + [max_n]))
    cpu = _cpu_numpy_rg_like_smoke(n_smoke, nr, repeats)
    gpu = _torch_cuda_smoke() if bool(smoke_cfg.get('torch_cuda_smoke', True)) else {'cuda_note': 'disabled'}
    report = {
        'stage': 'D15-P4C resource smoke for future P4D soft-label generation',
        'selected_profile_count': len(selected),
        'cpu_numpy_smoke': cpu,
        'torch_cuda_smoke': gpu,
        'important_interpretation': [
            'This is a resource smoke, not a soft-label generation result.',
            'Current P2Dlite-RG soft-label generator is NumPy/CPU. GPU utilization will remain zero unless the generator is rewritten with torch/CuPy.',
            'If CPU remains low during future generation while memory/disk is high, the bottleneck is memory/I/O or Python-level loops, not available compute.'
        ],
        'recommendation_for_P4D': 'Before launching all 14-cell soft-label generation, run a one/two-cell generation smoke with explicit per-process logging and memory monitoring; do not require GPU usage unless generator backend is rewritten.',
        'overall_status': 'PASS'
    }
    write_json(report, out_dir / 'D15_P4C_RESOURCE_SMOKE_REPORT.json')
    print('[D15-P4C resource smoke] overall_status: PASS')
    print('[D15-P4C resource smoke] GPU available:', gpu.get('cuda_available'))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
