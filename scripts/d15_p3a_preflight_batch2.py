from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2.batch2_io import discover_batch2_mat_files
from gv1.p2dlite_rg_batch2.utils import load_json, parse_battery_id, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='D15-P3A Batch-2 preflight: check raw files and D15 dependencies.')
    p.add_argument('--config', default='configs/d15_p3_batch2_applicability_config.json')
    p.add_argument('--raw-root', default=None)
    p.add_argument('--cache-root', default=None)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    raw_root = Path(args.raw_root or cfg['default_paths']['raw_root_windows'])
    cache_root = Path(args.cache_root or cfg['default_paths']['cache_root_windows'])
    reader_cfg = cfg.get('raw_reader', {})
    files = discover_batch2_mat_files(raw_root, reader_cfg.get('candidate_batch_dir_names', []))
    battery_ids = [parse_battery_id(p.name) for p in files]
    missing_battery_ids = [i for i in range(1, 16) if i not in set(x for x in battery_ids if x is not None)]
    dependency_paths = {
        'D15_P0_prior_json': Path(cfg['p2dlite_rg_generation']['prior_json']).exists(),
        'D15_P0_radial_audit_script': Path('scripts/d15_p0_radial_gradient_audit.py').exists(),
        'D15_P1_train_script': Path('scripts/d15_p1_train_rg_closedset_nn_smoke.py').exists(),
        'D15_P1_eval_script': Path('scripts/d15_p1_eval_rg_closedset_nn_smoke.py').exists(),
        'D15_P1_collect_script': Path('scripts/d15_p1_collect_scorecard.py').exists(),
        'P0_rg_solver_module': Path('gv1/p2dlite_rg/radial_solver.py').exists(),
        'P1_nn_module': Path('gv1/p2dlite_rg_nn/data.py').exists(),
    }
    try:
        import scipy  # noqa: F401
        scipy_ok = True
    except Exception:
        scipy_ok = False
    failures: List[str] = []
    warnings: List[str] = []
    if not raw_root.exists():
        failures.append(f'raw_root_missing: {raw_root}')
    if len(files) < 3:
        failures.append(f'found_less_than_3_batch2_mat_files: {len(files)}')
    if len(files) < int(cfg['batch2']['expected_profile_count_full']):
        warnings.append(f'found_batch2_mat_files={len(files)} expected={cfg["batch2"]["expected_profile_count_full"]}; can still run 3-cell if representative files exist')
    if missing_battery_ids:
        warnings.append('missing_or_unparsed_battery_ids=' + ','.join(map(str, missing_battery_ids)))
    if not scipy_ok:
        failures.append('scipy_missing_for_raw_mat_reader')
    for k, ok in dependency_paths.items():
        if not ok:
            failures.append(f'dependency_missing: {k}')
    report: Dict[str, Any] = {
        'stage': 'D15-P3A Batch-2 preflight',
        'raw_root': str(raw_root),
        'cache_root': str(cache_root),
        'found_mat_count': len(files),
        'found_files': [{'path': str(p), 'battery_id': parse_battery_id(p.name), 'size_bytes': p.stat().st_size if p.exists() else None} for p in files],
        'dependency_checks': dependency_paths,
        'scipy_available': scipy_ok,
        'warnings': warnings,
        'failures': failures,
        'overall_status': 'PASS' if not failures else 'FAIL',
    }
    write_json(report, args.out_json)
    print('[D15-P3A preflight] found_mat_count:', len(files))
    print('[D15-P3A preflight] overall_status:', report['overall_status'])
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
