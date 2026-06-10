from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2.batch2_io import build_replay_profile_from_mat, discover_batch2_mat_files
from gv1.p2dlite_rg_batch2.utils import format_profile_name, load_json, parse_battery_id, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='D15-P3A build Batch-2 replay profiles from raw .mat files.')
    p.add_argument('--config', default='configs/d15_p3_batch2_applicability_config.json')
    p.add_argument('--raw-root', default=None)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--allow-overwrite', action='store_true')
    p.add_argument('--max-cells', type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    raw_root = Path(args.raw_root or cfg['default_paths']['raw_root_windows'])
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {out_dir}. Use --allow-overwrite for deliberate rerun.')
    out_dir.mkdir(parents=True, exist_ok=True)
    reader_cfg = cfg.get('raw_reader', {})
    files = discover_batch2_mat_files(raw_root, reader_cfg.get('candidate_batch_dir_names', []))
    max_cells = args.max_cells if args.max_cells is not None else int(reader_cfg.get('max_cells_to_build', 15))
    files = files[:max_cells]
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for idx, mat_path in enumerate(files, start=1):
        bid = parse_battery_id(mat_path.name) or idx
        profile_id = format_profile_name('Batch-2', '3C', bid)
        profile_dir = out_dir / 'profiles' / profile_id
        out_npz = profile_dir / 'solution_replay_profile.npz'
        print(f'[D15-P3A build replay] {idx}/{len(files)} {mat_path} -> {profile_id}', flush=True)
        try:
            prof = build_replay_profile_from_mat(
                mat_path,
                out_npz,
                profile_id=profile_id,
                battery_id=int(bid),
                current_threshold_A=float(reader_cfg.get('current_rest_abs_A_threshold', 0.05)),
                temperature_fallback_C=float(reader_cfg.get('temperature_fallback_C', 25.0)),
                auto_flip_current=bool(reader_cfg.get('auto_flip_current_to_positive_charge', True)),
            )
            row = {
                'profile_id': prof.profile_id,
                'battery_id': prof.battery_id,
                'profile_npz': str(prof.npz_path),
                'source_mat': str(mat_path),
                'n_time': prof.n_time,
                'cycle_count': prof.cycle_count,
                'current_min_A': prof.current_min_A,
                'current_max_A': prof.current_max_A,
                'voltage_min_V': prof.voltage_min_V,
                'voltage_max_V': prof.voltage_max_V,
                'sign_flipped_to_positive_charge': prof.sign_flipped_to_positive_charge,
                'status': 'PASS',
            }
            rows.append(row)
        except Exception as exc:
            err = {'source_mat': str(mat_path), 'battery_id': bid, 'error': repr(exc), 'status': 'FAIL'}
            print('[D15-P3A build replay] ERROR:', err, flush=True)
            errors.append(err)
    manifest_csv = out_dir / 'xjtu_batch2_replay_profile_manifest.csv'
    write_csv(rows, manifest_csv)
    if errors:
        write_csv(errors, out_dir / 'D15_P3A_BATCH2_REPLAY_ERRORS.csv')
    min_points = int(reader_cfg.get('min_time_points_per_profile', 1000))
    too_short = [r for r in rows if int(r.get('n_time', 0)) < min_points]
    report = {
        'stage': 'D15-P3A Batch-2 replay profile build',
        'raw_root': str(raw_root),
        'output_dir': str(out_dir),
        'source_mat_count_attempted': len(files),
        'profile_count': len(rows),
        'error_count': len(errors),
        'too_short_count': len(too_short),
        'manifest_csv': str(manifest_csv),
        'errors': errors,
        'overall_status': 'PASS' if len(rows) >= 3 and not errors else ('REVIEW' if len(rows) >= 3 else 'FAIL'),
    }
    write_json(report, out_dir / 'D15_P3A_BATCH2_REPLAY_BUILD_REPORT.json')
    print('[D15-P3A build replay] profile_count:', len(rows), 'error_count:', len(errors), 'status:', report['overall_status'])
    return 0 if report['overall_status'] in {'PASS', 'REVIEW'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
