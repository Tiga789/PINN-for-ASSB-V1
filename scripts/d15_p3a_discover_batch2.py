from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2.batch2_io import discover_batch2_mat_files
from gv1.p2dlite_rg_batch2.utils import load_json, parse_battery_id, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='D15-P3A discover Batch-2 raw .mat files.')
    p.add_argument('--config', default='configs/d15_p3_batch2_applicability_config.json')
    p.add_argument('--raw-root', default=None)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    raw_root = Path(args.raw_root or cfg['default_paths']['raw_root_windows'])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_batch2_mat_files(raw_root, cfg.get('raw_reader', {}).get('candidate_batch_dir_names', []))
    rows = []
    for idx, p in enumerate(files, start=1):
        rows.append({'index': idx, 'path': str(p), 'name': p.name, 'battery_id': parse_battery_id(p.name), 'size_bytes': p.stat().st_size})
    write_csv(rows, out_dir / 'D15_P3A_BATCH2_DISCOVERY.csv')
    report = {
        'stage': 'D15-P3A Batch-2 discovery',
        'raw_root': str(raw_root),
        'found_mat_count': len(files),
        'rows_csv': str(out_dir / 'D15_P3A_BATCH2_DISCOVERY.csv'),
        'overall_status': 'PASS' if len(files) >= 3 else 'FAIL',
    }
    write_json(report, out_dir / 'D15_P3A_BATCH2_DISCOVERY_REPORT.json')
    print('[D15-P3A discovery] found_mat_count:', len(files))
    return 0 if len(files) >= 3 else 2


if __name__ == '__main__':
    raise SystemExit(main())
