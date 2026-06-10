from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2_15cell.utils import battery_sort_key, load_json, read_csv_rows, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P3C select all 15 PASS Batch-2 replay profiles for full-cell expansion.')
    p.add_argument('--config', default='configs/d15_p3c_batch2_15cell_applicability_config.json')
    p.add_argument('--replay-manifest-csv', required=True)
    p.add_argument('--out-csv', required=True)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    expected = int(cfg.get('batch2_scope', {}).get('expected_cell_count', 15))
    rows = [r for r in read_csv_rows(args.replay_manifest_csv) if str(r.get('status', 'PASS')).upper() == 'PASS']
    rows = sorted(rows, key=battery_sort_key)
    for i, r in enumerate(rows):
        r['d15_p3c_profile_index'] = i
        r['d15_p3c_selected'] = 'true'
        r['d15_p3c_scope'] = 'batch2_all15'
    status = 'PASS' if len(rows) == expected else ('REVIEW' if len(rows) >= 1 else 'FAIL')
    report = {
        'stage': 'D15-P3C all-15 Batch-2 manifest selection',
        'replay_manifest_csv': str(args.replay_manifest_csv),
        'out_csv': str(args.out_csv),
        'expected_cell_count': expected,
        'selected_count': len(rows),
        'selected_profile_ids': [r.get('profile_id') for r in rows],
        'selected_battery_ids': [r.get('battery_id') for r in rows],
        'overall_status': status,
        'important_boundary': 'All 15 PASS Batch-2 replay profiles are selected for closed-set applicability expansion.'
    }
    write_csv(rows, args.out_csv)
    write_json(report, args.out_json)
    print('[D15-P3C all15 manifest] selected_count:', len(rows), 'status:', status)
    return 0 if status in {'PASS', 'REVIEW'} else 2

if __name__ == '__main__':
    raise SystemExit(main())
