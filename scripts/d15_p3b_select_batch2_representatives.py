from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2.utils import load_json, read_csv_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='D15-P3B select Batch-2 representative cells for 3-cell RG smoke.')
    p.add_argument('--config', default='configs/d15_p3_batch2_applicability_config.json')
    p.add_argument('--manifest-csv', required=True)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def _int_or_none(x):
    try:
        return int(float(str(x)))
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    rows = [r for r in read_csv_rows(args.manifest_csv) if r.get('status', 'PASS') == 'PASS']
    rows = sorted(rows, key=lambda r: (_int_or_none(r.get('battery_id')) or 10**9, r.get('profile_id', '')))
    preferred = [int(x) for x in cfg['batch2'].get('representative_battery_ids_preferred', [1, 8, 15])]
    selected: List[Dict[str, str]] = []
    used = set()
    for bid in preferred:
        for r in rows:
            if (_int_or_none(r.get('battery_id')) == bid) and (r.get('profile_id') not in used):
                selected.append(r.copy())
                used.add(r.get('profile_id'))
                break
    if len(selected) < 3 and rows:
        idxs = sorted(set([0, len(rows)//2, len(rows)-1]))
        for i in idxs:
            r = rows[i]
            if r.get('profile_id') not in used:
                selected.append(r.copy())
                used.add(r.get('profile_id'))
            if len(selected) >= 3:
                break
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(selected, start=1):
        r['selection_rank'] = i
        r['selection_reason'] = 'preferred_1_8_15_or_first_middle_last'
    out_csv = out_dir / 'D15_P3B_BATCH2_REPRESENTATIVE_MANIFEST.csv'
    write_csv(selected, out_csv)
    report = {
        'stage': 'D15-P3B Batch-2 representative selection',
        'input_manifest_csv': str(args.manifest_csv),
        'available_profile_count': len(rows),
        'selected_count': len(selected),
        'selected_profiles': selected,
        'representative_manifest_csv': str(out_csv),
        'overall_status': 'PASS' if len(selected) == 3 else 'FAIL',
    }
    write_json(report, out_dir / 'D15_P3B_BATCH2_SELECTION_REPORT.json')
    print('[D15-P3B select] selected_count:', len(selected), 'status:', report['overall_status'])
    return 0 if len(selected) == 3 else 2


if __name__ == '__main__':
    raise SystemExit(main())
