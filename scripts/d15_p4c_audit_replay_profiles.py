from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d15_p4c_utils import audit_replay_npz, load_json, read_csv_rows, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P4C audit Batch-5/6 remaining14 replay profiles.')
    p.add_argument('--config', default='configs/d15_p4c_batch56_remaining14_replay_config.json')
    p.add_argument('--manifest-csv', required=True)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args(); cfg = load_json(args.config)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    build_rows = read_csv_rows(args.manifest_csv)
    rows = []
    for idx, r in enumerate(build_rows, start=1):
        if not r.get('npz_path'):
            rows.append({'canonical_cell_id': r.get('canonical_cell_id',''), 'status': 'FAIL', 'read_error': 'missing_npz_path'})
            continue
        print(f'[D15-P4C replay audit] {idx}/{len(build_rows)} {r.get("canonical_cell_id")}')
        rows.append(audit_replay_npz(r['npz_path'], r.get('canonical_cell_id')))
    csv_path = out_dir / 'D15_P4C_REPLAY_AUDIT_BY_PROFILE.csv'
    write_csv(rows, csv_path)
    pass_count = sum(1 for r in rows if r.get('status') == 'PASS')
    fail_count = len(rows) - pass_count
    by_batch = {}
    for r in rows:
        b = str(r.get('canonical_cell_id','')).split('_')[0]
        by_batch.setdefault(b, {'count':0, 'pass':0, 'fail':0})
        by_batch[b]['count'] += 1
        if r.get('status') == 'PASS': by_batch[b]['pass'] += 1
        else: by_batch[b]['fail'] += 1
    report = {
        'stage': 'D15-P4C replay audit',
        'profile_count': len(rows),
        'pass_count': pass_count,
        'fail_count': fail_count,
        'read_error_count': fail_count,
        'by_batch_counts': by_batch,
        'audit_csv': str(csv_path),
        'overall_status': 'PASS' if pass_count == int(cfg['thresholds']['replay_pass_required']) and fail_count == 0 else 'REVIEW'
    }
    write_json(report, out_dir / 'D15_P4C_REPLAY_AUDIT_SUMMARY.json')
    print('[D15-P4C replay audit] pass_count:', pass_count, 'fail_count:', fail_count, 'status:', report['overall_status'])
    return 0 if pass_count > 0 else 2

if __name__ == '__main__':
    raise SystemExit(main())
