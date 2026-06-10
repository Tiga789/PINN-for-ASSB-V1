from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running from project root without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg.audit import audit_npz_file, flatten_audit_for_csv
from gv1.p2dlite_rg.io_utils import discover_softlabel_npz, load_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='D15-P0 radial-gradient audit for XJTU P2Dlite soft labels.')
    p.add_argument('--source-dir', required=True, help='Directory containing per-profile solution_softlabels.npz files.')
    p.add_argument('--prior-json', default='configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json')
    p.add_argument('--out-dir', required=True, help='Audit output directory. New files will be written here.')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--allow-overwrite', action='store_true', help='Allow writing into an existing audit directory.')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_dir)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {out}. Use --allow-overwrite to append/replace audit files.')
    out.mkdir(parents=True, exist_ok=True)
    prior = load_json(args.prior_json)
    files = discover_softlabel_npz(source, filename=args.filename)
    if not files:
        raise FileNotFoundError(f'No soft-label npz files found under {source}')

    results = []
    rows = []
    for idx, npz_path in enumerate(files, start=1):
        print(f'[D15-P0 audit] {idx}/{len(files)} {npz_path}', flush=True)
        try:
            result = audit_npz_file(npz_path, source, prior)
        except Exception as exc:
            result = {
                'profile_id': str(npz_path),
                'npz_path': str(npz_path),
                'overall_flag': 'READ_ERROR',
                'error': repr(exc),
            }
        results.append(result)
        rows.append(flatten_audit_for_csv(result) if result.get('overall_flag') != 'READ_ERROR' else result)

    flags = [r.get('overall_flag', 'UNKNOWN') for r in results]
    summary = {
        'stage': 'D15-P0 radial-gradient audit',
        'source_dir': str(source),
        'prior_json': str(args.prior_json),
        'profile_count': len(results),
        'pass_count': sum(f == 'PASS' for f in flags),
        'warn_count': sum(f == 'WARN' for f in flags),
        'fail_count': sum(f == 'FAIL' for f in flags),
        'read_error_count': sum(f == 'READ_ERROR' for f in flags),
        'overall_status': 'PASS' if all(f == 'PASS' for f in flags) else ('FAIL' if any(f in ('FAIL', 'READ_ERROR') for f in flags) else 'WARN'),
        'results_json': 'radial_gradient_audit_by_profile.json',
        'results_csv': 'radial_gradient_audit_by_profile.csv',
    }
    write_json(summary, out / 'radial_gradient_audit_summary.json')
    write_json(results, out / 'radial_gradient_audit_by_profile.json')

    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out / 'radial_gradient_audit_by_profile.csv', 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print('[D15-P0 audit] wrote:', out)
    print('[D15-P0 audit] overall_status:', summary['overall_status'])
    return 0 if summary['read_error_count'] == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
