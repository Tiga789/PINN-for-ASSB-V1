from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg.io_utils import discover_softlabel_npz, load_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='D15-P0 no-overwrite and source-data preflight.')
    p.add_argument('--source-dir', required=True)
    p.add_argument('--prior-json', default='configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json')
    p.add_argument('--output-softlabels-dir', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--filename', default='solution_softlabels.npz')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    prior = load_json(args.prior_json)
    source = Path(args.source_dir)
    out = Path(args.output_softlabels_dir)
    files = discover_softlabel_npz(source, filename=args.filename) if source.exists() else []
    warnings = []
    failures = []
    if not source.exists():
        failures.append(f'source-dir does not exist: {source}')
    if not files:
        failures.append(f'no {args.filename} files found under source-dir')
    if out.exists() and any(out.iterdir()):
        failures.append(f'output-softlabels-dir exists and is not empty: {out}')
    for protected in ['xjtu_softlabels_p2dlite_v1_p4b_multicell_v3', 'p5b', 'p5c']:
        if protected.lower() in str(out).lower():
            failures.append(f'output path appears to target protected baseline name: {protected}')
    report = {
        'stage': 'D15-P0 preflight',
        'source_dir': str(source),
        'source_npz_count': len(files),
        'source_npz_preview': [str(p) for p in files[:12]],
        'output_softlabels_dir': str(out),
        'prior_json': str(args.prior_json),
        'prior_schema_version': prior.get('schema_version'),
        'warnings': warnings,
        'failures': failures,
        'overall_status': 'PASS' if not failures else 'FAIL',
    }
    write_json(report, args.out_json)
    print('[D15-P0 preflight] overall_status:', report['overall_status'])
    if failures:
        for f in failures:
            print('  FAIL:', f)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
