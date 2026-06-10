from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.utils import load_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description='Collect D15-P3B final scorecard.')
    p.add_argument('--repair-dir', required=True)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


def _metric(d: Dict[str, Any], path: list[str], default=None):
    x: Any = d
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


def main() -> int:
    args = parse_args()
    repair_dir = Path(args.repair_dir)
    summary_path = repair_dir / 'D15_P3B_BOUNDARY_REPAIR_SUMMARY.json'
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    s = load_json(summary_path)
    projected_status = _metric(s, ['projected_scorecard', 'overall_status'], 'UNKNOWN')
    raw_status = _metric(s, ['raw_scorecard', 'overall_status'], 'UNKNOWN')
    nonreg_status = _metric(s, ['nonregression_scorecard', 'overall_status'], 'UNKNOWN')
    projected_outside = _metric(s, ['projected_global_metrics', 'pred_theta_outside_fraction'], None)
    raw_outside = _metric(s, ['raw_global_metrics', 'pred_theta_outside_fraction'], None)
    final_status = 'PASS' if projected_status == 'PASS' and nonreg_status == 'PASS' else 'REVIEW'
    checks = [
        {'name': 'raw status recorded', 'value': raw_status, 'status': 'PASS' if raw_status in {'PASS', 'REVIEW'} else 'FAIL'},
        {'name': 'projected global scorecard', 'value': projected_status, 'status': 'PASS' if projected_status == 'PASS' else 'REVIEW'},
        {'name': 'projection nonregression', 'value': nonreg_status, 'status': 'PASS' if nonreg_status == 'PASS' else 'REVIEW'},
        {'name': 'projected theta outside fraction', 'value': projected_outside, 'status': 'PASS' if projected_outside is not None and float(projected_outside) <= 0.001 else 'REVIEW'},
    ]
    score = {
        'stage': 'D15-P3B Batch-2 theta boundary projection repair scorecard',
        'repair_dir': str(repair_dir),
        'summary_path': str(summary_path),
        'raw_status': raw_status,
        'projected_status': projected_status,
        'nonregression_status': nonreg_status,
        'raw_pred_theta_outside_fraction': raw_outside,
        'projected_pred_theta_outside_fraction': projected_outside,
        'final_status': final_status,
        'checks': checks,
        'interpretation': (
            'PASS means Batch-2 3-cell NN smoke can be promoted from REVIEW to projection-repaired PASS. '
            'The projection must be reported as inference-time theta projection; it does not alter P2Dlite-RG labels or prove true internal states.'
        ),
    }
    write_json(score, args.out_json)
    print('[D15-P3B scorecard] final_status:', final_status)
    print('[D15-P3B scorecard] wrote:', args.out_json)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
