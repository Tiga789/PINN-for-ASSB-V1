from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2_15cell.utils import load_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description='Collect D15-P3C Batch-2 15-cell applicability expansion scorecard.')
    p.add_argument('--scorecard-dir', required=True)
    p.add_argument('--all15-manifest-json', required=True)
    p.add_argument('--generation-json', required=True)
    p.add_argument('--radial-audit-json', required=True)
    p.add_argument('--nn-scorecard-json', default=None)
    p.add_argument('--projection-scorecard-json', default=None)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


def _load_optional(path: str | None) -> Dict[str, Any]:
    if not path:
        return {'overall_status': 'SKIPPED', 'path': None}
    p = Path(path)
    if not p.exists():
        return {'overall_status': 'MISSING', 'path': str(p)}
    return load_json(p)


def _status(d: Dict[str, Any]) -> str:
    return str(d.get('final_status') or d.get('overall_status') or d.get('status') or 'UNKNOWN').upper()


def _metric(d: Dict[str, Any], path: List[str], default=None):
    x: Any = d
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


def main() -> int:
    args = parse_args()
    parts = {
        'all15_manifest': _load_optional(args.all15_manifest_json),
        'rg_generation_15cell': _load_optional(args.generation_json),
        'radial_audit_15cell': _load_optional(args.radial_audit_json),
        'raw_nn_15cell': _load_optional(args.nn_scorecard_json),
        'projection_repair_15cell': _load_optional(args.projection_scorecard_json),
    }
    failures: List[str] = []
    reviews: List[str] = []

    for name in ['all15_manifest', 'rg_generation_15cell', 'radial_audit_15cell']:
        st = _status(parts[name])
        if st in {'FAIL', 'MISSING'}:
            failures.append(f'{name}_status={st}')
        elif st != 'PASS':
            reviews.append(f'{name}_status={st}')

    raw_st = _status(parts['raw_nn_15cell'])
    if raw_st in {'FAIL', 'MISSING'}:
        failures.append(f'raw_nn_15cell_status={raw_st}')
    elif raw_st == 'REVIEW':
        reviews.append('raw_nn_15cell_status=REVIEW_raw_boundary_or_max_error_requires_projection_review')
    elif raw_st not in {'PASS', 'SKIPPED'}:
        reviews.append(f'raw_nn_15cell_status={raw_st}')

    proj_st = _status(parts['projection_repair_15cell'])
    if proj_st in {'FAIL', 'MISSING', 'SKIPPED'}:
        failures.append(f'projection_repair_15cell_status={proj_st}')
    elif proj_st != 'PASS':
        reviews.append(f'projection_repair_15cell_status={proj_st}')

    projected_outside = _metric(parts['projection_repair_15cell'], ['projected_pred_theta_outside_fraction'], None)
    if projected_outside is not None and float(projected_outside) > 0.001:
        reviews.append(f'projected_pred_theta_outside_fraction={projected_outside}')

    final_status = 'FAIL' if failures else ('REVIEW' if reviews else 'PASS')
    scorecard = {
        'stage': 'D15-P3C XJTU Batch-2 15-cell P2Dlite-RG applicability expansion',
        'scope': 'Batch-2 full 15-cell closed-set soft-label generation + radial audit + NN benchmark + inference-time theta projection repair.',
        'parts': parts,
        'failures': failures,
        'reviews': reviews,
        'raw_nn_status': raw_st,
        'projection_repair_status': proj_st,
        'projected_pred_theta_outside_fraction': projected_outside,
        'final_status': final_status,
        'interpretation_if_pass': 'Batch-2 15-cell generator/radial-audit and projection-repaired NN closed-set applicability are established. This does not prove held-out generalization or experimental internal-state truth.',
        'next_step_if_pass': 'Optionally archive Batch-2 as covered, then decide whether to fold Batch-2 into a unified all-batch P2Dlite-RG scorecard or run held-out-cell splits later.',
        'important_boundary': 'P3C expands Batch-2 only; it does not change previous D15-P0/P1/P2/P3/P3B results.'
    }
    Path(args.scorecard_dir).mkdir(parents=True, exist_ok=True)
    write_json(scorecard, args.out_json)
    print('[D15-P3C scorecard] final_status:', final_status)
    return 0 if final_status in {'PASS', 'REVIEW'} else 2

if __name__ == '__main__':
    raise SystemExit(main())
