from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2.utils import load_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Collect D15-P3 Batch-2 applicability scorecard.')
    p.add_argument('--scorecard-dir', required=True)
    p.add_argument('--preflight-json', required=True)
    p.add_argument('--discovery-json', required=True)
    p.add_argument('--replay-json', required=True)
    p.add_argument('--selection-json', required=True)
    p.add_argument('--generation-json', required=True)
    p.add_argument('--radial-audit-json', required=True)
    p.add_argument('--nn-scorecard-json', default=None)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


def _load_optional(path: str | None) -> Dict[str, Any]:
    if not path:
        return {'overall_status': 'SKIPPED', 'path': None}
    p = Path(path)
    if not p.exists():
        return {'overall_status': 'MISSING', 'path': str(p)}
    return load_json(p)


def main() -> int:
    args = parse_args()
    parts = {
        'preflight': _load_optional(args.preflight_json),
        'discovery': _load_optional(args.discovery_json),
        'replay_build': _load_optional(args.replay_json),
        'representative_selection': _load_optional(args.selection_json),
        'rg_generation': _load_optional(args.generation_json),
        'radial_audit': _load_optional(args.radial_audit_json),
        'nn_smoke': _load_optional(args.nn_scorecard_json),
    }
    failures: List[str] = []
    reviews: List[str] = []
    required_pass_or_review = ['preflight', 'discovery', 'replay_build', 'representative_selection', 'rg_generation', 'radial_audit']
    for name in required_pass_or_review:
        status = str(parts[name].get('overall_status') or parts[name].get('final_status') or '').upper()
        if status == 'FAIL' or status == 'MISSING':
            failures.append(f'{name}_status={status}')
        elif status == 'REVIEW':
            reviews.append(f'{name}_status=REVIEW')
        elif status != 'PASS':
            reviews.append(f'{name}_status={status}')
    nn_status = str(parts['nn_smoke'].get('final_status') or parts['nn_smoke'].get('overall_status') or '').upper()
    if nn_status and nn_status not in {'PASS', 'REVIEW', 'SKIPPED', 'MISSING'}:
        failures.append(f'nn_smoke_status={nn_status}')
    elif nn_status == 'REVIEW':
        reviews.append('nn_smoke_status=REVIEW')
    elif nn_status in {'SKIPPED', 'MISSING'}:
        reviews.append('nn_smoke_skipped_or_missing')
    final_status = 'FAIL' if failures else ('REVIEW' if reviews else 'PASS')
    scorecard = {
        'stage': 'D15-P3 XJTU Batch-2 P2Dlite-RG applicability validation',
        'scope': 'Batch-2 3C charge / 1C discharge applicability extension; first 3 representative cells only unless expanded later.',
        'parts': parts,
        'failures': failures,
        'reviews': reviews,
        'final_status': final_status,
        'next_step_if_pass': 'Inspect review zip; if P3 is PASS/acceptable REVIEW, expand Batch-2 from 3 representative cells to 5-cell/15-cell closed-set benchmark.',
        'important_boundary': 'D15-P3 does not prove experimental radial internal-state truth or held-out Batch-2 generalization.'
    }
    Path(args.scorecard_dir).mkdir(parents=True, exist_ok=True)
    write_json(scorecard, args.out_json)
    print('[D15-P3 scorecard] final_status:', final_status)
    return 0 if final_status in {'PASS', 'REVIEW'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
