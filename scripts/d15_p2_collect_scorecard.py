from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn_precision.audit import read_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description='Collect D15-P2 precision benchmark scorecard.')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--audit-dir', required=True)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


def _load_optional(path: Path, failures: List[str]) -> Dict[str, Any]:
    if not path.exists():
        failures.append(f'missing {path}')
        return {}
    try:
        return read_json(path)
    except Exception as exc:
        failures.append(f'failed to read {path}: {exc!r}')
        return {}


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    audit_dir = Path(args.audit_dir)
    failures: List[str] = []
    train = _load_optional(run_dir / 'D15_P2_TRAINING_SUMMARY.json', failures)
    ev = _load_optional(eval_dir / 'D15_P2_EVAL_SUMMARY.json', failures)
    audit = _load_optional(audit_dir / 'D15_P2_PRECISION_AUDIT_SUMMARY.json', failures)
    train_status = train.get('overall_status_sampled_val', train.get('overall_status', 'MISSING'))
    eval_status = ev.get('overall_status', 'MISSING')
    audit_status = audit.get('overall_status', 'MISSING')
    if eval_status not in {'PASS', 'REVIEW'}:
        failures.append(f'eval_status={eval_status}')
    if audit_status not in {'PASS', 'REVIEW'}:
        failures.append(f'audit_status={audit_status}')
    if train_status == 'MISSING':
        failures.append('train_status missing')
    if failures:
        final = 'FAIL'
    elif eval_status == 'PASS' and audit_status == 'PASS':
        final = 'PASS'
    else:
        final = 'REVIEW'
    scorecard = {
        'stage': 'D15-P2 final precision benchmark scorecard',
        'run_dir': str(run_dir),
        'eval_dir': str(eval_dir),
        'audit_dir': str(audit_dir),
        'train_status_sampled_val': train_status,
        'eval_status_full_profile': eval_status,
        'precision_audit_status': audit_status,
        'final_status': final,
        'failures': failures,
        'key_eval_metrics': ev.get('global_metrics', {}),
        'eval_threshold_checks': ev.get('scorecard', {}).get('checks', []),
        'precision_audit_summary': audit.get('aggregate', {}),
        'precision_audit_checks': audit.get('precision_status', {}).get('checks', []),
        'interpretation': 'PASS means the 8-cell closed-set NN precision benchmark reproduced D15-P0 P2Dlite-RG labels under stricter global, per-profile, boundary and transition audits. This is not a held-out generalization proof and not experimental internal-state truth.',
    }
    write_json(scorecard, args.out_json)
    print('[D15-P2 scorecard] final_status:', final)
    print('[D15-P2 scorecard] wrote:', args.out_json)
    return 0 if final in {'PASS', 'REVIEW'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
