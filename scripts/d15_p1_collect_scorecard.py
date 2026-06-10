from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.utils import load_json, write_json


def parse_args():
    p = argparse.ArgumentParser(description='Collect D15-P1 training/eval artifacts into final scorecard.')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--eval-dir', required=True)
    p.add_argument('--out-json', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    train_path = run_dir / 'D15_P1_TRAINING_SUMMARY.json'
    eval_path = eval_dir / 'D15_P1_EVAL_SUMMARY.json'
    failures = []
    if not train_path.exists():
        failures.append(f'missing {train_path}')
        train = {}
    else:
        train = load_json(train_path)
    if not eval_path.exists():
        failures.append(f'missing {eval_path}')
        ev = {}
    else:
        ev = load_json(eval_path)
    eval_status = ev.get('overall_status', 'MISSING')
    train_status = train.get('overall_status_sampled_val', 'MISSING')
    if eval_status not in {'PASS', 'REVIEW'}:
        failures.append(f'eval_status={eval_status}')
    if train_status == 'MISSING':
        failures.append('train_status missing')
    final_status = 'PASS' if eval_status == 'PASS' and not failures else ('REVIEW' if eval_status == 'REVIEW' and not failures else 'FAIL')
    scorecard = {
        'stage': 'D15-P1 final scorecard',
        'run_dir': str(run_dir),
        'eval_dir': str(eval_dir),
        'train_status_sampled_val': train_status,
        'eval_status_full_profile': eval_status,
        'final_status': final_status,
        'failures': failures,
        'key_metrics': ev.get('global_metrics', {}),
        'threshold_checks': ev.get('scorecard', {}).get('checks', []),
        'interpretation': 'PASS means the closed-set NN smoke reproduced D15-P0 P2Dlite-RG labels within smoke thresholds. REVIEW means scripts ran but one or more strict smoke thresholds were not met; inspect metrics before retraining.',
    }
    write_json(scorecard, args.out_json)
    print('[D15-P1 scorecard] final_status:', final_status)
    print('[D15-P1 scorecard] wrote:', args.out_json)
    return 0 if final_status in {'PASS', 'REVIEW'} else 2

if __name__ == '__main__':
    raise SystemExit(main())
