from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn_precision.audit import write_json


def load_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(description='D15-P2 full-profile eval with prediction dump for precision audit.')
    p.add_argument('--softlabel-dir', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config', default='configs/d15_p2_precision_benchmark_config.json')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--allow-overwrite', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=65536)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    p1 = ROOT / 'scripts' / 'd15_p1_eval_rg_closedset_nn_smoke.py'
    if not p1.exists():
        raise FileNotFoundError(f'D15-P1 evaluator not found: {p1}')
    cmd = [
        sys.executable, str(p1),
        '--softlabel-dir', str(args.softlabel_dir),
        '--model-dir', str(args.model_dir),
        '--out-dir', str(args.out_dir),
        '--config', str(args.config),
        '--filename', str(args.filename),
        '--device', str(args.device),
        '--batch-size', str(args.batch_size),
        '--save-prediction-npz',
    ]
    if args.allow_overwrite:
        cmd.append('--allow-overwrite')
    print('[D15-P2 eval] invoking D15-P1 evaluator with full prediction dump:', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        print('[D15-P2 eval] underlying evaluator failed with code', rc, flush=True)
        return rc
    out_dir = Path(args.out_dir)
    p1_summary = out_dir / 'D15_P1_EVAL_SUMMARY.json'
    p2_summary = out_dir / 'D15_P2_EVAL_SUMMARY.json'
    if p1_summary.exists():
        summary = load_json(p1_summary)
        summary['stage_alias'] = 'D15-P2 precision benchmark full-profile evaluation summary'
        summary['p2_config'] = str(args.config)
        summary['prediction_npz_saved'] = True
        write_json(summary, p2_summary)
    else:
        write_json({'stage': 'D15-P2 eval', 'overall_status': 'FAIL', 'reason': f'missing {p1_summary}'}, p2_summary)
        return 2
    for src, dst in [
        ('D15_P1_METRICS_BY_PROFILE.csv', 'D15_P2_METRICS_BY_PROFILE.csv'),
        ('D15_P1_METRICS_BY_PROFILE.json', 'D15_P2_METRICS_BY_PROFILE.json'),
    ]:
        sp = out_dir / src
        if sp.exists():
            shutil.copyfile(sp, out_dir / dst)
    print('[D15-P2 eval] wrote:', p2_summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
