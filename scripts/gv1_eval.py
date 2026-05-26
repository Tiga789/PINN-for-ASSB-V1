#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.pipeline.npz_utils import load_npz_dict
from gv1.pipeline.metrics import regression_metrics, summarize_scorecard, write_metrics_json


def _candidate_pairs(true_arrays: dict, pred_arrays: dict) -> list[tuple[str, str, str]]:
    # name, true_key, pred_key
    pairs = []
    for name in ['voltage_exp', 'phis_c', 'phie', 'cs_a', 'cs_c', 'SOH', 'cbar_a_norm_replay', 'cbar_c_norm_replay']:
        pred_key = f'{name}_pred'
        if name in true_arrays and pred_key in pred_arrays:
            pairs.append((name, name, pred_key))
        if f'{name}_true' in true_arrays and pred_key in pred_arrays:
            pairs.append((name, f'{name}_true', pred_key))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description='GV1 evaluation entry for replay/profile or future prediction npz files.')
    ap.add_argument('--solution_npz', required=True, help='Reference solution/profile npz')
    ap.add_argument('--prediction_npz', default=None, help='Optional predictions npz. If omitted, only profile summary is written.')
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = load_npz_dict(args.solution_npz)
    summary = {
        'solution_npz': args.solution_npz,
        'solution_keys': sorted(ref.keys()),
        'n_time_points': int(len(ref['t_global_s'])) if 't_global_s' in ref else None,
        'has_prediction_npz': args.prediction_npz is not None,
    }
    rows = []
    if args.prediction_npz:
        pred = load_npz_dict(args.prediction_npz)
        for variable, true_key, pred_key in _candidate_pairs(ref, pred):
            m = regression_metrics(ref[true_key], pred[pred_key])
            row = {'variable': variable, 'true_key': true_key, 'pred_key': pred_key}
            row.update(m)
            rows.append(row)
        if rows:
            summarize_scorecard(rows, out_dir / 'scorecard.csv')
        summary['prediction_npz'] = args.prediction_npz
        summary['evaluated_variables'] = [r['variable'] for r in rows]
    write_metrics_json(summary, out_dir / 'eval_summary.json')
    print(json.dumps({'ok': True, 'output_dir': str(out_dir), 'evaluated_variables': summary.get('evaluated_variables', [])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
