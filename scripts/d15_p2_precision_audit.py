from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn_precision.audit import (
    aggregate_rows,
    audit_prediction_file,
    precision_status,
    read_json,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description='D15-P2 precision audit: per-profile/electrode boundary, transition, top-k error, cycle-level metrics.')
    p.add_argument('--softlabel-dir', required=True)
    p.add_argument('--eval-dir', required=True, help='D15-P2 eval dir containing predictions/*.npz')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config', default='configs/d15_p2_precision_benchmark_config.json')
    p.add_argument('--filename', default='solution_softlabels.npz')
    p.add_argument('--allow-overwrite', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = read_json(args.config)
    audit_cfg = cfg.get('precision_audit', {})
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'out_dir exists and is non-empty: {out_dir}; pass --allow-overwrite')
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = Path(args.eval_dir) / 'predictions'
    preds = sorted(pred_root.rglob('*.npz'))
    rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    print(f'[D15-P2 audit] found prediction files: {len(preds)}', flush=True)
    if not preds:
        failures.append(f'no prediction npz found under {pred_root}')
    for p in preds:
        try:
            row, top, cyc = audit_prediction_file(p, Path(args.softlabel_dir), audit_cfg, filename=args.filename)
            rows.append(row)
            top_rows.extend(top)
            cycle_rows.extend(cyc)
            print(f'[D15-P2 audit] {row.get("profile_id")}: phis_c_mae={row.get("phis_c_mae"):.6g} theta_a_mae={row.get("theta_a_mae"):.6g} theta_c_mae={row.get("theta_c_mae"):.6g} outside={row.get("pred_theta_outside_fraction"):.6g}', flush=True)
        except Exception as exc:
            failures.append(f'{p}: {exc!r}')
    aggregate = aggregate_rows(rows)
    status = precision_status(rows, aggregate, audit_cfg)
    if failures:
        status['overall_status'] = 'FAIL'
        status['read_failures'] = failures
    summary = {
        'stage': 'D15-P2 precision audit',
        'softlabel_dir': str(args.softlabel_dir),
        'eval_dir': str(args.eval_dir),
        'prediction_file_count': len(preds),
        'profile_count_audited': len(rows),
        'aggregate': aggregate,
        'precision_status': status,
        'overall_status': status.get('overall_status', 'FAIL'),
        'failures': failures,
        'notes': cfg.get('audit_notes', []),
    }
    write_csv(rows, out_dir / 'D15_P2_PRECISION_AUDIT_BY_PROFILE.csv')
    write_json(rows, out_dir / 'D15_P2_PRECISION_AUDIT_BY_PROFILE.json')
    write_csv(top_rows, out_dir / 'D15_P2_TOPK_ERROR_WINDOWS.csv')
    write_json(top_rows, out_dir / 'D15_P2_TOPK_ERROR_WINDOWS.json')
    write_csv(cycle_rows, out_dir / 'D15_P2_CYCLE_LEVEL_AUDIT.csv')
    write_json(summary, out_dir / 'D15_P2_PRECISION_AUDIT_SUMMARY.json')
    print('[D15-P2 audit] overall_status:', summary['overall_status'])
    print('[D15-P2 audit] wrote:', out_dir)
    return 0 if summary['overall_status'] in {'PASS', 'REVIEW'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
