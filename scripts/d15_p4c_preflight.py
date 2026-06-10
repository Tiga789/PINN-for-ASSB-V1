from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d15_p4c_utils import discover_raw_mat_for_targets, load_json, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P4C preflight: discover Batch-5/6 remaining 14 raw .mat files.')
    p.add_argument('--config', default='configs/d15_p4c_batch56_remaining14_replay_config.json')
    p.add_argument('--raw-root', default=None)
    p.add_argument('--out-dir', required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    raw_root = Path(args.raw_root or cfg['default_paths']['raw_root_windows'])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    target_cells = list(cfg['target_cells'])
    rows = discover_raw_mat_for_targets(raw_root, target_cells, cfg.get('batch_info', {}))
    found = {r['canonical_cell_id'] for r in rows}
    missing = [c for c in target_cells if c not in found]
    manifest_csv = out_dir / 'D15_P4C_RAW_TARGET_MANIFEST.csv'
    write_csv(rows, manifest_csv)
    report = {
        'stage': 'D15-P4C preflight raw discovery',
        'raw_root': str(raw_root),
        'target_count': len(target_cells),
        'found_raw_count': len(rows),
        'missing_raw_count': len(missing),
        'missing_cells': missing,
        'manifest_csv': str(manifest_csv),
        'overall_status': 'PASS' if len(rows) == len(target_cells) else 'REVIEW'
    }
    write_json(report, out_dir / 'D15_P4C_PREFLIGHT_REPORT.json')
    print('[D15-P4C preflight] found_raw_count:', len(rows), '/', len(target_cells))
    print('[D15-P4C preflight] overall_status:', report['overall_status'])
    return 0 if rows else 2

if __name__ == '__main__':
    raise SystemExit(main())
