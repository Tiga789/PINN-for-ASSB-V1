from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def pwin(path: str | Path) -> Path:
    s = str(path)
    if len(s) >= 3 and s[1] == ':' and s[2] in ('/', '\\'):
        return Path(s)
    return Path(s)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description='D15-P4B preflight for 18 replay-ready cells.')
    ap.add_argument('--config', default='configs/d15_p4b_ready18_generation_config.json')
    ap.add_argument('--manifest-csv', default=None)
    ap.add_argument('--prior-json', default=None)
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()

    cfg = load_json(args.config)
    manifest = pwin(args.manifest_csv or cfg['p4a_fix_manifest_csv'])
    prior = Path(args.prior_json or cfg['prior_json'])
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    warnings: List[str] = []

    if not manifest.exists():
        failures.append(f'manifest_missing:{manifest}')
    else:
        rows = read_csv(manifest)

    if not prior.exists():
        failures.append(f'prior_missing:{prior}')

    expected = int(cfg.get('expected_ready_cell_count', 18))
    ready_rows = [r for r in rows if str(r.get('p4b_ready', '')).lower() in ('true', '1', 'yes', 'pass')]
    if rows and len(rows) != expected:
        warnings.append(f'manifest_row_count_expected_{expected}_got_{len(rows)}')
    if ready_rows and len(ready_rows) != len(rows):
        failures.append(f'not_all_rows_p4b_ready:ready={len(ready_rows)} total={len(rows)}')

    missing_replay = []
    bad_replay = []
    duplicate_cells = []
    seen = set()
    for r in rows:
        can = r.get('canonical_cell_id', '')
        if can in seen:
            duplicate_cells.append(can)
        seen.add(can)
        npz = r.get('replay_npz', '')
        if not npz or not pwin(npz).exists():
            missing_replay.append(can or npz)
        elif pwin(npz).stat().st_size <= 0:
            bad_replay.append(can or npz)
    if missing_replay:
        failures.append('missing_replay:' + ';'.join(missing_replay[:20]))
    if bad_replay:
        failures.append('bad_replay:' + ';'.join(bad_replay[:20]))
    if duplicate_cells:
        failures.append('duplicate_canonical_cell_id:' + ';'.join(duplicate_cells))

    out = {
        'stage': 'D15-P4B preflight',
        'config': str(args.config),
        'manifest_csv': str(manifest),
        'prior_json': str(prior),
        'manifest_row_count': len(rows),
        'ready_row_count': len(ready_rows),
        'expected_ready_cell_count': expected,
        'missing_replay_count': len(missing_replay),
        'bad_replay_count': len(bad_replay),
        'duplicate_cell_count': len(duplicate_cells),
        'failures': failures,
        'warnings': warnings,
        'overall_status': 'PASS' if not failures else 'FAIL'
    }
    write_json(out, args.out_json)
    print('[D15-P4B preflight] row_count:', len(rows))
    print('[D15-P4B preflight] overall_status:', out['overall_status'])
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
