from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description='D15-P4D full: collect per-cell generation status files.')
    ap.add_argument('--config', default='configs/d15_p4d_full_remaining14_config.json')
    ap.add_argument('--status-dir', required=True)
    ap.add_argument('--output-softlabels-dir', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()
    cfg = load_json(args.config)
    cells = list(cfg.get('target_cells', []))
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for cell in cells:
        status_path = Path(args.status_dir) / (cell + '.json')
        if not status_path.exists():
            # clean_cell_id is same for our cells; this branch documents missing.
            missing.append(cell)
            rows.append({'canonical_cell_id': cell, 'status': 'MISSING_STATUS'})
            continue
        row = load_json(status_path, {})
        row.setdefault('canonical_cell_id', cell)
        rows.append(row)
    generated_rows = [r for r in rows if str(r.get('status') or r.get('standalone_process_status')).upper() in ('PASS', 'SKIPPED_ALREADY_COMPLETE')]
    fail_rows = [r for r in rows if str(r.get('status') or r.get('standalone_process_status')).upper() == 'FAIL']
    expected = int(cfg.get('target_cell_count', len(cells)))
    report = {
        'stage': 'D15-P4D full Batch-5/6 remaining14 P2Dlite-RG generation report',
        'expected_cell_count': expected,
        'requested_cell_count': len(cells),
        'generated_count': len(generated_rows),
        'error_count': len(fail_rows),
        'missing_status_count': len(missing),
        'missing_status_cells': missing,
        'output_softlabels_dir': str(args.output_softlabels_dir),
        'status_dir': str(args.status_dir),
        'overall_status': 'PASS' if len(generated_rows) == expected and not fail_rows and not missing else 'FAIL',
        'total_time_points': int(sum(int(float(r.get('time_points', 0) or 0)) for r in generated_rows)),
        'total_output_size_mb': float(sum(float(r.get('output_size_mb', 0) or 0) for r in generated_rows)),
    }
    write_json(report, args.out_json)
    write_csv(rows, args.out_csv)
    print('[D15-P4D generation report] overall_status:', report['overall_status'], 'generated:', report['generated_count'], 'errors:', report['error_count'])
    return 0 if report['overall_status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
