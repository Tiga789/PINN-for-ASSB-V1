from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.parent.relative_to(root)).replace('\\', '/')
    except Exception:
        return path.parent.name


def parse_batch_battery(s: str) -> Tuple[str, str]:
    ss = s.replace('\\', '/').replace('_', '-').replace(' ', '-')
    bm = re.search(r'(Batch-\d+)', ss, flags=re.IGNORECASE)
    batch = bm.group(1) if bm else ''
    am = re.search(r'battery-?(\d+)', ss, flags=re.IGNORECASE)
    battery = f'battery-{int(am.group(1))}' if am else ''
    return batch, battery


def inspect_npz_meta(npz_path: Path) -> Dict[str, Any]:
    # Keep this light. Only inspect zip member names and small scalar metadata if possible.
    meta: Dict[str, Any] = {'npz_path': str(npz_path), 'zip_members': []}
    with zipfile.ZipFile(npz_path, 'r') as zf:
        meta['zip_members'] = zf.namelist()
    # Avoid loading large arrays. We only need key list and path-derived fields.
    return meta


def discover_profiles(softlabel_root: Path) -> List[Dict[str, Any]]:
    files = sorted(softlabel_root.rglob('solution_softlabels.npz'))
    rows = []
    for p in files:
        pid = safe_rel(p, softlabel_root)
        batch, battery = parse_batch_battery(pid + '/' + str(p))
        rows.append({
            'profile_id': pid,
            'batch': batch,
            'battery': battery,
            'split': 'eval',
            'reason': 'heldout_eval49',
            'softlabel_npz': str(p),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='Build D16-P5E train6/eval49 manifest from ALL55 P2Dlite-RG soft-label directory.')
    ap.add_argument('--softlabel-root', required=True)
    ap.add_argument('--config', default='configs/d16_p5e_cathode_gauge_config.json')
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--allow-overwrite', action='store_true')
    args = ap.parse_args()

    soft_root = Path(args.softlabel_root)
    if not soft_root.exists():
        raise FileNotFoundError(f'softlabel root not found: {soft_root}')
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    if (out_csv.exists() or out_json.exists()) and not args.allow_overwrite:
        raise FileExistsError('manifest output exists; pass --allow-overwrite')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_json(Path(args.config))
    selected = cfg.get('selected_train_cells', [])
    wanted = {(str(x['batch']), str(x['battery'])): str(x.get('reason', 'train6_selected')) for x in selected}
    rows = discover_profiles(soft_root)
    if len(rows) != 55:
        print(f'[D16-P5E manifest] WARNING: expected 55 profiles, found {len(rows)}', flush=True)

    found_train = set()
    for r in rows:
        key = (r['batch'], r['battery'])
        if key in wanted:
            r['split'] = 'train'
            r['reason'] = wanted[key]
            found_train.add(key)

    missing = sorted([{'batch': b, 'battery': bat, 'reason': reason} for (b, bat), reason in wanted.items() if (b, bat) not in found_train], key=lambda x: (x['batch'], x['battery']))
    train_count = sum(1 for r in rows if r['split'] == 'train')
    eval_count = sum(1 for r in rows if r['split'] == 'eval')

    fieldnames = ['profile_id', 'batch', 'battery', 'split', 'reason', 'softlabel_npz']
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})

    report = {
        'stage': 'D16-P5E train6/eval49 manifest',
        'softlabel_root': str(soft_root),
        'profile_count': len(rows),
        'train_count': train_count,
        'eval_count': eval_count,
        'expected_train_count': 6,
        'missing_selected_train_cells': missing,
        'status': 'PASS' if train_count == 6 and eval_count == len(rows) - 6 and not missing else 'FAIL',
        'training_allowed_time_series': cfg.get('principles', {}).get('training_time_series_allowed', []),
        'training_forbidden_softlabel_keys': cfg.get('principles', {}).get('training_time_series_forbidden', []),
        'notes': [
            'Manifest construction reads file paths only; it does not read cs/theta/phie/phis targets for training.',
            'Soft-label arrays are reserved for later evaluation only.'
        ]
    }
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('[D16-P5E manifest] status:', report['status'], 'train_count=', train_count, 'eval_count=', eval_count, flush=True)
    print('[D16-P5E manifest] wrote:', out_csv, out_json, flush=True)
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
