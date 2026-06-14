from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open('r', encoding='utf-8') as f:
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


def discover_profiles(softlabel_root: Path) -> List[Dict[str, Any]]:
    files = sorted(softlabel_root.rglob('solution_softlabels.npz'))
    rows: List[Dict[str, Any]] = []
    for p in files:
        pid = safe_rel(p, softlabel_root)
        batch, battery = parse_batch_battery(pid + '/' + str(p))
        rows.append({
            'profile_id': pid,
            'batch': batch,
            'battery': battery,
            'split': 'eval',
            'reason': 'heldout_eval',
            'softlabel_npz': str(p),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='Build D16-P5K train6/train8/train10 manifest from ALL55 P2Dlite-RG soft-label directory.')
    ap.add_argument('--softlabel-root', required=True)
    ap.add_argument('--config', default='configs/d16_p5k_hard_cbar_ocp_residual_config.json')
    ap.add_argument('--train-set', default='', help='A_train6, B_train8, or C_train10. Defaults to config default_train_set.')
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

    cfg = load_json(args.config)
    train_set = args.train_set or cfg.get('default_train_set', 'C_train10')
    train_sets = cfg.get('train_sets', {})
    if train_set not in train_sets:
        raise KeyError(f'Unknown train set {train_set}; available={sorted(train_sets)}')

    selected = train_sets[train_set]
    wanted = {(str(x['batch']), str(x['battery'])): str(x.get('reason', f'{train_set}_selected')) for x in selected}
    rows = discover_profiles(soft_root)
    if len(rows) != 55:
        print(f'[D16-P5K manifest] WARNING: expected 55 profiles, found {len(rows)}', flush=True)

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
        'stage': 'D16-P5K train6to10 manifest',
        'softlabel_root': str(soft_root),
        'train_set': train_set,
        'profile_count': len(rows),
        'train_count': train_count,
        'eval_count': eval_count,
        'expected_train_count': len(selected),
        'missing_selected_train_cells': missing,
        'status': 'PASS' if train_count == len(selected) and eval_count == len(rows) - len(selected) and not missing else 'FAIL',
        'training_allowed_time_series': cfg.get('principles', {}).get('training_time_series_allowed', []),
        'training_forbidden_softlabel_keys': cfg.get('principles', {}).get('training_time_series_forbidden', []),
        'notes': [
            'Manifest construction reads file paths and metadata only; it does not read cs/theta/phie/phis targets for training.',
            'Soft-label arrays are reserved for evaluation/audit only.',
            'P5K-C train10 adds hard representative regimes to train data while retaining held-out eval for the remaining 45 cells.'
        ]
    }
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print('[D16-P5K manifest] status:', report['status'], 'train_set=', train_set, 'train_count=', train_count, 'eval_count=', eval_count, flush=True)
    print('[D16-P5K manifest] wrote:', out_csv, out_json, flush=True)
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
