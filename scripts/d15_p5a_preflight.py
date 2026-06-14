from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def discover_npz(root: Path, filename: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(filename), key=lambda p: str(p).lower())


def canonical_cell_id(name: str) -> str:
    s = str(name).replace('\\', '/').split('/')[-1]
    m = re.search(r'Batch-([1-6]).*battery-(\d+)', s)
    if m:
        return f'Batch-{m.group(1)}_battery-{int(m.group(2))}'
    m = re.search(r'^\d+_battery-(\d+)_2C_battery-\d+$', s)
    if m:
        return f'Batch-1_battery-{int(m.group(1))}'
    m = re.search(r'^\d+_battery-(\d+)_R2\.5_battery-\d+$', s)
    if m:
        return f'Batch-3_battery-{int(m.group(1))}'
    m = re.search(r'^\d+_battery-(\d+)_R3_battery-\d+$', s)
    if m:
        return f'Batch-4_battery-{int(m.group(1))}'
    raise ValueError(f'Cannot canonicalize {name!r}')


def model_file(model_dir: Path) -> Path | None:
    candidates = [model_dir / 'model' / 'best_with_state.pt', model_dir / 'best_with_state.pt']
    for c in candidates:
        if c.exists():
            return c
    return None


def expected_55() -> List[str]:
    out = []
    for i in range(1, 9):
        out.append(f'Batch-1_battery-{i}')
    for i in range(1, 16):
        out.append(f'Batch-2_battery-{i}')
    for b in [3, 4, 5, 6]:
        for i in range(1, 9):
            out.append(f'Batch-{b}_battery-{i}')
    return out


def parse_args():
    p = argparse.ArgumentParser(description='D15-P5A preflight for ALL55 existing-model transfer evaluation.')
    p.add_argument('--config', default='configs/d15_p5a_all55_existing_model_transfer_config.json')
    p.add_argument('--out-dir', default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    out_dir = Path(args.out_dir or cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(cfg['all55_softlabel_dir'])
    files = discover_npz(root, cfg.get('filename', 'solution_softlabels.npz'))
    rows = []
    canonical_ids = []
    read_errors = []
    for p in files:
        rel_parent = str(p.parent.relative_to(root)).replace('\\', '/') if root in p.parents else p.parent.name
        try:
            cid = canonical_cell_id(p.parent.name)
            canonical_ids.append(cid)
            size_gb = p.stat().st_size / (1024 ** 3)
            rows.append({'cell_id': cid, 'profile_folder': p.parent.name, 'npz_path': str(p), 'size_GB': round(size_gb, 6), 'status': 'FOUND'})
        except Exception as exc:
            read_errors.append({'npz_path': str(p), 'profile_folder': p.parent.name, 'error': repr(exc)})
    expected = expected_55()
    missing = [c for c in expected if c not in canonical_ids]
    extra = [c for c in canonical_ids if c not in expected]
    dupes = sorted([c for c in set(canonical_ids) if canonical_ids.count(c) > 1])
    model_rows = []
    for m in cfg.get('models', []):
        if not m.get('enabled', True):
            continue
        md = Path(m['model_dir'])
        mf = model_file(md)
        model_rows.append({
            'model_id': m.get('model_id', md.name),
            'model_dir': str(md),
            'model_dir_exists': md.exists(),
            'model_file': str(mf) if mf else '',
            'model_file_exists': bool(mf),
            'description': m.get('description', ''),
        })
    failures = []
    warnings = []
    if len(files) != 55:
        failures.append(f'Expected 55 soft-label npz files, found {len(files)}')
    if missing:
        failures.append(f'Missing expected cells: {missing[:10]}... count={len(missing)}')
    if extra:
        failures.append(f'Extra cells: {extra[:10]}... count={len(extra)}')
    if dupes:
        failures.append(f'Duplicate canonical ids: {dupes}')
    if read_errors:
        failures.append(f'Canonicalization/read errors count={len(read_errors)}')
    if not model_rows:
        failures.append('No enabled models configured')
    for r in model_rows:
        if not r['model_file_exists']:
            warnings.append(f"Model file missing for {r['model_id']}: {r['model_dir']}")
    status = 'PASS' if not failures else 'FAIL'
    report = {
        'stage': 'D15-P5A preflight',
        'all55_softlabel_dir': str(root),
        'npz_count': len(files),
        'canonical_cell_count': len(set(canonical_ids)),
        'expected_cell_count': 55,
        'missing_count': len(missing),
        'extra_count': len(extra),
        'duplicate_count': len(dupes),
        'model_count': len(model_rows),
        'model_ready_count': sum(1 for r in model_rows if r['model_file_exists']),
        'failures': failures,
        'warnings': warnings,
        'overall_status': status,
        'notes': cfg.get('notes', []),
    }
    write_csv(rows, out_dir / 'D15_P5A_ALL55_SOFTLABEL_FILE_AUDIT.csv')
    write_csv(model_rows, out_dir / 'D15_P5A_MODEL_PREFLIGHT.csv')
    write_json(read_errors, out_dir / 'D15_P5A_PREFLIGHT_READ_ERRORS.json')
    write_json(report, out_dir / 'D15_P5A_PREFLIGHT.json')
    print('[D15-P5A preflight] npz_count:', len(files))
    print('[D15-P5A preflight] model_ready_count:', report['model_ready_count'], '/', report['model_count'])
    print('[D15-P5A preflight] overall_status:', status)
    # Preflight returns 0 if soft-label set is valid even if some models are missing; eval will skip missing models.
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
