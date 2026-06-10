from __future__ import annotations
import argparse, csv, json, shutil, os, time
from pathlib import Path
from typing import Any, Dict, List


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


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


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def clean_cell_id(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ('-', '_', '.'):
            keep.append(ch)
        else:
            keep.append('_')
    return ''.join(keep).strip('_') or 'unknown_cell'


def same_file_size(src: Path, dst: Path) -> bool:
    try:
        return dst.exists() and src.stat().st_size == dst.stat().st_size
    except Exception:
        return False


def copy_file(src: Path, dst: Path, force: bool = False) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if same_file_size(src, dst) and not force:
        return {'status': 'SKIP_EXISTS', 'src': str(src), 'dst': str(dst), 'bytes': int(dst.stat().st_size)}
    t0 = time.time()
    shutil.copy2(src, dst)
    dt = max(time.time() - t0, 1e-9)
    n = int(dst.stat().st_size)
    return {'status': 'COPIED', 'src': str(src), 'dst': str(dst), 'bytes': n, 'seconds': dt, 'MBps': (n / 1e6) / dt}


def main() -> int:
    ap = argparse.ArgumentParser(description='Stage D15-P4B replay npz files onto local SSD and rewrite manifest replay_npz paths.')
    ap.add_argument('--manifest-csv', required=True)
    ap.add_argument('--staging-root', required=True)
    ap.add_argument('--out-manifest-csv', required=True)
    ap.add_argument('--out-report-json', required=True)
    ap.add_argument('--force-copy', action='store_true')
    args = ap.parse_args()

    manifest = Path(args.manifest_csv)
    staging_root = Path(args.staging_root)
    replay_stage = staging_root / 'replay_profiles'
    rows = read_csv_rows(manifest)
    ready = [r for r in rows if str(r.get('p4b_ready', '')).lower() in ('true', '1', 'yes', 'pass')]
    staged_rows: List[Dict[str, Any]] = []
    copies: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for i, row in enumerate(ready, start=1):
        can = row.get('canonical_cell_id') or row.get('profile_id') or f'cell_{i:03d}'
        src = Path(row.get('replay_npz', ''))
        if not src.exists():
            errors.append({'canonical_cell_id': can, 'src': str(src), 'error': 'source replay_npz missing'})
            continue
        dst = replay_stage / clean_cell_id(can) / src.name
        try:
            info = copy_file(src, dst, force=args.force_copy)
            info['canonical_cell_id'] = can
            copies.append(info)
            newrow = dict(row)
            newrow['replay_npz_original'] = str(src)
            newrow['replay_npz'] = str(dst)
            newrow['staged_on_ssd'] = 'true'
            staged_rows.append(newrow)
            print(f'[D15-P4B SSD staging] {i}/{len(ready)} {can}: {info["status"]} {info.get("MBps", "")}', flush=True)
        except Exception as exc:
            errors.append({'canonical_cell_id': can, 'src': str(src), 'dst': str(dst), 'error': repr(exc)})
            print(f'[D15-P4B SSD staging] ERROR {i}/{len(ready)} {can}: {exc!r}', flush=True)

    write_csv(staged_rows, args.out_manifest_csv)
    report = {
        'stage': 'D15-P4B SSD staging manifest rewrite',
        'input_manifest_csv': str(manifest),
        'staging_root': str(staging_root),
        'out_manifest_csv': str(args.out_manifest_csv),
        'ready_input_count': len(ready),
        'staged_count': len(staged_rows),
        'error_count': len(errors),
        'total_staged_bytes': sum(int(c.get('bytes', 0)) for c in copies),
        'overall_status': 'PASS' if len(errors) == 0 and len(staged_rows) == len(ready) else 'FAIL',
        'copies': copies,
        'errors': errors,
    }
    write_json(report, args.out_report_json)
    print('[D15-P4B SSD staging] overall_status:', report['overall_status'], 'staged_count:', report['staged_count'], flush=True)
    return 0 if report['overall_status'] == 'PASS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
