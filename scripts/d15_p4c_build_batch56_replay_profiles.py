from __future__ import annotations
import argparse, concurrent.futures as cf, sys, traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for p in [ROOT, SCRIPT_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d15_p4c_utils import build_replay_profile_from_mat, load_json, read_csv_rows, write_csv, write_json


def parse_args():
    p = argparse.ArgumentParser(description='D15-P4C build replay profiles for Batch-5/6 remaining 14 cells.')
    p.add_argument('--config', default='configs/d15_p4c_batch56_remaining14_replay_config.json')
    p.add_argument('--raw-manifest-csv', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--save-mode', choices=['compressed','uncompressed'], default=None)
    p.add_argument('--allow-overwrite', action='store_true')
    return p.parse_args()


def _one(row: Dict[str, Any], out_dir: str, reader: Dict[str, Any], save_mode: str) -> Dict[str, Any]:
    can = row['canonical_cell_id']
    out_npz = Path(out_dir) / 'profiles' / can / 'solution_replay_profile.npz'
    if out_npz.exists():
        out_npz.unlink()
    try:
        res = build_replay_profile_from_mat(
            mat_path=row['raw_mat_path'],
            out_npz=out_npz,
            canonical_cell_id=can,
            batch=row['batch'],
            protocol=row['protocol'],
            current_threshold_A=float(reader.get('current_rest_abs_A_threshold', 0.05)),
            temperature_fallback_C=float(reader.get('temperature_fallback_C', 25.0)),
            auto_flip_current=bool(reader.get('auto_flip_current_to_positive_charge', True)),
            save_mode=save_mode,
        )
        d = res.__dict__.copy()
        d['status'] = 'PASS'
        return d
    except Exception as exc:
        return {
            'canonical_cell_id': can, 'batch': row.get('batch'), 'protocol': row.get('protocol'),
            'status': 'FAIL', 'npz_path': str(out_npz), 'raw_mat_path': row.get('raw_mat_path'),
            'error': repr(exc), 'traceback_tail': traceback.format_exc()[-2000:]
        }


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {out_dir}. Use --allow-overwrite for deliberate rerun.')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_csv_rows(args.raw_manifest_csv) if r.get('status','READY_RAW')]
    reader = cfg.get('reader', {})
    save_mode = args.save_mode or reader.get('save_mode', 'compressed')
    workers = max(1, int(args.workers))
    print(f'[D15-P4C build replay] target_count={len(rows)} workers={workers} save_mode={save_mode}')
    results: List[Dict[str, Any]] = []
    if workers == 1:
        for idx, row in enumerate(rows, start=1):
            print(f'[D15-P4C build replay] {idx}/{len(rows)} {row["canonical_cell_id"]}')
            results.append(_one(row, str(out_dir), reader, save_mode))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_one, row, str(out_dir), reader, save_mode): row for row in rows}
            done = 0
            for fut in cf.as_completed(futs):
                done += 1
                res = fut.result()
                results.append(res)
                print(f'[D15-P4C build replay] done {done}/{len(rows)} {res.get("canonical_cell_id")} status={res.get("status")}')
    results = sorted(results, key=lambda r: str(r.get('canonical_cell_id')))
    csv_path = out_dir / 'xjtu_batch56_remaining14_replay_profile_manifest.csv'
    write_csv(results, csv_path)
    pass_count = sum(1 for r in results if r.get('status') == 'PASS')
    err_count = len(results) - pass_count
    report = {
        'stage': 'D15-P4C build Batch-5/6 remaining14 replay profiles',
        'target_count': len(rows),
        'profile_count': len(results),
        'pass_count': pass_count,
        'error_count': err_count,
        'manifest_csv': str(csv_path),
        'output_dir': str(out_dir),
        'save_mode': save_mode,
        'workers': workers,
        'overall_status': 'PASS' if pass_count == len(rows) and err_count == 0 else 'REVIEW'
    }
    write_json(report, out_dir / 'D15_P4C_REPLAY_BUILD_REPORT.json')
    print('[D15-P4C build replay] profile_count:', len(results), 'error_count:', err_count, 'status:', report['overall_status'])
    return 0 if pass_count > 0 else 2

if __name__ == '__main__':
    raise SystemExit(main())
