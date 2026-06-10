from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from gv1.p2dlite_rg_p4a.utils import batch_protocol, discover_raw_mat_cells, parse_cell_identity_from_text, read_json, write_csv, write_json


def find_npz_by_name(root: Path, filename: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(filename))


def index_existing_rg_softlabels(roots: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root_s in roots:
        root = Path(root_s)
        for npz in find_npz_by_name(root, 'solution_softlabels.npz'):
            b, bat, proto, key = parse_cell_identity_from_text(str(npz))
            if not key:
                b, bat, proto, key = parse_cell_identity_from_text(npz.parent.name)
            row: Dict[str, Any] = {
                'canonical_cell_id': key,
                'batch_id': b if b is not None else '',
                'battery_id': bat if bat is not None else '',
                'protocol_inferred': batch_protocol(b, proto),
                'softlabel_npz': str(npz),
                'softlabel_root': str(root),
                'softlabel_size_bytes': npz.stat().st_size if npz.exists() else '',
                'softlabel_read_ok': False,
                'softlabel_keys_ok': False,
                'softlabel_time_points': '',
                'softlabel_error': '',
            }
            try:
                with np.load(npz, allow_pickle=False) as z:
                    keys = set(z.files)
                    row['softlabel_keys_ok'] = all(k in keys for k in ['cs_a','cs_c','theta_a','theta_c','t_global_s','I_profile'])
                    row['softlabel_read_ok'] = True
                    if 't_global_s' in keys:
                        row['softlabel_time_points'] = int(z['t_global_s'].shape[0])
            except Exception as e:
                row['softlabel_error'] = repr(e)
            rows.append(row)
    dedup: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = r.get('canonical_cell_id') or r.get('softlabel_npz')
        if k not in dedup:
            dedup[k] = r
    out = list(dedup.values())
    out.sort(key=lambda r: (int(r.get('batch_id') or 999), int(r.get('battery_id') or 999)))
    return out


def should_skip_dir(path: str, skip_keywords: List[str]) -> bool:
    low = path.replace('\\','/').lower()
    return any(str(kw).lower() in low for kw in skip_keywords)


def discover_replay_profiles(cache_root: str, preferred_roots: List[str], scan_cache_root: bool, skip_keywords: List[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for root_s in preferred_roots:
        root = Path(root_s)
        if root.exists():
            for p in root.rglob('solution_replay_profile.npz'):
                rp = str(p.resolve()).lower()
                if rp not in seen:
                    found.append(p); seen.add(rp)
    if scan_cache_root:
        root = Path(cache_root)
        if root.exists():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not should_skip_dir(os.path.join(dirpath, d), skip_keywords)]
                if 'solution_replay_profile.npz' in filenames:
                    p = Path(dirpath) / 'solution_replay_profile.npz'
                    rp = str(p.resolve()).lower()
                    if rp not in seen:
                        found.append(p); seen.add(rp)
    return sorted(found)


def audit_replay_npz(npz: Path, required_keys: List[str], optional_keys: List[str]) -> Dict[str, Any]:
    b, bat, proto, key = parse_cell_identity_from_text(str(npz))
    row: Dict[str, Any] = {
        'canonical_cell_id': key,
        'batch_id': b if b is not None else '',
        'battery_id': bat if bat is not None else '',
        'protocol_inferred': batch_protocol(b, proto),
        'replay_npz': str(npz),
        'replay_size_bytes': npz.stat().st_size if npz.exists() else '',
        'replay_read_ok': False,
        'required_keys_ok': False,
        'missing_required_keys': '',
        'present_optional_keys': '',
        'time_points': '',
        'cycle_count': '',
        'current_min_A': '',
        'current_max_A': '',
        'voltage_min_V': '',
        'voltage_max_V': '',
        'time_monotonic_nondec': False,
        'finite_core_ok': False,
        'warnings': '',
        'error': '',
        'status': 'FAIL',
    }
    try:
        with np.load(npz, allow_pickle=False) as z:
            keys = set(z.files)
            missing = [k for k in required_keys if k not in keys]
            row['missing_required_keys'] = ';'.join(missing)
            row['present_optional_keys'] = ';'.join([k for k in optional_keys if k in keys])
            row['required_keys_ok'] = not missing
            if missing:
                return row
            t = np.asarray(z['t_global_s'])
            I = np.asarray(z['I_profile'])
            V = np.asarray(z['voltage_exp'])
            cyc = np.asarray(z['cycle_id'])
            n = int(t.shape[0])
            row['time_points'] = n
            row['cycle_count'] = int(len(np.unique(cyc))) if cyc.size else 0
            row['current_min_A'] = float(np.nanmin(I)) if I.size else float('nan')
            row['current_max_A'] = float(np.nanmax(I)) if I.size else float('nan')
            row['voltage_min_V'] = float(np.nanmin(V)) if V.size else float('nan')
            row['voltage_max_V'] = float(np.nanmax(V)) if V.size else float('nan')
            row['time_monotonic_nondec'] = bool(np.all(np.diff(t) >= -1e-9)) if n > 1 else True
            row['finite_core_ok'] = bool(np.isfinite(t).all() and np.isfinite(I).all() and np.isfinite(V).all())
            warnings = []
            if not row['time_monotonic_nondec']: warnings.append('time_not_monotonic')
            if not row['finite_core_ok']: warnings.append('nonfinite_core')
            if n < 1000: warnings.append('too_few_points')
            if int(row['cycle_count'] or 0) < 2: warnings.append('too_few_cycles')
            row['warnings'] = ';'.join(warnings)
            row['replay_read_ok'] = True
            row['status'] = 'PASS' if row['required_keys_ok'] and row['time_monotonic_nondec'] and row['finite_core_ok'] and n >= 1000 else 'WARN'
    except Exception as e:
        row['error'] = repr(e)
        row['status'] = 'FAIL'
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description='D15-P4A remaining 32-cell replay-profile audit')
    ap.add_argument('--config', default='configs/d15_p4a_remaining32_audit_config.json')
    ap.add_argument('--dataset-root', default=None)
    ap.add_argument('--cache-root', default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--allow-overwrite', action='store_true')
    args = ap.parse_args()
    cfg = read_json(args.config)
    dflt = cfg.get('defaults', {})
    dataset_root = args.dataset_root or dflt.get('dataset_root')
    cache_root = args.cache_root or dflt.get('cache_root')
    out_dir = Path(args.out_dir or dflt.get('out_dir'))
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {out_dir}. Use --allow-overwrite for deliberate rerun.')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[D15-P4A] discover raw XJTU .mat cells...')
    raw_rows = discover_raw_mat_cells(dataset_root)
    write_csv(out_dir / 'D15_P4A_RAW_CELL_INDEX.csv', raw_rows)

    print('[D15-P4A] index existing RG soft labels...')
    soft_rows = index_existing_rg_softlabels(cfg.get('existing_rg_softlabel_roots', []))
    write_csv(out_dir / 'D15_P4A_EXISTING_RG_SOFTLABEL_INDEX.csv', soft_rows)
    existing_keys = {r['canonical_cell_id'] for r in soft_rows if r.get('canonical_cell_id')}

    raw_by_key = {r['canonical_cell_id']: r for r in raw_rows if r.get('canonical_cell_id')}
    existing_cells_rows, remaining_rows = [], []
    for key, rr in raw_by_key.items():
        row = dict(rr)
        row['has_existing_rg_softlabel'] = key in existing_keys
        if key in existing_keys:
            existing_cells_rows.append(row)
        else:
            remaining_rows.append(row)
    existing_cells_rows.sort(key=lambda r: (int(r.get('batch_id') or 999), int(r.get('battery_id') or 999)))
    remaining_rows.sort(key=lambda r: (int(r.get('batch_id') or 999), int(r.get('battery_id') or 999)))
    write_csv(out_dir / 'D15_P4A_EXISTING_CELL_COVERAGE.csv', existing_cells_rows)
    write_csv(out_dir / 'D15_P4A_REMAINING32_CELL_MANIFEST.csv', remaining_rows)

    print('[D15-P4A] discover/audit replay profiles...')
    rs = cfg.get('replay_profile_scan', {})
    replay_paths = discover_replay_profiles(cache_root, rs.get('preferred_roots', []), bool(rs.get('scan_cache_root_for_solution_replay_profile', True)), rs.get('skip_dir_keywords', []))
    replay_audit_rows = [audit_replay_npz(p, cfg.get('required_replay_keys', []), cfg.get('optional_replay_keys', [])) for p in replay_paths]
    best: Dict[str, Dict[str, Any]] = {}
    score = {'PASS': 2, 'WARN': 1, 'FAIL': 0}
    for r in replay_audit_rows:
        key = r.get('canonical_cell_id')
        if not key:
            continue
        old = best.get(key)
        if old is None or (score.get(r.get('status'),0), int(r.get('replay_size_bytes') or 0)) > (score.get(old.get('status'),0), int(old.get('replay_size_bytes') or 0)):
            best[key] = r
    dedup = list(best.values())
    dedup.sort(key=lambda r: (int(r.get('batch_id') or 999), int(r.get('battery_id') or 999)))
    write_csv(out_dir / 'D15_P4A_REPLAY_PROFILE_AUDIT_ALL_DISCOVERED.csv', replay_audit_rows)
    write_csv(out_dir / 'D15_P4A_REPLAY_PROFILE_AUDIT_DEDUP.csv', dedup)

    p4b_rows, missing_rows = [], []
    for r in remaining_rows:
        key = r.get('canonical_cell_id')
        rep = best.get(key)
        row = dict(r)
        if rep is None:
            row.update({'replay_status': 'MISSING', 'replay_npz': '', 'p4b_ready': False})
            missing_rows.append(row)
        else:
            row.update({
                'replay_status': rep.get('status'),
                'replay_npz': rep.get('replay_npz'),
                'replay_time_points': rep.get('time_points'),
                'replay_cycle_count': rep.get('cycle_count'),
                'replay_current_min_A': rep.get('current_min_A'),
                'replay_current_max_A': rep.get('current_max_A'),
                'replay_voltage_min_V': rep.get('voltage_min_V'),
                'replay_voltage_max_V': rep.get('voltage_max_V'),
                'p4b_ready': rep.get('status') == 'PASS',
            })
            if rep.get('status') == 'PASS':
                p4b_rows.append(row)
            else:
                missing_rows.append(row)
    write_csv(out_dir / 'D15_P4A_P4B_INPUT_MANIFEST.csv', p4b_rows)
    write_csv(out_dir / 'D15_P4A_MISSING_OR_BAD_REPLAY_MANIFEST.csv', missing_rows)

    coverage_rows = []
    for batch_id in range(1, 7):
        rb = [r for r in raw_rows if int(r.get('batch_id') or -1) == batch_id]
        eb = [r for r in existing_cells_rows if int(r.get('batch_id') or -1) == batch_id]
        rem = [r for r in remaining_rows if int(r.get('batch_id') or -1) == batch_id]
        ready = [r for r in p4b_rows if int(r.get('batch_id') or -1) == batch_id]
        coverage_rows.append({
            'batch_id': batch_id,
            'protocol': batch_protocol(batch_id),
            'raw_cell_count': len(rb),
            'existing_rg_softlabel_cell_count': len(eb),
            'remaining_cell_count': len(rem),
            'remaining_p4b_ready_count': len(ready),
            'remaining_missing_or_bad_replay_count': len(rem)-len(ready),
            'remaining_cells': ';'.join(r['canonical_cell_id'] for r in rem),
        })
    write_csv(out_dir / 'D15_P4A_BATCH_COVERAGE_MATRIX.csv', coverage_rows)

    expected_total = int(dflt.get('expected_total_cells', 55))
    expected_existing = int(dflt.get('expected_existing_rg_cells', 23))
    expected_remaining = int(dflt.get('expected_remaining_cells', 32))
    failures, warnings = [], []
    if len(raw_rows) != expected_total:
        failures.append(f'raw_cell_count_expected_{expected_total}_got_{len(raw_rows)}')
    if len(existing_cells_rows) != expected_existing:
        warnings.append(f'existing_rg_cell_count_expected_{expected_existing}_got_{len(existing_cells_rows)}')
    if len(remaining_rows) != expected_remaining:
        failures.append(f'remaining_cell_count_expected_{expected_remaining}_got_{len(remaining_rows)}')
    if len(p4b_rows) != len(remaining_rows):
        warnings.append(f'not_all_remaining_cells_have_pass_replay_profiles_ready_{len(p4b_rows)}/{len(remaining_rows)}')
    if any(r.get('canonical_cell_id') == 'Batch-1_battery-8' for r in p4b_rows):
        warnings.append('Batch-1_battery-8_is_p4b_ready_but_must_remain_flagged_outlier_in_scorecards')
    final_status = 'PASS' if not failures and len(p4b_rows) == len(remaining_rows) else ('REVIEW' if not failures else 'FAIL')
    scorecard = {
        'stage': 'D15-P4A',
        'final_status': final_status,
        'raw_cell_count': len(raw_rows),
        'existing_rg_softlabel_cell_count': len(existing_cells_rows),
        'remaining_cell_count': len(remaining_rows),
        'p4b_ready_remaining_cell_count': len(p4b_rows),
        'missing_or_bad_replay_remaining_cell_count': len(missing_rows),
        'replay_profiles_discovered_total': len(replay_audit_rows),
        'replay_profiles_dedup_cell_count': len(dedup),
        'failures': failures,
        'warnings': warnings,
        'outputs': {'out_dir': str(out_dir)}
    }
    write_json(out_dir / 'D15_P4A_FINAL_SCORECARD.json', scorecard)
    print('[D15-P4A] final_status:', final_status)
    print('[D15-P4A] raw/existing/remaining/P4B-ready:', len(raw_rows), len(existing_cells_rows), len(remaining_rows), len(p4b_rows))
    return 0 if final_status in ('PASS','REVIEW') else 2

if __name__ == '__main__':
    raise SystemExit(main())
