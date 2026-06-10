from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg.io_utils import load_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compare D15-P0 old P2Dlite audit vs new P2Dlite-RG audit.')
    p.add_argument('--old-audit-dir', required=True)
    p.add_argument('--rg-audit-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--allow-overwrite', action='store_true')
    return p.parse_args()


def _load_by_profile(audit_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = audit_dir / 'radial_gradient_audit_by_profile.json'
    data = load_json(path)
    return {d.get('profile_id', d.get('npz_path', str(i))): d for i, d in enumerate(data)}


def _get(d: Dict[str, Any], path: str, default: float = float('nan')) -> Any:
    cur: Any = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def main() -> int:
    args = parse_args()
    old_dir = Path(args.old_audit_dir)
    rg_dir = Path(args.rg_audit_dir)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    old = _load_by_profile(old_dir)
    rg = _load_by_profile(rg_dir)
    rows: List[Dict[str, Any]] = []
    for profile_id in sorted(set(old) | set(rg)):
        o = old.get(profile_id, {})
        r = rg.get(profile_id, {})
        row = {
            'profile_id': profile_id,
            'old_flag': o.get('overall_flag'),
            'rg_flag': r.get('overall_flag'),
        }
        for el in ['a', 'c']:
            old_p95 = _get(o, f'{el}.active_abs_gradient_norm_p95')
            rg_p95 = _get(r, f'{el}.active_abs_gradient_norm_p95')
            old_mean = _get(o, f'{el}.active_abs_gradient_norm_mean')
            rg_mean = _get(r, f'{el}.active_abs_gradient_norm_mean')
            row[f'{el}_old_active_p95_grad_norm'] = old_p95
            row[f'{el}_rg_active_p95_grad_norm'] = rg_p95
            row[f'{el}_delta_active_p95_grad_norm'] = (rg_p95 - old_p95) if isinstance(rg_p95, (int, float)) and isinstance(old_p95, (int, float)) else float('nan')
            row[f'{el}_old_active_mean_grad_norm'] = old_mean
            row[f'{el}_rg_active_mean_grad_norm'] = rg_mean
            row[f'{el}_old_direction_match'] = _get(o, f'{el}.direction_match_fraction')
            row[f'{el}_rg_direction_match'] = _get(r, f'{el}.direction_match_fraction')
            row[f'{el}_old_mass_cbar_mae_norm'] = _get(o, f'{el}.mass_cbar_mae_norm')
            row[f'{el}_rg_mass_cbar_mae_norm'] = _get(r, f'{el}.mass_cbar_mae_norm')
        rows.append(row)

    promoted = [x for x in rows if x.get('rg_flag') == 'PASS']
    summary = {
        'stage': 'D15-P0 old-vs-RG radial-gradient comparison',
        'old_audit_dir': str(old_dir),
        'rg_audit_dir': str(rg_dir),
        'profile_count_compared': len(rows),
        'rg_pass_count': len(promoted),
        'rg_warn_count': sum(x.get('rg_flag') == 'WARN' for x in rows),
        'rg_fail_count': sum(x.get('rg_flag') == 'FAIL' for x in rows),
        'recommendation': 'Proceed to D15-P1 8-cell RG closed-set NN smoke if rg_fail_count == 0; inspect WARN profiles before expanding beyond 8 cells.',
        'comparison_csv': 'D15_P0_RADIAL_OLD_VS_RG_COMPARISON.csv',
    }
    write_json(summary, out / 'D15_P0_RADIAL_OLD_VS_RG_COMPARISON_SUMMARY.json')
    with open(out / 'D15_P0_RADIAL_OLD_VS_RG_COMPARISON.csv', 'w', encoding='utf-8-sig', newline='') as f:
        fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ['profile_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('[D15-P0 compare] wrote:', out)
    print('[D15-P0 compare] rg_pass_count:', summary['rg_pass_count'], 'of', summary['profile_count_compared'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
