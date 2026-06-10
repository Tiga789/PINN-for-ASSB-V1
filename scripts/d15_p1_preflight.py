from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn.data import load_profile_metas
from gv1.p2dlite_rg_nn.utils import load_json, write_json, write_csv

REQUIRED_KEYS = {'theta_a', 'theta_c', 'cs_a', 'cs_c'}
TARGET_ANY = [ {'phis_c_soft', 'phis_c', 'voltage_soft', 'V_soft', 'V_pred'}, {'phie', 'phi_e', 'phi_e_eff'} ]


def parse_args():
    p = argparse.ArgumentParser(description='D15-P1 preflight for P2Dlite-RG closed-set NN smoke.')
    p.add_argument('--softlabel-dir', required=True)
    p.add_argument('--prior-json', default='configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json')
    p.add_argument('--config', default='configs/d15_p1_nn_smoke_config.json')
    p.add_argument('--out-json', required=True)
    p.add_argument('--filename', default='solution_softlabels.npz')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(args.config)
    expected = int(cfg.get('data', {}).get('expected_profile_count', 8))
    failures = []
    warnings = []
    softlabel_dir = Path(args.softlabel_dir)
    prior_json = Path(args.prior_json)
    if not softlabel_dir.exists():
        failures.append(f'softlabel_dir does not exist: {softlabel_dir}')
    if not prior_json.exists():
        failures.append(f'prior_json does not exist: {prior_json}')
    metas = []
    rows = []
    if softlabel_dir.exists():
        metas = load_profile_metas(softlabel_dir, filename=args.filename)
        if len(metas) != expected:
            failures.append(f'profile count {len(metas)} != expected {expected}')
        for m in metas:
            keyset = set(m.keys)
            missing = sorted(k for k in REQUIRED_KEYS if k not in keyset)
            has_phis = any(bool(s & keyset) for s in [TARGET_ANY[0]])
            has_phie = any(bool(s & keyset) for s in [TARGET_ANY[1]])
            rg_version_ok = 'RG' in str(m.radial_solver_version) or 'P2Dlite-RG' in str(m.radial_solver_version)
            status = 'PASS'
            reasons = []
            if m.n_time <= 1:
                status = 'FAIL'; reasons.append('bad n_time')
            if m.nr_a < 17 or m.nr_c < 17:
                status = 'FAIL'; reasons.append('nr_a/nr_c < 17')
            if missing:
                status = 'FAIL'; reasons.append('missing ' + ','.join(missing))
            if not has_phis:
                status = 'FAIL'; reasons.append('missing phis_c target')
            if not has_phie:
                status = 'FAIL'; reasons.append('missing phie target')
            if not rg_version_ok:
                status = 'WARN'; reasons.append('radial_solver_version does not clearly mention RG')
            if status == 'FAIL':
                failures.append(f'{m.profile_id}: ' + '; '.join(reasons))
            elif status == 'WARN':
                warnings.append(f'{m.profile_id}: ' + '; '.join(reasons))
            rows.append({
                'profile_id': m.profile_id,
                'npz_path': m.npz_path,
                'n_time': m.n_time,
                'nr_a': m.nr_a,
                'nr_c': m.nr_c,
                'radial_solver_version': m.radial_solver_version,
                'key_count': len(m.keys),
                'status': status,
                'reasons': '; '.join(reasons),
            })
    report = {
        'stage': 'D15-P1 preflight',
        'softlabel_dir': str(softlabel_dir),
        'prior_json': str(prior_json),
        'config': str(args.config),
        'expected_profile_count': expected,
        'profile_count': len(metas),
        'failures': failures,
        'warnings': warnings,
        'overall_status': 'PASS' if not failures else 'FAIL',
        'profiles': rows,
    }
    out_json = Path(args.out_json)
    write_json(report, out_json)
    write_csv(rows, out_json.with_suffix('.profiles.csv'))
    print('[D15-P1 preflight] overall_status:', report['overall_status'])
    print('[D15-P1 preflight] wrote:', out_json)
    return 0 if report['overall_status'] == 'PASS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
