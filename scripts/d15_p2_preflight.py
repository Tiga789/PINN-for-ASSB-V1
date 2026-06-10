from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_nn_precision.audit import write_json


def load_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(description='D15-P2 precision benchmark preflight.')
    p.add_argument('--softlabel-dir', required=True)
    p.add_argument('--prior-json', default='configs/P2Dlite_prior_xjtu_lr18650la_rg_v1.json')
    p.add_argument('--config', default='configs/d15_p2_precision_benchmark_config.json')
    p.add_argument('--out-json', required=True)
    p.add_argument('--filename', default='solution_softlabels.npz')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    failures = []
    warnings = []
    checks: Dict[str, Any] = {}
    soft = Path(args.softlabel_dir)
    cfg_path = Path(args.config)
    prior = Path(args.prior_json)
    cfg = load_json(cfg_path) if cfg_path.exists() else {}
    expected = int(cfg.get('data', {}).get('expected_profile_count', 8))
    checks['softlabel_dir'] = str(soft)
    checks['softlabel_dir_exists'] = soft.exists()
    if not soft.exists():
        failures.append(f'softlabel_dir missing: {soft}')
        npz_files = []
    else:
        npz_files = sorted(soft.rglob(args.filename))
    checks['source_npz_count'] = len(npz_files)
    checks['expected_profile_count'] = expected
    if len(npz_files) != expected:
        failures.append(f'expected {expected} npz files, found {len(npz_files)}')
    checks['config'] = str(cfg_path)
    checks['config_exists'] = cfg_path.exists()
    if not cfg_path.exists():
        failures.append(f'config missing: {cfg_path}')
    checks['prior_json'] = str(prior)
    checks['prior_exists'] = prior.exists()
    if not prior.exists():
        warnings.append(f'prior json missing or not copied yet: {prior}')
    required_scripts = [
        'scripts/d15_p1_train_rg_closedset_nn_smoke.py',
        'scripts/d15_p1_eval_rg_closedset_nn_smoke.py',
        'gv1/p2dlite_rg_nn/data.py',
        'gv1/p2dlite_rg_nn/model.py',
        'gv1/p2dlite_rg_nn/train_eval.py',
    ]
    missing_required = []
    for rel in required_scripts:
        ok = (ROOT / rel).exists()
        checks[rel] = ok
        if not ok:
            missing_required.append(rel)
    if missing_required:
        failures.append('D15-P2 reuses D15-P1 NN modules, but these are missing: ' + ', '.join(missing_required))
    torch_spec = importlib.util.find_spec('torch')
    checks['torch_importable'] = torch_spec is not None
    if torch_spec is None:
        failures.append('torch is not importable in current environment')
    sample_profiles = []
    for p in npz_files[: min(20, len(npz_files))]:
        sample_profiles.append(str(p.parent.relative_to(soft)))
    checks['sample_profiles'] = sample_profiles
    out = {
        'stage': 'D15-P2 preflight',
        'overall_status': 'PASS' if not failures else 'FAIL',
        'failures': failures,
        'warnings': warnings,
        'checks': checks,
        'notes': cfg.get('audit_notes', []),
    }
    write_json(out, args.out_json)
    print('[D15-P2 preflight] overall_status:', out['overall_status'])
    print('[D15-P2 preflight] wrote:', args.out_json)
    return 0 if not failures else 2


if __name__ == '__main__':
    raise SystemExit(main())
