from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / 'configs' / 'd15_p4b_ready18_generation_config.json'
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    data = json.loads(cfg.read_text(encoding='utf-8'))
    required = ['p4a_fix_manifest_csv', 'prior_json', 'output_softlabels_dir', 'radial_audit_dir', 'scorecard_dir', 'review_zip']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f'Missing config keys: {missing}')
    if int(data.get('expected_ready_cell_count', -1)) != 18:
        raise ValueError('expected_ready_cell_count must be 18 for D15-P4B')
    print('[D15-P4B selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
