from __future__ import annotations
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = root / 'configs' / 'd15_p4d_smoke_config.json'
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    data = json.loads(cfg.read_text(encoding='utf-8'))
    required = ['p4c_replay_manifest_csv', 'prior_json', 'output_dir', 'review_zip', 'smoke_cells', 'generation']
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f'Missing config keys: {missing}')
    if not data['smoke_cells']:
        raise ValueError('smoke_cells must not be empty')
    if int(data['generation'].get('max_time_points_per_cell', 0)) <= 0:
        raise ValueError('max_time_points_per_cell must be positive')
    print('[D15-P4D-smoke selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
