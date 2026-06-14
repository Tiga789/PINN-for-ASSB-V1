from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def main() -> int:
    tests = {
        'profiles/Batch-2_3C_battery-15': 'Batch-2_battery-15',
        'profiles/Batch-4_R3_battery-2': 'Batch-4_battery-2',
        'profiles/0003_battery-3_2C_battery-3': 'Batch-1_battery-3',
        'profiles/0014_battery-6_R2.5_battery-6': 'Batch-3_battery-6',
        'profiles/0023_battery-7_R3_battery-7': 'Batch-4_battery-7',
        'Batch-5_battery-8': 'Batch-5_battery-8',
    }
    for raw, expected in tests.items():
        got = canonical_cell_id(raw)
        if got != expected:
            raise AssertionError(f'{raw}: got {got}, expected {expected}')
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f'Missing required dependency numpy/torch: {exc!r}')
    print('[D15-P5A selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
