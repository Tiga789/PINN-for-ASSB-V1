from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_batch2.utils import parse_battery_id, format_profile_name, write_json


def main() -> int:
    assert parse_battery_id('2C_battery-15.mat') == 15
    assert parse_battery_id('Batch-2_battery_8') == 8
    assert format_profile_name('Batch-2', '3C', 1) == 'Batch-2_3C_battery-1'
    tmp = ROOT / '_d15_p3_selftest_tmp.json'
    write_json({'ok': True, 'stage': 'D15-P3'}, tmp)
    with open(tmp, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    tmp.unlink(missing_ok=True)
    assert obj['ok'] is True
    print('[D15-P3 selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
