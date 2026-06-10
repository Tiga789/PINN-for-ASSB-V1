from __future__ import annotations

import json
import re
from pathlib import Path


def parse_cell_id(text: str):
    m = re.search(r'Batch[-_ ]?(?P<b>[1-6]).*?battery[-_ ]?(?P<n>\d+)', text, re.I)
    if m:
        return f"Batch-{int(m.group('b'))}_battery-{int(m.group('n'))}"
    m_b = re.search(r'Batch[-_ ]?(?P<b>[1-6])', text, re.I)
    m_n = re.search(r'battery[-_ ]?(?P<n>\d+)', text, re.I)
    if m_b and m_n:
        return f"Batch-{int(m_b.group('b'))}_battery-{int(m_n.group('n'))}"
    return None


def main() -> int:
    assert parse_cell_id('E:/XJTU battery dataset/Batch-5/random_walk_battery-8.mat') == 'Batch-5_battery-8'
    assert parse_cell_id('Batch_6/GEO_battery_7.mat') == 'Batch-6_battery-7'
    cfg = Path('configs/d15_p4c_batch56_remaining14_replay_config.json')
    assert cfg.exists(), f'missing config: {cfg}'
    obj = json.loads(cfg.read_text(encoding='utf-8'))
    assert len(obj['target_cells']) == 14
    print('[D15-P4C selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
