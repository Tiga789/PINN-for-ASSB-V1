from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from gv1.p2dlite_rg_p4a.utils import parse_cell_identity_from_text


def main() -> int:
    cases = {
        'Batch-2_3C_battery-15': (2, 15),
        'profiles/0003_battery-3_2C_battery-3': (1, 3),
        'profiles/0014_battery-6_R2.5_battery-6': (3, 6),
        'profiles/0023_battery-7_R3_battery-7': (4, 7),
        'profiles/Batch-5_battery-7': (5, 7),
        'E:/XJTU battery dataset/Batch-6/GEO_battery-3.mat': (6, 3),
    }
    for text, expected in cases.items():
        b, bat, proto, key = parse_cell_identity_from_text(text)
        assert (b, bat) == expected, (text, (b, bat, proto, key), expected)
    print('[D15-P4A selftest] PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
