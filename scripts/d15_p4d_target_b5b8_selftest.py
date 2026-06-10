from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    required = [
        ROOT / 'configs' / 'd15_p4d_batch5_battery8_targeted_fix_config.json',
        ROOT / 'scripts' / 'd15_p4d_target_b5b8_fix.py',
        ROOT / 'scripts' / 'd15_p4d_target_b5b8_pack_review.py',
        ROOT / 'scripts' / 'd15_p4d_full_generate_one_rg_softlabel.py',
        ROOT / 'scripts' / 'd15_p0_radial_gradient_audit.py',
        ROOT / 'configs' / 'P2Dlite_prior_xjtu_lr18650la_rg_v1.json',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print('[D15-P4D targeted selftest] missing required files:')
        for p in missing:
            print('  -', p)
        return 2
    cfg = json.loads((ROOT / 'configs' / 'd15_p4d_batch5_battery8_targeted_fix_config.json').read_text(encoding='utf-8'))
    if cfg.get('target_cell') != 'Batch-5_battery-8':
        print('[D15-P4D targeted selftest] unexpected target_cell:', cfg.get('target_cell'))
        return 3
    if any(Path(x).as_posix().startswith('gv1/') for x in []):
        return 4
    print('[D15-P4D targeted selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
