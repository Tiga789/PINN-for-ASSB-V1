from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg_boundary.projection import apply_theta_projection, theta_outside_counts


def main() -> int:
    slices = {'theta_a': (0, 3), 'theta_c': (3, 6), 'phie': (6, 7), 'phis_c': (7, 8)}
    y = np.array([
        [-0.2, 0.5, 1.2, 0.1, -0.1, 1.4, 0.0, 3.7],
        [0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.1, 3.6],
    ], dtype=np.float32)
    before = theta_outside_counts(y, slices)
    yp = apply_theta_projection(y, slices, theta_min=1e-4, theta_max=0.9999)
    after = theta_outside_counts(yp, slices)
    assert before['theta_outside_count'] == 4, before
    assert after['theta_outside_count'] == 0, after
    assert np.allclose(yp[:, 6:], y[:, 6:]), 'phie/phis_c should remain unchanged'
    assert float(yp.min()) >= -1e-6
    print('[D15-P3B selftest] PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
