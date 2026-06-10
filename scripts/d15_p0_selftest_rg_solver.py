from __future__ import annotations

import sys
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gv1.p2dlite_rg.radial_solver import ElectrodeRGParams, generate_rg_profile


def main() -> int:
    t = np.arange(0, 1200, dtype=float)
    # Synthetic cathode-like profile: charging for 600 s then rest.
    cbar = 30000.0 - 3.0 * np.minimum(t, 600.0)
    J = np.zeros_like(t)
    R = 6.5e-6
    # d cbar/dt=-3 mol/m3/s => J=R mol/m2/s approximately.
    J[:600] = R
    params = ElectrodeRGParams('selftest_positive', radius_m=R, diffusivity_m2_s=2.5e-14, csmax_mol_m3=50500.0, alpha_D=0.65)
    cs, diag = generate_rg_profile(t, cbar, J, None, params, nr=17, max_substep_s=10.0)
    assert cs.shape == (t.size, 17)
    assert np.all(np.isfinite(cs))
    # For positive electrode charging with J>0, surface-center should be negative after transient.
    assert float(np.nanmedian(diag['surface_center'][200:600])) < 0.0
    # During rest, gradient should relax toward zero.
    rest_abs_start = abs(float(diag['surface_center'][610]))
    rest_abs_end = abs(float(diag['surface_center'][-1]))
    assert rest_abs_end < rest_abs_start
    print('[D15-P0 selftest] PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
