from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import numpy as np
    from gv1.p2dlite_rg_batch2_15cell.utils import battery_sort_key, write_json
    from gv1.p2dlite_rg.radial_solver import ElectrodeRGParams, generate_rg_profile, infer_surface_flux_from_cbar

    t = np.linspace(0.0, 100.0, 101)
    cbar = 1000.0 + 2.0 * np.sin(t / 100.0 * np.pi)
    J = infer_surface_flux_from_cbar(t, cbar, 1.0e-5)
    p = ElectrodeRGParams(name='selftest', radius_m=1.0e-5, diffusivity_m2_s=1.0e-14, csmax_mol_m3=30000.0, alpha_D=1.0, alpha_J=1.0, gradient_clip_normalized=0.1)
    cs, diag = generate_rg_profile(t, cbar, J, np.full(9, cbar[0]), p, nr=9, max_substep_s=10.0)
    assert cs.shape == (101, 9)
    assert 'surface_center' in diag
    assert battery_sort_key({'battery_id': '15'}) == 15
    print('[D15-P3C selftest] PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
