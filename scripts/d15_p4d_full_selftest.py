from __future__ import annotations
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / 'configs' / 'd15_p4d_full_remaining14_config.json',
    ROOT / 'scripts' / 'd15_p4d_full_generate_one_rg_softlabel.py',
    ROOT / 'scripts' / 'd15_p4d_full_collect_generation_report.py',
    ROOT / 'scripts' / 'd15_p4d_full_collect_scorecard.py',
    ROOT / 'scripts' / 'd15_p4d_full_pack_review.py',
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    print('[D15-P4D full selftest] missing files:', missing)
    raise SystemExit(2)

# Check that existing RG solver can be imported in the user's project.
try:
    from gv1.p2dlite_rg.radial_solver import ElectrodeRGParams, generate_rg_profile, infer_surface_flux_from_cbar  # noqa: F401
except Exception as exc:
    print('[D15-P4D full selftest] could not import gv1.p2dlite_rg.radial_solver:', repr(exc))
    raise SystemExit(2)
print('[D15-P4D full selftest] PASS')
