from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
UTIL = ROOT / "util"
NAMES = [
    "init_pinn.py",
    "myNN.py",
    "_losses.py",
    "_rescale.py",
    "forwardPass.py",
    "load_pinn.py",
    "spm.py",
    "spm_simpler.py",
    "thermo.py",
    "uocp_cs.py",
]

print("=== Shadow-file check ===")
for name in NAMES:
    root_file = ROOT / name
    util_file = UTIL / name
    if root_file.exists():
        print(f"[ROOT SHADOW] {root_file}")
    else:
        print(f"[root clean] {name}")
    if util_file.exists():
        print(f"[util ok]     {util_file}")
    else:
        print(f"[util missing] {util_file}")
    print("-")
