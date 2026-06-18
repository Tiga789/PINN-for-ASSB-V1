# -*- coding: utf-8 -*-
"""Compatibility entrypoint for D17-P2 smoke training."""
from __future__ import annotations
import runpy
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "gv1_train_d17_pinn_rebuild.py"), run_name="__main__")
