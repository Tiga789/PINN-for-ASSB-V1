# -*- coding: utf-8 -*-
"""I/O helpers for XJTU P2Dlite soft labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

import numpy as np


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def save_softlabels(output_root: str | Path, soft: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    root = Path(output_root)
    cell = safe_name(str(soft.get("cell_uid", "unknown_cell")))
    if cell == "unknown_cell" or not cell:
        cell = safe_name(str(soft.get("batch", "unknown_batch")) + "_" + str(soft.get("protocol", "")))
    out_dir = root / cell
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "solution_softlabels.npz"
    np.savez_compressed(npz_path, **soft)

    summary = dict(summary)
    summary["solution_softlabels_npz"] = str(npz_path)
    (out_dir / "soft_label_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz_path
