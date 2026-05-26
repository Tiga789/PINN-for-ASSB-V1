from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with np.load(p, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def save_npz_dict(path: str | Path, arrays: Mapping[str, object], *, compressed: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in arrays.items():
        arr = np.asarray(v)
        if arr.dtype == object:
            arr = arr.astype(str)
        clean[k] = arr
    if compressed:
        np.savez_compressed(p, **clean)
    else:
        np.savez(p, **clean)


def list_npz_keys(path: str | Path) -> list[str]:
    with np.load(path, allow_pickle=True) as data:
        return list(data.files)
