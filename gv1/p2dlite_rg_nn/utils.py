from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


def norm_path(path_like: str | os.PathLike[str]) -> Path:
    return Path(str(path_like).replace('\\\\', '/'))


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj: Any, path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | os.PathLike[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()}) if rows else ["empty"]
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def discover_npz(root: str | os.PathLike[str], filename: str = 'solution_softlabels.npz') -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Soft-label directory does not exist: {root}')
    files = sorted(root.rglob(filename))
    if not files:
        files = sorted(root.rglob('*.npz'))
    good: List[Path] = []
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as z:
                keys = set(z.files)
            if ('cs_a' in keys or 'theta_a' in keys) and ('cs_c' in keys or 'theta_c' in keys):
                good.append(p)
        except Exception:
            continue
    return good


def first_key(d: Mapping[str, Any], candidates: Sequence[str]) -> str | None:
    for k in candidates:
        if k in d:
            return k
    return None


def scalar_string(x: Any, default: str = '') -> str:
    try:
        arr = np.asarray(x)
        if arr.shape == ():
            return str(arr.item())
        if arr.size == 1:
            return str(arr.reshape(-1)[0])
        return str(arr.reshape(-1)[0])
    except Exception:
        return default


def ensure_clean_or_allowed(path: str | os.PathLike[str], allow_overwrite: bool) -> Path:
    p = Path(path)
    if p.exists() and any(p.iterdir()) and not allow_overwrite:
        raise FileExistsError(f'Output directory exists and is not empty: {p}. Use --allow-overwrite for deliberate reruns.')
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: Any, default: float = float('nan')) -> float:
    try:
        return float(x)
    except Exception:
        return default
