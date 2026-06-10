from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def to_path(path_like: str | os.PathLike[str]) -> Path:
    return Path(str(path_like).replace('\\\\', '/'))


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(obj: Any, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def discover_softlabel_npz(root: str | os.PathLike[str], filename: str = 'solution_softlabels.npz') -> List[Path]:
    """Find source soft-label npz files.

    Preferred layout is one `solution_softlabels.npz` per profile directory.
    If none are found, fall back to all npz files that plausibly contain
    soft-label arrays.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'Source directory does not exist: {root}')
    files = sorted(root.rglob(filename))
    if files:
        return files
    candidates = sorted(root.rglob('*.npz'))
    good: List[Path] = []
    for p in candidates:
        try:
            with np.load(p, allow_pickle=True) as z:
                keys = set(z.files)
            if any(k in keys for k in ('cs_a', 'theta_a')) and any(k in keys for k in ('cs_c', 'theta_c')):
                good.append(p)
        except Exception:
            continue
    return good


def npz_to_dict(npz_path: str | os.PathLike[str]) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def save_npz_compressed(path: str | os.PathLike[str], arrays: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def first_existing_key(d: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def as_1d_float(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        raise TypeError(f'{name} is not numeric')
    arr = arr.astype(float).reshape(-1)
    return arr


def safe_numeric_array(d: Dict[str, Any], keys: Sequence[str], name: str, required: bool = True) -> Optional[np.ndarray]:
    k = first_existing_key(d, keys)
    if k is None:
        if required:
            raise KeyError(f'Missing required field for {name}; tried {keys}')
        return None
    return as_1d_float(d[k], k)


def infer_profile_id(npz_path: Path, source_root: Path) -> str:
    try:
        rel = npz_path.parent.relative_to(source_root)
        if str(rel) not in ('', '.'):
            return str(rel).replace('\\\\', '/').replace('\\', '/')
    except Exception:
        pass
    return npz_path.parent.name or npz_path.stem


def relative_output_dir(npz_path: Path, source_root: Path, output_root: Path) -> Path:
    profile_id = infer_profile_id(npz_path, source_root)
    safe_parts = [p for p in Path(profile_id).parts if p not in ('', '.', '..')]
    if not safe_parts:
        safe_parts = [npz_path.stem]
    return output_root.joinpath(*safe_parts)


def orient_time_radial(arr: Any, n_time: int, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        # Interpret 1D as a scalar state over time and add a single radial cell.
        if a.shape[0] != n_time:
            raise ValueError(f'{name}: 1D length {a.shape[0]} != n_time {n_time}')
        return a[:, None]
    if a.ndim != 2:
        raise ValueError(f'{name}: expected 1D/2D array, got shape {a.shape}')
    if a.shape[0] == n_time:
        return a.astype(float)
    if a.shape[1] == n_time:
        return a.T.astype(float)
    raise ValueError(f'{name}: cannot orient shape {a.shape} against n_time={n_time}')


def volume_weights_for_nr(nr: int) -> np.ndarray:
    # Equal-interval finite-volume shell weights over normalized radius [0, 1].
    edges = np.linspace(0.0, 1.0, nr + 1)
    vols = edges[1:] ** 3 - edges[:-1] ** 3
    vols = vols / np.sum(vols)
    return vols


def weighted_cbar(cs: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if cs.ndim != 2:
        raise ValueError('cs must be 2D [time, radial]')
    if weights is None:
        weights = volume_weights_for_nr(cs.shape[1])
    return np.sum(cs * weights[None, :], axis=1)


def get_time_and_current(d: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    t = safe_numeric_array(d, ['t_global_s', 'time_s', 't_s', 't', 'time'], 'time', required=True)
    I = safe_numeric_array(d, ['I_profile', 'current_A', 'I_A', 'current', 'I'], 'current', required=False)
    if I is None:
        I = np.zeros_like(t)
    if I.shape[0] != t.shape[0]:
        raise ValueError(f'current length {I.shape[0]} != time length {t.shape[0]}')
    return t, I


def get_cs_or_theta(d: Dict[str, Any], electrode: str, n_time: int, csmax: float) -> Tuple[np.ndarray, str]:
    if electrode == 'a':
        cs_keys = ['cs_a', 'cs_n', 'cs_negative', 'cs_anode']
        theta_keys = ['theta_a', 'theta_n', 'theta_negative', 'theta_anode']
    elif electrode == 'c':
        cs_keys = ['cs_c', 'cs_p', 'cs_positive', 'cs_cathode']
        theta_keys = ['theta_c', 'theta_p', 'theta_positive', 'theta_cathode']
    else:
        raise ValueError(electrode)
    k = first_existing_key(d, cs_keys)
    if k:
        return orient_time_radial(d[k], n_time, k), k
    k = first_existing_key(d, theta_keys)
    if k:
        theta = orient_time_radial(d[k], n_time, k)
        return theta * csmax, k
    raise KeyError(f'Missing cs/theta for electrode {electrode}')


def get_cbar_field(d: Dict[str, Any], electrode: str, n_time: int) -> Optional[np.ndarray]:
    keys = ['cbar_a', 'cbar_n', 'cbar_negative'] if electrode == 'a' else ['cbar_c', 'cbar_p', 'cbar_positive']
    k = first_existing_key(d, keys)
    if not k:
        return None
    arr = as_1d_float(d[k], k)
    if arr.shape[0] != n_time:
        return None
    return arr


def get_j_field(d: Dict[str, Any], electrode: str, n_time: int) -> Optional[np.ndarray]:
    keys = ['j_a', 'J_a_eff', 'J_a', 'j_n', 'J_n_eff'] if electrode == 'a' else ['j_c', 'J_c_eff', 'J_c', 'j_p', 'J_p_eff']
    k = first_existing_key(d, keys)
    if not k:
        return None
    try:
        arr = as_1d_float(d[k], k)
    except Exception:
        return None
    if arr.shape[0] != n_time:
        return None
    return arr


def copy_profile_metadata(arrays: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in arrays.items():
        if k.startswith('__'):
            continue
        out[k] = v
    return out
