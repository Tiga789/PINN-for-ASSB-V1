from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .field_mapper import infer_standard_column, standardize_dataframe
from .reader_base import ReadOptions, ReadResult, ensure_path


def _is_scalar_like(x: Any) -> bool:
    return isinstance(x, (str, bytes, int, float, np.number)) or x is None


def _mat_struct_to_dict(obj: Any) -> Any:
    """Recursively convert scipy MATLAB objects into Python containers."""
    # scipy.io.matlab.mat_struct uses _fieldnames.
    if hasattr(obj, "_fieldnames"):
        return {name: _mat_struct_to_dict(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:
            # Structured array.
            if obj.size == 1:
                return {name: _mat_struct_to_dict(obj[name].item()) for name in obj.dtype.names}
            return [{name: _mat_struct_to_dict(row[name]) for name in obj.dtype.names} for row in obj.reshape(-1)]
        if obj.dtype == object:
            if obj.size == 1:
                return _mat_struct_to_dict(obj.item())
            return [_mat_struct_to_dict(x) for x in obj.reshape(-1)]
        return np.asarray(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except Exception:
            return obj.decode("latin1", errors="replace")
    return obj


def _loadmat_v5(path: Path) -> dict[str, Any]:
    try:
        import scipy.io  # type: ignore
    except ImportError as exc:
        raise ImportError("Reading MATLAB v5/v7.2 files requires scipy. Install scipy or convert to CSV/Parquet.") from exc
    raw = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    return {k: _mat_struct_to_dict(v) for k, v in raw.items() if not k.startswith("__")}


def _loadmat_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError("Reading MATLAB v7.3/HDF5 files requires h5py or mat73. Install h5py/mat73.") from exc

    def read_node(node: Any) -> Any:
        if hasattr(node, "keys"):
            return {k: read_node(node[k]) for k in node.keys()}
        arr = np.array(node)
        # MATLAB HDF5 often stores arrays transposed.
        if arr.ndim == 2:
            arr = arr.T
        return arr

    with h5py.File(path, "r") as f:
        return {k: read_node(f[k]) for k in f.keys()}


def load_mat_any(path: str | Path) -> tuple[dict[str, Any], str, list[str]]:
    p = ensure_path(path)
    warnings: list[str] = []
    try:
        return _loadmat_v5(p), "scipy.io.loadmat", warnings
    except NotImplementedError as exc:
        warnings.append(f"scipy loadmat reports v7.3/HDF5 or unsupported format: {exc}")
    except ValueError as exc:
        # Common for HDF5 .mat: Unknown mat file type, version ...
        warnings.append(f"scipy loadmat failed: {exc}")
    except Exception as exc:
        warnings.append(f"scipy loadmat failed: {type(exc).__name__}: {exc}")
    try:
        return _loadmat_hdf5(p), "h5py", warnings
    except Exception as exc:
        warnings.append(f"h5py failed: {type(exc).__name__}: {exc}")
        raise ValueError(f"Could not read MATLAB file {p}. Attempts: {' | '.join(warnings)}") from exc


def _as_dataframe_from_dict(d: dict[str, Any]) -> pd.DataFrame | None:
    arrays: dict[str, Any] = {}
    n: int | None = None
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if _is_scalar_like(v):
            continue
        arr = np.asarray(v)
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            length = len(arr)
            if length <= 1:
                continue
            if n is None:
                n = length
            if length == n:
                arrays[k] = arr
        elif arr.ndim == 2:
            # If one dimension is 1, flatten. Otherwise keep small matrices as columns.
            if 1 in arr.shape:
                vec = arr.reshape(-1)
                if len(vec) > 1:
                    if n is None:
                        n = len(vec)
                    if len(vec) == n:
                        arrays[k] = vec
            elif min(arr.shape) <= 20:
                # Interpret as N x M if one dimension matches n or the larger dimension is likely N.
                mat = arr
                if n is not None and mat.shape[1] == n and mat.shape[0] != n:
                    mat = mat.T
                elif mat.shape[0] < mat.shape[1]:
                    mat = mat.T
                if n is None:
                    n = mat.shape[0]
                if mat.shape[0] == n:
                    for j in range(mat.shape[1]):
                        arrays[f"{k}_{j}"] = mat[:, j]
    if n is None or len(arrays) == 0:
        return None
    try:
        return pd.DataFrame(arrays)
    except Exception:
        return None


def _score_frame(df: pd.DataFrame) -> int:
    score = 0
    cols = [str(c) for c in df.columns]
    mapped = [infer_standard_column(c) for c in cols]
    for wanted, weight in [("time_s", 4), ("current_A", 5), ("voltage_V", 5), ("capacity_Ah", 2), ("temperature_C", 1)]:
        if wanted in mapped:
            score += weight
    if len(df) > 10:
        score += 1
    if len(df) > 100:
        score += 1
    return score


def _collect_candidate_frames(obj: Any, path: str = "root", max_depth: int = 8) -> list[tuple[str, pd.DataFrame, int]]:
    if max_depth < 0:
        return []
    candidates: list[tuple[str, pd.DataFrame, int]] = []
    if isinstance(obj, pd.DataFrame):
        candidates.append((path, obj, _score_frame(obj)))
        return candidates
    if isinstance(obj, dict):
        df = _as_dataframe_from_dict(obj)
        if df is not None:
            candidates.append((path, df, _score_frame(df)))
        for k, v in obj.items():
            candidates.extend(_collect_candidate_frames(v, f"{path}.{k}", max_depth - 1))
        return candidates
    if isinstance(obj, list):
        # List of dicts with table-like objects.
        frames: list[pd.DataFrame] = []
        for i, item in enumerate(obj):
            sub = _collect_candidate_frames(item, f"{path}[{i}]", max_depth - 1)
            for subpath, frame, score in sub:
                frame = frame.copy()
                frame["mat_subrecord_index"] = i
                frames.append(frame)
                candidates.append((subpath, frame, score))
        if frames:
            common_cols = set(frames[0].columns)
            for f in frames[1:]:
                common_cols &= set(f.columns)
            if common_cols:
                concat = pd.concat(frames, ignore_index=True, sort=False)
                candidates.append((path + "[*]", concat, _score_frame(concat)))
        return candidates
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if arr.dtype.names:
            try:
                df = pd.DataFrame({name: arr[name].reshape(-1) for name in arr.dtype.names})
                candidates.append((path, df, _score_frame(df)))
            except Exception:
                pass
        elif arr.ndim == 2 and min(arr.shape) <= 20 and max(arr.shape) > 10:
            mat = arr if arr.shape[0] >= arr.shape[1] else arr.T
            df = pd.DataFrame(mat, columns=[f"col_{i}" for i in range(mat.shape[1])])
            candidates.append((path, df, _score_frame(df)))
    return candidates


def _choose_candidate(candidates: list[tuple[str, pd.DataFrame, int]], preferred_path: str | None = None) -> tuple[str, pd.DataFrame, int]:
    if not candidates:
        raise ValueError("No table-like arrays found in .mat file")
    if preferred_path:
        for p, df, score in candidates:
            if p.endswith(preferred_path) or p == preferred_path:
                return p, df, score
    candidates_sorted = sorted(candidates, key=lambda x: (x[2], len(x[1])), reverse=True)
    return candidates_sorted[0]


def read_mat_battery_file(path: str | Path, options: ReadOptions | None = None) -> ReadResult:
    p = ensure_path(path)
    options = options or ReadOptions()
    mat, backend, warnings = load_mat_any(p)
    candidates = _collect_candidate_frames(mat)
    chosen_path, raw_df, score = _choose_candidate(candidates, options.mat_table_path)
    std, std_warnings = standardize_dataframe(
        raw_df,
        options,
        source_path=p,
        source_format="mat",
        metadata={"mat_backend": backend, "mat_table_path": chosen_path},
    )
    warnings.extend(std_warnings)
    metadata = {
        "reader": "mat",
        "backend": backend,
        "chosen_table_path": chosen_path,
        "chosen_score": score,
        "raw_shape": list(raw_df.shape),
        "raw_columns": list(map(str, raw_df.columns)),
        "candidate_tables": [
            {"path": cpath, "shape": list(cdf.shape), "score": int(cscore), "columns": list(map(str, cdf.columns))[:30]}
            for cpath, cdf, cscore in sorted(candidates, key=lambda x: (x[2], len(x[1])), reverse=True)[:20]
        ],
    }
    return ReadResult(std, str(p), "mat", metadata, warnings)
