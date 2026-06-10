from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import ensure_dir, parse_battery_id


@dataclass
class ReplayProfile:
    profile_id: str
    battery_id: int
    npz_path: Path
    n_time: int
    cycle_count: int
    current_min_A: float
    current_max_A: float
    voltage_min_V: float
    voltage_max_V: float
    sign_flipped_to_positive_charge: bool


def discover_batch2_mat_files(raw_root: str | Path, candidate_dirs: Sequence[str]) -> List[Path]:
    root = Path(raw_root)
    files: List[Path] = []
    candidates: List[Path] = []
    for name in candidate_dirs:
        candidates.append(root / name)
    if root.exists():
        for child in root.iterdir():
            if child.is_dir() and re.search(r'batch[-_ ]?2$', child.name, flags=re.IGNORECASE):
                candidates.append(child)
    seen = set()
    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        for p in d.rglob('*.mat'):
            if p.name.startswith('._'):
                continue
            key = str(p.resolve()).lower()
            if key not in seen:
                seen.add(key)
                files.append(p)
    return sorted(files, key=lambda p: (parse_battery_id(p.name) or 10**9, p.name.lower()))


def _is_mat_struct(obj: Any) -> bool:
    return hasattr(obj, '_fieldnames')


def _get_fields(obj: Any) -> Dict[str, Any]:
    if _is_mat_struct(obj):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: obj[name] for name in obj.dtype.names}
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        if obj.size == 1:
            rec = obj.reshape(-1)[0]
            return {name: rec[name] for name in obj.dtype.names}
    return {}


def _numeric_1d(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x)
        if arr.dtype.kind in {'U', 'S', 'O'}:
            return None
        arr = np.squeeze(arr).astype(float)
        if arr.ndim != 1 or arr.size < 10:
            return None
        if not np.isfinite(arr).any():
            return None
        return arr.reshape(-1)
    except Exception:
        return None


def _collect_table_candidates(obj: Any, prefix: str = '', depth: int = 0, max_depth: int = 8) -> List[Dict[str, np.ndarray]]:
    if depth > max_depth:
        return []
    out: List[Dict[str, np.ndarray]] = []
    fields = _get_fields(obj)
    if fields:
        cols: Dict[str, np.ndarray] = {}
        for k, v in fields.items():
            arr = _numeric_1d(v)
            if arr is not None:
                cols[f'{prefix}{k}' if prefix else k] = arr
        # A table candidate needs at least three equal-length numeric vectors.
        by_len: Dict[int, Dict[str, np.ndarray]] = {}
        for k, arr in cols.items():
            by_len.setdefault(arr.size, {})[k] = arr
        for n, group in by_len.items():
            if n >= 10 and len(group) >= 3:
                out.append(group)
        # Recurse into nested fields.
        for k, v in fields.items():
            out.extend(_collect_table_candidates(v, prefix=f'{prefix}{k}__' if prefix else f'{k}__', depth=depth+1, max_depth=max_depth))
        return out
    arr = np.asarray(obj) if not isinstance(obj, dict) else None
    if isinstance(obj, Mapping):
        # Root dict: skip MATLAB metadata.
        root_fields = {k: v for k, v in obj.items() if not str(k).startswith('__')}
        out.extend(_collect_table_candidates(type('Root', (), {'_fieldnames': list(root_fields.keys()), **root_fields})(), prefix=prefix, depth=depth+1, max_depth=max_depth))
        return out
    if isinstance(arr, np.ndarray):
        if arr.dtype.names:
            for item in arr.reshape(-1):
                out.extend(_collect_table_candidates(item, prefix=prefix, depth=depth+1, max_depth=max_depth))
            return out
        if arr.dtype.kind == 'O' or arr.size <= 2000:
            for item in arr.reshape(-1):
                out.extend(_collect_table_candidates(item, prefix=prefix, depth=depth+1, max_depth=max_depth))
            return out
    return out


def _score_name(name: str, keywords: Sequence[str], bad: Sequence[str] = ()) -> float:
    s = name.lower()
    score = 0.0
    for kw in keywords:
        if kw.lower() in s:
            score += 2.0 + len(kw) * 0.01
    for b in bad:
        if b.lower() in s:
            score -= 2.0
    if '__' not in s:
        score += 0.1
    return score


def _choose_col(cols: Mapping[str, np.ndarray], kind: str) -> Tuple[str, np.ndarray]:
    if kind == 'time':
        keys = ['time_s', 'test_time', 'elapsed', 'time', 't_s', 't']
        bad = ['date', 'system']
    elif kind == 'voltage':
        keys = ['voltage_v', 'voltage', 'volt', 'u', 'v']
        bad = ['cutoff']
    elif kind == 'current':
        keys = ['current_a', 'current', 'curr', 'i_a', 'i']
        bad = ['capacity']
    elif kind == 'temperature':
        keys = ['temperature_c', 'temperature', 'temp', 't_c']
        bad = ['time']
    else:
        raise ValueError(kind)
    best = None
    best_score = -1e9
    for k, arr in cols.items():
        sc = _score_name(k, keys, bad)
        if kind == 'time':
            dif = np.diff(arr)
            if np.nanmedian(np.abs(dif)) <= 0:
                sc -= 5
        if kind == 'voltage':
            finite = arr[np.isfinite(arr)]
            if finite.size and 1.0 <= np.nanmedian(finite) <= 5.5:
                sc += 4
        if kind == 'current':
            finite = arr[np.isfinite(arr)]
            if finite.size and 0.001 <= np.nanpercentile(np.abs(finite), 95) <= 20:
                sc += 4
        if kind == 'temperature':
            finite = arr[np.isfinite(arr)]
            if finite.size and -20 <= np.nanmedian(finite) <= 80:
                sc += 4
        if sc > best_score:
            best_score = sc
            best = k
    if best is None or best_score < 0:
        raise KeyError(f'Could not select {kind} column from {list(cols.keys())[:20]}')
    return best, cols[best]


def _local_time_to_seconds(t: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(t, dtype=float).reshape(-1)
    if x.size != n or not np.isfinite(x).any():
        return np.arange(n, dtype=float)
    x = x - x[0]
    dif = np.diff(x)
    if np.nanmedian(dif) <= 0 or np.nanmin(dif) < -1e-9:
        return np.arange(n, dtype=float)
    # MATLAB datenums converted to relative days sometimes have tiny steps; convert to seconds.
    med = float(np.nanmedian(dif)) if dif.size else 1.0
    span = float(np.nanmax(x) - np.nanmin(x))
    if span < 10 and med < 0.01 and n > 100:
        x = x * 86400.0
    return x.astype(float)


def _step_ids_from_current(I: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    I = np.asarray(I, dtype=float)
    st = np.empty(I.size, dtype=object)
    st[np.abs(I) <= threshold] = 'rest'
    st[I > threshold] = 'charge'
    st[I < -threshold] = 'discharge'
    step_id = np.zeros(I.size, dtype=int)
    cur = 1
    for i in range(I.size):
        if i > 0 and st[i] != st[i-1]:
            cur += 1
        step_id[i] = cur
    return step_id, st


def _normalize_current_positive_charge(t: np.ndarray, I: np.ndarray, V: np.ndarray, threshold: float) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    I = np.asarray(I, dtype=float).copy()
    V = np.asarray(V, dtype=float)
    if I.size < 3:
        return I, False, {}
    dV = np.diff(V, prepend=V[0])
    pos = I > threshold
    neg = I < -threshold
    stats: Dict[str, float] = {
        'median_dv_when_current_positive': float(np.nanmedian(dV[pos])) if pos.any() else float('nan'),
        'median_dv_when_current_negative': float(np.nanmedian(dV[neg])) if neg.any() else float('nan'),
        'positive_current_fraction': float(np.mean(pos)),
        'negative_current_fraction': float(np.mean(neg)),
    }
    flip = False
    if pos.any() and neg.any():
        # If positive-current segments mostly decrease voltage while negative-current segments increase it,
        # the raw sign is likely discharge-positive, so flip.
        if stats['median_dv_when_current_positive'] < -1e-5 and stats['median_dv_when_current_negative'] > 1e-5:
            flip = True
    if flip:
        I = -I
    return I, flip, stats


def load_mat_records_best_effort(mat_path: str | Path) -> List[Dict[str, np.ndarray]]:
    try:
        import scipy.io as sio
    except Exception as exc:
        raise RuntimeError('scipy is required to read raw XJTU .mat files. Install scipy or generate replay profiles with the existing GV1 pipeline first.') from exc
    raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    candidates = _collect_table_candidates(raw)
    if not candidates:
        raise ValueError(f'No table-like numeric records found in {mat_path}')
    records: List[Dict[str, np.ndarray]] = []
    for rec_idx, cols in enumerate(candidates):
        try:
            tk, t = _choose_col(cols, 'time')
            vk, V = _choose_col(cols, 'voltage')
            ik, I = _choose_col(cols, 'current')
            n = min(len(t), len(V), len(I))
            temp = None
            temp_key = None
            try:
                temp_key, temp = _choose_col(cols, 'temperature')
            except Exception:
                pass
            out = {
                'time_local_s': _local_time_to_seconds(t[:n], n),
                'voltage_exp': np.asarray(V[:n], dtype=float),
                'current_A_raw': np.asarray(I[:n], dtype=float),
                'temperature_C': np.asarray(temp[:n], dtype=float) if temp is not None and len(temp) >= n else np.full(n, 25.0, dtype=float),
                'raw_record_index': np.full(n, rec_idx + 1, dtype=int),
                'selected_time_key': np.array(tk),
                'selected_voltage_key': np.array(vk),
                'selected_current_key': np.array(ik),
                'selected_temperature_key': np.array(temp_key or '__fallback_25C__'),
            }
            # Filter impossible records.
            if n >= 10 and np.nanpercentile(out['voltage_exp'], 95) > 1.0 and np.nanpercentile(np.abs(out['current_A_raw']), 95) > 0.001:
                records.append(out)
        except Exception:
            continue
    if not records:
        raise ValueError(f'Table candidates found but none had usable time/voltage/current in {mat_path}')
    return records


def build_replay_profile_from_mat(
    mat_path: str | Path,
    out_npz: str | Path,
    profile_id: str,
    battery_id: int,
    current_threshold_A: float = 0.05,
    temperature_fallback_C: float = 25.0,
    auto_flip_current: bool = True,
) -> ReplayProfile:
    records = load_mat_records_best_effort(mat_path)
    t_all: List[np.ndarray] = []
    I_all: List[np.ndarray] = []
    V_all: List[np.ndarray] = []
    T_all: List[np.ndarray] = []
    cyc_all: List[np.ndarray] = []
    offset = 0.0
    sign_flipped_any = False
    for rec_i, rec in enumerate(records):
        tloc = np.asarray(rec['time_local_s'], dtype=float)
        V = np.asarray(rec['voltage_exp'], dtype=float)
        Iraw = np.asarray(rec['current_A_raw'], dtype=float)
        T = np.asarray(rec.get('temperature_C', np.full_like(tloc, temperature_fallback_C)), dtype=float)
        if T.size != tloc.size:
            T = np.full(tloc.size, temperature_fallback_C, dtype=float)
        I, flipped, _stats = _normalize_current_positive_charge(tloc, Iraw, V, current_threshold_A) if auto_flip_current else (Iraw, False, {})
        sign_flipped_any = sign_flipped_any or flipped
        if tloc.size < 2:
            continue
        # Use local elapsed time and append with a one-sample gap.
        tloc = tloc - tloc[0]
        if np.nanmax(tloc) <= 0 or np.nanmin(np.diff(tloc)) < -1e-9:
            tloc = np.arange(tloc.size, dtype=float)
        dt_med = np.nanmedian(np.diff(tloc)) if tloc.size > 1 else 1.0
        if not np.isfinite(dt_med) or dt_med <= 0:
            dt_med = 1.0
        tglob = offset + tloc
        offset = float(tglob[-1] + dt_med)
        n = tglob.size
        t_all.append(tglob.astype(np.float64))
        I_all.append(I[:n].astype(np.float64))
        V_all.append(V[:n].astype(np.float64))
        T_all.append(T[:n].astype(np.float64))
        cyc_all.append(np.full(n, rec_i + 1, dtype=np.int32))
    if not t_all:
        raise ValueError(f'No usable records from {mat_path}')
    t = np.concatenate(t_all)
    I = np.concatenate(I_all)
    V = np.concatenate(V_all)
    T = np.concatenate(T_all)
    cyc = np.concatenate(cyc_all)
    step_id, step_type = _step_ids_from_current(I, current_threshold_A)
    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t_global_s=t.astype(np.float32),
        time_s=t.astype(np.float32),
        I_profile=I.astype(np.float32),
        current_A=I.astype(np.float32),
        voltage_exp=V.astype(np.float32),
        voltage_V=V.astype(np.float32),
        temperature_C=T.astype(np.float32),
        cycle_id=cyc.astype(np.int32),
        step_id=step_id.astype(np.int32),
        step_type=step_type.astype(object),
        batch=np.array('Batch-2'),
        protocol=np.array('3C_charge_1C_discharge'),
        cell_uid=np.array(profile_id),
        source_file=np.array(str(mat_path)),
        source_profile_npz=np.array(str(out)),
        battery_id=np.array(int(battery_id)),
        sign_flipped_to_positive_charge=np.array(bool(sign_flipped_any)),
        builder_version=np.array('D15-P3A-best-effort-mat-to-replay-v1'),
    )
    return ReplayProfile(
        profile_id=profile_id,
        battery_id=int(battery_id),
        npz_path=out,
        n_time=int(t.size),
        cycle_count=int(len(np.unique(cyc))),
        current_min_A=float(np.nanmin(I)),
        current_max_A=float(np.nanmax(I)),
        voltage_min_V=float(np.nanmin(V)),
        voltage_max_V=float(np.nanmax(V)),
        sign_flipped_to_positive_charge=bool(sign_flipped_any),
    )
