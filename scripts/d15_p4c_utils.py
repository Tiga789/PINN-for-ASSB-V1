from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def ppath(p: str | Path) -> Path:
    return Path(str(p).replace('\\\\', '/'))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _jsonify(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return _jsonify(x.item())
        return [_jsonify(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(obj: Mapping[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonify(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path, fieldnames: Optional[Sequence[str]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(p, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _csv_value(r.get(k, '')) for k in fieldnames})


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def _csv_value(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return ';'.join(map(str, v))
    if isinstance(v, np.ndarray):
        return ';'.join(map(str, v.reshape(-1).tolist())) if v.ndim else v.item()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, Path):
        return str(v)
    return v


def parse_cell_id(text: str) -> Optional[str]:
    t = str(text).replace('\\', '/')
    b = None
    n = None
    m = re.search(r'Batch[-_ ]?(?P<b>[1-6])', t, flags=re.I)
    if m:
        b = int(m.group('b'))
    else:
        # Protocol-based fallback; only used if explicit Batch token is absent.
        low = t.lower()
        if 'random' in low:
            b = 5
        elif 'geo' in low:
            b = 6
        elif 'r2.5' in low or 'r2_5' in low:
            b = 3
        elif re.search(r'(^|[^\d])r3([^\d]|$)', low):
            b = 4
        elif '3c' in low:
            b = 2
        elif '2c' in low:
            b = 1
    m2 = re.search(r'battery[-_ ]?(?P<n>\d+)', t, flags=re.I)
    if m2:
        n = int(m2.group('n'))
    return f'Batch-{b}_battery-{n}' if b is not None and n is not None else None


def batch_protocol(batch: str) -> str:
    return {'Batch-5': 'random_walk', 'Batch-6': 'GEO'}.get(batch, '')


def parse_batch(cell_id: str) -> Optional[str]:
    m = re.match(r'(Batch-[1-6])_', str(cell_id))
    return m.group(1) if m else None


def parse_battery_num(cell_id: str) -> Optional[int]:
    m = re.search(r'battery-(\d+)', str(cell_id))
    return int(m.group(1)) if m else None


def discover_raw_mat_for_targets(raw_root: str | Path, target_cells: Sequence[str], batch_info: Mapping[str, Any]) -> List[Dict[str, Any]]:
    root = ppath(raw_root)
    rows: List[Dict[str, Any]] = []
    seen_paths = set()
    for batch in ['Batch-5', 'Batch-6']:
        info = batch_info.get(batch, {})
        candidates: List[Path] = []
        for dn in info.get('candidate_dirs', [batch]):
            candidates.append(root / dn)
        if root.exists():
            for child in root.iterdir():
                if child.is_dir() and re.search(batch.replace('-', '[-_ ]?') + r'$', child.name, flags=re.I):
                    candidates.append(child)
        for d in candidates:
            if not d.exists() or not d.is_dir():
                continue
            for p in d.rglob('*.mat'):
                if p.name.startswith('._'):
                    continue
                rp = str(p.resolve()).lower()
                if rp in seen_paths:
                    continue
                seen_paths.add(rp)
                can = parse_cell_id(str(p))
                if can in target_cells:
                    rows.append({
                        'canonical_cell_id': can,
                        'batch': batch,
                        'protocol': batch_protocol(batch),
                        'battery_id': parse_battery_num(can),
                        'raw_mat_path': str(p),
                        'raw_mat_name': p.name,
                        'raw_mat_size_bytes': p.stat().st_size,
                        'status': 'READY_RAW'
                    })
    rows = sorted(rows, key=lambda r: (str(r.get('batch')), int(r.get('battery_id') or 9999)))
    # Deduplicate by canonical id, prefer larger files if duplicates exist.
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = str(r['canonical_cell_id'])
        if k not in best or int(r.get('raw_mat_size_bytes') or 0) > int(best[k].get('raw_mat_size_bytes') or 0):
            best[k] = r
    out = [best[k] for k in sorted(best, key=lambda x: (parse_batch(x) or '', parse_battery_num(x) or 9999))]
    return out


# ---------- Best-effort MATLAB numeric table reader ----------

def _is_mat_struct(obj: Any) -> bool:
    return hasattr(obj, '_fieldnames')


def _get_fields(obj: Any) -> Dict[str, Any]:
    if _is_mat_struct(obj):
        return {name: getattr(obj, name) for name in obj._fieldnames}
    if isinstance(obj, np.void) and obj.dtype.names:
        return {name: obj[name] for name in obj.dtype.names}
    if isinstance(obj, np.ndarray) and obj.dtype.names and obj.size == 1:
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
    if isinstance(obj, Mapping):
        root_fields = {k: v for k, v in obj.items() if not str(k).startswith('__')}
        root = type('Root', (), {'_fieldnames': list(root_fields.keys()), **root_fields})()
        return _collect_table_candidates(root, prefix=prefix, depth=depth+1, max_depth=max_depth)
    fields = _get_fields(obj)
    if fields:
        cols: Dict[str, np.ndarray] = {}
        for k, v in fields.items():
            arr = _numeric_1d(v)
            if arr is not None:
                cols[f'{prefix}{k}' if prefix else k] = arr
        by_len: Dict[int, Dict[str, np.ndarray]] = {}
        for k, arr in cols.items():
            by_len.setdefault(arr.size, {})[k] = arr
        for n, group in by_len.items():
            if n >= 10 and len(group) >= 3:
                out.append(group)
        for k, v in fields.items():
            out.extend(_collect_table_candidates(v, prefix=f'{prefix}{k}__' if prefix else f'{k}__', depth=depth+1, max_depth=max_depth))
        return out
    try:
        arr = np.asarray(obj)
    except Exception:
        return out
    if isinstance(arr, np.ndarray):
        if arr.dtype.names:
            for item in arr.reshape(-1):
                out.extend(_collect_table_candidates(item, prefix=prefix, depth=depth+1, max_depth=max_depth))
        elif arr.dtype.kind == 'O' or arr.size <= 2000:
            for item in arr.reshape(-1):
                out.extend(_collect_table_candidates(item, prefix=prefix, depth=depth+1, max_depth=max_depth))
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
        finite = arr[np.isfinite(arr)]
        if kind == 'time':
            dif = np.diff(arr)
            if dif.size and np.nanmedian(np.abs(dif)) <= 0:
                sc -= 5
        elif kind == 'voltage':
            if finite.size and 1.0 <= np.nanmedian(finite) <= 5.5:
                sc += 4
        elif kind == 'current':
            if finite.size and 0.001 <= np.nanpercentile(np.abs(finite), 95) <= 20:
                sc += 4
        elif kind == 'temperature':
            if finite.size and -20 <= np.nanmedian(finite) <= 80:
                sc += 4
        if sc > best_score:
            best_score = sc
            best = k
    if best is None or best_score < 0:
        raise KeyError(f'Could not select {kind} column')
    return best, cols[best]


def _local_time_to_seconds(t: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(t, dtype=float).reshape(-1)
    if x.size != n or not np.isfinite(x).any():
        return np.arange(n, dtype=float)
    x = x - x[0]
    dif = np.diff(x)
    if dif.size and (np.nanmedian(dif) <= 0 or np.nanmin(dif) < -1e-9):
        return np.arange(n, dtype=float)
    med = float(np.nanmedian(dif)) if dif.size else 1.0
    span = float(np.nanmax(x) - np.nanmin(x)) if x.size else 0.0
    if span < 10 and med < 0.01 and n > 100:
        x = x * 86400.0
    return x.astype(float)


def _step_ids_from_current(I: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    I = np.asarray(I, dtype=float)
    st = np.empty(I.size, dtype=object)
    st[np.abs(I) <= threshold] = 'rest'
    st[I > threshold] = 'charge'
    st[I < -threshold] = 'discharge'
    step_id = np.zeros(I.size, dtype=np.int32)
    cur = 1
    for i in range(I.size):
        if i > 0 and st[i] != st[i-1]:
            cur += 1
        step_id[i] = cur
    return step_id, st


def _normalize_current_positive_charge(I: np.ndarray, V: np.ndarray, threshold: float, auto_flip: bool) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    I = np.asarray(I, dtype=float).copy()
    V = np.asarray(V, dtype=float)
    if not auto_flip or I.size < 3:
        return I, False, {}
    dV = np.diff(V, prepend=V[0])
    pos = I > threshold
    neg = I < -threshold
    stats = {
        'median_dv_when_current_positive': float(np.nanmedian(dV[pos])) if pos.any() else float('nan'),
        'median_dv_when_current_negative': float(np.nanmedian(dV[neg])) if neg.any() else float('nan'),
        'positive_current_fraction': float(np.mean(pos)),
        'negative_current_fraction': float(np.mean(neg)),
    }
    flip = bool(pos.any() and neg.any() and stats['median_dv_when_current_positive'] < -1e-5 and stats['median_dv_when_current_negative'] > 1e-5)
    return (-I if flip else I), flip, stats


def load_mat_records_best_effort(mat_path: str | Path, temperature_fallback_C: float = 25.0) -> List[Dict[str, np.ndarray]]:
    try:
        import scipy.io as sio
    except Exception as exc:
        raise RuntimeError('scipy is required to read XJTU .mat files') from exc
    raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    candidates = _collect_table_candidates(raw)
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
                'temperature_C': np.asarray(temp[:n], dtype=float) if temp is not None and len(temp) >= n else np.full(n, temperature_fallback_C, dtype=float),
                'raw_record_index': np.full(n, rec_idx + 1, dtype=np.int32),
                'selected_time_key': str(tk),
                'selected_voltage_key': str(vk),
                'selected_current_key': str(ik),
                'selected_temperature_key': str(temp_key or '__fallback_25C__'),
            }
            if n >= 10 and np.nanpercentile(out['voltage_exp'], 95) > 1.0 and np.nanpercentile(np.abs(out['current_A_raw']), 95) > 0.001:
                records.append(out)
        except Exception:
            continue
    if not records:
        raise ValueError(f'No usable time/voltage/current records in {mat_path}')
    return records


@dataclass
class ReplayBuildResult:
    canonical_cell_id: str
    batch: str
    protocol: str
    status: str
    npz_path: str
    raw_mat_path: str
    time_points: int = 0
    cycle_count: int = 0
    current_min_A: float = math.nan
    current_max_A: float = math.nan
    voltage_min_V: float = math.nan
    voltage_max_V: float = math.nan
    sign_flipped_to_positive_charge: bool = False
    error: str = ''


def build_replay_profile_from_mat(
    mat_path: str | Path,
    out_npz: str | Path,
    canonical_cell_id: str,
    batch: str,
    protocol: str,
    current_threshold_A: float = 0.05,
    temperature_fallback_C: float = 25.0,
    auto_flip_current: bool = True,
    save_mode: str = 'compressed',
) -> ReplayBuildResult:
    mat_path = Path(mat_path)
    out = Path(out_npz)
    records = load_mat_records_best_effort(mat_path, temperature_fallback_C=temperature_fallback_C)
    t_all: List[np.ndarray] = []
    I_all: List[np.ndarray] = []
    V_all: List[np.ndarray] = []
    T_all: List[np.ndarray] = []
    cyc_all: List[np.ndarray] = []
    step_record_all: List[np.ndarray] = []
    offset = 0.0
    sign_flipped_any = False
    for rec_i, rec in enumerate(records):
        tloc = np.asarray(rec['time_local_s'], dtype=float)
        V = np.asarray(rec['voltage_exp'], dtype=float)
        Iraw = np.asarray(rec['current_A_raw'], dtype=float)
        T = np.asarray(rec['temperature_C'], dtype=float)
        if T.size != tloc.size:
            T = np.full(tloc.size, temperature_fallback_C, dtype=float)
        I, flipped, _ = _normalize_current_positive_charge(Iraw, V, current_threshold_A, auto_flip_current)
        sign_flipped_any = sign_flipped_any or flipped
        if tloc.size < 2:
            continue
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
        step_record_all.append(np.full(n, rec_i + 1, dtype=np.int32))
    if not t_all:
        raise ValueError(f'No usable records from {mat_path}')
    t = np.concatenate(t_all)
    I = np.concatenate(I_all)
    V = np.concatenate(V_all)
    T = np.concatenate(T_all)
    cyc = np.concatenate(cyc_all)
    step_id, step_type = _step_ids_from_current(I, current_threshold_A)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
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
        batch=np.array(batch),
        protocol=np.array(protocol),
        cell_uid=np.array(canonical_cell_id),
        source_file=np.array(str(mat_path)),
        source_profile_npz=np.array(str(out)),
        battery_id=np.array(int(parse_battery_num(canonical_cell_id) or -1)),
        sign_flipped_to_positive_charge=np.array(bool(sign_flipped_any)),
        builder_version=np.array('D15-P4C-best-effort-batch56-mat-to-replay-v1'),
    )
    if save_mode == 'uncompressed':
        np.savez(out, **arrays)
    else:
        np.savez_compressed(out, **arrays)
    return ReplayBuildResult(
        canonical_cell_id=canonical_cell_id,
        batch=batch,
        protocol=protocol,
        status='PASS',
        npz_path=str(out),
        raw_mat_path=str(mat_path),
        time_points=int(t.size),
        cycle_count=int(len(np.unique(cyc))),
        current_min_A=float(np.nanmin(I)),
        current_max_A=float(np.nanmax(I)),
        voltage_min_V=float(np.nanmin(V)),
        voltage_max_V=float(np.nanmax(V)),
        sign_flipped_to_positive_charge=bool(sign_flipped_any),
    )


def audit_replay_npz(path: str | Path, canonical_cell_id: Optional[str] = None) -> Dict[str, Any]:
    p = Path(path)
    out: Dict[str, Any] = {'npz_path': str(p), 'canonical_cell_id': canonical_cell_id or parse_cell_id(str(p)) or ''}
    try:
        with np.load(p, allow_pickle=True) as z:
            keys = list(z.files)
            t = np.asarray(z['t_global_s'] if 't_global_s' in keys else z['time_s'], dtype=float)
            I = np.asarray(z['I_profile'] if 'I_profile' in keys else z['current_A'], dtype=float)
            V = np.asarray(z['voltage_exp'] if 'voltage_exp' in keys else z['voltage_V'], dtype=float)
            T = np.asarray(z['temperature_C'], dtype=float) if 'temperature_C' in keys else np.full_like(t, 25.0)
            cyc = np.asarray(z['cycle_id']) if 'cycle_id' in keys else np.array([], dtype=int)
            out.update({
                'status': 'PASS',
                'time_points': int(t.size),
                'cycle_count': int(len(np.unique(cyc))) if cyc.size else 0,
                'time_monotonic_nondec': bool(np.all(np.diff(t) >= -1e-9)) if t.size > 1 else True,
                'finite_core_ok': bool(np.isfinite(t).all() and np.isfinite(I).all() and np.isfinite(V).all() and np.isfinite(T).all()),
                'current_min_A': float(np.nanmin(I)),
                'current_max_A': float(np.nanmax(I)),
                'voltage_min_V': float(np.nanmin(V)),
                'voltage_max_V': float(np.nanmax(V)),
                'temperature_min_C': float(np.nanmin(T)),
                'temperature_max_C': float(np.nanmax(T)),
                'required_keys_present': all(k in keys for k in ['t_global_s', 'I_profile', 'voltage_exp', 'cycle_id', 'step_id']),
                'read_error': '',
            })
    except Exception as exc:
        out.update({'status': 'FAIL', 'read_error': repr(exc), 'time_points': 0})
    if out.get('status') == 'PASS':
        fails = []
        if not out.get('time_monotonic_nondec'):
            fails.append('time_not_monotonic')
        if not out.get('finite_core_ok'):
            fails.append('nonfinite_core')
        if not out.get('required_keys_present'):
            fails.append('missing_required_keys')
        if int(out.get('time_points', 0)) < 1000:
            fails.append('too_few_time_points')
        if fails:
            out['status'] = 'FAIL'
            out['read_error'] = ';'.join(fails)
    return out
