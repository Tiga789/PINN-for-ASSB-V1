from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import discover_npz, first_key, scalar_string


@dataclass
class ProfileMeta:
    profile_id: str
    npz_path: str
    n_time: int
    nr_a: int
    nr_c: int
    keys: List[str]
    radial_solver_version: str = ""
    cell_uid: str = ""
    batch: str = ""
    protocol: str = ""


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    profile_ids: List[str]
    profile_paths: List[str]
    nr_a: int
    nr_c: int
    target_slices: Dict[str, Tuple[int, int]]
    train_meta: Dict[str, Any]


def _as_1d_float(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        raise TypeError(f'{name} is not numeric')
    return arr.astype(np.float32).reshape(-1)


def _orient_time_radial(x: Any, n_time: int, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        raise TypeError(f'{name} is not numeric')
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != n_time:
            raise ValueError(f'{name}: 1D length {arr.shape[0]} != n_time {n_time}')
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f'{name}: expected 1D/2D, got {arr.shape}')
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return arr.T
    raise ValueError(f'{name}: cannot orient {arr.shape} for n_time={n_time}')


def _get_time(d: Mapping[str, Any]) -> np.ndarray:
    k = first_key(d, ['t_global_s', 'time_s', 't_s', 't', 'time'])
    if k is None:
        raise KeyError('Missing time key; tried t_global_s/time_s/t_s/t/time')
    t = _as_1d_float(d[k], k)
    if t.size < 2:
        raise ValueError('time array must have at least 2 points')
    return t


def _get_current(d: Mapping[str, Any], n_time: int) -> np.ndarray:
    k = first_key(d, ['I_profile', 'current_A', 'I_A', 'current', 'I'])
    if k is None:
        return np.zeros(n_time, dtype=np.float32)
    arr = _as_1d_float(d[k], k)
    if arr.size != n_time:
        raise ValueError(f'{k} length {arr.size} != n_time {n_time}')
    return arr


def _get_numeric_1d_optional(d: Mapping[str, Any], keys: Sequence[str], n_time: int, fill: float = 0.0) -> Tuple[np.ndarray, str]:
    k = first_key(d, keys)
    if k is None:
        return np.full(n_time, fill, dtype=np.float32), '__filled__'
    try:
        arr = _as_1d_float(d[k], k)
        if arr.size != n_time:
            return np.full(n_time, fill, dtype=np.float32), f'{k}__bad_length_filled'
        return arr, k
    except Exception:
        return np.full(n_time, fill, dtype=np.float32), f'{k}__non_numeric_filled'


def _get_theta(d: Mapping[str, Any], electrode: str, n_time: int) -> Tuple[np.ndarray, str]:
    if electrode == 'a':
        theta_keys = ['theta_a', 'theta_n', 'theta_negative']
        cs_keys = ['cs_a', 'cs_n', 'cs_negative']
        cmax_keys = ['csmax_a', 'csmax_n']
    else:
        theta_keys = ['theta_c', 'theta_p', 'theta_positive']
        cs_keys = ['cs_c', 'cs_p', 'cs_positive']
        cmax_keys = ['csmax_c', 'csmax_p']
    k = first_key(d, theta_keys)
    if k:
        return _orient_time_radial(d[k], n_time, k), k
    k = first_key(d, cs_keys)
    if k is None:
        raise KeyError(f'Missing theta/cs for electrode {electrode}')
    cs = _orient_time_radial(d[k], n_time, k)
    # Best-effort csmax detection. The D15-P0 RG labels should contain theta_* already;
    # this path is only for robustness.
    cmax = None
    for ck in cmax_keys:
        if ck in d:
            try:
                cmax = float(np.asarray(d[ck]).reshape(-1)[0])
                break
            except Exception:
                pass
    if cmax is None or not np.isfinite(cmax) or cmax <= 0:
        # Infer a scale from the observed max; avoid divide by zero.
        cmax = float(np.nanmax(cs)) if np.nanmax(cs) > 2.0 else 1.0
    return (cs / cmax).astype(np.float32), f'{k}__converted_to_theta'


def _get_target_scalar(d: Mapping[str, Any], n_time: int, keys: Sequence[str], required_name: str) -> Tuple[np.ndarray, str]:
    k = first_key(d, keys)
    if k is None:
        raise KeyError(f'Missing target {required_name}; tried {keys}')
    arr = _as_1d_float(d[k], k)
    if arr.size != n_time:
        raise ValueError(f'{k} length {arr.size} != n_time {n_time}')
    return arr.astype(np.float32), k


def _step_features(d: Mapping[str, Any], I: np.ndarray, n_time: int) -> Tuple[np.ndarray, List[str]]:
    # charge/rest/discharge indicators. Prefer step_type if it is readable; fall back to current sign.
    charge = np.zeros(n_time, dtype=np.float32)
    discharge = np.zeros(n_time, dtype=np.float32)
    rest = np.zeros(n_time, dtype=np.float32)
    if 'step_type' in d:
        try:
            st = np.asarray(d['step_type']).reshape(-1)
            if st.size == n_time:
                for i, val in enumerate(st):
                    s = str(val).lower()
                    if 'rest' in s or '搁' in s or '静' in s:
                        rest[i] = 1.0
                    elif 'dis' in s or '放' in s:
                        discharge[i] = 1.0
                    elif 'cha' in s or '充' in s:
                        charge[i] = 1.0
                if (charge + discharge + rest).sum() > 0:
                    unknown = (charge + discharge + rest) == 0
                    rest[unknown & (np.abs(I) < 1e-9)] = 1.0
                    charge[unknown & (I > 0)] = 1.0
                    discharge[unknown & (I < 0)] = 1.0
                    return np.stack([charge, rest, discharge], axis=1), ['is_charge', 'is_rest', 'is_discharge']
        except Exception:
            pass
    eps = max(1e-9, 0.001 * float(np.nanmax(np.abs(I)) + 1e-12))
    charge[I > eps] = 1.0
    discharge[I < -eps] = 1.0
    rest[np.abs(I) <= eps] = 1.0
    return np.stack([charge, rest, discharge], axis=1), ['is_charge', 'is_rest', 'is_discharge']


def _cumtrapz_charge(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    # Use rectangle rule for robustness with stepped data.
    q = np.cumsum(I.astype(np.float32) * dt) / 3600.0  # Ah up to a common sign scale
    scale = float(np.nanmax(np.abs(q)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(q, dtype=np.float32)
    return (q / scale).astype(np.float32)


def profile_id_from_path(npz_path: Path, root: Path) -> str:
    try:
        rel = npz_path.parent.relative_to(root)
        s = str(rel).replace('\\', '/')
        return s if s not in ('', '.') else npz_path.parent.name
    except Exception:
        return npz_path.parent.name or npz_path.stem


def load_profile_arrays(npz_path: Path, root: Path) -> Dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    t = _get_time(d)
    n_time = int(t.size)
    I = _get_current(d, n_time)
    theta_a, theta_key_a = _get_theta(d, 'a', n_time)
    theta_c, theta_key_c = _get_theta(d, 'c', n_time)
    phie, phie_key = _get_target_scalar(d, n_time, ['phie', 'phi_e', 'phi_e_eff'], 'phie')
    phis_c, phis_key = _get_target_scalar(d, n_time, ['phis_c_soft', 'phis_c', 'voltage_soft', 'V_soft', 'V_pred'], 'phis_c')
    voltage_exp, voltage_key = _get_numeric_1d_optional(d, ['voltage_exp', 'voltage_V', 'V_exp', 'V'], n_time, fill=float(np.nanmean(phis_c)))
    temp, temp_key = _get_numeric_1d_optional(d, ['temperature_C', 'temperature_K', 'temp_C', 'T_C', 'T'], n_time, fill=25.0)
    if theta_a.shape[0] != n_time or theta_c.shape[0] != n_time:
        raise ValueError('theta arrays are not time-major')
    return {
        'raw': d,
        'path': str(npz_path),
        'profile_id': profile_id_from_path(npz_path, root),
        't': t.astype(np.float32),
        'I': I.astype(np.float32),
        'theta_a': theta_a.astype(np.float32),
        'theta_c': theta_c.astype(np.float32),
        'phie': phie.astype(np.float32),
        'phis_c': phis_c.astype(np.float32),
        'voltage_exp': voltage_exp.astype(np.float32),
        'temperature': temp.astype(np.float32),
        'source_keys': {
            'theta_a': theta_key_a,
            'theta_c': theta_key_c,
            'phie': phie_key,
            'phis_c': phis_key,
            'voltage_exp': voltage_key,
            'temperature': temp_key,
        }
    }


def build_features(profile: Mapping[str, Any], profile_index: int, profile_count: int, include_profile_onehot: bool = True) -> Tuple[np.ndarray, List[str]]:
    t = np.asarray(profile['t'], dtype=np.float32)
    I = np.asarray(profile['I'], dtype=np.float32)
    voltage = np.asarray(profile['voltage_exp'], dtype=np.float32)
    temp = np.asarray(profile['temperature'], dtype=np.float32)
    n = t.size
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    tn = ((t - t[0]) / span).astype(np.float32)
    I_scale = float(np.nanpercentile(np.abs(I), 99.5))
    if not np.isfinite(I_scale) or I_scale <= 1e-12:
        I_scale = 1.0
    In = I / I_scale
    dI = np.diff(I, prepend=I[0]) / I_scale
    qn = _cumtrapz_charge(t, I)
    vmean = float(np.nanmean(voltage)) if np.isfinite(np.nanmean(voltage)) else 0.0
    vstd = float(np.nanstd(voltage))
    if not np.isfinite(vstd) or vstd <= 1e-9:
        vstd = 1.0
    vn = (voltage - vmean) / vstd
    tmean = float(np.nanmean(temp)) if np.isfinite(np.nanmean(temp)) else 25.0
    tstd = float(np.nanstd(temp))
    if not np.isfinite(tstd) or tstd <= 1e-9:
        tstd = 1.0
    Tn = (temp - tmean) / tstd
    step_feat, step_names = _step_features(profile.get('raw', {}), I, n)
    base = [
        tn,
        tn ** 2,
        np.sin(2 * np.pi * tn).astype(np.float32),
        np.cos(2 * np.pi * tn).astype(np.float32),
        In.astype(np.float32),
        np.abs(In).astype(np.float32),
        dI.astype(np.float32),
        qn.astype(np.float32),
        vn.astype(np.float32),
        Tn.astype(np.float32),
    ]
    names = ['t_norm', 't_norm2', 'sin_t', 'cos_t', 'I_norm', 'absI_norm', 'dI_norm', 'q_norm', 'voltage_exp_norm_local', 'temperature_norm_local']
    X = np.stack(base, axis=1)
    X = np.concatenate([X, step_feat], axis=1)
    names.extend(step_names)
    if include_profile_onehot:
        oh = np.zeros((n, profile_count), dtype=np.float32)
        oh[:, profile_index] = 1.0
        X = np.concatenate([X, oh], axis=1)
        names.extend([f'profile_onehot_{i:02d}' for i in range(profile_count)])
    return X.astype(np.float32), names


def build_targets(profile: Mapping[str, Any]) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[int, int]]]:
    th_a = np.asarray(profile['theta_a'], dtype=np.float32)
    th_c = np.asarray(profile['theta_c'], dtype=np.float32)
    phie = np.asarray(profile['phie'], dtype=np.float32).reshape(-1, 1)
    phis = np.asarray(profile['phis_c'], dtype=np.float32).reshape(-1, 1)
    nra = th_a.shape[1]
    nrc = th_c.shape[1]
    chunks = [th_a, th_c, phie, phis]
    names = [f'theta_a_r{i:02d}' for i in range(nra)] + [f'theta_c_r{i:02d}' for i in range(nrc)] + ['phie', 'phis_c']
    slices: Dict[str, Tuple[int, int]] = {
        'theta_a': (0, nra),
        'theta_c': (nra, nra + nrc),
        'phie': (nra + nrc, nra + nrc + 1),
        'phis_c': (nra + nrc + 1, nra + nrc + 2),
    }
    return np.concatenate(chunks, axis=1).astype(np.float32), names, slices


def _sample_indices(n: int, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count <= 0 or max_count >= n:
        return np.arange(n, dtype=np.int64)
    # Preserve endpoints and random middle points.
    if max_count <= 2:
        return np.linspace(0, n - 1, max_count).astype(np.int64)
    middle = rng.choice(np.arange(1, n - 1), size=max_count - 2, replace=False)
    idx = np.concatenate([[0], np.sort(middle), [n - 1]]).astype(np.int64)
    return idx


def load_profile_metas(softlabel_dir: str | Path, filename: str = 'solution_softlabels.npz') -> List[ProfileMeta]:
    root = Path(softlabel_dir)
    files = discover_npz(root, filename=filename)
    metas: List[ProfileMeta] = []
    for p in files:
        try:
            with np.load(p, allow_pickle=True) as z:
                keys = list(z.files)
                d = {k: z[k] for k in z.files}
            t = _get_time(d)
            th_a, _ = _get_theta(d, 'a', len(t))
            th_c, _ = _get_theta(d, 'c', len(t))
            metas.append(ProfileMeta(
                profile_id=profile_id_from_path(p, root),
                npz_path=str(p),
                n_time=int(len(t)),
                nr_a=int(th_a.shape[1]),
                nr_c=int(th_c.shape[1]),
                keys=keys,
                radial_solver_version=scalar_string(d.get('radial_solver_version', ''), ''),
                cell_uid=scalar_string(d.get('cell_uid', ''), ''),
                batch=scalar_string(d.get('batch', ''), ''),
                protocol=scalar_string(d.get('protocol', ''), ''),
            ))
        except Exception as exc:
            metas.append(ProfileMeta(
                profile_id=p.parent.name,
                npz_path=str(p),
                n_time=-1,
                nr_a=-1,
                nr_c=-1,
                keys=[],
                radial_solver_version=f'ERROR: {exc!r}',
            ))
    return metas


def build_dataset(
    softlabel_dir: str | Path,
    filename: str = 'solution_softlabels.npz',
    max_train_per_profile: int = 8192,
    max_val_per_profile: int = 2048,
    include_profile_onehot: bool = True,
    seed: int = 151,
) -> DatasetBundle:
    root = Path(softlabel_dir)
    files = discover_npz(root, filename=filename)
    if not files:
        raise FileNotFoundError(f'No soft-label npz files found under {root}')
    rng = np.random.default_rng(seed)
    profiles = [load_profile_arrays(p, root) for p in files]
    profile_ids = [str(p['profile_id']) for p in profiles]
    profile_paths = [str(p['path']) for p in profiles]
    Xtr_list: List[np.ndarray] = []
    Ytr_list: List[np.ndarray] = []
    Xva_list: List[np.ndarray] = []
    Yva_list: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    target_slices: Optional[Dict[str, Tuple[int, int]]] = None
    nr_a = nr_c = None
    sample_rows: List[Dict[str, Any]] = []
    for i, prof in enumerate(profiles):
        X, fnames = build_features(prof, i, len(profiles), include_profile_onehot=include_profile_onehot)
        Y, tnames, slices = build_targets(prof)
        if feature_names is None:
            feature_names = fnames
        elif feature_names != fnames:
            raise ValueError('Feature names changed across profiles')
        if target_names is None:
            target_names = tnames
            target_slices = slices
            nr_a = slices['theta_a'][1] - slices['theta_a'][0]
            nr_c = slices['theta_c'][1] - slices['theta_c'][0]
        elif target_names != tnames:
            raise ValueError('Target shape/names changed across profiles; ensure all D15-P0 RG labels use the same nr')
        n = X.shape[0]
        # Sample a superset, then split train/val within selected points.
        train_idx = _sample_indices(n, max_train_per_profile, rng)
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), train_idx, assume_unique=False)
        if remaining.size == 0:
            val_idx = train_idx[::max(1, train_idx.size // max(1, min(max_val_per_profile, train_idx.size)))]
        else:
            val_idx = _sample_indices(remaining.size, min(max_val_per_profile, remaining.size), rng)
            val_idx = remaining[val_idx]
        Xtr_list.append(X[train_idx])
        Ytr_list.append(Y[train_idx])
        Xva_list.append(X[val_idx])
        Yva_list.append(Y[val_idx])
        sample_rows.append({
            'profile_index': i,
            'profile_id': profile_ids[i],
            'path': profile_paths[i],
            'n_time': int(n),
            'train_points': int(len(train_idx)),
            'val_points': int(len(val_idx)),
        })
    X_train = np.concatenate(Xtr_list, axis=0).astype(np.float32)
    Y_train = np.concatenate(Ytr_list, axis=0).astype(np.float32)
    X_val = np.concatenate(Xva_list, axis=0).astype(np.float32)
    Y_val = np.concatenate(Yva_list, axis=0).astype(np.float32)
    x_mean = np.nanmean(X_train, axis=0).astype(np.float32)
    x_std = np.nanstd(X_train, axis=0).astype(np.float32)
    x_std[~np.isfinite(x_std) | (x_std < 1e-8)] = 1.0
    y_mean = np.nanmean(Y_train, axis=0).astype(np.float32)
    y_std = np.nanstd(Y_train, axis=0).astype(np.float32)
    y_std[~np.isfinite(y_std) | (y_std < 1e-8)] = 1.0
    train_meta = {
        'profile_count': len(files),
        'sample_rows': sample_rows,
        'train_points_total': int(X_train.shape[0]),
        'val_points_total': int(X_val.shape[0]),
        'feature_dim': int(X_train.shape[1]),
        'target_dim': int(Y_train.shape[1]),
    }
    return DatasetBundle(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        feature_names=feature_names or [],
        target_names=target_names or [],
        profile_ids=profile_ids,
        profile_paths=profile_paths,
        nr_a=int(nr_a or 0),
        nr_c=int(nr_c or 0),
        target_slices=target_slices or {},
        train_meta=train_meta,
    )
