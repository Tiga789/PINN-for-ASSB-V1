from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

STATE_KEYS = ["cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c"]
OBS_TIME_KEYS = ["t_global_s", "time_s", "t_s", "t", "time"]
OBS_I_KEYS = ["I_profile", "current_A", "I_A", "current", "I"]
OBS_V_KEYS = ["voltage_exp", "voltage_V", "V_exp", "V", "voltage"]
OBS_T_KEYS = ["temperature_C", "temp_C", "T_C", "temperature_K", "T", "temperature"]


@dataclass
class ProfilePack:
    split: str
    canonical_cell_uid: str
    cell_uid: str
    protocol: str
    branch: str
    softlabel_npz: str
    replay_npz: str
    features: np.ndarray
    targets: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    target_slices: Dict[str, Tuple[int, int]]
    t_global_s: np.ndarray
    source_info: Dict[str, Any]


@dataclass
class G1Dataset:
    train_profiles: List[ProfilePack]
    validation_profiles: List[ProfilePack]
    X_train: np.ndarray
    Y_train: np.ndarray
    X_validation: np.ndarray
    Y_validation: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    target_slices: Dict[str, Tuple[int, int]]
    manifest_summary: Dict[str, Any]


def json_load(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(p, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return default


def json_dump(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_csv_dicts(path: str | Path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _first_key(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in d:
            return k
    return None


def _to_1d_float(x: Any, name: str, n: Optional[int] = None, fill: Optional[float] = None) -> np.ndarray:
    try:
        arr = np.asarray(x)
        if arr.dtype.kind in {"U", "S", "O"}:
            raise TypeError(f"{name} is not numeric")
        out = arr.astype(np.float32).reshape(-1)
        if n is not None and out.size != n:
            raise ValueError(f"{name} length {out.size} != {n}")
        return out
    except Exception:
        if n is not None and fill is not None:
            return np.full(n, float(fill), dtype=np.float32)
        raise


def _orient_time_radial(x: Any, n_time: int, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind in {"U", "S", "O"}:
        raise TypeError(f"{name} is not numeric")
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != n_time:
            raise ValueError(f"{name}: 1D length {arr.shape[0]} != n_time {n_time}")
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 1D/2D, got {arr.shape}")
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return arr.T
    raise ValueError(f"{name}: cannot orient {arr.shape} for n_time={n_time}")


def _load_npz_dict(path: str | Path, keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, Any] = {}
    with np.load(p, allow_pickle=True) as z:
        if keys is None:
            use_keys = list(z.files)
        else:
            use_keys = [k for k in keys if k in z.files]
        for k in use_keys:
            out[k] = z[k]
    return out




def _candidate_target_lengths_from_soft(soft: Mapping[str, Any]) -> List[int]:
    """Infer the soft-label time length from state target arrays.

    Replay profiles can be much longer than soft labels.  G1 is a supervised
    generator-surrogate task, so the target arrays define the supervision grid.
    For radial arrays, the non-radial dimension is the time dimension.
    """
    cands: List[int] = []
    for k in ["cs_a", "cs_c", "theta_a", "theta_c"]:
        if k not in soft:
            continue
        a = np.asarray(soft[k])
        if a.dtype.kind in {"U", "S", "O"}:
            continue
        if a.ndim == 1 and a.size > 1:
            cands.append(int(a.size))
        elif a.ndim == 2:
            s0, s1 = int(a.shape[0]), int(a.shape[1])
            # Typical RG field is (time, n_r) or (n_r, time), with n_r <= 128.
            if s0 <= 128 < s1:
                cands.append(s1)
            elif s1 <= 128 < s0:
                cands.append(s0)
            else:
                cands.append(max(s0, s1))
    for k in ["phie", "phis_c"]:
        if k not in soft:
            continue
        a = np.asarray(soft[k])
        if a.dtype.kind in {"U", "S", "O"}:
            continue
        if a.ndim == 1 and a.size > 1:
            cands.append(int(a.size))
        elif a.ndim == 2 and 1 in a.shape:
            cands.append(int(max(a.shape)))
    return cands


def _infer_soft_target_n(soft: Mapping[str, Any]) -> int:
    cands = _candidate_target_lengths_from_soft(soft)
    if not cands:
        raise ValueError("Cannot infer soft-label target time length from cs/theta/phie/phis arrays")
    counts: Dict[int, int] = {}
    for n in cands:
        counts[int(n)] = counts.get(int(n), 0) + 1
    # Prefer the most common candidate; tie-break by larger length.
    return sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]


def _find_1d_exact(d: Mapping[str, Any], keys: Sequence[str], n: int) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for k in keys:
        if k not in d:
            continue
        try:
            arr = _to_1d_float(d[k], k)
        except Exception:
            continue
        if arr.size == int(n):
            return k, arr.astype(np.float32)
    return None, None


def _find_1d_any(d: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for k in keys:
        if k not in d:
            continue
        try:
            arr = _to_1d_float(d[k], k)
        except Exception:
            continue
        if arr.size > 0:
            return k, arr.astype(np.float32)
    return None, None


def _build_target_time(soft: Mapping[str, Any], replay: Mapping[str, Any], n_target: int) -> Tuple[str, np.ndarray]:
    # First choice: the soft-label time axis.  This is the generator output grid.
    k, t = _find_1d_exact(soft, OBS_TIME_KEYS, n_target)
    if t is not None:
        return f"soft:{k}", t
    # Second: replay already has the same length.
    k, t = _find_1d_exact(replay, OBS_TIME_KEYS, n_target)
    if t is not None:
        return f"replay:{k}", t
    # Third: use replay span and linearly place the soft-label samples across it.
    rk, rt = _find_1d_any(replay, OBS_TIME_KEYS)
    if rt is not None and rt.size > 1:
        return f"synthetic_from_replay_span:{rk}", np.linspace(float(rt[0]), float(rt[-1]), int(n_target), dtype=np.float32)
    # Last resort: integer index time.
    return "synthetic_index_time", np.arange(int(n_target), dtype=np.float32)


def _interp_to_target(src_y: np.ndarray, src_t: Optional[np.ndarray], target_t: np.ndarray, fill: float) -> np.ndarray:
    src_y = np.asarray(src_y, dtype=np.float32).reshape(-1)
    target_t = np.asarray(target_t, dtype=np.float32).reshape(-1)
    if src_y.size == target_t.size:
        return src_y.astype(np.float32)
    if src_y.size == 0:
        return np.full(target_t.size, float(fill), dtype=np.float32)
    if src_t is None or np.asarray(src_t).reshape(-1).size != src_y.size:
        # Fall back to index-domain interpolation. This is only for unusual
        # profiles where the time array is absent, and is recorded in source_info.
        x_old = np.linspace(0.0, 1.0, src_y.size, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, target_t.size, dtype=np.float32)
        return np.interp(x_new, x_old, src_y).astype(np.float32)
    src_t = np.asarray(src_t, dtype=np.float32).reshape(-1)
    order = np.argsort(src_t)
    x = src_t[order]
    y = src_y[order]
    # Remove duplicate/non-finite time points for np.interp stability.
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return np.full(target_t.size, float(fill), dtype=np.float32)
    uniq_x, uniq_idx = np.unique(x, return_index=True)
    y = y[uniq_idx]
    if uniq_x.size == 1:
        return np.full(target_t.size, float(y[0]), dtype=np.float32)
    return np.interp(target_t, uniq_x, y, left=float(y[0]), right=float(y[-1])).astype(np.float32)


def _aligned_observed_1d(
    soft: Mapping[str, Any],
    replay: Mapping[str, Any],
    keys: Sequence[str],
    target_t: np.ndarray,
    n_target: int,
    fill: float,
    prefer_replay: bool = True,
) -> Tuple[str, np.ndarray]:
    """Return an observed input aligned to the soft-label target grid.

    For observed voltage/current/temperature we prefer replay data because it is
    the actual time-series input.  Soft labels are only a fallback if they carry
    the same observed field already resampled by the generator.
    """
    sources = [("replay", replay), ("soft", soft)] if prefer_replay else [("soft", soft), ("replay", replay)]
    for src_name, src in sources:
        k, exact = _find_1d_exact(src, keys, n_target)
        if exact is not None:
            return f"{src_name}:{k}:exact", exact.astype(np.float32)
    for src_name, src in sources:
        k, arr = _find_1d_any(src, keys)
        if arr is None:
            continue
        tk, tt = _find_1d_exact(src, OBS_TIME_KEYS, arr.size)
        if tt is None:
            _, tt_any = _find_1d_any(src, OBS_TIME_KEYS)
            tt = tt_any if tt_any is not None and tt_any.size == arr.size else None
        return f"{src_name}:{k}:interpolated", _interp_to_target(arr, tt, target_t, fill)
    return "filled", np.full(int(n_target), float(fill), dtype=np.float32)


def _aligned_step_type(soft: Mapping[str, Any], replay: Mapping[str, Any], target_t: np.ndarray, n_target: int) -> Optional[np.ndarray]:
    for src_name, src in [("soft", soft), ("replay", replay)]:
        if "step_type" not in src:
            continue
        arr = np.asarray(src["step_type"]).reshape(-1)
        if arr.size == n_target:
            return arr
        tk, tt = _find_1d_exact(src, OBS_TIME_KEYS, arr.size)
        if tt is None:
            _, tt_any = _find_1d_any(src, OBS_TIME_KEYS)
            tt = tt_any if tt_any is not None and tt_any.size == arr.size else None
        if tt is not None and arr.size > 0:
            tt = np.asarray(tt, dtype=np.float32).reshape(-1)
            order = np.argsort(tt)
            tt_sorted = tt[order]
            arr_sorted = arr[order]
            pos = np.searchsorted(tt_sorted, target_t, side="left")
            pos = np.clip(pos, 0, arr_sorted.size - 1)
            return arr_sorted[pos]
    return None

def _safe_scalar_to_str(x: Any) -> str:
    try:
        a = np.asarray(x)
        if a.shape == ():
            v = a.item()
        elif a.size == 1:
            v = a.reshape(-1)[0].item()
        else:
            return ""
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)
    except Exception:
        return ""


def canonical_id(record: Mapping[str, Any]) -> str:
    return str(record.get("canonical_cell_uid") or record.get("cell_uid") or record.get("profile_id") or "UNKNOWN")


def load_split_records(split_manifest: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json_load(split_manifest, default={}) or {}
    recs = data.get("records", []) if isinstance(data, Mapping) else []
    return [dict(r) for r in recs if isinstance(r, Mapping)], dict(data) if isinstance(data, Mapping) else {}


def load_semantics_map(g0_profile_semantics_csv: str | Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv_dicts(g0_profile_semantics_csv)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        for k in ["canonical_cell_uid", "cell_uid", "softlabel_npz"]:
            v = str(row.get(k, "")).strip()
            if v:
                out[v] = row
                if k == "softlabel_npz":
                    out[str(Path(v).resolve())] = row
                    try:
                        out[Path(v).parent.name] = row
                    except Exception:
                        pass
    return out


def _semantics_for(record: Mapping[str, Any], sem_map: Mapping[str, Dict[str, str]]) -> Dict[str, str]:
    for key in [canonical_id(record), str(record.get("cell_uid", "")), str(record.get("softlabel_npz", ""))]:
        if key in sem_map:
            return sem_map[key]
        try:
            rkey = str(Path(key).resolve())
            if rkey in sem_map:
                return sem_map[rkey]
        except Exception:
            pass
    return {}


def _select_records(records: Sequence[Mapping[str, Any]], split: str, limit: int, exclude_flagged: bool = True) -> List[Dict[str, Any]]:
    rows = []
    for r in records:
        if exclude_flagged and (str(r.get("split")) == "flagged_probe" or bool(r.get("is_flagged_probe"))):
            continue
        if str(r.get("split")) == split:
            rows.append(dict(r))
    if limit and limit > 0:
        rows = rows[: int(limit)]
    return rows


def _linear_sample_indices(t: np.ndarray, max_points: int, time_window_s: float) -> np.ndarray:
    n = int(t.size)
    idx = np.arange(n, dtype=np.int64)
    if n == 0:
        return idx
    if time_window_s and time_window_s > 0:
        t0 = float(t[0])
        idx = idx[t <= t0 + float(time_window_s)]
        if idx.size == 0:
            idx = np.arange(n, dtype=np.int64)
    if max_points and max_points > 0 and idx.size > max_points:
        pos = np.linspace(0, idx.size - 1, int(max_points)).round().astype(np.int64)
        idx = idx[pos]
    return idx.astype(np.int64)


def _step_features(step_type: Optional[np.ndarray], I: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    n = I.size
    charge = np.zeros(n, dtype=np.float32)
    rest = np.zeros(n, dtype=np.float32)
    discharge = np.zeros(n, dtype=np.float32)
    if step_type is not None:
        try:
            st = np.asarray(step_type).reshape(-1)
            if st.size == n:
                for i, val in enumerate(st):
                    s = str(val).lower()
                    if "rest" in s or "静" in s or "搁" in s:
                        rest[i] = 1.0
                    elif "dis" in s or "放" in s:
                        discharge[i] = 1.0
                    elif "cha" in s or "充" in s:
                        charge[i] = 1.0
        except Exception:
            pass
    eps = max(1e-8, 0.001 * float(np.nanmax(np.abs(I)) + 1e-12))
    unknown = (charge + rest + discharge) == 0
    charge[unknown & (I > eps)] = 1.0
    discharge[unknown & (I < -eps)] = 1.0
    rest[unknown & (np.abs(I) <= eps)] = 1.0
    return np.stack([charge, rest, discharge], axis=1).astype(np.float32), ["is_charge", "is_rest", "is_discharge"]


def _onehot(value: str, vocab: Sequence[str], prefix: str, n: int) -> Tuple[np.ndarray, List[str]]:
    names = [f"{prefix}_{v}" for v in vocab]
    out = np.zeros((n, len(vocab)), dtype=np.float32)
    if value in vocab:
        out[:, vocab.index(value)] = 1.0
    return out, names


def _cum_charge_norm(t: np.ndarray, I: np.ndarray) -> np.ndarray:
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    q = np.cumsum(I.astype(np.float32) * dt) / 3600.0
    scale = float(np.nanmax(np.abs(q))) if q.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(q, dtype=np.float32)
    return (q / scale).astype(np.float32)


def _zscore_local(x: np.ndarray) -> np.ndarray:
    m = float(np.nanmean(x)) if x.size else 0.0
    s = float(np.nanstd(x)) if x.size else 1.0
    if not np.isfinite(s) or s <= 1e-8:
        s = 1.0
    return ((x - m) / s).astype(np.float32)


def _build_features(t: np.ndarray, I: np.ndarray, V: np.ndarray, T: np.ndarray, step_type: Optional[np.ndarray], protocol: str, branch: str, protocol_vocab: Sequence[str], branch_vocab: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    n = int(t.size)
    span = float(t[-1] - t[0]) if n > 1 else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    tn = ((t - t[0]) / span).astype(np.float32)
    I_scale = float(np.nanpercentile(np.abs(I), 99.5)) if I.size else 1.0
    if not np.isfinite(I_scale) or I_scale <= 1e-12:
        I_scale = 1.0
    In = (I / I_scale).astype(np.float32)
    dI = np.diff(In, prepend=In[0]).astype(np.float32)
    Vn = _zscore_local(V)
    dV = np.diff(Vn, prepend=Vn[0]).astype(np.float32)
    Tn = _zscore_local(T)
    qn = _cum_charge_norm(t, I)
    base = np.stack([
        tn,
        tn * tn,
        np.sqrt(np.clip(tn, 0.0, 1.0)).astype(np.float32),
        np.sin(2 * np.pi * tn).astype(np.float32),
        np.cos(2 * np.pi * tn).astype(np.float32),
        In,
        np.abs(In).astype(np.float32),
        dI,
        qn,
        Vn,
        dV,
        Tn,
    ], axis=1).astype(np.float32)
    names = ["t_norm", "t_norm2", "sqrt_t_norm", "sin_t", "cos_t", "I_norm", "absI_norm", "dI_norm", "q_norm", "voltage_exp_norm_local", "dV_norm_local", "temperature_norm_local"]
    step_feat, step_names = _step_features(step_type, I)
    proto, proto_names = _onehot(protocol, protocol_vocab, "protocol", n)
    br, br_names = _onehot(branch, branch_vocab, "branch", n)
    X = np.concatenate([base, step_feat, proto, br], axis=1).astype(np.float32)
    names = names + step_names + proto_names + br_names
    return X, names


def _target_arrays(soft: Mapping[str, Any], n: int) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[int, int]]]:
    cs_a = _orient_time_radial(soft["cs_a"], n, "cs_a")
    cs_c = _orient_time_radial(soft["cs_c"], n, "cs_c")
    th_a = _orient_time_radial(soft["theta_a"], n, "theta_a")
    th_c = _orient_time_radial(soft["theta_c"], n, "theta_c")
    phie = _to_1d_float(soft["phie"], "phie", n).reshape(n, 1)
    phis = _to_1d_float(soft["phis_c"], "phis_c", n).reshape(n, 1)
    names: List[str] = []
    slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    chunks = []
    for key, arr in [("theta_a", th_a), ("theta_c", th_c), ("cs_a", cs_a), ("cs_c", cs_c), ("phie", phie), ("phis_c", phis)]:
        start = cursor
        chunks.append(arr.astype(np.float32))
        if arr.shape[1] == 1:
            names.append(key)
        else:
            names.extend([f"{key}_r{i:02d}" for i in range(arr.shape[1])])
        cursor += arr.shape[1]
        slices[key] = (start, cursor)
    return np.concatenate(chunks, axis=1).astype(np.float32), names, slices


def load_profile_pack(record: Mapping[str, Any], sem_row: Mapping[str, str], protocol_vocab: Sequence[str], branch_vocab: Sequence[str], max_time_points: int, time_window_s: float) -> ProfilePack:
    soft_path = Path(str(record.get("softlabel_npz") or sem_row.get("softlabel_npz") or ""))
    replay_path = Path(str(record.get("replay_npz") or ""))
    if not soft_path.exists():
        raise FileNotFoundError(f"Missing softlabel_npz: {soft_path}")

    # G1 is supervised by generator soft labels; the soft-label arrays define
    # the target grid. Replay observed signals can be much longer and must be
    # interpolated onto this target grid.
    soft_keys = list(set(STATE_KEYS + OBS_TIME_KEYS + OBS_I_KEYS + OBS_V_KEYS + OBS_T_KEYS + ["step_type", "protocol", "batch", "cell_uid"]))
    soft = _load_npz_dict(soft_path, soft_keys)
    replay_keys = list(set(OBS_TIME_KEYS + OBS_I_KEYS + OBS_V_KEYS + OBS_T_KEYS + ["step_type", "protocol", "batch", "cell_uid"]))
    replay = _load_npz_dict(replay_path, replay_keys) if replay_path.exists() else {}

    n_target = _infer_soft_target_n(soft)
    t_key, t = _build_target_time(soft, replay, n_target)
    I_key, I = _aligned_observed_1d(soft, replay, OBS_I_KEYS, t, n_target, fill=0.0, prefer_replay=True)
    V_key, V = _aligned_observed_1d(soft, replay, OBS_V_KEYS, t, n_target, fill=0.0, prefer_replay=True)
    T_key, T = _aligned_observed_1d(soft, replay, OBS_T_KEYS, t, n_target, fill=25.0, prefer_replay=True)
    step_type = _aligned_step_type(soft, replay, t, n_target)

    # Protocol metadata can come from record/G0 semantics first, not from state arrays.
    protocol = str(record.get("protocol") or sem_row.get("protocol") or _safe_scalar_to_str(replay.get("protocol")) or _safe_scalar_to_str(soft.get("protocol")) or "UNKNOWN")
    branch = str(sem_row.get("semantic_branch") or "UNKNOWN_OR_MIXED_BRANCH")

    Y, target_names, slices = _target_arrays(soft, n_target)
    idx = _linear_sample_indices(t, max_points=max_time_points, time_window_s=time_window_s)
    step_idx = None if step_type is None else np.asarray(step_type).reshape(-1)[idx]
    X, feature_names = _build_features(t[idx], I[idx], V[idx], T[idx], step_idx, protocol, branch, protocol_vocab, branch_vocab)
    Y = Y[idx]
    return ProfilePack(
        split=str(record.get("split", "UNKNOWN")),
        canonical_cell_uid=canonical_id(record),
        cell_uid=str(record.get("cell_uid") or ""),
        protocol=protocol,
        branch=branch,
        softlabel_npz=str(soft_path),
        replay_npz=str(replay_path) if replay_path else "",
        features=X,
        targets=Y,
        feature_names=feature_names,
        target_names=target_names,
        target_slices=slices,
        t_global_s=t[idx].astype(np.float32),
        source_info={
            "target_grid_policy": "softlabel_state_arrays_define_time_grid",
            "n_target_softlabel": int(n_target),
            "n_replay_time_any": int(_find_1d_any(replay, OBS_TIME_KEYS)[1].size) if _find_1d_any(replay, OBS_TIME_KEYS)[1] is not None else 0,
            "t_key": t_key,
            "I_key": I_key,
            "V_key": V_key,
            "T_key": T_key,
            "semantic_branch": branch,
            "phie_source_semantics": sem_row.get("phie_source_semantics", ""),
            "phis_c_source_semantics": sem_row.get("phis_c_source_semantics", ""),
        },
    )


def build_g1_dataset(
    split_manifest: str | Path,
    g0_profile_semantics_csv: str | Path,
    train_profile_count: int = 12,
    validation_profile_count: int = 3,
    max_time_points: int = 512,
    time_window_s: float = 40000.0,
) -> G1Dataset:
    records, manifest = load_split_records(split_manifest)
    sem_map = load_semantics_map(g0_profile_semantics_csv)
    train_records = _select_records(records, "train", train_profile_count, exclude_flagged=True)
    validation_records = _select_records(records, "validation", validation_profile_count, exclude_flagged=True)
    selected_records = train_records + validation_records
    if not train_records:
        raise ValueError("No train records selected")
    protocols = sorted({str(r.get("protocol") or "UNKNOWN") for r in selected_records})
    branches = sorted({str(_semantics_for(r, sem_map).get("semantic_branch") or "UNKNOWN_OR_MIXED_BRANCH") for r in selected_records})
    if not branches:
        branches = ["UNKNOWN_OR_MIXED_BRANCH"]
    train_packs: List[ProfilePack] = []
    val_packs: List[ProfilePack] = []
    for r in train_records:
        train_packs.append(load_profile_pack(r, _semantics_for(r, sem_map), protocols, branches, max_time_points, time_window_s))
    for r in validation_records:
        val_packs.append(load_profile_pack(r, _semantics_for(r, sem_map), protocols, branches, max_time_points, time_window_s))
    feature_names = train_packs[0].feature_names
    target_names = train_packs[0].target_names
    target_slices = train_packs[0].target_slices
    for p in train_packs + val_packs:
        if p.feature_names != feature_names:
            raise ValueError("Feature names differ across profiles")
        if p.target_names != target_names:
            raise ValueError("Target names differ across profiles; check n_r and softlabel schema")
    X_train = np.concatenate([p.features for p in train_packs], axis=0).astype(np.float32)
    Y_train = np.concatenate([p.targets for p in train_packs], axis=0).astype(np.float32)
    if val_packs:
        X_val = np.concatenate([p.features for p in val_packs], axis=0).astype(np.float32)
        Y_val = np.concatenate([p.targets for p in val_packs], axis=0).astype(np.float32)
    else:
        X_val = np.zeros((0, X_train.shape[1]), dtype=np.float32)
        Y_val = np.zeros((0, Y_train.shape[1]), dtype=np.float32)
    x_mean = np.nanmean(X_train, axis=0).astype(np.float32)
    x_std = np.nanstd(X_train, axis=0).astype(np.float32)
    x_std[~np.isfinite(x_std) | (x_std < 1e-8)] = 1.0
    y_mean = np.nanmean(Y_train, axis=0).astype(np.float32)
    y_std = np.nanstd(Y_train, axis=0).astype(np.float32)
    y_std[~np.isfinite(y_std) | (y_std < 1e-8)] = 1.0
    summary = {
        "manifest_hash_sha256": manifest.get("manifest_hash_sha256"),
        "record_counts": manifest.get("counts"),
        "train_profile_count": len(train_packs),
        "validation_profile_count": len(val_packs),
        "train_points": int(X_train.shape[0]),
        "validation_points": int(X_val.shape[0]),
        "protocol_vocab": protocols,
        "semantic_branch_vocab": branches,
        "target_dim": int(Y_train.shape[1]),
        "feature_dim": int(X_train.shape[1]),
    }
    return G1Dataset(train_packs, val_packs, X_train, Y_train, X_val, Y_val, x_mean, x_std, y_mean, y_std, feature_names, target_names, target_slices, summary)


def save_profile_predictions(out_dir: str | Path, split_name: str, profiles: Sequence[ProfilePack], pred_arrays: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for i, (prof, pred) in enumerate(zip(profiles, pred_arrays)):
        fname = f"D17_G1_{split_name}_{i:03d}_{prof.canonical_cell_uid}_PRED.npz".replace("\\", "_").replace("/", "_")
        path = p / fname
        arrays: Dict[str, Any] = {
            "t_global_s": prof.t_global_s,
            "canonical_cell_uid": np.array(prof.canonical_cell_uid),
            "cell_uid": np.array(prof.cell_uid),
            "protocol": np.array(prof.protocol),
            "semantic_branch": np.array(prof.branch),
        }
        for key, (a, b) in prof.target_slices.items():
            arrays[f"{key}_pred"] = pred[:, a:b].astype(np.float32)
            arrays[f"{key}_true_report_only"] = prof.targets[:, a:b].astype(np.float32)
        np.savez_compressed(path, **arrays)
        rows.append({"split": split_name, "index": i, "canonical_cell_uid": prof.canonical_cell_uid, "pred_npz": str(path), "n_time": int(prof.targets.shape[0])})
    return rows
