from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STATE_KEYS: tuple[str, ...] = ("theta_a", "theta_c", "cs_a", "cs_c", "phie", "phis_c")
RADIAL_STATES: frozenset[str] = frozenset({"theta_a", "theta_c", "cs_a", "cs_c"})
POTENTIAL_STATES: frozenset[str] = frozenset({"phie", "phis_c"})

TIME_ALIASES: tuple[str, ...] = ("t_global_s", "time_s", "t_s", "time", "t")
CYCLE_ALIASES: tuple[str, ...] = ("cycle_id", "cycle", "cycle_index")
CURRENT_ALIASES: tuple[str, ...] = ("I_profile", "current_A", "I_A", "current", "I")
VOLTAGE_ALIASES: tuple[str, ...] = ("voltage_exp", "voltage_V", "V_exp", "voltage", "V")
TEMP_ALIASES: tuple[str, ...] = ("temperature_C", "temp_C", "T_C", "temperature_K", "temperature", "T")
STEP_ALIASES: tuple[str, ...] = ("step_type", "step", "state")
RADIAL_GRID_ALIASES: dict[str, tuple[str, ...]] = {
    "a": ("r_a", "rho_a", "radial_grid_a", "r_n"),
    "c": ("r_c", "rho_c", "radial_grid_c", "r_p"),
}

PRED_ALIASES: dict[str, tuple[str, ...]] = {
    state: (
        f"{state}_pred",
        f"pred_{state}",
        f"{state}_prediction",
        f"prediction_{state}",
        f"y_pred_{state}",
    )
    for state in STATE_KEYS
}
TRUE_ALIASES: dict[str, tuple[str, ...]] = {
    state: (
        f"{state}_true_report_only",
        f"true_{state}_report_only",
        f"{state}_true",
        f"true_{state}",
        f"{state}_target_report_only",
        f"{state}_target",
        f"target_{state}",
        f"teacher_{state}",
        f"{state}_teacher",
        f"y_true_{state}",
    )
    for state in STATE_KEYS
}
RAW_TRUE_ALIASES: dict[str, tuple[str, ...]] = {
    state: (state,) + TRUE_ALIASES[state] for state in STATE_KEYS
}

META_ALIASES: dict[str, tuple[str, ...]] = {
    "canonical_cell_uid": ("canonical_cell_uid", "cell_uid", "profile_id", "canonical_id"),
    "cell_uid": ("cell_uid", "canonical_cell_uid", "profile_id"),
    "protocol": ("protocol", "protocol_name", "experiment_protocol"),
    "branch": ("semantic_branch", "branch", "generator_branch"),
    "split": ("split", "dataset_split"),
    "source_softlabel_npz": ("source_softlabel_npz", "softlabel_npz", "truth_npz", "target_npz"),
    "branch_family": ("branch_family", "generator_branch_family"),
    "case_role": ("case_role", "diagnostic_role"),
    "case_id": ("case_id", "diagnostic_case_id"),
    "casepack_version": ("casepack_version",),
}


@dataclass
class ArrayCase:
    case_id: str
    prediction_path: str
    truth_path: str | None
    canonical_cell_uid: str
    cell_uid: str
    protocol: str
    branch: str
    split: str
    time_s: np.ndarray
    cycle_id: np.ndarray | None
    current_A: np.ndarray | None
    voltage_V: np.ndarray | None
    temperature_C: np.ndarray | None
    step_type: np.ndarray | None
    radial_grid_a: np.ndarray | None
    radial_grid_c: np.ndarray | None
    pred: dict[str, np.ndarray] = field(default_factory=dict)
    true: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return int(self.time_s.size)

    @property
    def available_states(self) -> list[str]:
        return [s for s in STATE_KEYS if s in self.pred and s in self.true]


def npz_keys(path: str | Path) -> list[str]:
    with np.load(Path(path), allow_pickle=True) as z:
        return list(z.files)


def load_npz_selected(path: str | Path, keys: Sequence[str] | None = None) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {}
    with np.load(p, allow_pickle=True) as z:
        selected = list(z.files) if keys is None else [k for k in keys if k in z.files]
        for key in selected:
            out[key] = z[key]
    return out


def find_key(mapping: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    for key in aliases:
        if key in mapping:
            return key
    lower = {str(k).lower(): str(k) for k in mapping.keys()}
    for key in aliases:
        match = lower.get(str(key).lower())
        if match is not None:
            return match
    return None


def scalar_string(value: Any) -> str:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return ""
        item = arr.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    except Exception:
        return ""


def extract_meta(mapping: Mapping[str, Any], name: str, default: str = "") -> str:
    key = find_key(mapping, META_ALIASES.get(name, (name,)))
    return scalar_string(mapping[key]) if key is not None else default


def as_1d_numeric(value: Any, name: str, dtype: np.dtype = np.float64) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind in {"O", "U", "S"}:
        raise TypeError(f"{name} is not a numeric array: dtype={arr.dtype}")
    return arr.astype(dtype, copy=False).reshape(-1)


def as_1d_any(value: Any) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def infer_time_length(arrays: Mapping[str, Any], prefer_pred: bool = True) -> int:
    candidate_groups: list[tuple[str, ...]] = []
    for state in STATE_KEYS:
        candidate_groups.append(PRED_ALIASES[state] if prefer_pred else RAW_TRUE_ALIASES[state])
    candidates: list[int] = []
    for aliases in candidate_groups:
        key = find_key(arrays, aliases)
        if key is None:
            continue
        arr = np.asarray(arrays[key])
        if arr.ndim == 1:
            candidates.append(int(arr.shape[0]))
        elif arr.ndim == 2:
            a, b = int(arr.shape[0]), int(arr.shape[1])
            if a <= 128 < b:
                candidates.append(b)
            elif b <= 128 < a:
                candidates.append(a)
            else:
                candidates.append(max(a, b))
    if not candidates:
        time_key = find_key(arrays, TIME_ALIASES)
        if time_key is not None:
            return int(np.asarray(arrays[time_key]).size)
        raise ValueError("Cannot infer time dimension from NPZ arrays")
    counts: dict[int, int] = {}
    for n in candidates:
        counts[n] = counts.get(n, 0) + 1
    return sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]


def orient_time_first(value: Any, n_time: int, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind in {"O", "U", "S"}:
        raise TypeError(f"{name} must be numeric")
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim == 1:
        if arr.shape[0] != n_time:
            raise ValueError(f"{name}: length {arr.shape[0]} != n_time {n_time}")
        return arr.reshape(n_time, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected 1D or 2D, got {arr.shape}")
    if arr.shape[0] == n_time:
        return arr
    if arr.shape[1] == n_time:
        return arr.T
    raise ValueError(f"{name}: cannot orient shape {arr.shape} to n_time={n_time}")


def get_state_arrays(mapping: Mapping[str, Any], mode: str, n_time: int) -> dict[str, np.ndarray]:
    if mode not in {"pred", "true", "raw_true"}:
        raise ValueError(f"Unknown mode: {mode}")
    aliases_map = PRED_ALIASES if mode == "pred" else TRUE_ALIASES if mode == "true" else RAW_TRUE_ALIASES
    out: dict[str, np.ndarray] = {}
    for state in STATE_KEYS:
        key = find_key(mapping, aliases_map[state])
        if key is None:
            continue
        out[state] = orient_time_first(mapping[key], n_time, f"{mode}:{state}")
    return out


def get_optional_1d(mapping: Mapping[str, Any], aliases: Sequence[str], n_time: int, numeric: bool = True) -> np.ndarray | None:
    key = find_key(mapping, aliases)
    if key is None:
        return None
    arr = as_1d_numeric(mapping[key], key) if numeric else as_1d_any(mapping[key])
    if arr.size == n_time:
        return arr
    return None


def get_time(mapping: Mapping[str, Any], n_time: int) -> np.ndarray:
    arr = get_optional_1d(mapping, TIME_ALIASES, n_time, numeric=True)
    if arr is None:
        return np.arange(n_time, dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def get_radial_grid(mapping: Mapping[str, Any], electrode: str, n_r: int) -> np.ndarray:
    key = find_key(mapping, RADIAL_GRID_ALIASES[electrode])
    if key is not None:
        arr = as_1d_numeric(mapping[key], key)
        if arr.size == n_r:
            if np.nanmax(np.abs(arr)) > 0:
                return arr.astype(np.float64)
    return np.linspace(0.0, 1.0, n_r, dtype=np.float64)


def nearest_align(source_t: np.ndarray, source_values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=np.float64).reshape(-1)
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    values = np.asarray(source_values)
    if source_t.size != values.shape[0]:
        raise ValueError("nearest_align: source time length does not match values")
    order = np.argsort(source_t)
    st = source_t[order]
    sv = values[order]
    pos = np.searchsorted(st, target_t, side="left")
    pos = np.clip(pos, 0, st.size - 1)
    left = np.clip(pos - 1, 0, st.size - 1)
    choose_left = np.abs(target_t - st[left]) <= np.abs(target_t - st[pos])
    idx = np.where(choose_left, left, pos)
    return sv[idx]


def linear_align(source_t: np.ndarray, source_values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    source_t = np.asarray(source_t, dtype=np.float64).reshape(-1)
    target_t = np.asarray(target_t, dtype=np.float64).reshape(-1)
    values = np.asarray(source_values, dtype=np.float64)
    if source_t.size != values.shape[0]:
        raise ValueError("linear_align: source time length does not match values")
    order = np.argsort(source_t)
    st = source_t[order]
    sv = values[order]
    finite_t = np.isfinite(st)
    st = st[finite_t]
    sv = sv[finite_t]
    if st.size == 0:
        return np.full((target_t.size,) + values.shape[1:], np.nan, dtype=np.float64)
    unique_t, unique_idx = np.unique(st, return_index=True)
    sv = sv[unique_idx]
    if unique_t.size == 1:
        return np.repeat(sv[:1], target_t.size, axis=0)
    flat = sv.reshape(sv.shape[0], -1)
    aligned = np.empty((target_t.size, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = np.isfinite(col)
        if np.count_nonzero(good) < 2:
            fill = float(col[good][0]) if np.count_nonzero(good) == 1 else np.nan
            aligned[:, j] = fill
        else:
            aligned[:, j] = np.interp(
                target_t,
                unique_t[good],
                col[good],
                left=float(col[good][0]),
                right=float(col[good][-1]),
            )
    return aligned.reshape((target_t.size,) + sv.shape[1:])


def step_labels_from_current(current_A: np.ndarray | None, n_time: int) -> np.ndarray:
    if current_A is None:
        return np.full(n_time, "unknown", dtype=object)
    current = np.asarray(current_A, dtype=np.float64).reshape(-1)
    scale = float(np.nanpercentile(np.abs(current), 99.0)) if current.size else 0.0
    eps = max(1e-9, 1e-3 * scale)
    labels = np.full(current.size, "rest", dtype=object)
    labels[current > eps] = "charge"
    labels[current < -eps] = "discharge"
    return labels


def normalize_step_labels(step_type: np.ndarray | None, current_A: np.ndarray | None, n_time: int) -> np.ndarray:
    fallback = step_labels_from_current(current_A, n_time)
    if step_type is None:
        return fallback
    arr = np.asarray(step_type).reshape(-1)
    if arr.size != n_time:
        return fallback
    out = fallback.copy()
    for i, value in enumerate(arr):
        text = str(value).strip().lower()
        if any(token in text for token in ("rest", "idle", "静", "搁")):
            out[i] = "rest"
        elif any(token in text for token in ("dis", "放")):
            out[i] = "discharge"
        elif any(token in text for token in ("cha", "充", "cv")):
            out[i] = "charge"
    return out
