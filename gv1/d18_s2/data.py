from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .common import ConfigError, dump_json, ensure_dir, sha256_file, write_csv
from .uid import CanonicalUID, canonical_from_record


STATE_KEYS = ("cs_a", "cs_c", "theta_a", "theta_c", "phie", "phis_c")
KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "time": ("t_global_s", "time_s", "t_s", "t"),
    "cycle_id": ("cycle_id", "cycle", "cycle_index"),
    "current": ("I_profile", "current_A", "current", "I"),
    "voltage": ("voltage_exp", "voltage_V", "voltage", "V"),
    "temperature": ("temperature_C", "temperature", "T_C", "T"),
    "step_type": ("step_type", "phase", "step_name"),
    "r_a": ("r_a", "rho_a", "radial_a"),
    "r_c": ("r_c", "rho_c", "radial_c"),
    "cs_a": ("cs_a", "c_s_a"),
    "cs_c": ("cs_c", "c_s_c"),
    "theta_a": ("theta_a",),
    "theta_c": ("theta_c",),
    "phie": ("phie", "phi_e"),
    "phis_c": ("phis_c", "phi_s_c", "voltage_soft", "phis_c_soft"),
}

CYCLE_FEATURE_NAMES = [
    "cycle_norm_profile",
    "log_cycle_norm",
    "cycle_position_norm",
    "cumulative_abs_Ah_norm",
    "cumulative_signed_Ah_norm",
    "cycle_charge_Ah_norm",
    "cycle_discharge_Ah_norm",
    "cycle_duration_norm",
    "charge_duration_fraction",
    "rest_duration_fraction",
    "discharge_duration_fraction",
    "mean_abs_current_norm",
    "max_abs_current_norm",
    "mean_voltage_centered",
    "min_voltage_centered",
    "max_voltage_centered",
    "mean_temperature_centered",
    "protocol_index_norm",
    "branch_is_rg",
    "branch_is_p4d",
]

LOCAL_FEATURE_NAMES = [
    "time_in_cycle_norm",
    "current_norm",
    "abs_current_norm",
    "dcurrent_norm",
    "q_signed_in_cycle_norm",
    "q_abs_in_cycle_norm",
    "cumulative_abs_Ah_profile_norm",
    "voltage_z_profile",
    "dvoltage_z_profile",
    "temperature_z_profile",
    "phase_charge",
    "phase_rest",
    "phase_discharge",
    "cycle_norm_profile",
]

PROTOCOL_ORDER = {"2c": 0, "3c": 1, "r2.5": 2, "r3": 3, "random_walk": 4, "geo": 5}


@dataclass
class SourceCycleAudit:
    canonical_cell_uid: str
    role: str
    protocol: str
    branch_family: str
    cycle_id: int
    cycle_position: str
    source_cycle_points: int
    replay_cycle_points: int | None
    preflight_examined_points: int
    preflight_downsampled: bool
    micro_smoke_exported_points: int
    micro_smoke_coverage_fraction: float
    micro_smoke_downsampled: bool
    source_to_replay_fraction: float | None
    source_generator_grid_complete: bool | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RawProfile:
    canonical_cell_uid: str
    role: str
    protocol: str
    branch_family: str
    split: str
    source_softlabel_npz: str
    source_replay_npz: str
    selected_cycle_ids: np.ndarray
    selected_cycle_positions: np.ndarray
    cycle_features: np.ndarray
    local_features: np.ndarray
    cycle_index: np.ndarray
    t_source_s: np.ndarray
    q_signed_global_Ah: np.ndarray
    cbar0_a: float
    cbar0_c: float
    cbar_true_a: np.ndarray
    cbar_true_c: np.ndarray
    potential_baseline: np.ndarray
    r_a: np.ndarray
    r_c: np.ndarray
    targets: dict[str, np.ndarray]
    audit_rows: list[dict[str, Any]]


@dataclass
class TrainPhysicalFit:
    theta_offset_a: float
    theta_scale_a: float
    theta_offset_c: float
    theta_scale_c: float
    dcbar_dAh_a: float
    dcbar_dAh_c: float
    target_scales: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "theta_offset_a": self.theta_offset_a,
            "theta_scale_a": self.theta_scale_a,
            "theta_offset_c": self.theta_offset_c,
            "theta_scale_c": self.theta_scale_c,
            "dcbar_dAh_a": self.dcbar_dAh_a,
            "dcbar_dAh_c": self.dcbar_dAh_c,
            "target_scales": dict(self.target_scales),
        }


def _first_key(files: Iterable[str], aliases: Sequence[str], *, required: bool = True) -> str | None:
    file_set = set(files)
    for key in aliases:
        if key in file_set:
            return key
    if required:
        raise ConfigError(f"Required NPZ field missing; expected one of {aliases}")
    return None


def inspect_npz_schema(path: str | Path) -> dict[str, str | None]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"NPZ not found: {p}")
    with np.load(p, allow_pickle=True) as data:
        files = data.files
    result: dict[str, str | None] = {}
    for logical, aliases in KEY_ALIASES.items():
        result[logical] = _first_key(files, aliases, required=logical not in {"temperature", "step_type"})
    return result


def _as_1d(value: Any, n: int | None = None, dtype: Any = np.float64) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[-1] == 1:
        arr = arr.reshape(-1)
    else:
        arr = arr.reshape(-1)
    if n is not None and arr.size != n:
        raise ConfigError(f"Array length mismatch: expected {n}, got {arr.size}")
    return arr.astype(dtype, copy=False)


def _as_state(value: Any, n: int, radial: bool) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if radial:
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ConfigError(f"Radial state must have shape [time, radial], got {arr.shape}")
    else:
        arr = arr.reshape(n, -1)
        if arr.shape[1] != 1:
            arr = arr[:, :1]
    if not np.isfinite(arr).all():
        raise ConfigError("State array contains non-finite values")
    return arr


def spherical_weights(r: np.ndarray) -> np.ndarray:
    rho = np.asarray(r, dtype=np.float64).reshape(-1)
    if rho.size < 2:
        return np.ones_like(rho)
    if rho[-1] <= 0:
        rho = np.linspace(0.0, 1.0, rho.size)
    else:
        rho = (rho - rho[0]) / max(1e-12, float(rho[-1] - rho[0]))
    edges = np.empty(rho.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (rho[:-1] + rho[1:])
    w = np.diff(np.clip(edges, 0.0, 1.0) ** 3)
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    return w / total if total > 0 else np.full(rho.size, 1.0 / rho.size)


def radial_mean(values: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(values, dtype=np.float64) * spherical_weights(r)[None, :], axis=1)




def safe_time_gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Finite gradient that remains defined when a source timestamp is repeated."""
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    t = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if y.size != t.size:
        raise ConfigError("gradient values/time length mismatch")
    if y.size < 2:
        return np.zeros_like(y)
    dt = np.diff(t)
    dy = np.diff(y)
    slope = np.zeros_like(dt)
    np.divide(dy, dt, out=slope, where=np.isfinite(dt) & (dt > 0))
    slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
    grad = np.empty_like(y)
    grad[0] = slope[0]
    grad[-1] = slope[-1]
    if y.size > 2:
        grad[1:-1] = 0.5 * (slope[:-1] + slope[1:])
    return grad

def cumulative_ah(t_s: np.ndarray, current_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    i = np.asarray(current_a, dtype=np.float64).reshape(-1)
    if t.size != i.size:
        raise ConfigError("time/current length mismatch")
    dt = np.diff(t, prepend=t[0])
    dt = np.where(np.isfinite(dt) & (dt >= 0), dt, 0.0)
    i_prev = np.concatenate([[i[0]], i[:-1]])
    dq = 0.5 * (i_prev + i) * dt / 3600.0
    return np.cumsum(dq), np.cumsum(np.abs(dq))


def classify_phase(step_type: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    i = np.asarray(current, dtype=np.float64).reshape(-1)
    out = np.full(i.size, "rest", dtype="U16")
    threshold = max(1e-8, float(np.nanpercentile(np.abs(i), 95)) * 1e-4) if i.size else 1e-8
    out[i > threshold] = "charge"
    out[i < -threshold] = "discharge"
    if step_type is None:
        return out
    raw = np.asarray(step_type).reshape(-1)
    if raw.size != i.size:
        return out
    text = np.char.lower(raw.astype("U64"))
    charge = np.char.find(text, "charge") >= 0
    discharge = np.char.find(text, "discharge") >= 0
    rest = (np.char.find(text, "rest") >= 0) | (np.char.find(text, "搁置") >= 0)
    out[charge] = "charge"
    out[discharge] = "discharge"
    out[rest] = "rest"
    return out


def valid_cycle_ids(cycle_id: np.ndarray, min_points: int) -> tuple[np.ndarray, dict[int, int]]:
    values, counts = np.unique(np.asarray(cycle_id, dtype=np.int64), return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    valid = np.asarray([int(v) for v, c in zip(values, counts) if int(c) >= min_points], dtype=np.int64)
    return valid, count_map


def select_position_cycles(valid: np.ndarray, per_position: int) -> tuple[np.ndarray, dict[int, str]]:
    cycles = np.sort(np.unique(np.asarray(valid, dtype=np.int64)))
    needed = 3 * per_position
    if cycles.size < needed:
        raise ConfigError(f"Need at least {needed} valid cycles, found {cycles.size}")
    early = cycles[:per_position]
    late = cycles[-per_position:]
    center = cycles.size // 2
    start = max(per_position, center - per_position // 2)
    stop = start + per_position
    if stop > cycles.size - per_position:
        stop = cycles.size - per_position
        start = stop - per_position
    middle = cycles[start:stop]
    selected = np.concatenate([early, middle, late])
    if np.unique(selected).size != selected.size:
        raise ConfigError("Early/middle/late cycle selection overlapped")
    positions = {int(c): "early" for c in early}
    positions.update({int(c): "middle" for c in middle})
    positions.update({int(c): "late" for c in late})
    return selected, positions


def stratified_cycle_indices(indices: np.ndarray, phases: np.ndarray, target_count: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size < target_count:
        raise ConfigError(f"Cycle has {idx.size} points, below micro-smoke target {target_count}")
    if idx.size == target_count:
        return idx.copy()
    local_phase = phases[idx]
    mandatory: set[int] = {0, idx.size - 1}
    changes = np.flatnonzero(local_phase[1:] != local_phase[:-1]) + 1
    for pos in changes:
        for q in (pos - 1, pos, pos + 1):
            if 0 <= q < idx.size:
                mandatory.add(int(q))
    # Ensure each phase contributes points.
    for phase in ("charge", "rest", "discharge"):
        where = np.flatnonzero(local_phase == phase)
        if where.size:
            for q in np.linspace(0, where.size - 1, min(4, where.size)).round().astype(int):
                mandatory.add(int(where[q]))
    uniform = np.linspace(0, idx.size - 1, target_count).round().astype(int)
    chosen = set(int(x) for x in uniform)
    chosen.update(mandatory)
    if len(chosen) > target_count:
        # Mandatory transitions are prioritized, then fill by evenly distributed mandatory positions.
        must = sorted(mandatory)
        if len(must) >= target_count:
            keep = np.linspace(0, len(must) - 1, target_count).round().astype(int)
            chosen = {must[int(k)] for k in keep}
        else:
            extras = [x for x in sorted(chosen) if x not in mandatory]
            need = target_count - len(must)
            keep = np.linspace(0, len(extras) - 1, need).round().astype(int) if need and extras else []
            chosen = set(must)
            chosen.update(extras[int(k)] for k in keep)
    elif len(chosen) < target_count:
        for q in range(idx.size):
            chosen.add(q)
            if len(chosen) == target_count:
                break
    local = np.asarray(sorted(chosen), dtype=np.int64)
    if local.size != target_count:
        raise ConfigError(f"Could not construct exactly {target_count} sample points")
    return idx[local]


def _safe_z(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mean = float(np.nanmean(arr)) if arr.size else 0.0
    std = float(np.nanstd(arr)) if arr.size else 1.0
    if not math.isfinite(std) or std < 1e-8:
        std = 1.0
    return (arr - mean) / std


def _normalize_phase(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(phase).astype("U16")
    return (p == "charge").astype(np.float64), (p == "rest").astype(np.float64), (p == "discharge").astype(np.float64)


def _resample_radial(values: np.ndarray, source_r: np.ndarray, target_r: np.ndarray) -> np.ndarray:
    if values.shape[1] == target_r.size and np.allclose(source_r, target_r, atol=1e-7, rtol=1e-6):
        return values
    src = np.asarray(source_r, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target_r, dtype=np.float64).reshape(-1)
    src = (src - src[0]) / max(1e-12, float(src[-1] - src[0]))
    tgt = (tgt - tgt[0]) / max(1e-12, float(tgt[-1] - tgt[0]))
    out = np.empty((values.shape[0], tgt.size), dtype=np.float64)
    for row in range(values.shape[0]):
        out[row] = np.interp(tgt, src, values[row])
    return out


def replay_cycle_counts(path: str | Path | None) -> dict[int, int]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=True) as data:
            key = _first_key(data.files, KEY_ALIASES["cycle_id"], required=False)
            if key is None:
                return {}
            cycle = _as_1d(data[key], dtype=np.int64)
        values, counts = np.unique(cycle, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}
    except Exception:
        return {}


def inspect_profile_cycles(
    record: Mapping[str, Any],
    *,
    min_source_points_per_cycle: int,
    cycles_per_position: int,
    micro_points_per_cycle: int,
    replay_path: str | Path | None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    uid = canonical_from_record(record)
    path = Path(str(record["softlabel_npz"]))
    schema = inspect_npz_schema(path)
    with np.load(path, allow_pickle=True) as data:
        cycle = _as_1d(data[str(schema["cycle_id"])], dtype=np.int64)
        time = _as_1d(data[str(schema["time"])], n=cycle.size)
        current = _as_1d(data[str(schema["current"])], n=cycle.size)
    if not np.isfinite(time).all() or not np.isfinite(current).all():
        raise ConfigError(f"Non-finite time/current in {uid.canonical}")
    if np.any(np.diff(time) < 0):
        raise ConfigError(f"Non-monotonic source time in {uid.canonical}")
    valid, source_counts = valid_cycle_ids(cycle, min_source_points_per_cycle)
    selected, positions = select_position_cycles(valid, cycles_per_position)
    replay_counts = replay_cycle_counts(replay_path)
    rows: list[dict[str, Any]] = []
    for cid in selected:
        source_count = source_counts[int(cid)]
        replay_count = replay_counts.get(int(cid))
        ratio = source_count / replay_count if replay_count else None
        rows.append(
            SourceCycleAudit(
                canonical_cell_uid=uid.canonical,
                role=str(record.get("d18_s2_role", "")),
                protocol=uid.protocol,
                branch_family=uid.branch_family,
                cycle_id=int(cid),
                cycle_position=positions[int(cid)],
                source_cycle_points=source_count,
                replay_cycle_points=replay_count,
                preflight_examined_points=source_count,
                preflight_downsampled=False,
                micro_smoke_exported_points=micro_points_per_cycle,
                micro_smoke_coverage_fraction=micro_points_per_cycle / source_count,
                micro_smoke_downsampled=micro_points_per_cycle < source_count,
                source_to_replay_fraction=ratio,
                source_generator_grid_complete=(source_count == replay_count) if replay_count else None,
            ).as_dict()
        )
    return rows, selected


def _cycle_features(
    *,
    selected_cycles: np.ndarray,
    all_cycles: np.ndarray,
    cycle_id: np.ndarray,
    time: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    temperature: np.ndarray,
    phase: np.ndarray,
    q_signed: np.ndarray,
    q_abs: np.ndarray,
    protocol: str,
    branch_family: str,
) -> np.ndarray:
    unique_all = np.sort(np.unique(all_cycles))
    min_cycle = float(unique_all[0])
    max_cycle = float(unique_all[-1])
    cycle_span = max(1.0, max_cycle - min_cycle)
    max_abs_i = max(1e-8, float(np.max(np.abs(current))))
    total_abs_q = max(1e-8, float(q_abs[-1]))
    durations: list[float] = []
    for cid in unique_all:
        idx = np.flatnonzero(cycle_id == cid)
        durations.append(float(time[idx[-1]] - time[idx[0]]) if idx.size > 1 else 0.0)
    max_duration = max(1.0, max(durations) if durations else 1.0)
    protocol_idx = PROTOCOL_ORDER.get(protocol.lower(), len(PROTOCOL_ORDER))
    protocol_norm = protocol_idx / max(1, len(PROTOCOL_ORDER) - 1)
    rows: list[list[float]] = []
    for cid in selected_cycles:
        idx = np.flatnonzero(cycle_id == cid)
        tseg = time[idx]
        iseg = current[idx]
        vseg = voltage[idx]
        tempseg = temperature[idx]
        pseg = phase[idx]
        dt = np.diff(tseg, prepend=tseg[0])
        dt = np.maximum(dt, 0.0)
        charge_ah = float(np.sum(np.maximum(iseg, 0.0) * dt / 3600.0))
        discharge_ah = float(np.sum(np.maximum(-iseg, 0.0) * dt / 3600.0))
        duration = float(tseg[-1] - tseg[0]) if idx.size > 1 else 0.0
        duration_safe = max(1e-8, float(np.sum(dt)))
        charge_frac = float(np.sum(dt[pseg == "charge"]) / duration_safe)
        rest_frac = float(np.sum(dt[pseg == "rest"]) / duration_safe)
        discharge_frac = float(np.sum(dt[pseg == "discharge"]) / duration_safe)
        cyc_norm = (float(cid) - min_cycle) / cycle_span
        rows.append(
            [
                cyc_norm,
                math.log1p(max(0.0, float(cid) - min_cycle)) / math.log1p(cycle_span),
                cyc_norm,
                float(q_abs[idx[-1]]) / total_abs_q,
                float(q_signed[idx[-1]]) / total_abs_q,
                charge_ah / total_abs_q,
                discharge_ah / total_abs_q,
                duration / max_duration,
                charge_frac,
                rest_frac,
                discharge_frac,
                float(np.mean(np.abs(iseg))) / max_abs_i,
                float(np.max(np.abs(iseg))) / max_abs_i,
                (float(np.mean(vseg)) - 3.35) / 1.0,
                (float(np.min(vseg)) - 3.35) / 1.0,
                (float(np.max(vseg)) - 3.35) / 1.0,
                (float(np.mean(tempseg)) - 25.0) / 20.0,
                protocol_norm,
                1.0 if branch_family == "RG" else 0.0,
                1.0 if branch_family == "P4D" else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def load_raw_profile(
    record: Mapping[str, Any],
    *,
    selected_cycles: np.ndarray,
    micro_points_per_cycle: int,
    target_radial_points: int,
    replay_path: str | Path | None,
    cycle_audit_rows: list[dict[str, Any]],
) -> RawProfile:
    uid = canonical_from_record(record)
    p = Path(str(record["softlabel_npz"]))
    schema = inspect_npz_schema(p)
    with np.load(p, allow_pickle=True) as data:
        time = _as_1d(data[str(schema["time"])])
        n = time.size
        cycle = _as_1d(data[str(schema["cycle_id"])], n=n, dtype=np.int64)
        current = _as_1d(data[str(schema["current"])], n=n)
        voltage = _as_1d(data[str(schema["voltage"])], n=n)
        temp_key = schema["temperature"]
        temperature = _as_1d(data[str(temp_key)], n=n) if temp_key else np.full(n, 25.0)
        step_key = schema["step_type"]
        step = np.asarray(data[str(step_key)]).reshape(-1) if step_key else None
        r_a = _as_1d(data[str(schema["r_a"])])
        r_c = _as_1d(data[str(schema["r_c"])])
        cs_a_full = _as_state(data[str(schema["cs_a"])], n=n, radial=True)
        cs_c_full = _as_state(data[str(schema["cs_c"])], n=n, radial=True)
        theta_a_full = _as_state(data[str(schema["theta_a"])], n=n, radial=True)
        theta_c_full = _as_state(data[str(schema["theta_c"])], n=n, radial=True)
        phie_full = _as_state(data[str(schema["phie"])], n=n, radial=False)
        phis_c_full = _as_state(data[str(schema["phis_c"])], n=n, radial=False)
    phase = classify_phase(step, current)
    q_signed, q_abs = cumulative_ah(time, current)
    cbar_a_full = radial_mean(cs_a_full, r_a)
    cbar_c_full = radial_mean(cs_c_full, r_c)
    target_r_a = np.linspace(float(r_a[0]), float(r_a[-1]), target_radial_points)
    target_r_c = np.linspace(float(r_c[0]), float(r_c[-1]), target_radial_points)

    selected_idx_parts: list[np.ndarray] = []
    cycle_positions: list[str] = []
    audit_by_cycle = {int(row["cycle_id"]): row for row in cycle_audit_rows}
    for cid in selected_cycles:
        idx = np.flatnonzero(cycle == cid)
        picked = stratified_cycle_indices(idx, phase, micro_points_per_cycle)
        selected_idx_parts.append(picked)
        cycle_positions.append(str(audit_by_cycle[int(cid)]["cycle_position"]))
    selected_idx = np.concatenate(selected_idx_parts)
    if np.unique(selected_idx).size != selected_idx.size:
        raise ConfigError(f"Duplicate sampled time indices for {uid.canonical}")

    cycle_index = np.repeat(np.arange(selected_cycles.size, dtype=np.int64), micro_points_per_cycle)
    t_sel = time[selected_idx]
    i_sel = current[selected_idx]
    v_sel = voltage[selected_idx]
    temp_sel = temperature[selected_idx]
    phase_sel = phase[selected_idx]
    q_abs_total = max(1e-8, float(q_abs[-1]))
    max_abs_i = max(1e-8, float(np.max(np.abs(current))))
    voltage_z = _safe_z(voltage)
    temp_z = _safe_z(temperature)
    d_i = safe_time_gradient(current, time)
    d_v = safe_time_gradient(voltage, time)
    d_i_scale = max(1e-8, float(np.nanpercentile(np.abs(d_i), 99)))
    d_v_scale = max(1e-8, float(np.nanpercentile(np.abs(d_v), 99)))

    local_rows: list[np.ndarray] = []
    for cid, picked in zip(selected_cycles, selected_idx_parts):
        all_idx = np.flatnonzero(cycle == cid)
        t0 = float(time[all_idx[0]])
        duration = max(1e-8, float(time[all_idx[-1]] - t0))
        q_cyc_signed, q_cyc_abs = cumulative_ah(time[all_idx], current[all_idx])
        lookup = {int(global_idx): local_pos for local_pos, global_idx in enumerate(all_idx)}
        local_pos = np.asarray([lookup[int(x)] for x in picked], dtype=np.int64)
        q_scale = max(1e-8, float(q_cyc_abs[-1]))
        charge, rest, discharge = _normalize_phase(phase[picked])
        cyc_norm = (float(cid) - float(np.min(cycle))) / max(1.0, float(np.max(cycle) - np.min(cycle)))
        block = np.column_stack(
            [
                (time[picked] - t0) / duration,
                current[picked] / max_abs_i,
                np.abs(current[picked]) / max_abs_i,
                np.clip(d_i[picked] / d_i_scale, -5.0, 5.0),
                q_cyc_signed[local_pos] / q_scale,
                q_cyc_abs[local_pos] / q_scale,
                q_abs[picked] / q_abs_total,
                voltage_z[picked],
                np.clip(d_v[picked] / d_v_scale, -5.0, 5.0),
                temp_z[picked],
                charge,
                rest,
                discharge,
                np.full(picked.size, cyc_norm),
            ]
        )
        local_rows.append(block)
    local_features = np.concatenate(local_rows, axis=0).astype(np.float32)

    cycle_features = _cycle_features(
        selected_cycles=selected_cycles,
        all_cycles=cycle,
        cycle_id=cycle,
        time=time,
        current=current,
        voltage=voltage,
        temperature=temperature,
        phase=phase,
        q_signed=q_signed,
        q_abs=q_abs,
        protocol=uid.protocol,
        branch_family=uid.branch_family,
    )

    targets = {
        "cs_a": _resample_radial(cs_a_full[selected_idx], r_a, target_r_a).astype(np.float32),
        "cs_c": _resample_radial(cs_c_full[selected_idx], r_c, target_r_c).astype(np.float32),
        "theta_a": _resample_radial(theta_a_full[selected_idx], r_a, target_r_a).astype(np.float32),
        "theta_c": _resample_radial(theta_c_full[selected_idx], r_c, target_r_c).astype(np.float32),
        "phie": phie_full[selected_idx].astype(np.float32),
        "phis_c": phis_c_full[selected_idx].astype(np.float32),
    }
    potential_baseline = np.column_stack([np.zeros(selected_idx.size), v_sel]).astype(np.float32)
    result = RawProfile(
        canonical_cell_uid=uid.canonical,
        role=str(record.get("d18_s2_role", "")),
        protocol=uid.protocol,
        branch_family=uid.branch_family,
        split=str(record.get("split", "")),
        source_softlabel_npz=str(p),
        source_replay_npz=str(replay_path or ""),
        selected_cycle_ids=np.asarray(selected_cycles, dtype=np.int64),
        selected_cycle_positions=np.asarray(cycle_positions, dtype="U8"),
        cycle_features=cycle_features,
        local_features=local_features,
        cycle_index=cycle_index,
        t_source_s=t_sel.astype(np.float64),
        q_signed_global_Ah=q_signed[selected_idx].astype(np.float64),
        cbar0_a=float(cbar_a_full[0]),
        cbar0_c=float(cbar_c_full[0]),
        cbar_true_a=cbar_a_full[selected_idx].astype(np.float64),
        cbar_true_c=cbar_c_full[selected_idx].astype(np.float64),
        potential_baseline=potential_baseline,
        r_a=target_r_a.astype(np.float32),
        r_c=target_r_c.astype(np.float32),
        targets=targets,
        audit_rows=cycle_audit_rows,
    )
    del cs_a_full, cs_c_full, theta_a_full, theta_c_full, phie_full, phis_c_full
    gc.collect()
    return result


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    if xv.size < 2:
        raise ConfigError("Not enough points for linear physical fit")
    A = np.column_stack([np.ones(xv.size), xv])
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_train_physical_parameters(profiles: Sequence[RawProfile]) -> TrainPhysicalFit:
    train = [p for p in profiles if p.role == "fit_train"]
    if not train:
        raise ConfigError("No fit_train profiles available")
    cs_a = np.concatenate([p.targets["cs_a"].reshape(-1) for p in train])
    th_a = np.concatenate([p.targets["theta_a"].reshape(-1) for p in train])
    cs_c = np.concatenate([p.targets["cs_c"].reshape(-1) for p in train])
    th_c = np.concatenate([p.targets["theta_c"].reshape(-1) for p in train])
    offset_a, scale_a = _linear_fit(cs_a, th_a)
    offset_c, scale_c = _linear_fit(cs_c, th_c)
    if scale_a <= 0 or scale_c <= 0:
        raise ConfigError("Theta-vs-cs scale must be positive")

    qa: list[np.ndarray] = []
    ya: list[np.ndarray] = []
    qc: list[np.ndarray] = []
    yc: list[np.ndarray] = []
    for p in train:
        q = p.q_signed_global_Ah
        qa.append(q)
        ya.append(p.cbar_true_a - p.cbar0_a)
        qc.append(q)
        yc.append(p.cbar_true_c - p.cbar0_c)
    q_a = np.concatenate(qa)
    y_a = np.concatenate(ya)
    q_c = np.concatenate(qc)
    y_c = np.concatenate(yc)
    denom_a = max(1e-12, float(np.dot(q_a, q_a)))
    denom_c = max(1e-12, float(np.dot(q_c, q_c)))
    slope_a = float(np.dot(q_a, y_a) / denom_a)
    slope_c = float(np.dot(q_c, y_c) / denom_c)

    scales: dict[str, float] = {}
    for key in STATE_KEYS:
        arr = np.concatenate([p.targets[key].reshape(-1) for p in train])
        q05, q95 = np.nanpercentile(arr, [5, 95])
        scale = max(1e-6, float(q95 - q05), float(np.nanstd(arr)))
        scales[key] = scale
    return TrainPhysicalFit(offset_a, scale_a, offset_c, scale_c, slope_a, slope_c, scales)


def save_prepared_profiles(
    profiles: Sequence[RawProfile],
    fit: TrainPhysicalFit,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    out = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        cbar_a = profile.cbar0_a + fit.dcbar_dAh_a * profile.q_signed_global_Ah
        cbar_c = profile.cbar0_c + fit.dcbar_dAh_c * profile.q_signed_global_Ah
        arrays: dict[str, Any] = {
            "canonical_cell_uid": np.array(profile.canonical_cell_uid),
            "role": np.array(profile.role),
            "protocol": np.array(profile.protocol),
            "branch_family": np.array(profile.branch_family),
            "split": np.array(profile.split),
            "source_softlabel_npz": np.array(profile.source_softlabel_npz),
            "source_replay_npz": np.array(profile.source_replay_npz),
            "selected_cycle_ids": profile.selected_cycle_ids,
            "selected_cycle_positions": profile.selected_cycle_positions,
            "cycle_features": profile.cycle_features,
            "local_features": profile.local_features,
            "cycle_index": profile.cycle_index,
            "t_source_s": profile.t_source_s,
            "q_signed_global_Ah": profile.q_signed_global_Ah,
            "cbar_baseline": np.column_stack([cbar_a, cbar_c]).astype(np.float32),
            "cbar_true_report_only": np.column_stack([profile.cbar_true_a, profile.cbar_true_c]).astype(np.float32),
            "potential_baseline": profile.potential_baseline,
            "theta_offset": np.array([fit.theta_offset_a, fit.theta_offset_c], dtype=np.float32),
            "theta_scale": np.array([fit.theta_scale_a, fit.theta_scale_c], dtype=np.float32),
            "r_a": profile.r_a,
            "r_c": profile.r_c,
            "teacher_initial_cbar_anchor_used": np.array(True),
            "formal_s2_training_eligible": np.array(False),
        }
        for key, value in profile.targets.items():
            arrays[f"{key}_true"] = value
        role_dir = ensure_dir(out / profile.role)
        path = role_dir / f"{profile.canonical_cell_uid}.npz"
        np.savez_compressed(path, **arrays)
        rows.append(
            {
                "canonical_cell_uid": profile.canonical_cell_uid,
                "role": profile.role,
                "protocol": profile.protocol,
                "branch_family": profile.branch_family,
                "selected_cycle_count": int(profile.selected_cycle_ids.size),
                "time_point_count": int(profile.local_features.shape[0]),
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    write_csv(rows, out / "D18_S2_MICRO_CASEPACK_MANIFEST.csv")
    dump_json(
        {
            "status": "PASS",
            "profile_count": len(rows),
            "roles": sorted({r["role"] for r in rows}),
            "protocols": sorted({r["protocol"] for r in rows}),
            "branches": sorted({r["branch_family"] for r in rows}),
            "teacher_initial_cbar_anchor_used": True,
            "formal_s2_training_eligible": False,
            "physical_fit": fit.as_dict(),
            "cycle_feature_names": CYCLE_FEATURE_NAMES,
            "local_feature_names": LOCAL_FEATURE_NAMES,
        },
        out / "D18_S2_MICRO_CASEPACK_SUMMARY.json",
    )
    return rows


def load_prepared_npz(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with np.load(p, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def build_micro_casepack(
    records: Sequence[Mapping[str, Any]],
    replay_paths: Mapping[str, str | Path | None],
    *,
    output_dir: str | Path,
    min_source_points_per_cycle: int,
    cycles_per_position: int,
    micro_points_per_cycle: int,
    target_radial_points: int,
    progress: callable | None = None,
) -> dict[str, Any]:
    all_audit_rows: list[dict[str, Any]] = []
    raw_profiles: list[RawProfile] = []
    selected_by_uid: dict[str, np.ndarray] = {}
    for idx, record in enumerate(records, start=1):
        uid = canonical_from_record(record).canonical
        if progress:
            progress(f"[{idx}/{len(records)}] inspecting cycles: {uid}")
        replay = replay_paths.get(uid)
        audit_rows, selected = inspect_profile_cycles(
            record,
            min_source_points_per_cycle=min_source_points_per_cycle,
            cycles_per_position=cycles_per_position,
            micro_points_per_cycle=micro_points_per_cycle,
            replay_path=replay,
        )
        all_audit_rows.extend(audit_rows)
        selected_by_uid[uid] = selected
    write_csv(all_audit_rows, Path(output_dir) / "D18_S2_PER_CYCLE_SOURCE_COVERAGE.csv")

    for idx, record in enumerate(records, start=1):
        uid = canonical_from_record(record).canonical
        if progress:
            progress(f"[{idx}/{len(records)}] loading selected generator states: {uid}")
        profile_rows = [r for r in all_audit_rows if r["canonical_cell_uid"] == uid]
        raw_profiles.append(
            load_raw_profile(
                record,
                selected_cycles=selected_by_uid[uid],
                micro_points_per_cycle=micro_points_per_cycle,
                target_radial_points=target_radial_points,
                replay_path=replay_paths.get(uid),
                cycle_audit_rows=profile_rows,
            )
        )
    fit = fit_train_physical_parameters(raw_profiles)
    manifest_rows = save_prepared_profiles(raw_profiles, fit, Path(output_dir) / "profiles")
    source_min = min(int(r["source_cycle_points"]) for r in all_audit_rows)
    source_max = max(int(r["source_cycle_points"]) for r in all_audit_rows)
    summary = {
        "status": "PASS",
        "profile_count": len(raw_profiles),
        "fit_train_count": sum(p.role == "fit_train" for p in raw_profiles),
        "internal_heldout_count": sum(p.role == "internal_heldout" for p in raw_profiles),
        "validation_report_only_count": sum(p.role == "validation_report_only" for p in raw_profiles),
        "protocols": sorted({p.protocol for p in raw_profiles}),
        "branches": sorted({p.branch_family for p in raw_profiles}),
        "cycle_audit_row_count": len(all_audit_rows),
        "source_cycle_points_min": source_min,
        "source_cycle_points_max": source_max,
        "micro_points_per_cycle": micro_points_per_cycle,
        "micro_smoke_is_downsampled_view": any(bool(r["micro_smoke_downsampled"]) for r in all_audit_rows),
        "preflight_source_counts_are_not_downsampled": all(not bool(r["preflight_downsampled"]) for r in all_audit_rows),
        "teacher_initial_cbar_anchor_used": True,
        "formal_s2_training_eligible": False,
        "physical_fit": fit.as_dict(),
        "manifest_rows": manifest_rows,
    }
    dump_json(summary, Path(output_dir) / "D18_S2_MICRO_CASEPACK_BUILD_SUMMARY.json")
    return summary
