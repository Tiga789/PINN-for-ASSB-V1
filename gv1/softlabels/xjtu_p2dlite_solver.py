# -*- coding: utf-8 -*-
"""P2Dlite low-r solid diffusion and soft-label construction.

D14-P4A changes
---------------
1. Replay metadata (`batch`, `protocol`, `cell_uid`) is inferred from source
   path/parent directory when older replay-profile NPZ files do not contain
   scalar metadata fields.
2. `phis_c_soft` is generated through the bounded D14-P4A voltage residual
   closure, and the raw unbounded value is saved as `phis_c_soft_raw`.
3. `voltage_bound_correction = phis_c_soft - phis_c_soft_raw` is saved so the
   audit can quantify whether the voltage label was clipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import re

from .xjtu_lowr_grid import build_diffusion_matrix, shell_average
from .xjtu_p2dlite_voltage import voltage_closure, apply_s1k_style_residual


def _clean_scalar_string(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        s = str(value)
    except Exception:
        s = str(value)
    s = s.strip()
    if s in {"", "None", "none", "nan", "NaN", "[]"}:
        return default
    return s


def infer_xjtu_metadata_from_path(path: str | Path) -> Dict[str, str]:
    """Infer batch/protocol/cell_uid for old replay profiles with missing metadata.

    Batch-1/3/4 replay profiles often have parent names like
    `0003_battery-3_2C_battery-3` and no scalar `batch`/`protocol` NPZ fields.
    Batch-5/6 P3B profiles usually include `Batch-5_battery-7` or similar.
    """
    p = Path(path)
    text = str(p).replace("\\", "/")
    parent = p.parent.name
    lower = text.lower()

    batch = ""
    protocol = ""

    m = re.search(r"batch[-_ ]?([1-6])", text, flags=re.I)
    if m:
        batch = f"Batch-{int(m.group(1))}"

    # Protocol names used in this project.
    if re.search(r"(^|[_/\-])2c($|[_/\-])", lower):
        protocol = "2C"
        if not batch:
            batch = "Batch-1"
    elif re.search(r"r2[._-]?5", lower):
        protocol = "R2.5"
        if not batch:
            batch = "Batch-3"
    elif re.search(r"(^|[_/\-])r3($|[_/\-])", lower):
        protocol = "R3"
        if not batch:
            batch = "Batch-4"
    elif "random" in lower or "batch-5" in lower or "batch_5" in lower:
        protocol = "random_walk"
        if not batch:
            batch = "Batch-5"
    elif "geo" in lower or "batch-6" in lower or "batch_6" in lower:
        protocol = "GEO"
        if not batch:
            batch = "Batch-6"

    if not protocol:
        protocol = {"Batch-1": "2C", "Batch-3": "R2.5", "Batch-4": "R3", "Batch-5": "random_walk", "Batch-6": "GEO"}.get(batch, "")

    cell_uid = parent
    if parent in {"profiles", "0000", ""}:
        cell_uid = p.stem
    return {"batch": batch, "protocol": protocol, "cell_uid": cell_uid}


def initial_thetas_from_voltage(v0: float, resolved: Dict[str, Any]) -> tuple[float, float]:
    vmin = float(resolved["cell"]["voltage_min_V"])
    vmax = float(resolved["cell"]["voltage_max_V"])
    soc = (float(v0) - vmin) / max(vmax - vmin, 1e-9)
    soc = float(np.clip(soc, 0.0, 1.0))

    p = resolved["solid_phase"]["positive"]
    n = resolved["solid_phase"]["negative"]
    theta_p = p["theta_max"] - soc * (p["theta_max"] - p["theta_min"])
    theta_n = n["theta_min"] + soc * (n["theta_max"] - n["theta_min"])
    return float(theta_n), float(theta_p)


def _downsample_indices(t, v, step_id, max_points: int, preserve_low_voltage: bool = True, low_thr: float = 3.05):
    N = len(t)
    if N <= max_points:
        return np.arange(N, dtype=int)
    base_stride = int(np.ceil(N / max_points))
    idx = set(range(0, N, base_stride))
    idx.add(0)
    idx.add(N - 1)

    if preserve_low_voltage:
        low = np.where(np.asarray(v) <= low_thr)[0]
        if low.size:
            low_stride = max(1, int(np.ceil(low.size / max(1, max_points // 4))))
            idx.update(low[::low_stride].tolist())

    if step_id is not None and len(step_id) == N:
        step = np.asarray(step_id)
        trans = np.where(step[1:] != step[:-1])[0] + 1
        for k in trans:
            for j in range(max(0, k - 2), min(N, k + 3)):
                idx.add(j)

    arr = np.array(sorted(idx), dtype=int)
    if arr.size > max_points:
        keep = np.linspace(0, arr.size - 1, max_points).round().astype(int)
        arr = arr[keep]
    return np.unique(arr)


def load_profile_npz(path: str | Path, max_points: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    required = ["t_global_s", "I_profile", "voltage_exp", "cycle_id", "step_id"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"Replay profile missing required key {k}: {path}")

    t = np.asarray(data["t_global_s"], dtype=float)
    I = np.asarray(data["I_profile"], dtype=float)
    V = np.asarray(data["voltage_exp"], dtype=float)
    T = np.asarray(data["temperature_C"], dtype=float) if "temperature_C" in data.files else np.full_like(t, np.nan)
    cycle = np.asarray(data["cycle_id"])
    step = np.asarray(data["step_id"])
    stype = np.asarray(data["step_type"]) if "step_type" in data.files else np.array(["unknown"] * len(t), dtype="<U16")

    N = min(len(t), len(I), len(V), len(T), len(cycle), len(step), len(stype))
    t, I, V, T, cycle, step, stype = t[:N], I[:N], V[:N], T[:N], cycle[:N], step[:N], stype[:N]

    finite = np.isfinite(t) & np.isfinite(I) & np.isfinite(V)
    t, I, V, T, cycle, step, stype = t[finite], I[finite], V[finite], T[finite], cycle[finite], step[finite], stype[finite]

    gen_cfg = cfg.get("generator", {})
    idx = _downsample_indices(
        t, V, step,
        int(max_points),
        preserve_low_voltage=bool(gen_cfg.get("preserve_low_voltage", True)),
        low_thr=float(gen_cfg.get("low_voltage_threshold_V", 3.05)),
    )

    t, I, V, T, cycle, step, stype = t[idx], I[idx], V[idx], T[idx], cycle[idx], step[idx], stype[idx]
    t = t - t[0]
    if len(t) >= 2 and np.any(np.diff(t) < 0):
        t = np.arange(len(t), dtype=float)

    def _scalar_string(key, default=""):
        if key in data.files:
            return _clean_scalar_string(data[key], default)
        return default

    inferred = infer_xjtu_metadata_from_path(path)
    batch = _scalar_string("batch", inferred["batch"])
    protocol = _scalar_string("protocol", inferred["protocol"])
    cell_uid = _scalar_string("cell_uid", inferred["cell_uid"])

    # Fill any partially missing scalar.
    if not batch:
        batch = inferred["batch"]
    if not protocol:
        protocol = inferred["protocol"]
    if not cell_uid:
        cell_uid = inferred["cell_uid"]

    return {
        "t_global_s": t,
        "I_profile": I,
        "voltage_exp": V,
        "temperature_C": T,
        "cycle_id": cycle.astype(np.int32, copy=False),
        "step_id": step.astype(np.int32, copy=False),
        "step_type": stype.astype("<U32"),
        "source_profile_npz": str(path),
        "batch": batch,
        "protocol": protocol,
        "cell_uid": cell_uid,
        "metadata_inferred_from_path": bool(
            ("batch" not in data.files or not _clean_scalar_string(data["batch"], "")) or
            ("protocol" not in data.files or not _clean_scalar_string(data["protocol"], ""))
        ),
    }


def _solve_electrode(t, I, resolved, side: str):
    assert side in ("negative", "positive")
    F = float(resolved["constants"]["F_C_per_mol"])
    Aeff = float(resolved["geometry"]["effective_area_m2"])
    geom = resolved["geometry"][side]
    solid = resolved["solid_phase"][side]
    R_particle = float(geom["R_particle_m"])
    L = float(geom["L_m"])
    eps = float(geom["eps_s"])
    cmax = float(solid["cmax_mol_m3"])
    D = float(solid["D_m2_s"])
    n_r = int(resolved["n_r"])

    A_mtx, b_flux, grid = build_diffusion_matrix(R_particle, n_r, D)
    eye = np.eye(n_r)
    return A_mtx, b_flux, grid, eye, F, Aeff, L, eps, R_particle, cmax


def generate_softlabels(profile: Dict[str, Any], resolved: Dict[str, Any]) -> Dict[str, Any]:
    t = profile["t_global_s"].astype(float)
    I = profile["I_profile"].astype(float)
    V = profile["voltage_exp"].astype(float)
    temp_C = profile["temperature_C"].astype(float)
    T_K = np.where(np.isfinite(temp_C), temp_C + 273.15, float(resolved["cell"]["fallback_temperature_K"]))

    N = len(t)
    n_r = int(resolved["n_r"])

    theta_n0, theta_p0 = initial_thetas_from_voltage(float(V[0]), resolved)

    A_n, b_n, grid_n, eye_n, F, Aeff, L_n, eps_n, Rn, cmax_n = _solve_electrode(t, I, resolved, "negative")
    A_p, b_p, grid_p, eye_p, _, _, L_p, eps_p, Rp, cmax_p = _solve_electrode(t, I, resolved, "positive")

    c_n = np.full(n_r, theta_n0 * cmax_n, dtype=float)
    c_p = np.full(n_r, theta_p0 * cmax_p, dtype=float)

    cs_n = np.zeros((N, n_r), dtype=np.float32)
    cs_p = np.zeros((N, n_r), dtype=np.float32)
    j_n = np.zeros(N, dtype=np.float32)
    j_p = np.zeros(N, dtype=np.float32)

    cache_n = {}
    cache_p = {}

    cs_n[0] = c_n.astype(np.float32)
    cs_p[0] = c_p.astype(np.float32)

    p_bounds = resolved["solid_phase"]["positive"]
    n_bounds = resolved["solid_phase"]["negative"]

    for k in range(1, N):
        dt = float(t[k] - t[k - 1])
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0

        Ik = float(I[k - 1])
        Jn = -Ik * Rn / max(3.0 * eps_n * F * Aeff * L_n, 1e-30)
        Jp = +Ik * Rp / max(3.0 * eps_p * F * Aeff * L_p, 1e-30)
        j_n[k] = Jn
        j_p[k] = Jp

        key = round(dt, 6)
        if key not in cache_n:
            cache_n[key] = np.linalg.inv(eye_n - dt * A_n)
        if key not in cache_p:
            cache_p[key] = np.linalg.inv(eye_p - dt * A_p)

        c_n = cache_n[key] @ (c_n + dt * b_n * Jn)
        c_p = cache_p[key] @ (c_p + dt * b_p * Jp)

        c_n = np.clip(c_n, n_bounds["theta_min"] * cmax_n, n_bounds["theta_max"] * cmax_n)
        c_p = np.clip(c_p, p_bounds["theta_min"] * cmax_p, p_bounds["theta_max"] * cmax_p)

        cs_n[k] = c_n.astype(np.float32)
        cs_p[k] = c_p.astype(np.float32)

    theta_n = cs_n.astype(float) / cmax_n
    theta_p = cs_p.astype(float) / cmax_p
    cbar_n = shell_average(cs_n.astype(float), grid_n["volumes"])
    cbar_p = shell_average(cs_p.astype(float), grid_p["volumes"])

    theta_n_surf = theta_n[:, -1]
    theta_p_surf = theta_p[:, -1]

    V_base, phie, U_n, U_p, eta_n, eta_p = voltage_closure(
        theta_n_surf, theta_p_surf, j_n.astype(float), j_p.astype(float), I, T_K, resolved
    )
    V_soft, residual, weight, V_soft_raw = apply_s1k_style_residual(V_base, V, resolved, return_raw=True)
    voltage_bound_correction = V_soft - V_soft_raw

    out = {
        "t_global_s": t.astype(np.float32),
        "I_profile": I.astype(np.float32),
        "voltage_exp": V.astype(np.float32),
        "temperature_C": temp_C.astype(np.float32),
        "cycle_id": profile["cycle_id"].astype(np.int32),
        "step_id": profile["step_id"].astype(np.int32),
        "step_type": profile["step_type"].astype("<U32"),
        "r_a": grid_n["centers"].astype(np.float32),
        "r_c": grid_p["centers"].astype(np.float32),
        "cs_a": cs_n.astype(np.float32),
        "cs_c": cs_p.astype(np.float32),
        "theta_a": theta_n.astype(np.float32),
        "theta_c": theta_p.astype(np.float32),
        "phie": phie.astype(np.float32),
        "phis_c": V_soft.astype(np.float32),
        "phis_c_base": V_base.astype(np.float32),
        "phis_c_soft": V_soft.astype(np.float32),
        "phis_c_soft_raw": V_soft_raw.astype(np.float32),
        "voltage_bound_correction": voltage_bound_correction.astype(np.float32),
        "cbar_a": cbar_n.astype(np.float32),
        "cbar_c": cbar_p.astype(np.float32),
        "cs_a_surface": cs_n[:, -1].astype(np.float32),
        "cs_c_surface": cs_p[:, -1].astype(np.float32),
        "Uocp_a": U_n.astype(np.float32),
        "Uocp_c": U_p.astype(np.float32),
        "eta_a": eta_n.astype(np.float32),
        "eta_c": eta_p.astype(np.float32),
        "j_a": j_n.astype(np.float32),
        "j_c": j_p.astype(np.float32),
        "voltage_residual_s1k": residual.astype(np.float32),
        "low_transition_weight": weight.astype(np.float32),
        "cell_uid": str(profile.get("cell_uid", "")),
        "batch": str(profile.get("batch", "")),
        "protocol": str(profile.get("protocol", "")),
        "metadata_inferred_from_path": str(profile.get("metadata_inferred_from_path", False)),
        "source_profile_npz": str(profile.get("source_profile_npz", "")),
        "resolved_spec_hash": str(resolved["prior_hash"]),
    }
    return out
