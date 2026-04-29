#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASSB effective-SPM parameter builder with explicit soft-label provenance.

Replacement location:
    util/spm_assb_train_discharge.py

Main change in this version:
    ASSB_SOFT_LABEL_DIR and an explicit soft-label summary take precedence over
    stale config.json/train_summary_json paths. The actual soft-label directory,
    actual soft_label_summary.json and actual current-profile source are written
    into params so ModelFin_*/config.json and evaluators can diagnose mismatches.

The physical model conventions are unchanged:
    J_a(t) = -I(t) * R_a / (3 * eps_a * F * V_a)
    J_c(t) =  I(t) * R_c / (3 * eps_c * F * V_c)
"""
from __future__ import annotations

import importlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

print("INFO: USING ASSB PROVENANCE-AWARE TRAINING SPM PRIOR")


def _infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "util" else here.parent


ROOT = _infer_repo_root()
DEFAULT_SOFT_LABEL_DIR_CYCLE5 = ROOT / "Data" / "assb_soft_labels_cycle5_v3"
DEFAULT_SOFT_LABEL_DIR_CYCLES5PLUS = ROOT / "Data" / "assb_soft_labels_cycles5plus_v3"
DEFAULT_SUMMARY_JSON = DEFAULT_SOFT_LABEL_DIR_CYCLE5 / "soft_label_summary.json"
DEFAULT_AREA_M2 = np.float64(7.853981633974483e-05)  # 10 mm diameter area
DEFAULT_TMAX_S = np.float64(3600.0)
DEFAULT_I_A = np.float64(-3.3e-4)
DEFAULT_PROFILE_CYCLE_FROM = 5
ASSB_T_K = np.float64(303.15)
ASSB_CE_CODE = np.float64(1.2)
ASSB_THETA_A0_FULL = np.float64(0.55)
ASSB_THETA_C0_FULL = np.float64(0.27)
ASSB_THETA_A0_DISCHARGED = np.float64(0.45)
ASSB_THETA_C0_DISCHARGED = np.float64(0.90)


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.upper() in {"NONE", "NULL"}:
        return None
    return Path(text)


def _load_summary(summary_json: Path | None) -> dict:
    if summary_json is None or not Path(summary_json).exists():
        return {}
    try:
        with open(summary_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _parse_time_to_seconds(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except Exception:
        pass
    parts = text.split(":")
    try:
        nums = [float(p) for p in parts]
    except Exception:
        return float("nan")
    if len(nums) == 3:
        h, m, s = nums
        return float(h * 3600.0 + m * 60.0 + s)
    if len(nums) == 2:
        m, s = nums
        return float(m * 60.0 + s)
    return float("nan")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if p is None:
            continue
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _candidate_soft_label_dirs(summary: dict, summary_path: Path | None = None) -> list[Path]:
    """Return candidate soft-label dirs.

    IMPORTANT: environment override comes first. This prevents a stale
    ModelFin_*/config.json train_summary_json from silently forcing old data.
    """
    candidates: list[Path] = []
    env = os.environ.get("ASSB_SOFT_LABEL_DIR")
    if env:
        candidates.append(Path(env))

    if summary_path is not None:
        p = Path(summary_path)
        if p.name == "soft_label_summary.json":
            candidates.append(p.parent)

    for key in ("soft_label_dir", "soft_labels_dir", "assb_soft_label_dir"):
        val = summary.get(key)
        if val:
            candidates.append(Path(str(val)))

    candidates.extend([DEFAULT_SOFT_LABEL_DIR_CYCLE5, DEFAULT_SOFT_LABEL_DIR_CYCLES5PLUS])
    return _dedupe_paths(candidates)


def _resolve_soft_label_context(summary_json: Path | None) -> tuple[Path | None, Path | None, dict, str]:
    original_summary = Path(summary_json) if summary_json is not None else None
    original_data = _load_summary(original_summary)

    for d in _candidate_soft_label_dirs(original_data, original_summary):
        summary_file = d / "soft_label_summary.json"
        if summary_file.exists():
            data = _load_summary(summary_file)
            data.setdefault("soft_label_dir", str(d.resolve()))
            return d.resolve(), summary_file.resolve(), data, "soft_label_dir/soft_label_summary.json"

    if original_summary is not None and original_summary.exists():
        return None, original_summary.resolve(), original_data, "original summary_json"

    return None, None, {}, "fallback defaults"


def _load_thermo(soft_label_dir: Path | None):
    """Load thermo after selecting ASSB_SOFT_LABEL_DIR.

    thermo_assb reads v3 constants from soft_label_summary.json at import time.
    Reloading after the environment is set makes summary provenance explicit.
    """
    if soft_label_dir is not None:
        os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_label_dir)
    try:
        import util.thermo_assb as thermo  # type: ignore
    except Exception:
        import thermo_assb as thermo  # type: ignore
    return importlib.reload(thermo)


def _load_profile_from_summary_arrays(summary: dict):
    profile = summary.get("current_profile") or summary.get("I_profile") or summary.get("I_app_profile")
    if isinstance(profile, dict):
        t = profile.get("t") or profile.get("time") or profile.get("time_s") or profile.get("t_s")
        i = profile.get("I") or profile.get("current") or profile.get("current_A") or profile.get("I_A")
        if t is not None and i is not None:
            t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
            i_arr = np.asarray(i, dtype=np.float64).reshape(-1)
            if t_arr.size >= 2 and t_arr.size == i_arr.size:
                return t_arr, i_arr, "summary.current_profile", None

    t = None
    for key in ("time_profile", "t_profile", "current_time_profile", "I_time_profile"):
        if key in summary:
            t = summary[key]
            break
    i = None
    for key in ("current_profile_A", "I_profile_A", "I_values", "current_values", "I_app_values"):
        if key in summary:
            i = summary[key]
            break
    if t is not None and i is not None:
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
        i_arr = np.asarray(i, dtype=np.float64).reshape(-1)
        if t_arr.size >= 2 and t_arr.size == i_arr.size:
            return t_arr, i_arr, "summary.separate_profile_keys", None
    return None


def _load_profile_from_soft_labels(soft_label_dir: Path | None):
    if soft_label_dir is None:
        return None
    sol = soft_label_dir / "solution.npz"
    if not sol.exists():
        return None
    try:
        data = np.load(sol)
        t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
        i = np.asarray(data["I_profile"], dtype=np.float64).reshape(-1)
        if t.size >= 2 and t.size == i.size:
            v0 = None
            for key in ("voltage_exp", "V_exp", "voltage", "voltage_V"):
                if key in data.files:
                    v = np.asarray(data[key], dtype=np.float64).reshape(-1)
                    valid = np.where(np.isfinite(v))[0]
                    if valid.size:
                        v0 = float(v[valid[0]])
                    break
            return t, i, f"{sol}", v0
    except Exception:
        return None
    return None


def _candidate_record_paths(summary: dict, summary_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    for key in (
        "record_csv",
        "record_csv_path",
        "record_path",
        "current_profile_csv",
        "current_profile_path",
        "assb_record_csv",
        "assb_record_path",
    ):
        value = summary.get(key)
        if value:
            candidates.append(Path(str(value)))
    for env_key in ("ASSB_RECORD_CSV", "ASSB_RECORD_ZIP", "ASSB_CURRENT_PROFILE_CSV"):
        value = os.environ.get(env_key)
        if value:
            candidates.append(Path(value))

    roots = [ROOT, ROOT / "Data"]
    if summary_path is not None:
        sp = Path(summary_path)
        roots.extend([sp.parent, sp.parent.parent])
    for base in roots:
        candidates.extend([
            base / "record_extracted.csv",
            base / "record_extracted.zip",
            base / "record" / "record_extracted.csv",
            base / "records" / "record_extracted.csv",
            base / "Data" / "record_extracted.csv",
            base / "Data" / "record_extracted.zip",
        ])
    candidates.extend([
        Path(r"C:/Users/Tiga_QJW/Desktop/ZHB_realDATA/record_extracted.csv"),
        Path(r"C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/record_extracted.csv"),
        Path(r"C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/record_extracted/record_extracted.csv"),
        Path(r"C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/record_extracted.zip"),
    ])
    return _dedupe_paths(candidates)


def _read_record_dataframe(path: Path):
    try:
        import pandas as pd
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    return None
                csv_name = "record_extracted.csv" if "record_extracted.csv" in [Path(n).name for n in csv_names] else csv_names[0]
                for name in csv_names:
                    if Path(name).name == "record_extracted.csv":
                        csv_name = name
                        break
                with zf.open(csv_name) as f:
                    return pd.read_csv(f)
        return pd.read_csv(path)
    except Exception:
        return None


def _compress_profile(t: np.ndarray, i: np.ndarray, max_points: int = 5000):
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    i = np.asarray(i, dtype=np.float64).reshape(-1)
    valid = np.isfinite(t) & np.isfinite(i)
    t = t[valid]
    i = i[valid]
    if t.size < 2:
        return None
    order = np.argsort(t)
    t = t[order]
    i = i[order]
    unique_t, inverse = np.unique(t, return_inverse=True)
    if unique_t.size != t.size:
        i_last = np.zeros_like(unique_t)
        for idx, val in zip(inverse, i):
            i_last[idx] = val
        t, i = unique_t, i_last
    rounded = np.round(i, decimals=10)
    change_idx = np.where(np.abs(np.diff(rounded)) > 1e-12)[0]
    keep = {0, t.size - 1}
    for idx in change_idx:
        for j in (idx - 2, idx - 1, idx, idx + 1, idx + 2, idx + 3):
            if 0 <= j < t.size:
                keep.add(int(j))
    if t.size > max_points:
        anchors = np.linspace(0, t.size - 1, max_points, dtype=int)
        keep.update(int(a) for a in anchors)
    else:
        keep.update(range(t.size))
    keep_idx = np.array(sorted(keep), dtype=int)
    return t[keep_idx], i[keep_idx]


def _build_record_current_profile(summary: dict, summary_path: Path | None, soft_label_dir: Path | None):
    # Priority 1: solution.npz inside the actual selected soft-label directory.
    soft = _load_profile_from_soft_labels(soft_label_dir)
    if soft is not None:
        return soft

    # Priority 2: explicit arrays inside summary.
    direct = _load_profile_from_summary_arrays(summary)
    if direct is not None:
        return direct

    # Priority 3: record_extracted.csv fallback.
    df = None
    source = None
    for path in _candidate_record_paths(summary, summary_path):
        df = _read_record_dataframe(path)
        if df is not None:
            source = str(path)
            break
    if df is None:
        return None

    cycle_col = "循环号" if "循环号" in df.columns else None
    time_col = "总时间" if "总时间" in df.columns else ("时间" if "时间" in df.columns else None)
    current_col = "电流(A)" if "电流(A)" in df.columns else None
    voltage_col = "电压(V)" if "电压(V)" in df.columns else None
    if cycle_col is None or time_col is None or current_col is None:
        return None

    cycle_from = int(summary.get("cycle_from", summary.get("current_profile_cycle_from", DEFAULT_PROFILE_CYCLE_FROM)))
    cycle_to = summary.get("cycle_to", summary.get("current_profile_cycle_to", None))
    merge = _parse_bool(summary.get("merge_cycles", os.environ.get("ASSB_MERGE_CYCLES")), True)
    df = df.copy()
    cycles = df[cycle_col].astype(int)
    if merge:
        chosen = df[cycles >= cycle_from].copy()
        source_note = f"{source} | cycles >= {cycle_from}"
        if cycle_to is not None:
            try:
                cycle_to_i = int(cycle_to)
                chosen = chosen[chosen[cycle_col].astype(int) <= cycle_to_i].copy()
                source_note = f"{source} | cycles {cycle_from}-{cycle_to_i}"
            except Exception:
                pass
    else:
        cycle_value = int(summary.get("cycle", summary.get("current_profile_cycle", cycle_from)))
        chosen = df[cycles == cycle_value].copy()
        source_note = f"{source} | cycle {cycle_value}"
    if chosen.empty:
        return None

    t_abs = np.array([_parse_time_to_seconds(v) for v in chosen[time_col].to_numpy()], dtype=np.float64)
    i_arr = chosen[current_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(t_abs) & np.isfinite(i_arr)
    if np.count_nonzero(valid) < 2:
        return None
    t_abs = t_abs[valid]
    i_arr = i_arr[valid]
    t_arr = t_abs - t_abs[0]
    compressed = _compress_profile(t_arr, i_arr)
    if compressed is None:
        return None
    t_arr, i_arr = compressed
    v0 = None
    if voltage_col is not None:
        try:
            v_values = chosen[voltage_col].to_numpy(dtype=np.float64)
            idx = np.where(np.isfinite(v_values))[0]
            if idx.size:
                v0 = float(v_values[idx[0]])
        except Exception:
            pass
    return t_arr, i_arr, source_note, v0


def _profile_capacity_ah(t_s: np.ndarray, i_a: np.ndarray) -> np.float64:
    t_s = np.asarray(t_s, dtype=np.float64).reshape(-1)
    i_a = np.asarray(i_a, dtype=np.float64).reshape(-1)
    if t_s.size < 2 or t_s.size != i_a.size:
        return np.float64(1.0)
    i_pos = np.clip(i_a, 0.0, None)
    i_neg = np.clip(-i_a, 0.0, None)
    q_chg = float(np.trapezoid(i_pos, t_s) / 3600.0)
    q_dis = float(np.trapezoid(i_neg, t_s) / 3600.0)
    vals = [v for v in (q_chg, q_dis) if np.isfinite(v) and v > 0.0]
    return np.float64(np.mean(vals)) if vals else np.float64(1.0)


def _current_ref_from_profile(profile, fallback=DEFAULT_I_A):
    if profile is None:
        return np.float64(fallback)
    i = np.asarray(profile[1], dtype=np.float64).reshape(-1)
    nz = i[np.abs(i) > 1.0e-12]
    if nz.size:
        mag = float(np.median(np.abs(nz)))
        sign = 1.0 if float(nz[0]) >= 0.0 else -1.0
        return np.float64(sign * mag)
    return np.float64(fallback)


def _infer_capacity_ah(summary: dict, tmax_s, current_ref_A, profile=None):
    for key in ("capacity_Ah", "C_Ah", "capacity_ah", "C"):
        if key in summary:
            try:
                val = float(summary[key])
                if np.isfinite(val) and val > 0:
                    return np.float64(val)
            except Exception:
                pass
    if profile is not None:
        q = _profile_capacity_ah(profile[0], profile[1])
        if np.isfinite(q) and q > 0:
            return q
    return np.float64(abs(float(current_ref_A)) * float(tmax_s) / 3600.0) if abs(float(current_ref_A)) > 0 else np.float64(1.0)


def _first_nonzero_current(profile):
    if profile is None:
        return float(DEFAULT_I_A)
    i = np.asarray(profile[1], dtype=np.float64).reshape(-1)
    nz = i[np.abs(i) > 1.0e-10]
    if nz.size:
        return float(nz[0])
    return 0.0


def _initial_thetas(summary: dict, current_profile=None, v0=None):
    theta_a_override = summary.get("theta_a0", summary.get("theta_n0", None))
    theta_c_override = summary.get("theta_c0", summary.get("theta_p0", None))
    if theta_a_override is not None and theta_c_override is not None:
        try:
            return np.float64(theta_a_override), np.float64(theta_c_override), "soft-label summary override"
        except Exception:
            pass
    first_i = _first_nonzero_current(current_profile)
    if first_i > 0 or (v0 is not None and v0 < 2.5):
        return ASSB_THETA_A0_DISCHARGED, ASSB_THETA_C0_DISCHARGED, "charge-first cycle initial state"
    return ASSB_THETA_A0_FULL, ASSB_THETA_C0_FULL, "discharge/full initial state"


def _summary_float(summary: dict, keys: tuple[str, ...], default: float) -> np.float64:
    for key in keys:
        if key in summary:
            try:
                val = float(summary[key])
                if np.isfinite(val):
                    return np.float64(val)
            except Exception:
                pass
    return np.float64(default)


def _to_jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def makeParams(summary_json=None):
    soft_label_dir, actual_summary_path, summary, summary_source = _resolve_soft_label_context(
        Path(summary_json) if summary_json is not None else DEFAULT_SUMMARY_JSON
    )
    thermo = _load_thermo(soft_label_dir)

    record_profile_info = _build_record_current_profile(summary, actual_summary_path, soft_label_dir)
    current_profile = None
    profile_source = None
    profile_v0 = None
    if record_profile_info is not None:
        t_profile, i_profile, profile_source, profile_v0 = record_profile_info
        current_profile = (np.asarray(t_profile, dtype=np.float64), np.asarray(i_profile, dtype=np.float64))

    current_ref_A = np.float64(summary.get("current_ref_A", _current_ref_from_profile(current_profile, DEFAULT_I_A)))
    if current_profile is not None and "tmax_train_s" not in summary:
        tmax_s = np.float64(np.max(current_profile[0]))
    else:
        tmax_s = np.float64(summary.get("tmax_train_s", DEFAULT_TMAX_S))
    capacity_ah = _infer_capacity_ah(summary, tmax_s=tmax_s, current_ref_A=current_ref_A, profile=current_profile)
    theta_a0, theta_c0, theta_source = _initial_thetas(summary, current_profile, profile_v0)
    csanmax_eff = _summary_float(summary, ("csanmax", "csanmax_eff"), float(getattr(thermo, "CSANMAX_EFF", 6.0)))
    cscamax_eff = _summary_float(summary, ("cscamax", "cscamax_eff", "cscmax"), 51.8)

    class Degradation:
        def __init__(self):
            self.n_params = 2
            self.ind_i0_a = 0
            self.ind_ds_c = 1
            self.bounds = [[] for _ in range(self.n_params)]
            self.ref_vals = [0 for _ in range(self.n_params)]
            self.eff = np.float64(0.0)
            self.bounds[self.ind_i0_a] = [np.float64(0.5), np.float64(4.0)]
            self.bounds[self.ind_ds_c] = [np.float64(0.5), np.float64(40.0)]
            self.ref_vals[self.ind_i0_a] = np.float64(1.0)
            self.ref_vals[self.ind_ds_c] = np.float64(1.0)

    class Macroscopic:
        def __init__(self):
            self.F = np.float64(96485.3321e3)  # C / kmol
            self.R = np.float64(8.3145e3)      # J / (kmol K)
            self.T = ASSB_T_K
            self.T_const = ASSB_T_K
            self.T_ref = ASSB_T_K
            self.C = np.float64(capacity_ah)
            self.tmin = np.float64(0.0)
            self.tmax = np.float64(tmax_s)
            self.rmin = np.float64(0.0)
            self.I = np.float64(current_ref_A)

    class Anode:
        def __init__(self):
            self.thickness = np.float64(100e-6)
            self.solids = self.Anode_solids()
            self.A = DEFAULT_AREA_M2
            self.alpha = np.float64(0.5)
            self.D50 = np.float64(100e-6)  # Rs_a = 50 um equivalent Li-In diffusion length
            self.csmax = np.float64(csanmax_eff)
            self.uocp = thermo.uocp_a_fun
            self.i0 = thermo.i0_a_degradation_param_fun
            self.ds = thermo.ds_a_fun

        class Anode_solids:
            def __init__(self):
                self.eps = np.float64(0.95)

    class Cathode:
        def __init__(self):
            self.thickness = np.float64(16e-6)
            self.A = DEFAULT_AREA_M2
            self.solids = self.Cathode_solids()
            self.alpha = np.float64(0.5)
            self.D50 = np.float64(3.6e-6)  # Rs_c = 1.8 um
            self.csmax = np.float64(cscamax_eff)
            self.uocp = thermo.uocp_c_fun
            self.i0 = thermo.i0_c_fun
            self.ds = thermo.ds_c_degradation_param_fun

        class Cathode_solids:
            def __init__(self):
                self.eps = np.float64(0.55)

    deg = Degradation()
    bat = Macroscopic()
    an = Anode()
    ca = Cathode()

    class IC:
        def __init__(self):
            self.an = self.Anode_IC()
            self.ca = self.Cathode_IC(self.an.cs)
            self.ce = ASSB_CE_CODE
            self.phie = -float(an.uocp(self.an.cs, an.csmax))
            self.phis_c = float(ca.uocp(self.ca.cs, ca.csmax) - an.uocp(self.an.cs, an.csmax))

        class Anode_IC:
            def __init__(self):
                self.ce = ASSB_CE_CODE
                self.cs = np.float64(theta_a0 * an.csmax)
                self.phis = np.float64(0.0)

        class Cathode_IC:
            def __init__(self, cs_a0):
                self.ce = ASSB_CE_CODE
                self.cs = np.float64(theta_c0 * ca.csmax)
                self.phis = float(ca.uocp(self.cs, ca.csmax) - an.uocp(cs_a0, an.csmax))

    ic = IC()
    params = thermo.setParams({}, deg, bat, an, ca, ic)

    # Explicitly override v3 constants from the selected soft-label summary so the
    # saved config reflects the actual data source, even if thermo_assb was already
    # imported elsewhere in the process.
    params["csanmax"] = np.float64(csanmax_eff)
    params["cscamax"] = np.float64(cscamax_eff)
    for key, aliases, default in [
        ("R_ohm_eff", ("R_ohm_eff_v3", "R_ohm_eff"), params.get("R_ohm_eff", 105.0)),
        ("voltage_alignment_offset_V", ("voltage_alignment_offset_V", "voltage_offset_V"), params.get("voltage_alignment_offset_V", -0.11588681607942332)),
        ("theta_c_bottom", ("theta_c_bottom_v3", "theta_c_bottom"), params.get("theta_c_bottom", 0.834)),
        ("theta_c_top", ("theta_c_top_v3", "theta_c_top"), params.get("theta_c_top", 0.432)),
    ]:
        params[key] = _summary_float(summary, aliases, float(default))

    # Provenance keys used by training config, evaluator and checks.
    params["train_summary_json"] = str(actual_summary_path) if actual_summary_path is not None else "NONE"
    params["actual_train_summary_json"] = params["train_summary_json"]
    params["soft_label_summary_json"] = params["train_summary_json"]
    params["soft_label_dir"] = str(soft_label_dir) if soft_label_dir is not None else "NONE"
    params["actual_soft_label_dir"] = params["soft_label_dir"]
    params["summary_source"] = summary_source
    params["theta_a0"] = np.float64(theta_a0)
    params["theta_c0"] = np.float64(theta_c0)
    params["initial_state_source"] = theta_source
    params["provenance_summary_subset"] = _to_jsonable({k: summary.get(k) for k in [
        "cycle", "cycle_from", "cycle_to", "merge_cycles", "tmax_train_s", "R_ohm_eff",
        "voltage_alignment_offset_V", "theta_c_bottom", "theta_c_top", "csanmax", "cscamax"
    ] if k in summary})

    if current_profile is not None:
        t_profile, i_profile = current_profile
        params["current_profile"] = (t_profile, i_profile)
        params["time_profile"] = t_profile
        params["current_profile_A"] = i_profile
        params["current_profile_source"] = profile_source
        params["actual_current_profile_source"] = profile_source
        params["I_app"] = np.float64(current_ref_A)
    else:
        params["current_profile_source"] = "constant-current fallback"
        params["actual_current_profile_source"] = "constant-current fallback"

    print(
        f"INFO: ASSB actual summary = {params['train_summary_json']} | "
        f"soft_label_dir = {params['soft_label_dir']} | "
        f"tmax = {float(tmax_s):.6g} s | I_ref = {float(current_ref_A):.6g} A | "
        f"C = {float(capacity_ah):.6g} Ah | T = {float(ASSB_T_K):.2f} K | "
        f"csanmax = {float(csanmax_eff):.3g} | theta_a0 = {float(theta_a0):.3f} | "
        f"theta_c0 = {float(theta_c0):.3f} | IC = {theta_source}"
    )
    if current_profile is not None:
        print(
            f"INFO: ASSB current profile = {profile_source} | points = {len(current_profile[0])} | "
            f"I_min = {float(np.min(current_profile[1])):.6g} A | I_max = {float(np.max(current_profile[1])):.6g} A"
        )
    else:
        print("INFO: No ASSB soft-label/current profile found; using constant-current fallback.")

    return params
