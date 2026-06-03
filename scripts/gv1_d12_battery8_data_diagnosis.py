#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
D12 battery-8 data diagnosis for GV1 / XJTU Batch-1 2C.

This script does NOT train a model. It reads existing replay-profile NPZ files and
optional cycle/SOH CSV caches, compares Batch-1_2C_battery-8 against its 2C peers,
and writes a diagnostic report focused on:
  - data-acquisition anomalies,
  - battery/profile behavior outliers,
  - model-boundary clues for effective-SPM voltage inversion.

Expected cache layout:
  <CacheRoot>/xjtu_batch134_replay_profiles/.../solution_replay_profile.npz

The script is intentionally self-contained and conservative: it never modifies
existing GV1/ASSB mainline files.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("pandas is required for this diagnostic script. Install pandas first.") from exc


PROTOCOL_TO_BATCH = {
    "2C": "Batch-1",
    "R2.5": "Batch-3",
    "R2p5": "Batch-3",
    "R3": "Batch-4",
}


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def canonical_protocol(p: str) -> str:
    p = str(p).strip()
    return "R2.5" if p in {"R2p5", "R2_5", "R25", "R2.5"} else p


def protocol_alias(p: str) -> str:
    p = canonical_protocol(p)
    return "R2p5" if p == "R2.5" else p


@dataclass
class ProfileRef:
    canonical: str
    batch: str
    protocol: str
    battery: str
    npz_path: str
    aliases: List[str]


def parse_profile_from_path(npz_path: Path) -> Optional[ProfileRef]:
    parent = npz_path.parent.name

    # Direct canonical form: Batch-1_2C_battery-8
    m = re.search(r"(Batch-\d+)_(2C|R2\.5|R2p5|R3)_(battery-\d+)", parent)
    if m:
        batch = m.group(1)
        protocol = canonical_protocol(m.group(2))
        battery = m.group(3)
        canonical = f"{batch}_{protocol}_{battery}"
        aliases = sorted(set([
            canonical,
            f"{batch}_{protocol_alias(protocol)}_{battery}",
            parent,
        ]))
        return ProfileRef(canonical, batch, protocol, battery, str(npz_path), aliases)

    # Actual generated profile form: 0008_battery-8_2C_battery-8
    m = re.search(r"(?:^\d+_)?(battery-\d+)_(2C|R2\.5|R2p5|R3)_(battery-\d+)$", parent)
    if m:
        # The first and last battery are normally identical. Use the last one.
        protocol = canonical_protocol(m.group(2))
        battery = m.group(3)
        batch = PROTOCOL_TO_BATCH.get(protocol, "Batch-unknown")
        canonical = f"{batch}_{protocol}_{battery}"
        aliases = sorted(set([
            canonical,
            f"{batch}_{protocol_alias(protocol)}_{battery}",
            parent,
            parent.replace("R2.5", "R2p5"),
        ]))
        return ProfileRef(canonical, batch, protocol, battery, str(npz_path), aliases)

    # Fallback: inspect parent string for protocol/battery
    if "2C" in parent and "battery-" in parent:
        bm = re.findall(r"battery-\d+", parent)
        if bm:
            battery = bm[-1]
            batch, protocol = "Batch-1", "2C"
            canonical = f"{batch}_{protocol}_{battery}"
            return ProfileRef(canonical, batch, protocol, battery, str(npz_path), [canonical, parent])
    return None


def build_profile_index(profiles_root: Path) -> Dict[str, ProfileRef]:
    if not profiles_root.exists():
        raise FileNotFoundError(f"ProfilesRoot does not exist: {profiles_root}")
    refs: List[ProfileRef] = []
    for npz_path in profiles_root.rglob("solution_replay_profile.npz"):
        ref = parse_profile_from_path(npz_path)
        if ref is not None:
            refs.append(ref)
    if not refs:
        raise FileNotFoundError(f"No solution_replay_profile.npz found under {profiles_root}")

    index: Dict[str, ProfileRef] = {}
    for ref in refs:
        for a in ref.aliases:
            index[a] = ref
    return index


def choose_key(data, candidates: Iterable[str]) -> str:
    keys = set(data.files)
    for k in candidates:
        if k in keys:
            return k
    raise KeyError(f"None of keys {list(candidates)} found. Available keys: {sorted(keys)[:80]}")


def load_profile_arrays(npz_path: Path):
    with np.load(npz_path, allow_pickle=False) as z:
        k_t = choose_key(z, ["t_global_s", "time_s", "t_s", "time"])
        k_i = choose_key(z, ["I_profile", "current_A", "I_A", "current"])
        k_v = choose_key(z, ["voltage_exp", "voltage_V", "voltage", "V"])
        t = np.asarray(z[k_t], dtype=np.float64)
        I = np.asarray(z[k_i], dtype=np.float64)
        V = np.asarray(z[k_v], dtype=np.float64)
        if "temperature_C" in z.files:
            T = np.asarray(z["temperature_C"], dtype=np.float64)
        elif "T_C" in z.files:
            T = np.asarray(z["T_C"], dtype=np.float64)
        else:
            T = np.full_like(t, np.nan, dtype=np.float64)
        if "cycle_id" in z.files:
            cycle_id = np.asarray(z["cycle_id"])
        else:
            cycle_id = np.full_like(t, -1, dtype=np.int64)
        if "step_id" in z.files:
            step_id = np.asarray(z["step_id"])
        else:
            step_id = np.full_like(t, -1, dtype=np.int64)
        if "step_type" in z.files:
            step_type = np.asarray(z["step_type"])
        else:
            step_type = np.array(["unknown"] * len(t), dtype=object)

    n = min(len(t), len(I), len(V), len(T), len(cycle_id), len(step_id))
    return {
        "t": t[:n],
        "I": I[:n],
        "V": V[:n],
        "T": T[:n],
        "cycle_id": cycle_id[:n],
        "step_id": step_id[:n],
        "step_type": step_type[:n],
    }


def interval_quantities(t: np.ndarray, I: np.ndarray, V: np.ndarray, T: np.ndarray):
    if len(t) < 2:
        return {}
    dt = np.diff(t)
    I_mid = 0.5 * (I[:-1] + I[1:])
    V_mid = 0.5 * (V[:-1] + V[1:])
    T_mid = 0.5 * (T[:-1] + T[1:])
    valid_dt = np.isfinite(dt) & (dt > 0)
    return {
        "dt": dt,
        "I_mid": I_mid,
        "V_mid": V_mid,
        "T_mid": T_mid,
        "valid_dt": valid_dt,
    }


def safe_min(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.min(x)) if x.size else float("nan")


def safe_max(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else float("nan")


def safe_mean(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def safe_median(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def summarize_profile(ref: ProfileRef, arrays: Dict[str, np.ndarray], current_eps: float = 1e-6) -> Dict[str, float]:
    t, I, V, T = arrays["t"], arrays["I"], arrays["V"], arrays["T"]
    cycle_id = arrays["cycle_id"]
    q = interval_quantities(t, I, V, T)
    out = {
        "profile": ref.canonical,
        "batch": ref.batch,
        "protocol": ref.protocol,
        "battery": ref.battery,
        "npz_path": ref.npz_path,
        "n_time_points": int(len(t)),
        "duration_s": float(t[-1] - t[0]) if len(t) else float("nan"),
        "time_nonmonotonic_count": int(np.sum(np.diff(t) <= 0)) if len(t) > 1 else 0,
        "cycle_count": int(len(np.unique(cycle_id[cycle_id >= 0]))) if np.issubdtype(cycle_id.dtype, np.number) else int(len(np.unique(cycle_id))),
        "voltage_min_V": safe_min(V),
        "voltage_max_V": safe_max(V),
        "voltage_mean_V": safe_mean(V),
        "voltage_range_V": float(safe_max(V) - safe_min(V)),
        "current_min_A": safe_min(I),
        "current_max_A": safe_max(I),
        "current_abs_max_A": safe_max(np.abs(I)),
        "temperature_min_C": safe_min(T),
        "temperature_max_C": safe_max(T),
        "temperature_mean_C": safe_mean(T),
        "temperature_range_C": float(safe_max(T) - safe_min(T)) if np.any(np.isfinite(T)) else float("nan"),
        "nan_count_time": int(np.sum(~np.isfinite(t))),
        "nan_count_current": int(np.sum(~np.isfinite(I))),
        "nan_count_voltage": int(np.sum(~np.isfinite(V))),
        "nan_count_temperature": int(np.sum(~np.isfinite(T))),
    }
    if len(t) > 1:
        dt = q["dt"]
        valid = q["valid_dt"]
        I_mid, V_mid, T_mid = q["I_mid"], q["V_mid"], q["T_mid"]
        charge = valid & (I_mid > current_eps)
        discharge = valid & (I_mid < -current_eps)
        rest = valid & (np.abs(I_mid) <= current_eps)
        out.update({
            "dt_min_s": safe_min(dt[valid]),
            "dt_median_s": safe_median(dt[valid]),
            "dt_max_s": safe_max(dt[valid]),
            "large_dt_gap_count_gt_60s": int(np.sum(valid & (dt > 60.0))),
            "charge_duration_s": float(np.sum(dt[charge])),
            "discharge_duration_s": float(np.sum(dt[discharge])),
            "rest_duration_s": float(np.sum(dt[rest])),
            "charge_fraction": float(np.sum(dt[charge]) / np.sum(dt[valid])) if np.sum(dt[valid]) > 0 else float("nan"),
            "discharge_fraction": float(np.sum(dt[discharge]) / np.sum(dt[valid])) if np.sum(dt[valid]) > 0 else float("nan"),
            "rest_fraction": float(np.sum(dt[rest]) / np.sum(dt[valid])) if np.sum(dt[valid]) > 0 else float("nan"),
            "q_charge_Ah": float(np.sum(np.maximum(I_mid[valid], 0.0) * dt[valid]) / 3600.0),
            "q_discharge_Ah": float(np.sum(np.maximum(-I_mid[valid], 0.0) * dt[valid]) / 3600.0),
            "energy_charge_Wh": float(np.sum(np.maximum(I_mid[valid], 0.0) * V_mid[valid] * dt[valid]) / 3600.0),
            "energy_discharge_Wh": float(np.sum(np.maximum(-I_mid[valid], 0.0) * V_mid[valid] * dt[valid]) / 3600.0),
            "max_abs_dV_V": safe_max(np.abs(np.diff(V))),
            "max_abs_dI_A": safe_max(np.abs(np.diff(I))),
            "max_abs_dT_C": safe_max(np.abs(np.diff(T))) if np.any(np.isfinite(T)) else float("nan"),
            "voltage_jump_count_gt_80mV": int(np.sum(np.abs(np.diff(V)) > 0.08)),
            "voltage_jump_count_gt_150mV": int(np.sum(np.abs(np.diff(V)) > 0.15)),
            "current_jump_count_gt_0p5A": int(np.sum(np.abs(np.diff(I)) > 0.5)),
            "temperature_jump_count_gt_3C": int(np.sum(np.abs(np.diff(T)) > 3.0)) if np.any(np.isfinite(T)) else 0,
            "low_voltage_point_fraction_le_2p75": float(np.mean(V <= 2.75)),
            "high_voltage_point_fraction_ge_4p10": float(np.mean(V >= 4.10)),
            "voltage_outside_2p4_4p7_count": int(np.sum((V < 2.4) | (V > 4.7))),
        })
    return out


def robust_z_table(df: pd.DataFrame, target_profile: str, peer_profiles: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    target_row = df[df["profile"] == target_profile]
    if target_row.empty:
        return pd.DataFrame()
    target_row = target_row.iloc[0]
    peer_df = df[df["profile"].isin(peer_profiles)]
    for m in metrics:
        if m not in df.columns:
            continue
        vals = pd.to_numeric(peer_df[m], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        target_val = float(pd.to_numeric(pd.Series([target_row[m]]), errors="coerce").iloc[0])
        if vals.size:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            scale = 1.4826 * mad
            robust_z = (target_val - med) / scale if scale > 1e-12 else float("nan")
            peer_min, peer_max = float(np.min(vals)), float(np.max(vals))
        else:
            med = mad = scale = peer_min = peer_max = robust_z = float("nan")
        rows.append({
            "metric": m,
            "target_value": target_val,
            "peer_median_excluding_target": med,
            "peer_mad_excluding_target": mad,
            "peer_min_excluding_target": peer_min,
            "peer_max_excluding_target": peer_max,
            "robust_z_vs_peers": robust_z,
            "abs_robust_z": abs(robust_z) if np.isfinite(robust_z) else float("nan"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_robust_z", ascending=False)
    return out


def target_anomaly_events(ref: ProfileRef, arrays: Dict[str, np.ndarray], max_events: int = 5000) -> pd.DataFrame:
    t, I, V, T = arrays["t"], arrays["I"], arrays["V"], arrays["T"]
    cycle_id, step_id = arrays["cycle_id"], arrays["step_id"]
    if len(t) < 2:
        return pd.DataFrame()
    dt = np.diff(t)
    dV = np.diff(V)
    dI = np.diff(I)
    dT = np.diff(T)
    rows = []
    for idx in np.where((dt <= 0) | (dt > 60.0) | (np.abs(dV) > 0.08) | (np.abs(dI) > 0.5) | (np.abs(dT) > 3.0) | (V[:-1] < 2.4) | (V[:-1] > 4.7))[0][:max_events]:
        tags = []
        if dt[idx] <= 0:
            tags.append("nonpositive_dt")
        if dt[idx] > 60.0:
            tags.append("large_dt_gap")
        if abs(dV[idx]) > 0.08:
            tags.append("voltage_jump_gt_80mV")
        if abs(dV[idx]) > 0.15:
            tags.append("voltage_jump_gt_150mV")
        if abs(dI[idx]) > 0.5:
            tags.append("current_jump_gt_0p5A")
        if np.isfinite(dT[idx]) and abs(dT[idx]) > 3.0:
            tags.append("temperature_jump_gt_3C")
        if V[idx] < 2.4 or V[idx] > 4.7:
            tags.append("voltage_outside_2p4_4p7")
        rows.append({
            "profile": ref.canonical,
            "idx": int(idx),
            "t_s": float(t[idx]),
            "cycle_id": str(cycle_id[idx]),
            "step_id": str(step_id[idx]),
            "dt_s": float(dt[idx]),
            "I_A": float(I[idx]),
            "V_V": float(V[idx]),
            "T_C": float(T[idx]) if np.isfinite(T[idx]) else float("nan"),
            "dI_A": float(dI[idx]),
            "dV_V": float(dV[idx]),
            "dT_C": float(dT[idx]) if np.isfinite(dT[idx]) else float("nan"),
            "tags": ";".join(tags),
        })
    return pd.DataFrame(rows)


def segment_summary(ref: ProfileRef, arrays: Dict[str, np.ndarray], current_eps: float = 1e-6) -> pd.DataFrame:
    t, I, V, T = arrays["t"], arrays["I"], arrays["V"], arrays["T"]
    q = interval_quantities(t, I, V, T)
    if not q:
        return pd.DataFrame()
    dt, I_mid, V_mid, T_mid, valid = q["dt"], q["I_mid"], q["V_mid"], q["T_mid"], q["valid_dt"]

    masks = {
        "charge": I_mid > current_eps,
        "discharge": I_mid < -current_eps,
        "rest": np.abs(I_mid) <= current_eps,
        "low_voltage_le_2p75": V_mid <= 2.75,
        "normal_voltage_2p75_4p10": (V_mid > 2.75) & (V_mid < 4.10),
        "high_voltage_ge_4p10": V_mid >= 4.10,
        "discharge_low_voltage_le_2p75": (I_mid < -current_eps) & (V_mid <= 2.75),
        "discharge_high_voltage_ge_4p10": (I_mid < -current_eps) & (V_mid >= 4.10),
        "charge_high_voltage_ge_4p10": (I_mid > current_eps) & (V_mid >= 4.10),
    }
    rows = []
    total_dt = float(np.sum(dt[valid])) if np.any(valid) else 0.0
    for seg, m in masks.items():
        mask = valid & m
        if not np.any(mask):
            rows.append({
                "profile": ref.canonical,
                "segment": seg,
                "n_intervals": 0,
                "duration_s": 0.0,
                "duration_fraction": 0.0,
                "I_mean_A": float("nan"),
                "I_abs_mean_A": float("nan"),
                "V_mean_V": float("nan"),
                "V_min_V": float("nan"),
                "V_max_V": float("nan"),
                "T_mean_C": float("nan"),
                "q_abs_Ah": 0.0,
            })
        else:
            rows.append({
                "profile": ref.canonical,
                "segment": seg,
                "n_intervals": int(np.sum(mask)),
                "duration_s": float(np.sum(dt[mask])),
                "duration_fraction": float(np.sum(dt[mask]) / total_dt) if total_dt > 0 else float("nan"),
                "I_mean_A": float(np.average(I_mid[mask], weights=dt[mask])),
                "I_abs_mean_A": float(np.average(np.abs(I_mid[mask]), weights=dt[mask])),
                "V_mean_V": float(np.average(V_mid[mask], weights=dt[mask])),
                "V_min_V": safe_min(V_mid[mask]),
                "V_max_V": safe_max(V_mid[mask]),
                "T_mean_C": float(np.average(T_mid[mask], weights=dt[mask])) if np.any(np.isfinite(T_mid[mask])) else float("nan"),
                "q_abs_Ah": float(np.sum(np.abs(I_mid[mask]) * dt[mask]) / 3600.0),
            })
    return pd.DataFrame(rows)


def cycle_summary_for_target(ref: ProfileRef, arrays: Dict[str, np.ndarray], current_eps: float = 1e-6) -> pd.DataFrame:
    t, I, V, T = arrays["t"], arrays["I"], arrays["V"], arrays["T"]
    cycle_id = arrays["cycle_id"]
    if len(t) < 2:
        return pd.DataFrame()
    q = interval_quantities(t, I, V, T)
    dt, I_mid, V_mid, T_mid, valid = q["dt"], q["I_mid"], q["V_mid"], q["T_mid"], q["valid_dt"]
    cy_mid = cycle_id[:-1]
    rows = []
    # limit to numeric cycle ids where possible
    for cy in np.unique(cy_mid):
        mask = valid & (cy_mid == cy)
        if not np.any(mask):
            continue
        charge = mask & (I_mid > current_eps)
        discharge = mask & (I_mid < -current_eps)
        rest = mask & (np.abs(I_mid) <= current_eps)
        rows.append({
            "profile": ref.canonical,
            "cycle_id": str(cy),
            "duration_s": float(np.sum(dt[mask])),
            "charge_duration_s": float(np.sum(dt[charge])),
            "discharge_duration_s": float(np.sum(dt[discharge])),
            "rest_duration_s": float(np.sum(dt[rest])),
            "q_charge_Ah": float(np.sum(np.maximum(I_mid[mask], 0.0) * dt[mask]) / 3600.0),
            "q_discharge_Ah": float(np.sum(np.maximum(-I_mid[mask], 0.0) * dt[mask]) / 3600.0),
            "V_min_V": safe_min(V_mid[mask]),
            "V_max_V": safe_max(V_mid[mask]),
            "V_mean_V": float(np.average(V_mid[mask], weights=dt[mask])),
            "T_max_C": safe_max(T_mid[mask]),
            "T_mean_C": float(np.average(T_mid[mask], weights=dt[mask])) if np.any(np.isfinite(T_mid[mask])) else float("nan"),
            "low_voltage_fraction_le_2p75": float(np.sum(dt[mask & (V_mid <= 2.75)]) / np.sum(dt[mask])),
            "high_voltage_fraction_ge_4p10": float(np.sum(dt[mask & (V_mid >= 4.10)]) / np.sum(dt[mask])),
        })
    return pd.DataFrame(rows)


def make_plots(output_dir: Path, target_ref: ProfileRef, target_arrays: Dict[str, np.ndarray], peer_summaries: pd.DataFrame) -> List[str]:
    plots = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return plots

    t = target_arrays["t"]
    I = target_arrays["I"]
    V = target_arrays["V"]
    T = target_arrays["T"]
    n = len(t)
    if n == 0:
        return plots
    step = max(1, n // 20000)
    idx = np.arange(0, n, step)
    x_h = (t[idx] - t[0]) / 3600.0

    # One file per plot, no subplots, simple defaults.
    for y, ylabel, fname in [
        (V[idx], "Voltage (V)", "battery8_voltage_time.png"),
        (I[idx], "Current (A)", "battery8_current_time.png"),
        (T[idx], "Temperature (C)", "battery8_temperature_time.png"),
    ]:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(x_h, y)
        plt.xlabel("Time from profile start (h)")
        plt.ylabel(ylabel)
        plt.title(f"{target_ref.canonical}: {ylabel} over time")
        plt.tight_layout()
        path = output_dir / fname
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots.append(str(path))

    if not peer_summaries.empty:
        metric = "q_discharge_Ah"
        if metric in peer_summaries.columns:
            fig = plt.figure(figsize=(8, 4))
            labels = peer_summaries["battery"].astype(str).tolist()
            vals = pd.to_numeric(peer_summaries[metric], errors="coerce").tolist()
            plt.bar(labels, vals)
            plt.xlabel("Battery")
            plt.ylabel(metric)
            plt.title("Batch-1 2C peer comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            path = output_dir / "peer_q_discharge_Ah_bar.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plots.append(str(path))
    return plots


def main():
    ap = argparse.ArgumentParser(description="Diagnose XJTU Batch-1 2C battery-8 data/profile abnormality.")
    ap.add_argument("--cache-root", required=True, help="GV1 cache root, e.g. E:\\XJTU battery dataset\\_gv1_cache")
    ap.add_argument("--profiles-root", default=None, help="Replay profiles root. Default: <cache-root>/xjtu_batch134_replay_profiles")
    ap.add_argument("--target-profile", default="Batch-1_2C_battery-8")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--current-eps", type=float, default=1e-6)
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    profiles_root = Path(args.profiles_root) if args.profiles_root else cache_root / "xjtu_batch134_replay_profiles"
    output_dir = Path(args.output_dir) if args.output_dir else cache_root / "xjtu_batch134_d12_battery8_data_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    index = build_profile_index(profiles_root)
    if args.target_profile not in index:
        available = sorted({ref.canonical for ref in index.values()})
        raise KeyError(f"Target profile {args.target_profile!r} not found. Available canonical profiles: {available}")

    target_ref = index[args.target_profile]
    # Peer profiles = same batch/protocol. Use unique canonical refs.
    unique_refs = {}
    for ref in index.values():
        unique_refs[ref.canonical] = ref
    peers = sorted(
        [r for r in unique_refs.values() if r.batch == target_ref.batch and r.protocol == target_ref.protocol],
        key=lambda r: int(re.search(r"(\d+)$", r.battery).group(1)) if re.search(r"(\d+)$", r.battery) else 999,
    )
    peer_profiles_excluding_target = [r.canonical for r in peers if r.canonical != target_ref.canonical]

    profile_rows = []
    target_arrays = None
    for ref in peers:
        arrays = load_profile_arrays(Path(ref.npz_path))
        if ref.canonical == target_ref.canonical:
            target_arrays = arrays
        profile_rows.append(summarize_profile(ref, arrays, current_eps=args.current_eps))
    if target_arrays is None:
        raise RuntimeError("Internal error: target arrays not loaded.")

    profile_df = pd.DataFrame(profile_rows)
    profile_csv = output_dir / "D12_B8_profile_peer_summary.csv"
    profile_df.to_csv(profile_csv, index=False, encoding="utf-8-sig")

    metrics_for_z = [
        "n_time_points", "duration_s", "cycle_count",
        "voltage_min_V", "voltage_max_V", "voltage_range_V",
        "current_abs_max_A", "temperature_max_C", "temperature_range_C",
        "charge_duration_s", "discharge_duration_s", "rest_duration_s",
        "charge_fraction", "discharge_fraction", "rest_fraction",
        "q_charge_Ah", "q_discharge_Ah", "energy_discharge_Wh",
        "dt_median_s", "dt_max_s", "large_dt_gap_count_gt_60s",
        "max_abs_dV_V", "max_abs_dI_A", "max_abs_dT_C",
        "voltage_jump_count_gt_80mV", "voltage_jump_count_gt_150mV",
        "current_jump_count_gt_0p5A", "temperature_jump_count_gt_3C",
        "low_voltage_point_fraction_le_2p75", "high_voltage_point_fraction_ge_4p10",
        "voltage_outside_2p4_4p7_count",
    ]
    robust_df = robust_z_table(profile_df, target_ref.canonical, peer_profiles_excluding_target, metrics_for_z)
    robust_csv = output_dir / "D12_B8_robust_peer_outlier_scores.csv"
    robust_df.to_csv(robust_csv, index=False, encoding="utf-8-sig")

    events_df = target_anomaly_events(target_ref, target_arrays)
    events_csv = output_dir / "D12_B8_target_anomaly_events.csv"
    events_df.to_csv(events_csv, index=False, encoding="utf-8-sig")

    seg_df = segment_summary(target_ref, target_arrays, current_eps=args.current_eps)
    seg_csv = output_dir / "D12_B8_target_segment_summary.csv"
    seg_df.to_csv(seg_csv, index=False, encoding="utf-8-sig")

    cyc_df = cycle_summary_for_target(target_ref, target_arrays, current_eps=args.current_eps)
    cyc_csv = output_dir / "D12_B8_target_cycle_summary.csv"
    cyc_df.to_csv(cyc_csv, index=False, encoding="utf-8-sig")

    # Optional SOH label table join if present.
    soh_join_info = {}
    soh_csv_candidates = [
        cache_root / "xjtu_batch134_soh_labels" / "xjtu_batch134_soh_label_table.csv",
        cache_root / "xjtu_batch134_soh_labels" / "soh_label_table.csv",
    ]
    for sp in soh_csv_candidates:
        if sp.exists():
            try:
                soh_df = pd.read_csv(sp)
                # write simple filtered rows if a likely id column exists
                cols = list(soh_df.columns)
                mask = None
                for c in ["profile", "profile_uid", "cell_uid", "cell_id"]:
                    if c in cols:
                        mask = soh_df[c].astype(str).str.contains("battery-8", case=False, na=False) & soh_df[c].astype(str).str.contains("2C|Batch-1", case=False, regex=True, na=False)
                        break
                if mask is not None:
                    filtered = soh_df[mask]
                    outp = output_dir / "D12_B8_soh_label_rows_if_available.csv"
                    filtered.to_csv(outp, index=False, encoding="utf-8-sig")
                    soh_join_info = {"soh_label_source": str(sp), "filtered_rows": int(len(filtered)), "filtered_output": str(outp)}
                else:
                    soh_join_info = {"soh_label_source": str(sp), "note": "No recognized profile/cell id column for automatic filtering.", "columns": cols}
            except Exception as exc:
                soh_join_info = {"soh_label_source": str(sp), "error": repr(exc)}
            break

    # Interpret flags.
    top_outliers = robust_df.head(12).to_dict(orient="records") if not robust_df.empty else []
    target_summary = profile_df[profile_df["profile"] == target_ref.canonical].iloc[0].to_dict()

    data_flags = []
    behavior_flags = []
    model_boundary_flags = []

    if target_summary.get("time_nonmonotonic_count", 0) > 0:
        data_flags.append("time_nonmonotonic")
    if target_summary.get("nan_count_voltage", 0) > 0 or target_summary.get("nan_count_current", 0) > 0:
        data_flags.append("nan_in_core_channels")
    if target_summary.get("voltage_outside_2p4_4p7_count", 0) > 0:
        data_flags.append("voltage_outside_expected_range_2p4_4p7")
    if target_summary.get("voltage_jump_count_gt_150mV", 0) > 0:
        data_flags.append("large_voltage_jumps_gt_150mV")
    if target_summary.get("large_dt_gap_count_gt_60s", 0) > 0:
        data_flags.append("large_time_gaps_gt_60s")
    if target_summary.get("temperature_jump_count_gt_3C", 0) > 0:
        data_flags.append("large_temperature_jumps_gt_3C")

    # behavior/model flags based on robust outlier metrics
    if not robust_df.empty:
        for _, row in robust_df.iterrows():
            z = row.get("abs_robust_z", np.nan)
            metric = str(row.get("metric"))
            if np.isfinite(z) and z >= 3.0:
                if metric in {"q_discharge_Ah", "energy_discharge_Wh", "duration_s", "cycle_count", "temperature_max_C", "temperature_range_C"}:
                    behavior_flags.append(f"peer_outlier_{metric}_z{z:.2f}")
                elif metric in {"low_voltage_point_fraction_le_2p75", "high_voltage_point_fraction_ge_4p10", "rest_fraction", "discharge_fraction", "voltage_range_V"}:
                    model_boundary_flags.append(f"profile_regime_outlier_{metric}_z{z:.2f}")
                elif "jump" in metric or "dt" in metric or "nan" in metric:
                    data_flags.append(f"data_shape_outlier_{metric}_z{z:.2f}")

    verdict = "undetermined"
    if data_flags and len(data_flags) >= 2:
        verdict = "possible_data_acquisition_or_preprocessing_issue"
    if behavior_flags and not data_flags:
        verdict = "likely_battery_or_cell_behavior_outlier"
    if model_boundary_flags and not data_flags:
        verdict = "likely_special_regime_or_model_boundary_case"
    if behavior_flags and model_boundary_flags and not data_flags:
        verdict = "likely_real_outlier_cell_or_special_regime_not_general_pipeline_error"
    if not data_flags and not behavior_flags and not model_boundary_flags:
        verdict = "no_clear_raw_data_outlier_detected_needs_model_residual_inspection"

    plots = make_plots(output_dir, target_ref, target_arrays, profile_df) if args.make_plots else []

    summary = {
        "ok": True,
        "stage": "D12 battery-8 data/profile anomaly diagnosis",
        "target_profile": target_ref.canonical,
        "target_npz": target_ref.npz_path,
        "profiles_root": str(profiles_root),
        "peer_profiles_same_batch_protocol": [p.canonical for p in peers],
        "peer_profiles_excluding_target": peer_profiles_excluding_target,
        "output_dir": str(output_dir),
        "files": {
            "profile_peer_summary_csv": str(profile_csv),
            "robust_peer_outlier_scores_csv": str(robust_csv),
            "target_anomaly_events_csv": str(events_csv),
            "target_segment_summary_csv": str(seg_csv),
            "target_cycle_summary_csv": str(cyc_csv),
        },
        "soh_join_info": soh_join_info,
        "target_profile_summary": target_summary,
        "top_robust_outlier_metrics": top_outliers,
        "data_acquisition_flags": sorted(set(data_flags)),
        "battery_behavior_flags": sorted(set(behavior_flags)),
        "model_boundary_flags": sorted(set(model_boundary_flags)),
        "verdict": verdict,
        "interpretation": {
            "data_acquisition_flags": "Evidence such as nonmonotonic time, NaN, large voltage jumps, or impossible voltage range suggests acquisition/preprocessing issues.",
            "battery_behavior_flags": "Capacity, duration, temperature, or energy outliers relative to same-protocol peers suggest the cell is genuinely different or degraded.",
            "model_boundary_flags": "Unusual low/high voltage fraction, rest fraction, or voltage range suggests this profile is outside the effective-SPM surrogate regime.",
            "important_boundary": "A lack of raw data anomalies does not prove the cell is normal; it means the issue likely lies in cell behavior or model expressiveness rather than a broken file.",
        },
        "plots": plots,
    }
    write_json(output_dir / "D12_B8_diagnostic_summary.json", summary)

    # Human-readable recommendation
    rec_lines = [
        "# D12 Battery-8 Data Diagnosis Recommendation",
        "",
        f"Target profile: `{target_ref.canonical}`",
        f"Verdict: `{verdict}`",
        "",
        "## Short interpretation",
    ]
    if verdict == "possible_data_acquisition_or_preprocessing_issue":
        rec_lines.append("There are raw-data shape anomalies. Inspect anomaly events and raw standard parquet before using this battery as a model-failure case.")
    elif verdict == "likely_battery_or_cell_behavior_outlier":
        rec_lines.append("The profile is a peer-level behavior outlier. This points more to battery/cell behavior than to a universal code or model pipeline issue.")
    elif verdict == "likely_special_regime_or_model_boundary_case":
        rec_lines.append("The profile has unusual voltage/regime composition. This points to an effective-SPM/model-boundary issue for this specific regime.")
    elif verdict == "likely_real_outlier_cell_or_special_regime_not_general_pipeline_error":
        rec_lines.append("The profile is both behaviorally and regime-wise unusual versus peers, without strong raw acquisition evidence. Treat it as a real outlier/special regime, not as a general pipeline error.")
    else:
        rec_lines.append("No decisive raw-data anomaly was found. Compare model residual plots and cycle-level behavior before deciding whether this is a model-boundary case.")

    rec_lines += [
        "",
        "## Files to inspect first",
        "- `D12_B8_robust_peer_outlier_scores.csv`: which metrics make battery-8 different from Batch-1 2C peers.",
        "- `D12_B8_target_anomaly_events.csv`: possible raw voltage/current/time/temperature jumps.",
        "- `D12_B8_target_segment_summary.csv`: charge/discharge/rest and low/high voltage regime composition.",
        "- `D12_B8_target_cycle_summary.csv`: cycle-level capacity, voltage, and temperature behavior.",
        "",
        "## Recommended next action",
        "If data flags are weak but behavior/model-boundary flags are strong, keep battery-8 flagged and analyze it separately rather than forcing the 23-profile wrapper to fit it.",
        "If large raw data jumps or impossible ranges are present, audit the original `.mat`/standard parquet around the flagged indices.",
        "",
        "## Top robust outlier metrics",
    ]
    for row in top_outliers[:10]:
        rec_lines.append(
            f"- {row.get('metric')}: target={row.get('target_value')}, peer_median={row.get('peer_median_excluding_target')}, robust_z={row.get('robust_z_vs_peers')}"
        )
    (output_dir / "D12_B8_RECOMMENDATION.md").write_text("\n".join(rec_lines), encoding="utf-8")

    print(json.dumps(_json_safe({
        "ok": True,
        "target_profile": target_ref.canonical,
        "output_dir": str(output_dir),
        "verdict": verdict,
        "data_acquisition_flags": sorted(set(data_flags)),
        "battery_behavior_flags_count": len(set(behavior_flags)),
        "model_boundary_flags_count": len(set(model_boundary_flags)),
        "profile_count_compared": len(peers),
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
