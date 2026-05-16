# -*- coding: utf-8 -*-
r"""Prepare cycle-level aging table for ASSB aging-fix1 / ModelFin_110.

Example
-------
D:\Anaconda\envs\torchgpu\python.exe .\scripts\prepare_assb_aging_fix1_cycle_table.py `
  --solution_npz "..\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  --capacity_target_csv ".\Data\assb_capacity_soh_targets\capacity_soh_targets.csv" `
  --cycle_from 5 --cycle_to 522 --train_to 300 --val_to 420 `
  --output_csv ".\Data\assb_aging_fix1\cycle_table.csv" `
  --output_json ".\Data\assb_aging_fix1\cycle_table_summary.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb_aging_capacity import load_capacity_targets, q_ref_from_targets


def _first_key(npz, candidates: Iterable[str]) -> Optional[str]:
    keys = set(npz.files)
    for key in candidates:
        if key in keys:
            return key
    return None


def _as_1d(arr, name: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.ndim == 0:
        out = out.reshape(1)
    if out.ndim > 1:
        out = np.ravel(out)
    if out.size == 0:
        raise RuntimeError(f"Array {name} is empty")
    return out


def _load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        t_key = _first_key(z, ["t_global_s", "time_s", "t_s", "t", "time"])
        c_key = _first_key(z, ["cycle_id", "cycle", "cycle_index"])
        i_key = _first_key(z, ["I_profile", "I_A", "current_A", "I", "current"])
        if t_key is None or c_key is None or i_key is None:
            raise KeyError(f"solution.npz must contain time, cycle_id and current arrays. Available: {z.files}")
        t = _as_1d(z[t_key], t_key).astype(float)
        cycle_raw = _as_1d(z[c_key], c_key).astype(int)
        I = _as_1d(z[i_key], i_key).astype(float)
        if cycle_raw.size == 1 and t.size > 1:
            cycle = np.full(t.size, int(cycle_raw[0]), dtype=int)
        elif cycle_raw.size == t.size:
            cycle = cycle_raw
        else:
            raise RuntimeError(f"cycle length mismatch: {cycle_raw.size} vs time {t.size}")
        if I.size != t.size:
            raise RuntimeError(f"current length mismatch: {I.size} vs time {t.size}")
        step_key = _first_key(z, ["step_id", "step", "step_index"])
        type_key = _first_key(z, ["step_type", "step_name", "mode"])
        voltage_key = _first_key(z, ["voltage_exp", "V_exp", "U_exp", "voltage", "V"])
        out = {"t": t, "cycle_id": cycle, "I": I}
        if step_key:
            step = _as_1d(z[step_key], step_key)
            out["step_id"] = step if step.size == t.size else np.full(t.size, np.nan)
        if type_key:
            stype = _as_1d(z[type_key], type_key)
            out["step_type"] = stype if stype.size == t.size else np.full(t.size, "", dtype=object)
        if voltage_key:
            v = _as_1d(z[voltage_key], voltage_key).astype(float)
            if v.size == t.size:
                out["voltage"] = v
    return out


def _trapz_by_cycle(t: np.ndarray, y: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    order = np.argsort(t)
    yy = np.asarray(y[order], dtype=float)
    tt = np.asarray(t[order], dtype=float)
    if yy.size < 2:
        return 0.0
    # NumPy >= 2.x may remove np.trapz; use an explicit trapezoidal rule.
    return float(np.sum((tt[1:] - tt[:-1]) * (yy[1:] + yy[:-1]) * 0.5))


def _complete_from_step_types(step_type: Optional[np.ndarray], I: np.ndarray) -> bool:
    if step_type is not None:
        text = " ".join(str(x).lower() for x in step_type)
        has_charge = "充" in text or "charge" in text or np.any(I > 1e-12)
        has_discharge = "放" in text or "discharge" in text or np.any(I < -1e-12)
        return bool(has_charge and has_discharge)
    return bool(np.any(I > 1e-12) and np.any(I < -1e-12))


def _normalize(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return vals
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return np.zeros_like(vals)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi - lo <= 1e-30:
        return np.zeros_like(vals)
    out = (vals - lo) / (hi - lo)
    out[~np.isfinite(out)] = 0.0
    return out


def make_cycle_table(arrays: Dict[str, np.ndarray], targets: pd.DataFrame, *, cycle_from: int, cycle_to: int, train_to: int, val_to: int) -> pd.DataFrame:
    t = arrays["t"].astype(float)
    cycle = arrays["cycle_id"].astype(int)
    I = arrays["I"].astype(float)
    step_type = arrays.get("step_type")
    voltage = arrays.get("voltage")
    mask = (cycle >= int(cycle_from)) & (cycle <= int(cycle_to))
    if not mask.any():
        raise RuntimeError(f"No solution rows in cycle range {cycle_from}..{cycle_to}")
    t = t[mask]
    cycle = cycle[mask]
    I = I[mask]
    if step_type is not None:
        step_type = step_type[mask]
    if voltage is not None:
        voltage = voltage[mask]

    rows = []
    q_net_running = 0.0
    throughput_running = 0.0
    for cid in sorted(np.unique(cycle)):
        m = cycle == cid
        tt = t[m]
        ii = I[m]
        if tt.size == 0:
            continue
        charge_C = _trapz_by_cycle(tt, np.clip(ii, 0.0, None))
        discharge_C = _trapz_by_cycle(tt, np.clip(-ii, 0.0, None))
        net_C = _trapz_by_cycle(tt, ii)
        through_C = _trapz_by_cycle(tt, np.abs(ii))
        duration = float(np.nanmax(tt) - np.nanmin(tt)) if tt.size else 0.0
        stype_slice = step_type[m] if step_type is not None else None
        rest_fraction = float(np.mean(np.abs(ii) <= 1e-12)) if ii.size else 0.0
        row = {
            "cycle_id": int(cid),
            "t_start_s": float(np.nanmin(tt)),
            "t_end_s": float(np.nanmax(tt)),
            "duration_s": duration,
            "n_points": int(tt.size),
            "q_charge_cycle_C": charge_C,
            "q_discharge_cycle_C": discharge_C,
            "q_net_cycle_C": net_C,
            "throughput_cycle_C": through_C,
            "q_net_start_C": q_net_running,
            "q_net_end_C": q_net_running + net_C,
            "throughput_start_C": throughput_running,
            "throughput_end_C": throughput_running + through_C,
            "I_abs_mean_A": float(np.mean(np.abs(ii))),
            "I_abs_max_A": float(np.max(np.abs(ii))) if ii.size else 0.0,
            "I_charge_mean_A": float(np.mean(ii[ii > 1e-12])) if np.any(ii > 1e-12) else 0.0,
            "I_discharge_abs_mean_A": float(np.mean(np.abs(ii[ii < -1e-12]))) if np.any(ii < -1e-12) else 0.0,
            "rest_fraction": rest_fraction,
            "complete_cycle_from_solution": _complete_from_step_types(stype_slice, ii),
            "split": "train" if cid <= int(train_to) else ("val" if cid <= int(val_to) else "test"),
        }
        if voltage is not None:
            vv = voltage[m]
            finite = np.isfinite(vv)
            row.update(
                voltage_mean=float(np.nanmean(vv[finite])) if finite.any() else np.nan,
                voltage_min=float(np.nanmin(vv[finite])) if finite.any() else np.nan,
                voltage_max=float(np.nanmax(vv[finite])) if finite.any() else np.nan,
            )
        rows.append(row)
        q_net_running += net_C
        throughput_running += through_C

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("Empty cycle table")
    frame = frame.merge(targets, on="cycle_id", how="left", suffixes=("", "_target"))
    if "complete_cycle" in frame.columns:
        frame["complete_cycle"] = frame["complete_cycle"].fillna(False).astype(bool) & frame["complete_cycle_from_solution"].astype(bool)
    else:
        frame["complete_cycle"] = frame["complete_cycle_from_solution"].astype(bool)

    # Normalized features for Stage-B mechanism training.  Keep this list in sync
    # with util.assb_aging_fix1_config.AgingFix1Config.feature_dim default.
    frame["cycle_norm"] = _normalize(frame["cycle_id"].to_numpy(dtype=float))
    frame["throughput_norm"] = _normalize(frame["throughput_end_C"].to_numpy(dtype=float))
    frame["duration_norm"] = _normalize(frame["duration_s"].to_numpy(dtype=float))
    frame["I_abs_mean_norm"] = _normalize(frame["I_abs_mean_A"].to_numpy(dtype=float))
    frame["I_abs_max_norm"] = _normalize(frame["I_abs_max_A"].to_numpy(dtype=float))
    frame["q_charge_norm"] = _normalize(frame["q_charge_cycle_C"].to_numpy(dtype=float))
    frame["q_discharge_norm"] = _normalize(frame["q_discharge_cycle_C"].to_numpy(dtype=float))
    frame["rest_fraction_norm"] = np.clip(frame["rest_fraction"].to_numpy(dtype=float), 0.0, 1.0)
    return frame


def summarize(frame: pd.DataFrame, *, solution_npz: Path, capacity_target_csv: Path, feature_columns) -> Dict[str, object]:
    q_ref = q_ref_from_targets(frame)
    out = {
        "solution_npz": str(solution_npz),
        "capacity_target_csv": str(capacity_target_csv),
        "n_cycles": int(len(frame)),
        "cycle_min": int(frame["cycle_id"].min()),
        "cycle_max": int(frame["cycle_id"].max()),
        "split_counts": {str(k): int(v) for k, v in frame["split"].value_counts().to_dict().items()},
        "complete_cycles": int(frame["complete_cycle"].sum()),
        "capacity_target_rows_with_Q": int(pd.to_numeric(frame.get("Q_obs_Ah", pd.Series(dtype=float)), errors="coerce").notna().sum()),
        "capacity_target_rows_with_SOH": int(pd.to_numeric(frame.get("SOH_obs", pd.Series(dtype=float)), errors="coerce").notna().sum()),
        "q_ref_Ah": float(q_ref),
        "q_ref_mAh": float(q_ref * 1000.0),
        "feature_columns": list(feature_columns),
    }
    if "SOH_obs" in frame.columns:
        vals = pd.to_numeric(frame["SOH_obs"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out.update(SOH_obs_min=float(np.nanmin(vals)), SOH_obs_max=float(np.nanmax(vals)))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare ASSB aging-fix1 cycle table")
    parser.add_argument("--solution_npz", required=True)
    parser.add_argument("--capacity_target_csv", required=True)
    parser.add_argument("--cycle_from", type=int, default=5)
    parser.add_argument("--cycle_to", type=int, default=522)
    parser.add_argument("--train_to", type=int, default=300)
    parser.add_argument("--val_to", type=int, default=420)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args(argv)

    solution_npz = Path(args.solution_npz)
    capacity_csv = Path(args.capacity_target_csv)
    arrays = _load_npz_arrays(solution_npz)
    targets = load_capacity_targets(capacity_csv, cycle_from=args.cycle_from, cycle_to=args.cycle_to)
    frame = make_cycle_table(arrays, targets, cycle_from=args.cycle_from, cycle_to=args.cycle_to, train_to=args.train_to, val_to=args.val_to)
    feature_columns = [
        "cycle_norm",
        "throughput_norm",
        "duration_norm",
        "I_abs_mean_norm",
        "I_abs_max_norm",
        "q_charge_norm",
        "q_discharge_norm",
        "rest_fraction_norm",
    ]
    frame["feature_columns"] = ";".join(feature_columns)
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary = summarize(frame, solution_npz=solution_npz, capacity_target_csv=capacity_csv, feature_columns=feature_columns)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("[prepare_assb_aging_fix1_cycle_table] wrote", out_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
