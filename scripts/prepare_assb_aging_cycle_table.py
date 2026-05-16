# -*- coding: utf-8 -*-
r"""
Prepare a cycle-level aging table for ASSB ModelFin_109.

Example:
D:\Anaconda\envs\torchgpu\python.exe .\scripts\prepare_assb_aging_cycle_table.py `
  --solution_npz "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate\solution.npz" `
  --capacity_target_csv ".\Data\assb_capacity_soh_targets\capacity_soh_targets.csv" `
  --cycle_from 5 --cycle_to 522 --train_to 300 --val_to 420 `
  --output_csv ".\Data\assb_aging_ModelFin109\cycle_table.csv" `
  --output_json ".\Data\assb_aging_ModelFin109\cycle_table_summary.json"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pandas is required for prepare_assb_aging_cycle_table.py") from exc

# Make the script runnable from project root without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.assb_capacity_from_states import load_capacity_targets_simple
from util.assb_cycle_table import make_cycle_features


def _first_existing_key(npz, candidates: Iterable[str]) -> Optional[str]:
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
    if len(out) == 0:
        raise RuntimeError(f"Array {name} is empty.")
    return out


def _load_solution_arrays(solution_npz: Path) -> Dict[str, np.ndarray]:
    if not solution_npz.exists():
        raise FileNotFoundError(f"solution_npz not found: {solution_npz}")
    with np.load(solution_npz, allow_pickle=True) as z:
        t_key = _first_existing_key(z, ["t_global_s", "time_s", "t_s", "t", "time"])
        c_key = _first_existing_key(z, ["cycle_id", "cycle", "cycle_index"])
        i_key = _first_existing_key(z, ["I_profile", "I_A", "current_A", "I", "current"])
        if t_key is None or c_key is None or i_key is None:
            raise KeyError(
                "solution.npz must contain time, cycle_id and current arrays. "
                f"Available keys: {z.files}"
            )
        t = _as_1d(z[t_key], t_key).astype(float)
        cycle_raw = _as_1d(z[c_key], c_key).astype(int)
        current = _as_1d(z[i_key], i_key).astype(float)
        if len(cycle_raw) == 1 and len(t) > 1:
            cycle = np.full(len(t), int(cycle_raw[0]), dtype=int)
        elif len(cycle_raw) == len(t):
            cycle = cycle_raw
        else:
            raise RuntimeError(
                f"Length mismatch: {t_key}={len(t)}, {c_key}={len(cycle_raw)}, {i_key}={len(current)}. "
                "cycle arrays must either match time length or contain one cycle id."
            )
        if len(current) != len(t):
            raise RuntimeError(f"Length mismatch: {t_key}={len(t)}, {i_key}={len(current)}")
        step_type_key = _first_existing_key(z, ["step_type", "step_types", "工步类型"])
        step_id_key = _first_existing_key(z, ["step_id", "step_index", "工步号"])
        voltage_key = _first_existing_key(z, ["voltage_exp", "V_exp", "voltage_V", "V", "phis_c_exp"])
        out = {"t_global_s": t, "cycle_id": cycle, "I_profile": current}
        if step_type_key is not None:
            out["step_type"] = _as_1d(z[step_type_key], step_type_key)
        if step_id_key is not None:
            out["step_id"] = _as_1d(z[step_id_key], step_id_key)
        if voltage_key is not None:
            v = _as_1d(z[voltage_key], voltage_key).astype(float)
            if len(v) == len(t):
                out["voltage_exp"] = v
        return out


def _time_deltas(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if len(t) == 1:
        return np.array([0.0], dtype=float)
    dt = np.diff(t, prepend=t[0])
    # The first point receives the median positive dt so that per-cycle charge
    # estimates are not biased by a zero first sample when data are sparse.
    pos = dt[dt > 0]
    if len(pos):
        dt[0] = float(np.median(pos))
    dt[~np.isfinite(dt)] = 0.0
    dt[dt < 0] = 0.0
    return dt


def _classify_step_type(values: Optional[np.ndarray], current: np.ndarray) -> Tuple[float, float, float, int, int, int]:
    if values is None or len(values) != len(current):
        charge_mask = current > 1.0e-12
        discharge_mask = current < -1.0e-12
        rest_mask = ~(charge_mask | discharge_mask)
    else:
        s = np.asarray(values).astype(str)
        charge_mask = np.array([(str(v).find("充") >= 0) or ("charge" in str(v).lower() and "dis" not in str(v).lower()) for v in s])
        discharge_mask = np.array([(str(v).find("放") >= 0) or ("discharge" in str(v).lower()) for v in s])
        rest_mask = np.array([(str(v).find("搁") >= 0) or (str(v).find("置") >= 0) or ("rest" in str(v).lower()) for v in s])
        # Fall back to current sign when labels are not informative.
        if not (charge_mask.any() or discharge_mask.any() or rest_mask.any()):
            charge_mask = current > 1.0e-12
            discharge_mask = current < -1.0e-12
            rest_mask = ~(charge_mask | discharge_mask)
    return charge_mask, discharge_mask, rest_mask


def build_cycle_table(
    arrays: Dict[str, np.ndarray],
    *,
    cycle_from: Optional[int] = None,
    cycle_to: Optional[int] = None,
    train_to: int = 300,
    val_to: int = 420,
) -> pd.DataFrame:
    t_all = arrays["t_global_s"].astype(float)
    c_all = arrays["cycle_id"].astype(int)
    i_all = arrays["I_profile"].astype(float)

    mask = np.ones(len(t_all), dtype=bool)
    if cycle_from is not None:
        mask &= c_all >= int(cycle_from)
    if cycle_to is not None:
        mask &= c_all <= int(cycle_to)
    if not mask.any():
        raise RuntimeError("No solution rows remain after cycle filtering.")

    t = t_all[mask]
    cycle = c_all[mask]
    current = i_all[mask]
    # Re-zero the local timeline only if the source is not already zeroed.
    t0_global = float(np.nanmin(t))
    if t0_global != 0.0:
        t = t - t0_global
    dt = _time_deltas(t)

    step_type = arrays.get("step_type")
    step_type = step_type[mask] if step_type is not None and len(step_type) == len(mask) else None
    step_id = arrays.get("step_id")
    step_id = step_id[mask] if step_id is not None and len(step_id) == len(mask) else None
    voltage = arrays.get("voltage_exp")
    voltage = voltage[mask] if voltage is not None and len(voltage) == len(mask) else None

    rows: List[Dict[str, object]] = []
    q_net_cum = 0.0
    throughput_cum = 0.0
    for cid in np.unique(cycle):
        idx = np.where(cycle == cid)[0]
        if len(idx) == 0:
            continue
        tt = t[idx]
        ii = current[idx]
        dti = dt[idx]
        st = step_type[idx] if step_type is not None else None
        sid = step_id[idx] if step_id is not None else None
        v = voltage[idx] if voltage is not None else None
        charge_mask, discharge_mask, rest_mask = _classify_step_type(st, ii)

        q_net = float(np.sum(ii * dti))
        throughput = float(np.sum(np.abs(ii) * dti))
        charge_C = float(np.sum(np.maximum(ii, 0.0) * dti))
        discharge_C = float(np.sum(np.maximum(-ii, 0.0) * dti))
        duration = float(np.nanmax(tt) - np.nanmin(tt)) if len(tt) else 0.0
        rest_time = float(np.sum(dti[rest_mask])) if len(rest_mask) == len(dti) else 0.0
        row: Dict[str, object] = {
            "cycle_id": int(cid),
            "t_start_s": float(np.nanmin(tt)),
            "t_end_s": float(np.nanmax(tt)),
            "duration_s": duration,
            "n_points": int(len(idx)),
            "q_net_cycle_C": q_net,
            "q_net_start_C": q_net_cum,
            "q_net_end_C": q_net_cum + q_net,
            "throughput_cycle_C": throughput,
            "throughput_start_C": throughput_cum,
            "throughput_end_C": throughput_cum + throughput,
            "charge_C": charge_C,
            "discharge_C": discharge_C,
            "rest_time_s": rest_time,
            "charge_time_s": float(np.sum(dti[charge_mask])) if len(charge_mask) == len(dti) else 0.0,
            "discharge_time_s": float(np.sum(dti[discharge_mask])) if len(discharge_mask) == len(dti) else 0.0,
            "I_abs_mean_A": float(np.nanmean(np.abs(ii))),
            "I_charge_mean_A": float(np.nanmean(ii[ii > 1.0e-12])) if np.any(ii > 1.0e-12) else 0.0,
            "I_discharge_mean_A": float(np.nanmean(ii[ii < -1.0e-12])) if np.any(ii < -1.0e-12) else 0.0,
            "complete_cycle": bool(charge_C > 0.0 and discharge_C > 0.0),
            "split": "train" if int(cid) <= int(train_to) else ("val" if int(cid) <= int(val_to) else "test"),
        }
        if sid is not None and len(sid):
            row["step_count"] = int(len(np.unique(sid)))
        if v is not None and len(v):
            row["voltage_min_V"] = float(np.nanmin(v))
            row["voltage_max_V"] = float(np.nanmax(v))
            row["voltage_mean_V"] = float(np.nanmean(v))
        rows.append(row)
        q_net_cum += q_net
        throughput_cum += throughput

    frame = pd.DataFrame(rows)
    if len(frame) == 0:
        raise RuntimeError("No cycle rows were built from solution arrays.")
    frame, _ = make_cycle_features(frame)
    return frame


def attach_capacity_targets(frame: pd.DataFrame, capacity_target_csv: Optional[Path]) -> pd.DataFrame:
    out = frame.copy()
    if capacity_target_csv is None:
        return out
    cap_path = Path(capacity_target_csv)
    if not cap_path.exists():
        raise FileNotFoundError(f"capacity_target_csv not found: {cap_path}")
    cap = load_capacity_targets_simple(cap_path)
    keep_cols = [c for c in ["cycle_id", "Q_dis_Ah", "SOH", "Q_ref_Ah", "complete_cycle"] if c in cap.columns]
    cap_small = cap.loc[:, keep_cols].copy()
    rename = {"Q_dis_Ah": "Q_obs_Ah", "SOH": "SOH_obs", "complete_cycle": "capacity_complete_cycle"}
    cap_small = cap_small.rename(columns=rename)
    out = out.merge(cap_small, on="cycle_id", how="left")
    if "Q_ref_Ah" in out.columns and out["Q_ref_Ah"].notna().any():
        q_ref = float(out["Q_ref_Ah"].dropna().iloc[0])
        out["Q_ref_Ah"] = out["Q_ref_Ah"].fillna(q_ref)
    return out


def compute_summary(frame: pd.DataFrame, *, solution_npz: Path, capacity_target_csv: Optional[Path]) -> Dict[str, object]:
    split_counts = frame["split"].astype(str).value_counts().to_dict() if "split" in frame.columns else {}
    out: Dict[str, object] = {
        "solution_npz": str(solution_npz),
        "capacity_target_csv": str(capacity_target_csv) if capacity_target_csv else None,
        "n_cycles": int(len(frame)),
        "cycle_min": int(frame["cycle_id"].min()),
        "cycle_max": int(frame["cycle_id"].max()),
        "t_start_s": float(frame["t_start_s"].min()),
        "t_end_s": float(frame["t_end_s"].max()),
        "throughput_total_C": float(frame["throughput_cycle_C"].sum()),
        "q_net_total_C": float(frame["q_net_cycle_C"].sum()),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "complete_cycle_count": int(pd.Series(frame.get("complete_cycle", False)).fillna(False).astype(bool).sum()),
    }
    if "Q_obs_Ah" in frame.columns and frame["Q_obs_Ah"].notna().any():
        out.update(
            {
                "capacity_rows_matched": int(frame["Q_obs_Ah"].notna().sum()),
                "Q_obs_mAh_min": float(frame["Q_obs_Ah"].min() * 1000.0),
                "Q_obs_mAh_max": float(frame["Q_obs_Ah"].max() * 1000.0),
                "SOH_obs_min": float(frame["SOH_obs"].min()) if "SOH_obs" in frame.columns else None,
                "SOH_obs_max": float(frame["SOH_obs"].max()) if "SOH_obs" in frame.columns else None,
            }
        )
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare ASSB ModelFin_109 aging cycle table.")
    p.add_argument("--solution_npz", required=True, help="Continuous soft-label solution.npz path.")
    p.add_argument("--capacity_target_csv", default=None, help="Cycle-level capacity/SOH target CSV.")
    p.add_argument("--cycle_from", type=int, default=5)
    p.add_argument("--cycle_to", type=int, default=522)
    p.add_argument("--train_to", type=int, default=300)
    p.add_argument("--val_to", type=int, default=420)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_json", required=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    solution_npz = Path(args.solution_npz)
    cap_path = Path(args.capacity_target_csv) if args.capacity_target_csv else None
    arrays = _load_solution_arrays(solution_npz)
    frame = build_cycle_table(
        arrays,
        cycle_from=args.cycle_from,
        cycle_to=args.cycle_to,
        train_to=args.train_to,
        val_to=args.val_to,
    )
    frame = attach_capacity_targets(frame, cap_path)
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")
    summary = compute_summary(frame, solution_npz=solution_npz, capacity_target_csv=cap_path)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[prepare_assb_aging_cycle_table] wrote:")
    print(f"  CSV : {out_csv}")
    print(f"  JSON: {out_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
