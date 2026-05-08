# -*- coding: utf-8 -*-
r"""
Repair ASSB soft-label solution.npz so cs_a/cs_c spherical averages exactly follow
stored surface flux j_a/j_c, while preserving each time step's radial deviation shape.

This is intended for solutions used with the ID101/ID102 I(t)-cbar hard baseline.
It does NOT refit voltage, OCP, kinetics, or aging parameters. It only shifts the
solid concentration profiles by a time-dependent uniform offset:

    cs_new(t, r) = target_cbar(t) + [cs_old(t, r) - old_cbar(t)]

so that sum_r w(r) cs_new(t,r) = target_cbar(t).

Example PowerShell:
  D:\Anaconda\envs\torchgpu\python.exe .\repair_assb_solution_mass_closure.py `
    --solution "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1\solution.npz" `
    --output_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1_massclosed" `
    --electrodes a c `
    --integration right
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def pick_name(files: Iterable[str], names: Iterable[str], required: bool = True) -> Optional[str]:
    fs = set(files)
    for n in names:
        if n in fs:
            return n
    if required:
        raise KeyError(f"Cannot find any of {list(names)}. Available={sorted(fs)}")
    return None


def as_Nt_Nr(arr: np.ndarray, Nt: int, Nr: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (Nt, Nr):
        return arr.astype(np.float64, copy=False)
    if arr.shape == (Nr, Nt):
        return arr.T.astype(np.float64, copy=False)
    if arr.ndim == 1 and arr.size == Nt * Nr:
        return arr.reshape(Nt, Nr).astype(np.float64, copy=False)
    raise ValueError(f"Cannot reshape {name}: {arr.shape}; expected ({Nt},{Nr}) or ({Nr},{Nt})")


def weights_spherical_trapz(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    coeff = np.zeros_like(r)
    dr = np.diff(r)
    coeff[:-1] += 0.5 * dr
    coeff[1:] += 0.5 * dr
    w = coeff * r**2
    return w / np.sum(w)


def weights_finite_volume_shell(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    R = float(r[-1])
    edges = np.empty(r.size + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[-1] = R
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    w = (edges[1:] ** 3 - edges[:-1] ** 3) / (R**3)
    return w / np.sum(w)


def metric_dict(old: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    e = old - target
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "maxabs": float(np.max(np.abs(e))),
        "corr": float(np.corrcoef(old, target)[0, 1]) if np.std(old) > 0 and np.std(target) > 0 else float("nan"),
    }


def target_cbar_from_flux(t: np.ndarray, j: np.ndarray, R: float, cbar0: float, integration: str) -> np.ndarray:
    dt = np.diff(t)
    rate = -3.0 * np.asarray(j, dtype=np.float64).reshape(-1) / float(R)
    if integration == "right":
        inc = rate[1:] * dt
    elif integration == "left":
        inc = rate[:-1] * dt
    elif integration == "trapezoid":
        inc = 0.5 * (rate[:-1] + rate[1:]) * dt
    else:
        raise ValueError(f"Unknown integration mode: {integration}")
    out = np.empty_like(t, dtype=np.float64)
    out[0] = float(cbar0)
    out[1:] = out[0] + np.cumsum(inc)
    return out


def load_summary_values(source_dir: Path) -> Dict[str, float]:
    vals: Dict[str, float] = {}
    for name in ["soft_label_summary.json", "summary.json", "record_profile_summary.json"]:
        p = source_dir / name
        if not p.exists():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [obj]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    lk = str(k).lower()
                    if isinstance(v, (int, float)):
                        vals[lk] = float(v)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(x, list):
                stack.extend(x)
    return vals


def guess_csmax(data: Dict[str, np.ndarray], electrode: str, cs: np.ndarray, source_dir: Path,
                override: Optional[float]) -> Optional[float]:
    if override is not None and override > 0:
        return float(override)

    vals = load_summary_values(source_dir)
    candidates = []
    if electrode == "a":
        candidates = ["csmax_a", "c_s_max_a", "csanmax", "cs_a_max", "csa_max"]
        theta_names = ["theta_a", "theta_s_a", "theta_anode"]
        fallback = 6.0
    else:
        candidates = ["csmax_c", "c_s_max_c", "cscamax", "cs_c_max", "csc_max"]
        theta_names = ["theta_c", "theta_s_c", "theta_cathode"]
        fallback = 51.8

    for k in candidates:
        if k in vals and np.isfinite(vals[k]) and vals[k] > 0:
            return float(vals[k])

    for tn in theta_names:
        if tn in data:
            th = np.asarray(data[tn])
            if th.shape == cs.shape:
                mask = np.isfinite(th) & np.isfinite(cs) & (np.abs(th) > 1e-8)
                if np.count_nonzero(mask) > 100:
                    ratio = cs[mask] / th[mask]
                    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                    if ratio.size > 100:
                        return float(np.median(ratio))
    return fallback


def update_theta_and_cbar_fields(data: Dict[str, np.ndarray], electrode: str, cs_new: np.ndarray,
                                 cbar_target: np.ndarray, csmax: Optional[float], report: Dict) -> None:
    Nt, Nr = cs_new.shape
    suffix_names = {
        "a": {
            "theta2d": ["theta_a", "theta_s_a", "theta_anode"],
            "theta1d": ["theta_a_surf", "theta_surf_a", "theta_surface_a", "theta_a_surface"],
            "cbar": ["cbar_a", "csbar_a", "cs_a_bar", "cs_mean_a", "cbar_anode"],
            "theta_bar": ["theta_bar_a", "theta_a_bar", "theta_mean_a"],
        },
        "c": {
            "theta2d": ["theta_c", "theta_s_c", "theta_cathode"],
            "theta1d": ["theta_c_surf", "theta_surf_c", "theta_surface_c", "theta_c_surface"],
            "cbar": ["cbar_c", "csbar_c", "cs_c_bar", "cs_mean_c", "cbar_cathode"],
            "theta_bar": ["theta_bar_c", "theta_c_bar", "theta_mean_c"],
        },
    }[electrode]

    updated = []
    if csmax is not None and csmax > 0:
        theta_new = cs_new / float(csmax)
        theta_surf = cs_new[:, -1] / float(csmax)
        theta_bar = cbar_target / float(csmax)

        for n in suffix_names["theta2d"]:
            if n in data and np.asarray(data[n]).shape == (Nt, Nr):
                data[n] = theta_new.astype(np.asarray(data[n]).dtype, copy=False)
                updated.append(n)
        for n in suffix_names["theta1d"]:
            if n in data and np.asarray(data[n]).shape == (Nt,):
                data[n] = theta_surf.astype(np.asarray(data[n]).dtype, copy=False)
                updated.append(n)
        for n in suffix_names["theta_bar"]:
            if n in data and np.asarray(data[n]).shape == (Nt,):
                data[n] = theta_bar.astype(np.asarray(data[n]).dtype, copy=False)
                updated.append(n)

    for n in suffix_names["cbar"]:
        if n in data and np.asarray(data[n]).shape == (Nt,):
            data[n] = cbar_target.astype(np.asarray(data[n]).dtype, copy=False)
            updated.append(n)

    report["updated_related_fields"] = updated
    report["csmax_used"] = None if csmax is None else float(csmax)


def repair_one(data: Dict[str, np.ndarray], files: Iterable[str], electrode: str, integration: str,
               weight_mode: str, source_dir: Path, csmax_override: Optional[float]) -> Dict:
    t_name = pick_name(files, ["t_global_s", "t", "time_s"])
    t = np.asarray(data[t_name], dtype=np.float64).reshape(-1)
    Nt = t.size

    if electrode == "a":
        r_name = pick_name(files, ["r_a", "r_grid_a", "r_anode"])
        j_name = pick_name(files, ["j_a", "J_a", "flux_a"])
        cs_name = pick_name(files, ["cs_a", "c_s_a", "cs_a_full", "anode_cs", "c_s_anode"])
    else:
        r_name = pick_name(files, ["r_c", "r_grid_c", "r_cathode"])
        j_name = pick_name(files, ["j_c", "J_c", "flux_c"])
        cs_name = pick_name(files, ["cs_c", "c_s_c", "cs_c_full", "cathode_cs", "c_s_cathode"])

    r = np.asarray(data[r_name], dtype=np.float64).reshape(-1)
    j = np.asarray(data[j_name], dtype=np.float64).reshape(-1)
    Nr = r.size
    R = float(r[-1])
    cs_old = as_Nt_Nr(np.asarray(data[cs_name]), Nt, Nr, cs_name)

    w = weights_finite_volume_shell(r) if weight_mode == "fv" else weights_spherical_trapz(r)
    cbar_old = cs_old @ w
    cbar_target = target_cbar_from_flux(t, j, R, cbar_old[0], integration=integration)

    radial_dev = cs_old - cbar_old[:, None]
    cs_new = cbar_target[:, None] + radial_dev
    cbar_new = cs_new @ w

    orig_dtype = np.asarray(data[cs_name]).dtype
    data[cs_name] = cs_new.astype(orig_dtype, copy=False)

    rep: Dict = {
        "electrode": electrode,
        "arrays": {"t": t_name, "r": r_name, "j": j_name, "cs": cs_name},
        "weight_mode": weight_mode,
        "integration": integration,
        "R": R,
        "before_cbar_vs_flux": metric_dict(cbar_old, cbar_target),
        "after_cbar_vs_flux": metric_dict(cbar_new, cbar_target),
        "old_cbar_range": [float(np.min(cbar_old)), float(np.max(cbar_old))],
        "target_cbar_range": [float(np.min(cbar_target)), float(np.max(cbar_target))],
        "new_cs_range": [float(np.min(cs_new)), float(np.max(cs_new))],
        "uniform_shift_range": [float(np.min(cbar_target - cbar_old)), float(np.max(cbar_target - cbar_old))],
    }

    csmax = guess_csmax(data, electrode, cs_new, source_dir, csmax_override)
    update_theta_and_cbar_fields(data, electrode, cs_new, cbar_target, csmax, rep)
    if csmax is not None and csmax > 0:
        rep["new_theta_range_from_csmax"] = [float(np.min(cs_new / csmax)), float(np.max(cs_new / csmax))]

    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True, help="Input solution.npz")
    ap.add_argument("--output_dir", required=True, help="Output directory for repaired solution.npz")
    ap.add_argument("--electrodes", nargs="+", choices=["a", "c"], default=["a", "c"])
    ap.add_argument("--weights", choices=["fv", "trapz"], default="fv")
    ap.add_argument("--integration", choices=["right", "left", "trapezoid"], default="right")
    ap.add_argument("--csmax_a", type=float, default=None)
    ap.add_argument("--csmax_c", type=float, default=None)
    ap.add_argument("--compressed", action="store_true", help="Use np.savez_compressed; slower but smaller")
    args = ap.parse_args()

    src = Path(args.solution)
    if not src.exists():
        raise FileNotFoundError(src)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = np.load(src, allow_pickle=True)
    data: Dict[str, np.ndarray] = {k: sd[k] for k in sd.files}
    files = list(data.keys())

    reports = []
    for e in args.electrodes:
        reports.append(
            repair_one(
                data=data,
                files=files,
                electrode=e,
                integration=args.integration,
                weight_mode=args.weights,
                source_dir=src.parent,
                csmax_override=args.csmax_a if e == "a" else args.csmax_c,
            )
        )

    out_npz = out_dir / "solution.npz"
    if args.compressed:
        np.savez_compressed(out_npz, **data)
    else:
        np.savez(out_npz, **data)

    # Copy sidecar files for traceability; do not overwrite the new solution.npz.
    for p in src.parent.iterdir():
        if p.name == "solution.npz" or p.is_dir():
            continue
        dst = out_dir / p.name
        try:
            shutil.copy2(p, dst)
        except Exception:
            pass

    report = {
        "source_solution": str(src),
        "output_solution": str(out_npz),
        "electrodes_repaired": args.electrodes,
        "weights": args.weights,
        "integration": args.integration,
        "reports": reports,
    }
    (out_dir / "mass_closure_repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nDONE. Repaired solution:", out_npz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
