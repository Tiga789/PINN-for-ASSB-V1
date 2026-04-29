#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASSB soft-label residual audit.

Recommended location:
    project root / assb_softlabel_residual_audit.py

This script does not train and does not use data loss. It checks whether the
cycle5_v3 soft-label concentration fields are broadly consistent with the
current training-side SPM flux closure.

It computes two finite-difference diagnostics for cs_a and cs_c:
    1. surface boundary residual: D_s * dc/dr + J(t)
    2. spherical-average balance residual: d<c>/dt + 3*J(t)/R

With the convention used in this project:
    D_s * dc/dr | r=R = -J(t)
    J_a(t) = -I(t) R_a / (3 eps_a F V_a)
    J_c(t) =  I(t) R_c / (3 eps_c F V_c)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _repo_root_from_script() -> Path:
    return Path(__file__).resolve().parent


def _add_paths(repo_root: Path) -> None:
    for p in [repo_root, repo_root / "util"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean_abs": float("nan"), "rmse": float("nan"), "max_abs": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(x.size),
        "mean_abs": float(np.mean(np.abs(x))),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "max_abs": float(np.max(np.abs(x))),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _rel_stats(res: np.ndarray, scale: np.ndarray | float) -> Dict[str, float]:
    res = np.asarray(res, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.size == 1:
        denom = float(np.abs(scale.reshape(-1)[0]))
        denom = max(denom, 1.0e-30)
        rel = res / denom
    else:
        scale = scale.reshape(-1)
        denom = np.maximum(np.abs(scale), 1.0e-30)
        rel = res / denom
    return _stats(rel)


def _grid_from_data(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(data["x_train"], dtype=np.float64)
    y = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("cs data x_train must have columns t and r")
    t = x[:, 0]
    r = x[:, 1]
    t_unique = np.unique(t)
    r_unique = np.unique(r)
    nt, nr = t_unique.size, r_unique.size
    if nt * nr != y.size:
        # General reconstruction by sorting.
        order = np.lexsort((r, t))
        t_s = t[order]
        r_s = r[order]
        y_s = y[order]
        if np.unique(t_s).size * np.unique(r_s).size != y.size:
            raise ValueError("Could not reshape cs data into a full t-r grid")
        t_unique = np.unique(t_s)
        r_unique = np.unique(r_s)
        nt, nr = t_unique.size, r_unique.size
        c = y_s.reshape(nt, nr)
    else:
        order = np.lexsort((r, t))
        c = y[order].reshape(nt, nr)
    return t_unique, r_unique, c


def _spherical_average(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    c = np.asarray(c, dtype=np.float64)
    weights = r ** 2
    denom = np.trapezoid(weights, r)
    if denom <= 0:
        return np.mean(c, axis=1)
    return np.trapezoid(c * weights.reshape(1, -1), r, axis=1) / denom


def _surface_gradient(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """dc/dr at r=R using numpy's edge-order finite difference.

    The audit is diagnostic, not a training loss. A robust, transparent
    finite-difference derivative is preferable to a hand-written stencil.
    """
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    c = np.asarray(c, dtype=np.float64)
    if r.size < 2:
        raise ValueError("At least two radial points are required for surface-gradient audit")
    edge_order = 2 if r.size >= 3 else 1
    grad = np.gradient(c, r, axis=1, edge_order=edge_order)
    return np.asarray(grad[:, -1], dtype=np.float64)


def _interp_profile(t_query: np.ndarray, t_profile: np.ndarray, i_profile: np.ndarray) -> np.ndarray:
    t_profile = np.asarray(t_profile, dtype=np.float64).reshape(-1)
    i_profile = np.asarray(i_profile, dtype=np.float64).reshape(-1)
    order = np.argsort(t_profile)
    t_profile = t_profile[order]
    i_profile = i_profile[order]
    return np.interp(np.asarray(t_query, dtype=np.float64), t_profile, i_profile)


def _current_profile(params: Dict[str, Any], solution: Dict[str, np.ndarray] | None = None):
    if "time_profile" in params and "current_profile_A" in params:
        t = np.asarray(params["time_profile"], dtype=np.float64).reshape(-1)
        i = np.asarray(params["current_profile_A"], dtype=np.float64).reshape(-1)
        if t.size == i.size and t.size >= 2:
            return t, i, "params.time_profile/current_profile_A"
    if "current_profile" in params:
        prof = params["current_profile"]
        if isinstance(prof, (tuple, list)) and len(prof) == 2:
            t = np.asarray(prof[0], dtype=np.float64).reshape(-1)
            i = np.asarray(prof[1], dtype=np.float64).reshape(-1)
            if t.size == i.size and t.size >= 2:
                return t, i, "params.current_profile"
    if solution and "t" in solution and "I_profile" in solution:
        return np.asarray(solution["t"], dtype=np.float64).reshape(-1), np.asarray(solution["I_profile"], dtype=np.float64).reshape(-1), "solution.npz"
    return None


def _flux(params: Dict[str, Any], I: np.ndarray, electrode: str) -> np.ndarray:
    F = float(params["F"])
    if electrode == "a":
        V = float(params.get("V_a", params["A_a"] * params["L_a"]))
        return -np.asarray(I, dtype=np.float64) * float(params["Rs_a"]) / (3.0 * float(params["eps_s_a"]) * F * V)
    V = float(params.get("V_c", params["A_c"] * params["L_c"]))
    return np.asarray(I, dtype=np.float64) * float(params["Rs_c"]) / (3.0 * float(params["eps_s_c"]) * F * V)


def _eval_D(params: Dict[str, Any], electrode: str, cs_surface: np.ndarray) -> np.ndarray:
    import torch

    T = params.get("T", 303.15)
    R = params.get("R", 8.3145e3)
    cs_surface_t = torch.as_tensor(cs_surface.reshape(-1, 1), dtype=torch.float64)
    try:
        if electrode == "a":
            fn = params.get("D_s_a")
            if callable(fn):
                out = fn(T, R)
            else:
                out = float(fn)
        else:
            fn = params.get("D_s_c")
            if callable(fn):
                deg = torch.ones_like(cs_surface_t)
                out = fn(cs_surface_t, T, R, params["cscamax"], deg)
            else:
                out = float(fn)
        if isinstance(out, torch.Tensor):
            arr = out.detach().cpu().numpy().astype(np.float64).reshape(-1)
        else:
            arr = np.asarray(out, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full(cs_surface.size, float(arr[0]), dtype=np.float64)
        return arr
    except Exception:
        if electrode == "a":
            return np.full(cs_surface.size, 5.0e-13, dtype=np.float64)
        return np.full(cs_surface.size, 5.0e-15, dtype=np.float64)


def _audit_one(name: str, data: Dict[str, np.ndarray], params: Dict[str, Any], t_prof: np.ndarray, i_prof: np.ndarray) -> Dict[str, Any]:
    electrode = "a" if name == "cs_a" else "c"
    t, r, c = _grid_from_data(data)
    I_t = _interp_profile(t, t_prof, i_prof)
    J = _flux(params, I_t, electrode)
    dcs_dr_R = _surface_gradient(r, c)
    cs_R = c[:, -1]
    D = _eval_D(params, electrode, cs_R)
    bc_res = D * dcs_dr_R + J

    avg_c = _spherical_average(r, c)
    davg_dt = np.gradient(avg_c, t, edge_order=1)
    Rpart = float(params["Rs_a"] if electrode == "a" else params["Rs_c"])
    mass_res = davg_dt + 3.0 * J / Rpart

    return {
        "name": name,
        "n_time": int(t.size),
        "n_r": int(r.size),
        "t_stats": {"min": float(np.min(t)), "max": float(np.max(t))},
        "r_min": float(np.min(r)),
        "r_max": float(np.max(r)),
        "I_stats": {"min": float(np.min(I_t)), "max": float(np.max(I_t)), "mean": float(np.mean(I_t))},
        "J_stats": _stats(J),
        "D_stats": _stats(D),
        "surface_bc_residual_abs": _stats(bc_res),
        "surface_bc_residual_relative_to_J": _rel_stats(bc_res, J),
        "mass_balance_residual_abs": _stats(mass_res),
        "mass_balance_residual_relative_to_3J_over_R": _rel_stats(mass_res, 3.0 * J / Rpart),
        "surface_theta_stats": _stats(cs_R / float(params["csanmax"] if electrode == "a" else params["cscamax"])),
    }


def _diagnose(report: Dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ["cs_a", "cs_c"]:
        r = report.get(key, {})
        rel_bc = r.get("surface_bc_residual_relative_to_J", {}).get("mean_abs", float("nan"))
        rel_mass = r.get("mass_balance_residual_relative_to_3J_over_R", {}).get("mean_abs", float("nan"))
        if np.isfinite(rel_bc) and rel_bc > 0.5:
            out.append(f"{key}: surface flux residual is large relative to J. Check D_s, sign convention, and r-grid ordering.")
        if np.isfinite(rel_mass) and rel_mass > 0.5:
            out.append(f"{key}: spherical-average mass balance residual is large. This strongly suggests soft-label dynamics and training-side flux closure differ.")
        if np.isfinite(rel_bc) and rel_bc <= 0.2 and np.isfinite(rel_mass) and rel_mass <= 0.2:
            out.append(f"{key}: finite-difference residuals look broadly consistent with the current training-side flux closure.")
    if not out:
        out.append("No simple diagnosis. Inspect residual JSON and plots.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ASSB soft-label residual consistency with current training-side SPM")
    parser.add_argument("--repo_root", default=None)
    parser.add_argument("--soft_label_dir", default="Data/assb_soft_labels_cycle5_v3")
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--ocp_dir", default=None)
    parser.add_argument("--output_dir", default="Eval_assb_softlabel_residual_audit")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root_from_script().resolve()
    soft_dir = Path(args.soft_label_dir)
    if not soft_dir.is_absolute():
        soft_dir = repo_root / soft_dir
    soft_dir = soft_dir.resolve()
    summary_path = Path(args.summary_json) if args.summary_json else soft_dir / "soft_label_summary.json"
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    summary_path = summary_path.resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ocp_dir:
        os.environ["ASSB_OCP_DIR"] = str(Path(args.ocp_dir).resolve())
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_dir)
    _add_paths(repo_root)
    try:
        from util.spm_assb_train_discharge import makeParams  # type: ignore
    except Exception:
        from spm_assb_train_discharge import makeParams  # type: ignore

    params = makeParams(summary_json=str(summary_path))
    solution = _load_npz(soft_dir / "solution.npz") if (soft_dir / "solution.npz").exists() else {}
    prof = _current_profile(params, solution)
    if prof is None:
        raise RuntimeError("No current profile found in params or solution.npz")
    t_prof, i_prof, prof_src = prof

    report: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "soft_label_dir": str(soft_dir),
        "summary_json": str(summary_path),
        "current_profile_source": prof_src,
        "params": {k: _jsonable(params.get(k)) for k in [
            "train_summary_json", "soft_label_dir", "rescale_T", "rescale_R", "Rs_a", "Rs_c", "eps_s_a", "eps_s_c",
            "V_a", "V_c", "F", "R", "T", "csanmax", "cscamax", "R_ohm_eff", "voltage_alignment_offset_V",
        ] if k in params},
    }
    for name, fname in [("cs_a", "data_cs_a.npz"), ("cs_c", "data_cs_c.npz")]:
        report[name] = _audit_one(name, _load_npz(soft_dir / fname), params, t_prof, i_prof)
    report["diagnosis"] = _diagnose(report)

    (out_dir / "softlabel_residual_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["ASSB soft-label residual audit", "=" * 60, f"soft_label_dir: {soft_dir}", f"summary_json: {summary_path}", f"current_profile_source: {prof_src}", ""]
    for name in ["cs_a", "cs_c"]:
        r = report[name]
        lines.append(f"[{name}]")
        lines.append(f"  surface BC rel mean_abs: {r['surface_bc_residual_relative_to_J']['mean_abs']}")
        lines.append(f"  mass balance rel mean_abs: {r['mass_balance_residual_relative_to_3J_over_R']['mean_abs']}")
        lines.append(f"  surface theta min/max: {r['surface_theta_stats']['mean']} mean, max_abs_res_BC {r['surface_bc_residual_abs']['max_abs']}")
        lines.append("")
    lines.append("[diagnosis]")
    for item in report["diagnosis"]:
        lines.append(f"  - {item}")
    (out_dir / "softlabel_residual_audit.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote: {out_dir / 'softlabel_residual_audit.txt'}")
    print(f"Wrote: {out_dir / 'softlabel_residual_audit.json'}")


if __name__ == "__main__":
    main()
