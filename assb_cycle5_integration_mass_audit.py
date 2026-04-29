#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit whether the cycle5 soft labels follow the same spherical-average
mass-balance sign implied by integration_spm/spm_int_assb_cycle.py.

This script does not train. It deliberately imports
integration_spm.spm_int_assb_cycle.surface_flux_from_current so that the
surface-flux sign used for the audit is the same as the soft-label generator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _insert_repo_paths(repo: Path) -> None:
    for p in (repo, repo / "util", repo / "integration_spm"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(np.asarray(v).reshape(-1)[0])
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _find_current(solution: dict[str, np.ndarray], t_data: np.ndarray) -> tuple[np.ndarray, str]:
    # Common keys written by ASSB soft-label generators.
    current_keys = [
        "I_profile", "current_profile_A", "current_A", "I_A", "I", "I_app_profile",
    ]
    time_keys = ["t", "time_s", "time_profile", "t_profile", "current_time_profile"]
    current = None
    current_key = None
    for k in current_keys:
        if k in solution:
            arr = np.asarray(solution[k], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] == 2:
                # Some files may store [t, I].
                t_src, i_src = arr[0].reshape(-1), arr[1].reshape(-1)
                return np.interp(t_data, t_src, i_src), f"solution.{k}[2xN]"
            if arr.ndim == 2 and arr.shape[1] == 2:
                t_src, i_src = arr[:, 0].reshape(-1), arr[:, 1].reshape(-1)
                return np.interp(t_data, t_src, i_src), f"solution.{k}[Nx2]"
            current = arr.reshape(-1)
            current_key = k
            break
    if current is None:
        raise KeyError(f"Could not find current in solution.npz. Available keys: {list(solution.keys())}")

    t_src = None
    for k in time_keys:
        if k in solution:
            arr = np.asarray(solution[k], dtype=np.float64).reshape(-1)
            if arr.size == current.size:
                t_src = arr
                break
    if current.size == t_data.size:
        return current, f"solution.{current_key} aligned_to_data_t"
    if t_src is not None:
        return np.interp(t_data, t_src, current), f"solution.{current_key} interpolated_from_{k}"
    raise ValueError(
        f"Current length {current.size} does not match data time length {t_data.size}, "
        "and no compatible time vector was found."
    )


def _group_spherical_average(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(data["x_train"], dtype=np.float64)
    y = np.asarray(data["y_train"], dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("Expected x_train with columns [t, r].")
    t_all = x[:, 0]
    r_all = x[:, 1]
    t_unique = np.unique(t_all)
    avg = []
    Rs = float(np.nanmax(r_all))
    if not np.isfinite(Rs) or Rs <= 0:
        raise ValueError("Invalid particle radius inferred from data file.")
    for t in t_unique:
        mask = np.isclose(t_all, t, rtol=0.0, atol=1.0e-9)
        r = r_all[mask]
        c = y[mask]
        order = np.argsort(r)
        r = r[order]
        c = c[order]
        integrand = c * r * r
        try:
            integ = np.trapezoid(integrand, r)
        except AttributeError:
            integ = np.trapz(integrand, r)
        avg.append(3.0 * integ / (Rs ** 3))
    return t_unique.astype(np.float64), np.asarray(avg, dtype=np.float64), Rs


def _gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if y.size < 3:
        return np.zeros_like(y)
    return np.gradient(y, x, edge_order=2)


def _stats(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    return {
        "mae": float(np.nanmean(np.abs(v))),
        "rmse": float(np.sqrt(np.nanmean(v * v))),
        "maxabs": float(np.nanmax(np.abs(v))),
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) <= 0 or np.nanstd(b[mask]) <= 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _build_params(soft_label_dir: Path, ocp_dir: Path | None) -> dict[str, Any]:
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_label_dir.resolve())
    if ocp_dir is not None:
        os.environ["ASSB_OCP_DIR"] = str(ocp_dir.resolve())
    from util.spm_assb_train_discharge import makeParams  # type: ignore
    params = makeParams()
    summary = _load_json(soft_label_dir / "soft_label_summary.json")
    # Apply summary overrides that matter for provenance/debugging. Flux uses geometry,
    # F and eps, but keeping these fields visible helps catch stale config issues.
    for key in [
        "R_ohm_eff", "voltage_alignment_offset_V", "theta_c_bottom", "theta_c_top",
        "csanmax", "cscamax", "Rs_a", "Rs_c", "eps_s_a", "eps_s_c", "V_a", "V_c",
    ]:
        if key in summary:
            try:
                params[key] = np.float64(summary[key])
            except Exception:
                params[key] = summary[key]
    params["V_a"] = np.float64(params.get("V_a", params["A_a"] * params["L_a"]))
    params["V_c"] = np.float64(params.get("V_c", params["A_c"] * params["L_c"]))
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_label_dir", default="Data/assb_soft_labels_cycle5_v3")
    ap.add_argument("--ocp_dir", default=None)
    ap.add_argument("--output_dir", default="Eval_cycle5_integration_mass_audit")
    args = ap.parse_args()

    repo = _repo_root()
    _insert_repo_paths(repo)

    soft_dir = Path(args.soft_label_dir)
    if not soft_dir.is_absolute():
        soft_dir = repo / soft_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ocp_dir = Path(args.ocp_dir) if args.ocp_dir else None

    from integration_spm.spm_int_assb_cycle import surface_flux_from_current  # type: ignore

    params = _build_params(soft_dir, ocp_dir)
    solution = _load_npz(soft_dir / "solution.npz")
    data_a = _load_npz(soft_dir / "data_cs_a.npz")
    data_c = _load_npz(soft_dir / "data_cs_c.npz")

    t_a, cbar_a, Rs_a_data = _group_spherical_average(data_a)
    t_c, cbar_c, Rs_c_data = _group_spherical_average(data_c)
    I_a, I_src_a = _find_current(solution, t_a)
    I_c, I_src_c = _find_current(solution, t_c)

    j_a = np.array([surface_flux_from_current(float(i), params)[0] for i in I_a], dtype=np.float64)
    j_c = np.array([surface_flux_from_current(float(i), params)[1] for i in I_c], dtype=np.float64)
    dcbar_a_dt = _gradient(cbar_a, t_a)
    dcbar_c_dt = _gradient(cbar_c, t_c)

    Rs_a = float(params.get("Rs_a", Rs_a_data))
    Rs_c = float(params.get("Rs_c", Rs_c_data))

    audit = {
        "soft_label_dir": str(soft_dir),
        "generator_flux_function": "integration_spm.spm_int_assb_cycle.surface_flux_from_current",
        "current_source_a": I_src_a,
        "current_source_c": I_src_c,
        "params": {
            "Rs_a": _float(params.get("Rs_a")),
            "Rs_c": _float(params.get("Rs_c")),
            "eps_s_a": _float(params.get("eps_s_a")),
            "eps_s_c": _float(params.get("eps_s_c")),
            "V_a": _float(params.get("V_a")),
            "V_c": _float(params.get("V_c")),
            "F": _float(params.get("F")),
            "R_ohm_eff": _float(params.get("R_ohm_eff")),
            "voltage_alignment_offset_V": _float(params.get("voltage_alignment_offset_V")),
            "theta_c_bottom": _float(params.get("theta_c_bottom")),
            "theta_c_top": _float(params.get("theta_c_top")),
        },
        "anode": {},
        "cathode": {},
    }

    # With D dc/dr|R = -J, the expected spherical-average equation is
    # d<c>/dt + 3J/R = 0. We also report the opposite sign as a diagnostic.
    for name, t, cbar, dcbar_dt, j, Rs in [
        ("anode", t_a, cbar_a, dcbar_a_dt, j_a, Rs_a),
        ("cathode", t_c, cbar_c, dcbar_c_dt, j_c, Rs_c),
    ]:
        source = -3.0 * j / Rs
        res_plus = dcbar_dt + 3.0 * j / Rs
        res_minus = dcbar_dt - 3.0 * j / Rs
        audit[name] = {
            "n_time": int(t.size),
            "t_min": float(np.nanmin(t)),
            "t_max": float(np.nanmax(t)),
            "cbar_min": float(np.nanmin(cbar)),
            "cbar_max": float(np.nanmax(cbar)),
            "dcbar_dt_stats": _stats(dcbar_dt),
            "minus_3J_over_R_stats": _stats(source),
            "corr_dcbar_dt_with_minus_3J_over_R": _corr(dcbar_dt, source),
            "residual_expected_plus_stats": _stats(res_plus),
            "residual_opposite_minus_stats": _stats(res_minus),
            "preferred_sign_by_rmse": "expected_plus" if _stats(res_plus)["rmse"] <= _stats(res_minus)["rmse"] else "opposite_minus",
            "rmse_ratio_expected_over_opposite": float(_stats(res_plus)["rmse"] / max(_stats(res_minus)["rmse"], 1.0e-300)),
        }
        csv = np.column_stack([t, I_a if name == "anode" else I_c, cbar, dcbar_dt, j, source, res_plus, res_minus])
        np.savetxt(
            out_dir / f"{name}_mass_balance_timeseries.csv",
            csv,
            delimiter=",",
            header="t_s,I_A,cbar,dcbar_dt,J,minus_3J_over_R,res_expected_plus,res_opposite_minus",
            comments="",
        )

    with (out_dir / "integration_mass_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("ASSB cycle5 integration mass-balance audit")
    lines.append("================================================")
    lines.append(f"soft_label_dir: {soft_dir}")
    lines.append("flux function: integration_spm.spm_int_assb_cycle.surface_flux_from_current")
    lines.append("")
    for name in ("anode", "cathode"):
        d = audit[name]
        lines.append(f"[{name}]")
        lines.append(f"  cbar range: {d['cbar_min']:.8g} to {d['cbar_max']:.8g}")
        lines.append(f"  corr dcbar_dt vs -3J/R: {d['corr_dcbar_dt_with_minus_3J_over_R']:.6g}")
        lines.append(f"  expected plus residual RMSE: {d['residual_expected_plus_stats']['rmse']:.8g}")
        lines.append(f"  opposite minus residual RMSE: {d['residual_opposite_minus_stats']['rmse']:.8g}")
        lines.append(f"  preferred sign by RMSE: {d['preferred_sign_by_rmse']}")
        lines.append(f"  rmse expected/opposite: {d['rmse_ratio_expected_over_opposite']:.6g}")
        lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("expected_plus means d<c>/dt + 3J/R is smaller, consistent with the generator boundary convention D dc/dr = -J.")
    lines.append("opposite_minus means the sign used by the generator-derived current and the soft-label average trend disagree and should be inspected before more training.")
    txt = "\n".join(lines)
    (out_dir / "integration_mass_audit.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
