#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch/enrich soft_label_summary.json after spm_int_assb_cycle.py generation.

Recommended location:
    integration_spm/spm_int_assb_cycle_patch_summary.py

Why this is separate from the heavy generator:
    The current generator already produces usable v3 soft labels. This helper
    only adds provenance fields required by the training/evaluation loop, so it
    is safer than replacing the 1500-line integrator while debugging.

Example:
    D:\\Anaconda\\envs\\torchgpu\\python.exe integration_spm\\spm_int_assb_cycle_patch_summary.py ^
      --soft_label_dir Data\\assb_soft_labels_cycle5_v3 ^
      --ocp_dir C:\\Users\\Tiga_QJW\\Desktop\\ASSB_Scheme_V1\\ocp_estimation_outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "integration_spm" else here.parent


def _add_paths(root: Path) -> None:
    for p in [root, root / "util"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def _npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {"n": int(x.size), "min": float(np.min(x)), "max": float(np.max(x)), "mean": float(np.mean(x)), "std": float(np.std(x))}


def _first_last(arr: np.ndarray, n: int = 5):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {"first": arr[:n].tolist(), "last": arr[-n:].tolist() if arr.size else []}


def _soft_dataset_stats(soft_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    mapping = {
        "phie": "data_phie.npz",
        "phis_c": "data_phis_c.npz",
        "cs_a": "data_cs_a.npz",
        "cs_c": "data_cs_c.npz",
    }
    for name, fname in mapping.items():
        data = _npz(soft_dir / fname)
        if not data:
            out[name] = {"available": False}
            continue
        rep: Dict[str, Any] = {"available": True, "keys": list(data.keys())}
        if "x_train" in data:
            x = np.asarray(data["x_train"], dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            rep["x_train_shape"] = list(x.shape)
            rep["t_stats"] = _stats(x[:, 0])
            if x.shape[1] > 1:
                rep["r_stats"] = _stats(x[:, 1])
        if "y_train" in data:
            rep["y_train_shape"] = list(np.asarray(data["y_train"]).shape)
            rep["y_stats"] = _stats(data["y_train"])
        if "x_params_train" in data:
            rep["x_params_train_shape"] = list(np.asarray(data["x_params_train"]).shape)
        out[name] = rep
    return out


def _solution_stats(soft_dir: Path) -> Dict[str, Any]:
    sol = _npz(soft_dir / "solution.npz")
    if not sol:
        return {"available": False}
    out: Dict[str, Any] = {"available": True, "keys": list(sol.keys())}
    for key in ["t", "I_profile", "voltage_exp", "phis_c", "phis_c_raw", "phie", "theta_a_surf", "theta_c_surf"]:
        if key in sol:
            out[key] = _stats(sol[key])
            if key in {"t", "I_profile", "voltage_exp"}:
                out[f"{key}_head_tail"] = _first_last(sol[key])
    if "t" in sol:
        t = np.asarray(sol["t"], dtype=np.float64).reshape(-1)
        if t.size:
            out["t_start_s"] = float(t[0])
            out["t_end_s"] = float(t[-1])
            out["tmax_train_s"] = float(t[-1])
    if "I_profile" in sol and "t" in sol:
        t = np.asarray(sol["t"], dtype=np.float64).reshape(-1)
        i = np.asarray(sol["I_profile"], dtype=np.float64).reshape(-1)
        if t.size == i.size and t.size > 1:
            out["charge_capacity_Ah"] = float(np.trapezoid(np.clip(i, 0, None), t) / 3600.0)
            out["discharge_capacity_Ah"] = float(np.trapezoid(np.clip(-i, 0, None), t) / 3600.0)
            nz = i[np.abs(i) > 1.0e-12]
            out["current_ref_A"] = float(np.sign(nz[0]) * np.median(np.abs(nz))) if nz.size else 0.0
    return out


def _params_snapshot(root: Path, soft_dir: Path, summary_path: Path) -> Dict[str, Any]:
    _add_paths(root)
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_dir.resolve())
    try:
        from util.spm_assb_train_discharge import makeParams  # type: ignore
    except Exception:
        from spm_assb_train_discharge import makeParams  # type: ignore
    params = makeParams(summary_json=str(summary_path))
    keys = [
        "train_summary_json", "soft_label_dir", "current_profile_source", "rescale_T", "rescale_R",
        "tmax", "I_discharge", "I_app", "C", "T", "F", "R", "A_a", "A_c", "L_a", "L_c",
        "V_a", "V_c", "Rs_a", "Rs_c", "eps_s_a", "eps_s_c", "csanmax", "cscamax",
        "theta_a0", "theta_c0", "cs_a0", "cs_c0", "R_ohm_eff", "voltage_alignment_offset_V",
        "theta_c_bottom", "theta_c_top", "cs_rescale_mode"
    ]
    snap = {k: _jsonable(params.get(k)) for k in keys if k in params}
    if "time_profile" in params and "current_profile_A" in params:
        t = np.asarray(params["time_profile"], dtype=np.float64).reshape(-1)
        i = np.asarray(params["current_profile_A"], dtype=np.float64).reshape(-1)
        snap["time_profile_head_tail"] = _first_last(t)
        snap["current_profile_A_head_tail"] = _first_last(i)
        snap["time_profile_n"] = int(t.size)
        snap["current_profile_A_stats"] = _stats(i)
    return snap


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch ASSB soft_label_summary.json with provenance fields")
    parser.add_argument("--repo_root", default=None)
    parser.add_argument("--soft_label_dir", default="Data/assb_soft_labels_cycle5_v3")
    parser.add_argument("--ocp_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve() if args.repo_root else _repo_root().resolve()
    soft_dir = Path(args.soft_label_dir)
    if not soft_dir.is_absolute():
        soft_dir = root / soft_dir
    soft_dir = soft_dir.resolve()
    if args.ocp_dir:
        os.environ["ASSB_OCP_DIR"] = str(Path(args.ocp_dir).resolve())
    os.environ["ASSB_SOFT_LABEL_DIR"] = str(soft_dir)

    summary_path = soft_dir / "soft_label_summary.json"
    summary = _load_json(summary_path)
    sol_stats = _solution_stats(soft_dir)
    data_stats = _soft_dataset_stats(soft_dir)
    params = _params_snapshot(root, soft_dir, summary_path)

    patched = dict(summary)
    patched.update({
        "soft_label_dir": str(soft_dir),
        "soft_label_summary_json": str(summary_path),
        "repo_root_at_patch_time": str(root),
        "ocp_source_dir": str(Path(args.ocp_dir).resolve()) if args.ocp_dir else os.environ.get("ASSB_OCP_DIR", ""),
        "provenance_patch_applied": True,
        "provenance_patch_script": "integration_spm/spm_int_assb_cycle_patch_summary.py",
        "solution_stats": sol_stats,
        "dataset_stats": data_stats,
        "training_param_snapshot": params,
    })
    # Top-level convenience fields consumed by makeParams/evaluator.
    for k in [
        "tmax_train_s", "current_ref_A", "charge_capacity_Ah", "discharge_capacity_Ah",
    ]:
        if k in sol_stats:
            patched[k] = sol_stats[k]
    for k in [
        "R_ohm_eff", "voltage_alignment_offset_V", "theta_c_bottom", "theta_c_top",
        "csanmax", "cscamax", "theta_a0", "theta_c0", "Rs_a", "Rs_c", "eps_s_a", "eps_s_c", "V_a", "V_c",
    ]:
        if k in params:
            patched[k] = params[k]

    out_path = soft_dir / "soft_label_summary.provenance_preview.json" if args.dry_run else summary_path
    _write_json(out_path, patched)
    print(f"Patched summary written to: {out_path}")
    print(f"soft_label_dir: {soft_dir}")
    print(f"train_summary_json: {params.get('train_summary_json')}")
    print(f"current_profile_source: {params.get('current_profile_source')}")


if __name__ == "__main__":
    main()
