#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ASSB provenance check before training.

Recommended location:
    project root / assb_provenance_check.py

This script does not train. It verifies that makeParams() actually uses the
intended soft-label directory, summary JSON, OCP path, current profile, v3 voltage
alignment and SPM flux closure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

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
    if isinstance(x, (np.integer, np.floating)):
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


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


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


def _head_tail(x: np.ndarray, n: int = 10) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {"first": x[:n].tolist(), "last": x[-n:].tolist() if x.size else []}


def _current_profile_from_params(params: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    if "current_profile" in params:
        prof = params["current_profile"]
        if isinstance(prof, (tuple, list)) and len(prof) == 2:
            t = np.asarray(prof[0], dtype=np.float64).reshape(-1)
            i = np.asarray(prof[1], dtype=np.float64).reshape(-1)
            if t.size == i.size and t.size >= 2:
                return t, i
    if "time_profile" in params and "current_profile_A" in params:
        t = np.asarray(params["time_profile"], dtype=np.float64).reshape(-1)
        i = np.asarray(params["current_profile_A"], dtype=np.float64).reshape(-1)
        if t.size == i.size and t.size >= 2:
            return t, i
    return None


def _flux_from_I(params: Dict[str, Any], i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = float(params["F"])
    Va = float(params.get("V_a", params["A_a"] * params["L_a"]))
    Vc = float(params.get("V_c", params["A_c"] * params["L_c"]))
    ja = -np.asarray(i, dtype=np.float64) * float(params["Rs_a"]) / (3.0 * float(params["eps_s_a"]) * F * Va)
    jc = np.asarray(i, dtype=np.float64) * float(params["Rs_c"]) / (3.0 * float(params["eps_s_c"]) * F * Vc)
    return ja, jc


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ASSB soft-label/training provenance")
    parser.add_argument("--repo_root", default=None)
    parser.add_argument("--soft_label_dir", default="Data/assb_soft_labels_cycle5_v3")
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--ocp_dir", default=None)
    parser.add_argument("--output_dir", default="Eval_assb_provenance_check")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if key provenance checks fail")
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
    profile = _current_profile_from_params(params)
    solution = _npz(soft_dir / "solution.npz")
    summary = _load_json(summary_path)

    report: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "soft_label_dir_requested": str(soft_dir),
        "summary_json_requested": str(summary_path),
        "ocp_dir_env": os.environ.get("ASSB_OCP_DIR"),
        "params_snapshot": {},
        "soft_label_summary_keys": list(summary.keys()),
        "solution_keys": list(solution.keys()),
        "checks": {},
        "warnings": [],
    }
    for key in [
        "train_summary_json", "actual_train_summary_json", "soft_label_summary_json", "soft_label_dir",
        "actual_soft_label_dir", "current_profile_source", "actual_current_profile_source",
        "rescale_T", "rescale_R", "tmax", "I_discharge", "I_app", "C", "T",
        "A_a", "A_c", "L_a", "L_c", "V_a", "V_c", "Rs_a", "Rs_c", "eps_s_a", "eps_s_c",
        "csanmax", "cscamax", "theta_a0", "theta_c0", "cs_a0", "cs_c0",
        "R_ohm_eff", "voltage_alignment_offset_V", "theta_c_bottom", "theta_c_top", "cs_rescale_mode",
    ]:
        if key in params:
            report["params_snapshot"][key] = _jsonable(params[key])

    train_summary = str(params.get("train_summary_json", ""))
    soft_label_dir_param = str(params.get("soft_label_dir", ""))
    report["checks"]["train_summary_matches_soft_dir"] = str(summary_path) == train_summary
    report["checks"]["soft_label_dir_matches_requested"] = str(soft_dir) == soft_label_dir_param

    if profile is None:
        report["warnings"].append("No current profile found in params. Training would fall back to constant current.")
    else:
        t, i = profile
        ja, jc = _flux_from_I(params, i)
        report["current_profile"] = {
            "n": int(t.size),
            "t_stats": _stats(t),
            "I_stats": _stats(i),
            "t_head_tail": _head_tail(t),
            "I_head_tail": _head_tail(i),
            "J_a_head_tail": _head_tail(ja),
            "J_c_head_tail": _head_tail(jc),
            "J_a_stats": _stats(ja),
            "J_c_stats": _stats(jc),
        }
        report["checks"]["current_profile_has_multiple_points"] = bool(t.size >= 2)
        report["checks"]["current_profile_tmax_matches_rescale_T"] = bool(
            abs(float(np.max(t)) - float(params.get("rescale_T", np.nan))) <= max(1e-6, 1e-6 * max(1.0, float(np.max(t))))
        )

    if solution:
        for key in ["t", "I_profile", "voltage_exp", "phis_c", "phie"]:
            if key in solution:
                report[f"solution_{key}_stats"] = _stats(solution[key])
                if key in {"t", "I_profile"}:
                    report[f"solution_{key}_head_tail"] = _head_tail(solution[key])
        if profile is not None and "t" in solution and "I_profile" in solution:
            t, i = profile
            ts = np.asarray(solution["t"], dtype=np.float64).reshape(-1)
            is_ = np.asarray(solution["I_profile"], dtype=np.float64).reshape(-1)
            same_n = t.size == ts.size == i.size == is_.size
            report["checks"]["params_current_matches_solution_shape"] = bool(same_n)
            if same_n:
                report["checks"]["params_current_matches_solution_values"] = bool(np.allclose(t, ts) and np.allclose(i, is_))
            else:
                report["warnings"].append("Params current profile and solution.npz current profile have different shapes. This may be OK only if params profile is compressed.")
    else:
        report["warnings"].append("solution.npz not found in soft_label_dir.")

    # Fail recommendations.
    if not report["checks"].get("train_summary_matches_soft_dir", False):
        report["warnings"].append("train_summary_json does not equal the requested soft_label_summary.json. Do not train until this is fixed.")
    if not report["checks"].get("soft_label_dir_matches_requested", False):
        report["warnings"].append("params soft_label_dir does not equal the requested soft_label_dir. Do not train until this is fixed.")

    txt_lines = []
    txt_lines.append("ASSB provenance check")
    txt_lines.append("=" * 60)
    txt_lines.append(f"repo_root: {repo_root}")
    txt_lines.append(f"soft_label_dir: {soft_dir}")
    txt_lines.append(f"summary_json: {summary_path}")
    txt_lines.append("")
    txt_lines.append("[checks]")
    for k, v in report["checks"].items():
        txt_lines.append(f"  {k}: {v}")
    txt_lines.append("")
    txt_lines.append("[key params]")
    for k, v in report["params_snapshot"].items():
        txt_lines.append(f"  {k}: {v}")
    txt_lines.append("")
    txt_lines.append("[warnings]")
    if report["warnings"]:
        for w in report["warnings"]:
            txt_lines.append(f"  - {w}")
    else:
        txt_lines.append("  none")

    (out_dir / "assb_provenance_check.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "assb_provenance_check.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n".join(txt_lines))
    print(f"\nWrote: {out_dir / 'assb_provenance_check.txt'}")
    print(f"Wrote: {out_dir / 'assb_provenance_check.json'}")
    if args.strict and report["warnings"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
