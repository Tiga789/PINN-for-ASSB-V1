#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnose common-mode and differential potential bias from an EvalFin npz."""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


def _metrics(err: np.ndarray) -> dict:
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias_mean": None, "std": None}
    return {
        "n": int(err.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias_mean": float(np.mean(err)),
        "std": float(np.std(err)),
    }


def _find_eval_npz(eval_dir: Path, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = sorted(eval_dir.glob("eval_sampled_arrays*.npz"))
    if not candidates:
        candidates = [Path(p) for p in sorted(glob.glob(str(eval_dir / "**" / "eval_sampled_arrays*.npz"), recursive=True))]
    if not candidates:
        raise FileNotFoundError(f"No eval_sampled_arrays*.npz found under {eval_dir}")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--eval_npz", default=None)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_npz = _find_eval_npz(eval_dir, args.eval_npz)
    out_dir = Path(args.output_dir) if args.output_dir else eval_dir / "potential_common_mode_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(eval_npz, allow_pickle=False) as z:
        t = np.asarray(z["t_potential"], dtype=np.float64).reshape(-1)
        cycle = np.asarray(z["cycle_id_potential"]).reshape(-1).astype(int)
        phie_true = np.asarray(z["phie_true"], dtype=np.float64).reshape(-1)
        phie_pred = np.asarray(z["phie_pred"], dtype=np.float64).reshape(-1)
        phis_true = np.asarray(z["phis_c_true"], dtype=np.float64).reshape(-1)
        phis_pred = np.asarray(z["phis_c_pred"], dtype=np.float64).reshape(-1)

    err_phie = phie_pred - phie_true
    err_phis = phis_pred - phis_true
    cm_true = 0.5 * (phie_true + phis_true)
    cm_pred = 0.5 * (phie_pred + phis_pred)
    diff_true = phis_true - phie_true
    diff_pred = phis_pred - phie_pred
    err_cm = cm_pred - cm_true
    err_diff = diff_pred - diff_true

    global_summary = {
        "eval_npz": str(eval_npz),
        "interpretation_hint": "If phie and phis_c have similar bias but the phis_c-phie differential is accurate, the main issue is a common-mode/gauge-like potential offset rather than a wrong current-dependent shape.",
        "phie_error": _metrics(err_phie),
        "phis_c_error": _metrics(err_phis),
        "common_mode_error": _metrics(err_cm),
        "differential_phis_minus_phie_error": _metrics(err_diff),
    }
    with open(out_dir / "potential_common_mode_global.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

    rows = []
    for cy in sorted(set(int(c) for c in cycle)):
        m = cycle == cy
        row = {"cycle_id": cy, "n": int(np.count_nonzero(m))}
        for name, arr in [
            ("phie", err_phie[m]),
            ("phis_c", err_phis[m]),
            ("common_mode", err_cm[m]),
            ("differential", err_diff[m]),
        ]:
            met = _metrics(arr)
            row[f"{name}_mae"] = met["mae"]
            row[f"{name}_rmse"] = met["rmse"]
            row[f"{name}_bias_mean"] = met["bias_mean"]
        rows.append(row)

    if rows:
        with open(out_dir / "potential_common_mode_by_cycle.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("Potential common-mode diagnostic written to:", out_dir)
    print(json.dumps(global_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
