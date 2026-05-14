#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a compact soft-label data folder for ModelFin_105 potential-gauge anchoring.

This script creates the four NPZ files expected by util/init_pinn.py when main.py
is launched with -df/--dataFolder:

  data_phie.npz
  data_phis_c.npz
  data_cs_a.npz
  data_cs_c.npz

ModelFin_105 uses only the potential data weights by default. The concentration
files are still generated for compatibility with the existing loader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _find_key(npz: np.lib.npyio.NpzFile, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    files = set(npz.files)
    lower = {k.lower(): k for k in npz.files}
    for cand in candidates:
        if cand in files:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"None of the keys were found: {tuple(candidates)}. Available keys: {npz.files}")
    return None


def _regular_sample_indices(n: int, n_data: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        raise ValueError("No rows available for sampling.")
    n_data = int(max(n_data, 1))
    if n >= n_data:
        return np.linspace(0, n - 1, n_data, dtype=np.int64)
    return rng.choice(n, size=n_data, replace=True).astype(np.int64)


def _save_npz(path: Path, x_train: np.ndarray, y_train: np.ndarray, x_params_train: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=np.asarray(x_train, dtype=np.float64),
        y_train=np.asarray(y_train, dtype=np.float64),
        x_params_train=np.asarray(x_params_train, dtype=np.float64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ModelFin_105 potential-gauge data NPZ files from ASSB soft labels.")
    parser.add_argument("--soft_label_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cycle_from", type=int, default=5)
    parser.add_argument("--cycle_to", type=int, default=20)
    parser.add_argument("--n_data", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--deg_i0_a", type=float, default=1.0)
    parser.add_argument("--deg_ds_c", type=float, default=1.0)
    args = parser.parse_args()

    soft_dir = Path(args.soft_label_dir)
    sol = soft_dir / "solution.npz"
    out_dir = Path(args.output_dir)
    if not sol.exists():
        raise FileNotFoundError(f"solution.npz not found: {sol}")

    rng = np.random.default_rng(int(args.seed))
    with np.load(sol, allow_pickle=False) as data:
        t_key = _find_key(data, ("t_global_s", "t", "time_s", "t_s", "time"))
        cycle_key = _find_key(data, ("cycle_id", "cycle", "cycle_index"))
        phie_key = _find_key(data, ("phie", "phi_e"))
        phis_key = _find_key(data, ("phis_c", "phi_s_c", "phis"))
        csa_key = _find_key(data, ("cs_a", "csa", "cs_an"))
        csc_key = _find_key(data, ("cs_c", "csc", "cs_ca"))
        ra_key = _find_key(data, ("r_a", "ra", "r_an"))
        rc_key = _find_key(data, ("r_c", "rc", "r_ca"))

        t_all = np.asarray(data[t_key], dtype=np.float64).reshape(-1)
        cycle_all = np.asarray(data[cycle_key]).reshape(-1).astype(int)
        mask = (cycle_all >= int(args.cycle_from)) & (cycle_all <= int(args.cycle_to))
        idx_all = np.where(mask)[0]
        if idx_all.size < 2:
            raise ValueError(f"No data found for cycles {args.cycle_from}-{args.cycle_to} in {sol}")

        # Potential data: one input coordinate, physical time in seconds.
        tidx = idx_all[_regular_sample_indices(idx_all.size, int(args.n_data), rng)]
        x_phie = t_all[tidx].reshape(-1, 1)
        y_phie = np.asarray(data[phie_key], dtype=np.float64).reshape(-1)[tidx].reshape(-1, 1)
        x_phis = t_all[tidx].reshape(-1, 1)
        y_phis = np.asarray(data[phis_key], dtype=np.float64).reshape(-1)[tidx].reshape(-1, 1)

        # Concentration data: two input coordinates, physical time and radius.
        cs_a = np.asarray(data[csa_key], dtype=np.float64)
        cs_c = np.asarray(data[csc_key], dtype=np.float64)
        r_a = np.asarray(data[ra_key], dtype=np.float64).reshape(-1)
        r_c = np.asarray(data[rc_key], dtype=np.float64).reshape(-1)
        if cs_a.ndim != 2 or cs_c.ndim != 2:
            raise ValueError(f"Expected cs_a/cs_c rank-2 arrays; got {cs_a.shape} and {cs_c.shape}")
        if cs_a.shape[0] != t_all.size or cs_c.shape[0] != t_all.size:
            raise ValueError("cs_a/cs_c first dimension must match the time vector length.")
        if cs_a.shape[1] != r_a.size or cs_c.shape[1] != r_c.size:
            raise ValueError("cs_a/cs_c radial dimension must match r_a/r_c length.")

        csa_tidx = idx_all[rng.choice(idx_all.size, size=int(args.n_data), replace=idx_all.size < int(args.n_data))]
        csa_ridx = rng.integers(0, r_a.size, size=int(args.n_data))
        csc_tidx = idx_all[rng.choice(idx_all.size, size=int(args.n_data), replace=idx_all.size < int(args.n_data))]
        csc_ridx = rng.integers(0, r_c.size, size=int(args.n_data))
        x_csa = np.column_stack([t_all[csa_tidx], r_a[csa_ridx]])
        y_csa = cs_a[csa_tidx, csa_ridx].reshape(-1, 1)
        x_csc = np.column_stack([t_all[csc_tidx], r_c[csc_ridx]])
        y_csc = cs_c[csc_tidx, csc_ridx].reshape(-1, 1)

    x_params = np.column_stack([
        np.full(int(args.n_data), float(args.deg_i0_a), dtype=np.float64),
        np.full(int(args.n_data), float(args.deg_ds_c), dtype=np.float64),
    ])

    _save_npz(out_dir / "data_phie.npz", x_phie, y_phie, x_params)
    _save_npz(out_dir / "data_phis_c.npz", x_phis, y_phis, x_params)
    _save_npz(out_dir / "data_cs_a.npz", x_csa, y_csa, x_params)
    _save_npz(out_dir / "data_cs_c.npz", x_csc, y_csc, x_params)

    summary = {
        "script": Path(__file__).name,
        "soft_label_dir": str(soft_dir),
        "solution_npz": str(sol),
        "output_dir": str(out_dir),
        "cycle_from": int(args.cycle_from),
        "cycle_to": int(args.cycle_to),
        "n_data_per_variable": int(args.n_data),
        "seed": int(args.seed),
        "deg_i0_a": float(args.deg_i0_a),
        "deg_ds_c": float(args.deg_ds_c),
        "note": "Potential data are used to anchor the phie/phis_c common-mode bias in ModelFin_105. Concentration data files are generated for loader compatibility; their weights are zero in the default ID105 input.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Built ModelFin_105 data folder:", out_dir)
    print("  data_phie.npz   ", x_phie.shape, y_phie.shape)
    print("  data_phis_c.npz ", x_phis.shape, y_phis.shape)
    print("  data_cs_a.npz   ", x_csa.shape, y_csa.shape)
    print("  data_cs_c.npz   ", x_csc.shape, y_csc.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
