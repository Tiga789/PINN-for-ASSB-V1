# -*- coding: utf-8 -*-
r"""
Diagnose whether cs_c/cs_a radial mass averaging is consistent with stored j_c/j_a.

Usage example on Windows PowerShell:
  D:\Anaconda\envs\torchgpu\python.exe .\diagnose_cbar_mass_weights.py --solution "C:\\Users\\Tiga_QJW\\Desktop\\ASSB_Scheme_V1\\assb_soft_lable_cycle5-522_v1\\solution.npz" --electrode c
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def pick(d: np.lib.npyio.NpzFile, names: Iterable[str], required: bool = True) -> Tuple[Optional[np.ndarray], Optional[str]]:
    for n in names:
        if n in d.files:
            return d[n], n
    if required:
        raise KeyError(f"Cannot find any of {list(names)}. Available={list(d.files)}")
    return None, None


def as_Nt_Nr(arr: np.ndarray, Nt: int, Nr: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape == (Nt, Nr):
        return arr.astype(np.float64)
    if arr.shape == (Nr, Nt):
        return arr.T.astype(np.float64)
    if arr.ndim == 1 and arr.size == Nt * Nr:
        return arr.reshape(Nt, Nr).astype(np.float64)
    raise ValueError(f"Cannot reshape {name}: {arr.shape}; expected ({Nt}, {Nr}) or ({Nr}, {Nt})")


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


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    e = pred - true
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "bias": float(np.mean(e)),
        "corr": float(np.corrcoef(true, pred)[0, 1]) if np.std(true) > 0 and np.std(pred) > 0 else float("nan"),
    }


def evaluate_weights(label: str, w: np.ndarray, cs: np.ndarray, inc_right: np.ndarray, I: Optional[np.ndarray] = None):
    w = np.asarray(w, dtype=np.float64)
    w = w / np.sum(w)

    cbar = cs @ w
    dc = np.diff(cbar)
    pred = np.empty_like(cbar)
    pred[0] = cbar[0]
    pred[1:] = cbar[0] + np.cumsum(inc_right)

    resid = dc - inc_right
    m = metrics(cbar, pred)

    print(f"\n=== {label} ===")
    print("w_sum/min/max =", float(w.sum()), float(w.min()), float(w.max()))
    print(
        f"cbar_vs_j_right: MAE={m['MAE']:.8g} RMSE={m['RMSE']:.8g} "
        f"corr={m['corr']:.8g} bias={m['bias']:.8g}"
    )
    print(
        "interval_resid:",
        "sum=", float(np.sum(resid)),
        "MAE=", float(np.mean(np.abs(resid))),
        "RMSE=", float(np.sqrt(np.mean(resid**2))),
        "maxabs=", float(np.max(np.abs(resid))),
    )

    if I is not None:
        I = np.asarray(I, dtype=np.float64).reshape(-1)
        eps = 1e-12
        I_change = np.abs(np.diff(I)) > eps
        rest = (np.abs(I[:-1]) <= eps) | (np.abs(I[1:]) <= eps)
        pos_same = (~I_change) & (I[:-1] > eps) & (I[1:] > eps)
        neg_same = (~I_change) & (I[:-1] < -eps) & (I[1:] < -eps)

        for name, mask in [
            ("positive same I", pos_same),
            ("negative same I", neg_same),
            ("I change", I_change),
            ("rest involved", rest),
        ]:
            rr = resid[mask]
            if rr.size:
                print(
                    f"{name:18s} n={rr.size:8d} "
                    f"sum={np.sum(rr): .8g} "
                    f"mean={np.mean(rr): .8g} "
                    f"MAE={np.mean(np.abs(rr)): .8g} "
                    f"maxabs={np.max(np.abs(rr)): .8g}"
                )

    return cbar, pred, resid, w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--solution",
        default=r"C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1\solution.npz",
        help="Path to solution.npz",
    )
    ap.add_argument("--electrode", choices=["a", "c"], default="c", help="a=negative/anode side, c=positive/cathode side")
    args = ap.parse_args()

    soft_npz = Path(args.solution)
    if not soft_npz.exists():
        raise FileNotFoundError(f"solution.npz not found: {soft_npz}")

    sd = np.load(soft_npz, allow_pickle=True)

    t, t_name = pick(sd, ["t_global_s", "t", "time_s"])
    assert t is not None and t_name is not None
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    Nt = len(t)
    dt = np.diff(t)

    if args.electrode == "c":
        r, r_name = pick(sd, ["r_c", "r_grid_c", "r_cathode"])
        j, j_name = pick(sd, ["j_c", "J_c", "flux_c"])
        cs, cs_name = pick(sd, ["cs_c", "c_s_c", "cs_c_full", "cathode_cs", "c_s_cathode"])
        sign_factor = -3.0  # dcbar/dt = -3*j_c/R under current convention used in previous checks
        label = "cathode / positive electrode"
    else:
        r, r_name = pick(sd, ["r_a", "r_grid_a", "r_anode"])
        j, j_name = pick(sd, ["j_a", "J_a", "flux_a"])
        cs, cs_name = pick(sd, ["cs_a", "c_s_a", "cs_a_full", "anode_cs", "c_s_anode"])
        sign_factor = -3.0  # boundary convention is D dc/dr = -j, so dcbar/dt = -3*j/R
        label = "anode / negative electrode"

    assert r is not None and j is not None and cs is not None and r_name is not None and j_name is not None and cs_name is not None
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    j = np.asarray(j, dtype=np.float64).reshape(-1)
    Nr = len(r)
    R = float(r[-1])
    cs = as_Nt_Nr(cs, Nt, Nr, cs_name)

    I, I_name = pick(sd, ["I_profile", "I", "current_A"], required=False)
    I_arr = np.asarray(I, dtype=np.float64).reshape(-1) if I is not None else None

    rate = sign_factor * j / R
    inc_right = rate[1:] * dt

    print("solution:", soft_npz)
    print("electrode:", label)
    print("arrays:", {"t": t_name, "r": r_name, "j": j_name, "cs": cs_name, "I": I_name})
    print("Nt, Nr:", Nt, Nr)
    print("t range:", float(t[0]), float(t[-1]))
    print("dt min/median/max:", float(dt.min()), float(np.median(dt)), float(dt.max()))
    print("R:", R)
    print("j range:", float(j.min()), float(j.max()))

    w_trapz = weights_spherical_trapz(r)
    w_fv = weights_finite_volume_shell(r)

    evaluate_weights("spherical trapz weights", w_trapz, cs, inc_right, I_arr)
    evaluate_weights("finite-volume shell weights", w_fv, cs, inc_right, I_arr)

    print("\n=== fitting mass weights from dc = inc_right ===")
    A = np.diff(cs, axis=0)
    b = inc_right.copy()
    mask = np.all(np.isfinite(A), axis=1) & np.isfinite(b)
    A = A[mask]
    b = b[mask]

    ATA = A.T @ A
    ATb = A.T @ b
    ones = np.ones(Nr, dtype=np.float64)

    K = np.zeros((Nr + 1, Nr + 1), dtype=np.float64)
    K[:Nr, :Nr] = ATA
    K[:Nr, Nr] = ones
    K[Nr, :Nr] = ones
    rhs = np.zeros(Nr + 1, dtype=np.float64)
    rhs[:Nr] = ATb
    rhs[Nr] = 1.0

    # The KKT matrix can be singular/ill-conditioned for smooth radial profiles.
    # That is a diagnostic result, not necessarily a data-loading error.  Use
    # lstsq fallback so the script can continue and report whether any mass
    # weights can explain dc = -3*j/R.
    try:
        sol = np.linalg.solve(K, rhs)
        fit_solver = "solve"
        fit_rank = Nr + 1
        fit_lstsq_resid = np.nan
    except np.linalg.LinAlgError as exc:
        sol, residuals, rank, svals = np.linalg.lstsq(K, rhs, rcond=1e-12)
        fit_solver = f"lstsq fallback after {type(exc).__name__}"
        fit_rank = int(rank)
        fit_lstsq_resid = float(np.linalg.norm(K @ sol - rhs))
        print("KKT solve failed; using np.linalg.lstsq fallback.")
        print("KKT rank / size =", fit_rank, "/", Nr + 1)
        print("KKT residual norm =", fit_lstsq_resid)
    w_eq = sol[:Nr]
    print("fit_eq solver =", fit_solver)
    print("fit_eq raw sum/min/max =", float(np.sum(w_eq)), float(np.min(w_eq)), float(np.max(w_eq)))
    evaluate_weights("fit weights, equality only", w_eq, cs, inc_right, I_arr)
    print("fit_eq negative_weight_count =", int(np.sum(w_eq < -1e-12)))
    print("fit_eq L1 distance to FV =", float(np.sum(np.abs(w_eq - w_fv))))
    print("fit_eq L1 distance to trapz =", float(np.sum(np.abs(w_eq - w_trapz))))

    if HAS_SCIPY:
        def obj(w: np.ndarray) -> float:
            return 0.5 * float(w @ (ATA @ w)) - float(ATb @ w)

        def jac(w: np.ndarray) -> np.ndarray:
            return ATA @ w - ATb

        cons = ({
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: ones,
        })

        res = minimize(
            obj,
            x0=w_fv.copy(),
            jac=jac,
            bounds=[(0.0, None)] * Nr,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-14, "disp": False},
        )

        print("\nnonnegative fit success:", bool(res.success), res.message)
        w_nn = res.x / np.sum(res.x)
        evaluate_weights("fit weights, nonnegative + sum=1", w_nn, cs, inc_right, I_arr)
        print("fit_nonneg L1 distance to FV =", float(np.sum(np.abs(w_nn - w_fv))))
        print("fit_nonneg L1 distance to trapz =", float(np.sum(np.abs(w_nn - w_trapz))))
    else:
        print("\nscipy not available; skipped nonnegative constrained fit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
