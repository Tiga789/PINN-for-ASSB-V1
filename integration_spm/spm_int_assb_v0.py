#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Current-profile aware SPM integration utility for ASSB v0.

GPU patch notes
---------------
This version keeps the original ASSB SPM logic, but adds an optional torch
backend for the most expensive implicit linear solves.

What is actually moved to GPU when backend='cuda' or backend='auto' with CUDA:
- tridiagonal matrix assembly for the implicit step
- right-hand-side vector assembly for the implicit step
- torch.linalg.solve for anode/cathode concentration updates

What still stays on CPU:
- thermo / OCP / i0 helper functions from thermo_assb.py
- pandas / file IO / cycle orchestration
- most scalar bookkeeping in the outer loop

So this is a partial GPU acceleration patch, not a full solver rewrite.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from prettyPlot.progressBar import print_progress_bar
except Exception:  # pragma: no cover
    def print_progress_bar(iteration, total, prefix="", suffix="", length=20):
        if total <= 0:
            return
        if iteration == 0 or iteration == total or iteration % max(1, total // 10) == 0:
            print(f"{prefix} {iteration}/{total} {suffix}")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_scalar(x: Any) -> np.float64:
    arr = _to_numpy(x).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot convert empty value to scalar.")
    return np.float64(arr[0])


def _resolve_backend(backend: str = "auto", device: str | None = None) -> tuple[str, str | None]:
    backend = (backend or "auto").lower()
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "cpu":
        return "cpu", None
    if torch is None:
        if backend == "cuda":
            raise RuntimeError("backend='cuda' 但未安装 torch。")
        return "cpu", None
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("backend='cuda' 但 torch.cuda.is_available() 为 False。")
        return "cuda", device or "cuda:0"
    if torch.cuda.is_available():
        return "cuda", device or "cuda:0"
    return "cpu", None


def _torch_tensor(x: Any, device: str):
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def _grad1d_torch(y, dr: float):
    n = int(y.shape[0])
    g = torch.empty_like(y)
    if n == 1:
        g[0] = 0.0
        return g
    if n == 2:
        val = (y[1] - y[0]) / dr
        g[0] = val
        g[1] = val
        return g
    g[1:-1] = (y[2:] - y[:-2]) / (2.0 * dr)
    g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dr)
    g[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * dr)
    return g


def get_r_domain(n_r, params):
    r_a = np.linspace(0, params["Rs_a"], n_r)
    dR_a = params["Rs_a"] / (n_r - 1)
    r_c = np.linspace(0, params["Rs_c"], n_r)
    dR_c = params["Rs_c"] / (n_r - 1)
    return {"n_r": n_r, "r_a": r_a, "dR_a": dR_a, "r_c": r_c, "dR_c": dR_c}


def get_t_domain(n_t, params, time_profile=None):
    if time_profile is None:
        t = np.linspace(0, params["tmax"], n_t)
    else:
        t = np.asarray(time_profile, dtype=np.float64).reshape(-1)
        if len(t) != n_t:
            raise ValueError("len(time_profile) must equal n_t.")
    if len(t) <= 1:
        dt = np.float64(0.0)
    else:
        dt = np.float64(np.median(np.diff(t)))
    return {"t": t, "dt": dt, "n_t": n_t}


def get_nt_from_dt(dt, params):
    return int(round(params["tmax"] / dt)) + 1


def get_expl_nt(r_dom, params, deg_ds_c):
    r_a = r_dom["r_a"]
    dR_a = r_dom["dR_a"]
    dR_c = r_dom["dR_c"]
    mindR = min(dR_a, dR_c)
    Ds_ave = 0.5 * _to_scalar(params["D_s_a"](params["T"], params["R"])) + 0.5 * (
        np.mean(
            _to_numpy(
                params["D_s_c"](
                    params["cs_c0"],
                    params["T"],
                    params["R"],
                    params["cscamax"],
                    deg_ds_c,
                )
            )
        )
    )
    dt_target = mindR**2 / (2 * max(Ds_ave, 1e-20))
    n_t = int(2 * params["tmax"] // dt_target)
    return max(n_t, 2)


def make_sim_config(t_dom, r_dom):
    return {**t_dom, **r_dom}


def init_arrays(n_t, n_r):
    phie = np.zeros(n_t, dtype=np.float64)
    phis_c = np.zeros(n_t, dtype=np.float64)
    cs_a = np.zeros((n_t, n_r), dtype=np.float64)
    cs_c = np.zeros((n_t, n_r), dtype=np.float64)
    Ds_a = np.zeros(n_r, dtype=np.float64)
    Ds_c = np.zeros(n_r, dtype=np.float64)
    rhs_a = np.zeros(n_r, dtype=np.float64)
    rhs_c = np.zeros(n_r, dtype=np.float64)
    A = np.zeros((n_r, n_r), dtype=np.float64)
    B_a = np.zeros(n_r, dtype=np.float64)
    B_c = np.zeros(n_r, dtype=np.float64)

    return {
        "ce": 0.0,
        "phis_a": 0.0,
        "phie": phie,
        "phis_c": phis_c,
        "cs_a": cs_a,
        "cs_c": cs_c,
        "Ds_a": Ds_a,
        "Ds_c": Ds_c,
        "rhs_a": rhs_a,
        "rhs_c": rhs_c,
        "A": A,
        "B_a": B_a,
        "B_c": B_c,
    }


def tridiag(ds, dt, dr):
    a = 1 + 2 * ds * dt / (dr**2)
    b = -ds * dt / (dr**2)
    bup = b[:-1]
    bdown = b[1:]
    main_mat = np.diag(a, 0) + np.diag(bdown, -1) + np.diag(bup, 1)
    main_mat[0, :] = 0
    main_mat[-1, :] = 0
    main_mat[0, 0] = -1 / dr
    main_mat[0, 1] = 1 / dr
    main_mat[-1, -1] = 1 / dr
    main_mat[-1, -2] = -1 / dr
    return main_mat


def tridiag_torch(ds, dt, dr, device: str):
    ds = _torch_tensor(ds, device)
    a = 1.0 + 2.0 * ds * dt / (dr**2)
    b = -ds * dt / (dr**2)
    main_mat = torch.diag(a)
    if ds.numel() > 1:
        main_mat = main_mat + torch.diag(b[1:], diagonal=-1) + torch.diag(b[:-1], diagonal=1)
    main_mat[0, :] = 0.0
    main_mat[-1, :] = 0.0
    main_mat[0, 0] = -1.0 / dr
    main_mat[0, 1] = 1.0 / dr
    main_mat[-1, -1] = 1.0 / dr
    main_mat[-1, -2] = -1.0 / dr
    return main_mat


def rhs(dt, r, ddr_cs, ds, ddDs_cs, cs, bound_grad):
    rhs_col = (
        dt
        * (np.float64(2.0) / np.clip(r, a_min=1e-12, a_max=None))
        * ddr_cs
        * ds
    )
    rhs_col += dt * ddr_cs**2 * ddDs_cs
    rhs_col += cs
    rhs_col[0] = 0
    rhs_col[-1] = bound_grad
    return rhs_col


def rhs_torch(dt, r, ddr_cs, ds, ddDs_cs, cs, bound_grad, device: str):
    r_t = _torch_tensor(r, device)
    ddr_t = _torch_tensor(ddr_cs, device)
    ds_t = _torch_tensor(ds, device)
    dd_t = _torch_tensor(ddDs_cs, device)
    cs_t = _torch_tensor(cs, device)
    rhs_col = dt * (2.0 / torch.clamp(r_t, min=1e-12)) * ddr_t * ds_t
    rhs_col = rhs_col + dt * (ddr_t**2) * dd_t + cs_t
    rhs_col[0] = 0.0
    rhs_col[-1] = float(bound_grad)
    return rhs_col


def _flux_from_current(I_now, params):
    V_a = params.get("V_a", params["A_a"] * params["L_a"])
    V_c = params.get("V_c", params["A_c"] * params["L_c"])

    j_a = (
            -I_now * params["Rs_a"]
            / (np.float64(3.0) * params["eps_s_a"] * params["F"] * V_a)
    )
    j_c = (
            I_now * params["Rs_c"]
            / (np.float64(3.0) * params["eps_s_c"] * params["F"] * V_c)
    )
    return np.float64(j_a), np.float64(j_c)


def init_sol(n_t, n_r, params, deg_i0_a, deg_ds_c, current_profile=None):
    sol = init_arrays(n_t, n_r)
    sol["ce"] = np.float64(params["ce0"])
    sol["phis_a"] = np.float64(0.0)
    sol["cs_a"][0, :] = np.float64(params["cs_a0"])
    sol["cs_c"][0, :] = np.float64(params["cs_c0"])

    if current_profile is None:
        I_profile = np.ones(n_t, dtype=np.float64) * np.float64(params["I_discharge"])
    else:
        I_profile = np.asarray(current_profile, dtype=np.float64).reshape(-1)
        if len(I_profile) != n_t:
            raise ValueError("len(current_profile) must equal n_t.")
    sol["I_profile"] = I_profile

    I0 = np.float64(I_profile[0])
    sol["j_a"], sol["j_c"] = _flux_from_current(I0, params)

    Uocp_a = _to_scalar(params["Uocp_a"](params["cs_a0"], params["csanmax"]))
    i0_a = _to_scalar(
        params["i0_a"](
            params["cs_a0"],
            sol["ce"],
            params["T"],
            params["alpha_a"],
            params["csanmax"],
            params["R"],
            deg_i0_a,
        )
    )
    sol["phie"][0] = _to_scalar(
        params["phie0"](
            i0_a,
            sol["j_a"],
            params["F"],
            params["R"],
            params["T"],
            Uocp_a,
        )
    )

    Uocp_c = _to_scalar(params["Uocp_c"](params["cs_c0"], params["cscamax"]))
    i0_c = _to_scalar(
        params["i0_c"](
            params["cs_c0"],
            params["ce0"],
            params["T"],
            params["alpha_c"],
            params["cscamax"],
            params["R"],
        )
    )
    sol["phis_c"][0] = _to_scalar(
        params["phis_c0"](
            i0_a,
            sol["j_a"],
            params["F"],
            params["R"],
            params["T"],
            Uocp_a,
            sol["j_c"],
            i0_c,
            Uocp_c,
        )
    )
    return sol


def _implicit_step_cpu(sol, i_t, dt, r_a, dR_a, r_c, dR_c, params, deg_ds_c, EXACT_GRAD_DS_CS, GRAD_STEP):
    sol["Ds_a"][:] = _to_numpy(params["D_s_a"](params["T"], params["R"]))
    if sol["Ds_a"].ndim == 0:
        sol["Ds_a"][:] = np.float64(sol["Ds_a"])
    if EXACT_GRAD_DS_CS:
        try:
            from thermo_assb import grad_ds_a_cs_a
            gradDs_a_cs_a = np.array(grad_ds_a_cs_a(params["T"], params["R"]), dtype=np.float64)
        except Exception:
            gradDs_a_cs_a = np.zeros(len(sol["Ds_a"]), dtype=np.float64)
    else:
        gradDs_a_cs_a = np.zeros(len(sol["Ds_a"]), dtype=np.float64)

    ddr_csa = np.gradient(sol["cs_a"][i_t - 1, :], r_a, axis=0, edge_order=2)
    ddr_csa[0] = 0.0
    ddr_csa[-1] = -sol["j_a"] / sol["Ds_a"][-1]

    A_a = tridiag(sol["Ds_a"], dt, dR_a)
    B_a = rhs(
        dt=dt,
        r=r_a,
        ddr_cs=ddr_csa,
        ds=sol["Ds_a"],
        ddDs_cs=gradDs_a_cs_a,
        cs=sol["cs_a"][i_t - 1, :],
        bound_grad=-sol["j_a"] / sol["Ds_a"][-1],
    )
    cs_a_tmp = np.linalg.solve(A_a, B_a)
    sol["cs_a"][i_t, :] = np.clip(cs_a_tmp, a_min=0.0, a_max=params["csanmax"])

    sol["Ds_c"][:] = _to_numpy(
        params["D_s_c"](
            sol["cs_c"][i_t - 1, :],
            params["T"],
            params["R"],
            params["cscamax"],
            deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape, dtype=np.float64),
        )
    )
    if EXACT_GRAD_DS_CS:
        try:
            from thermo_assb import grad_ds_c_cs_c
            gradDs_c_cs_c = _to_numpy(
                grad_ds_c_cs_c(
                    sol["cs_c"][i_t - 1, :],
                    params["T"],
                    params["R"],
                    params["cscamax"],
                )
            )
        except Exception:
            gradDs_c_cs_c = np.zeros(len(sol["Ds_c"]), dtype=np.float64)
    else:
        Ds_c_tmp1 = _to_numpy(
            params["D_s_c"](
                np.clip(
                    sol["cs_c"][i_t - 1, :] + np.ones(sol["cs_c"].shape[1], dtype=np.float64) * GRAD_STEP,
                    a_min=0,
                    a_max=params["cscamax"],
                ),
                params["T"],
                params["R"],
                params["cscamax"],
                deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape, dtype=np.float64),
            )
        )
        Ds_c_tmp2 = _to_numpy(
            params["D_s_c"](
                np.clip(
                    sol["cs_c"][i_t - 1, :] - np.ones(sol["cs_c"].shape[1], dtype=np.float64) * GRAD_STEP,
                    a_min=0,
                    a_max=params["cscamax"],
                ),
                params["T"],
                params["R"],
                params["cscamax"],
                deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape, dtype=np.float64),
            )
        )
        gradDs_c_cs_c = (Ds_c_tmp1 - Ds_c_tmp2) / (2 * GRAD_STEP)

    ddr_csc = np.gradient(sol["cs_c"][i_t - 1, :], r_c, axis=0, edge_order=2)
    ddr_csc[0] = 0.0
    ddr_csc[-1] = -sol["j_c"] / sol["Ds_c"][-1]

    A_c = tridiag(sol["Ds_c"], dt, dR_c)
    B_c = rhs(
        dt=dt,
        r=r_c,
        ddr_cs=ddr_csc,
        ds=sol["Ds_c"],
        ddDs_cs=gradDs_c_cs_c,
        cs=sol["cs_c"][i_t - 1, :],
        bound_grad=-sol["j_c"] / sol["Ds_c"][-1],
    )
    cs_c_tmp = np.linalg.solve(A_c, B_c)
    sol["cs_c"][i_t, :] = np.clip(cs_c_tmp, a_min=0.0, a_max=params["cscamax"])


def _implicit_step_cuda(sol, i_t, dt, r_a, dR_a, r_c, dR_c, params, deg_ds_c, EXACT_GRAD_DS_CS, GRAD_STEP, device: str):
    # anode diffusion coeff / gradient stay on CPU because thermo helpers are NumPy-based
    sol["Ds_a"][:] = _to_numpy(params["D_s_a"](params["T"], params["R"]))
    if sol["Ds_a"].ndim == 0:
        sol["Ds_a"][:] = np.float64(sol["Ds_a"])
    if EXACT_GRAD_DS_CS:
        try:
            from thermo_assb import grad_ds_a_cs_a
            gradDs_a_cs_a = np.array(grad_ds_a_cs_a(params["T"], params["R"]), dtype=np.float64)
        except Exception:
            gradDs_a_cs_a = np.zeros(len(sol["Ds_a"]), dtype=np.float64)
    else:
        gradDs_a_cs_a = np.zeros(len(sol["Ds_a"]), dtype=np.float64)

    ddr_csa = np.gradient(sol["cs_a"][i_t - 1, :], r_a, axis=0, edge_order=2)
    ddr_csa[0] = 0.0
    ddr_csa[-1] = -sol["j_a"] / max(sol["Ds_a"][-1], 1e-30)

    A_a = tridiag_torch(sol["Ds_a"], dt, dR_a, device)
    B_a = rhs_torch(
        dt=dt,
        r=r_a,
        ddr_cs=ddr_csa,
        ds=sol["Ds_a"],
        ddDs_cs=gradDs_a_cs_a,
        cs=sol["cs_a"][i_t - 1, :],
        bound_grad=-sol["j_a"] / max(sol["Ds_a"][-1], 1e-30),
        device=device,
    )
    cs_a_tmp = torch.linalg.solve(A_a, B_a.unsqueeze(-1)).squeeze(-1)
    sol["cs_a"][i_t, :] = np.clip(cs_a_tmp.detach().cpu().numpy(), a_min=0.0, a_max=params["csanmax"])

    sol["Ds_c"][:] = _to_numpy(
        params["D_s_c"](
            sol["cs_c"][i_t - 1, :],
            params["T"],
            params["R"],
            params["cscamax"],
            deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape, dtype=np.float64),
        )
    )
    if EXACT_GRAD_DS_CS:
        try:
            from thermo_assb import grad_ds_c_cs_c
            gradDs_c_cs_c = _to_numpy(
                grad_ds_c_cs_c(
                    sol["cs_c"][i_t - 1, :],
                    params["T"],
                    params["R"],
                    params["cscamax"],
                )
            )
        except Exception:
            gradDs_c_cs_c = np.zeros(len(sol["Ds_c"]), dtype=np.float64)
    else:
        Ds_c_tmp1 = _to_numpy(
            params["D_s_c"](
                np.clip(
                    sol["cs_c"][i_t - 1, :] + np.ones(sol["cs_c"].shape[1], dtype=np.float64) * GRAD_STEP,
                    a_min=0,
                    a_max=params["cscamax"],
                ),
                params["T"],
                params["R"],
                params["cscamax"],
                deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape, dtype=np.float64),
            )
        )
        Ds_c_tmp2 = _to_numpy(
            params["D_s_c"](
                np.clip(
                    sol["cs_c"][i_t - 1, :] - np.ones(sol["cs_c"].shape[1], dtype=np.float64) * GRAD_STEP,
                    a_min=0,
                    a_max=params["cscamax"],
                ),
                params["T"],
                params["R"],
                params["cscamax"],
                deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape, dtype=np.float64),
            )
        )
        gradDs_c_cs_c = (Ds_c_tmp1 - Ds_c_tmp2) / (2 * GRAD_STEP)

    ddr_csc = np.gradient(sol["cs_c"][i_t - 1, :], r_c, axis=0, edge_order=2)
    ddr_csc[0] = 0.0
    ddr_csc[-1] = -sol["j_c"] / max(sol["Ds_c"][-1], 1e-30)

    A_c = tridiag_torch(sol["Ds_c"], dt, dR_c, device)
    B_c = rhs_torch(
        dt=dt,
        r=r_c,
        ddr_cs=ddr_csc,
        ds=sol["Ds_c"],
        ddDs_cs=gradDs_c_cs_c,
        cs=sol["cs_c"][i_t - 1, :],
        bound_grad=-sol["j_c"] / max(sol["Ds_c"][-1], 1e-30),
        device=device,
    )
    cs_c_tmp = torch.linalg.solve(A_c, B_c.unsqueeze(-1)).squeeze(-1)
    sol["cs_c"][i_t, :] = np.clip(cs_c_tmp.detach().cpu().numpy(), a_min=0.0, a_max=params["cscamax"])


def integration(
    sol,
    config,
    params,
    deg_i0_a,
    deg_ds_c,
    explicit=False,
    verbose=False,
    LINEARIZE_J=True,
    EXACT_GRAD_DS_CS=False,
    GRAD_STEP=0.1,
    backend: str = "auto",
    device: str | None = None,
):
    n_t = config["n_t"]
    t = config["t"]
    r_a = config["r_a"]
    dR_a = config["dR_a"]
    r_c = config["r_c"]
    dR_c = config["dR_c"]

    solver_backend, solver_device = _resolve_backend(backend=backend, device=device)
    sol["solver_backend"] = solver_backend
    sol["solver_device"] = solver_device
    config["solver_backend"] = solver_backend
    config["solver_device"] = solver_device

    if verbose:
        suffix = f"Start [{solver_backend}]"
        print_progress_bar(0, n_t - 1, prefix="Step=", suffix=suffix, length=20)

    for i_t in range(1, n_t):
        dt = np.float64(t[i_t] - t[i_t - 1])
        if dt <= 0:
            raise ValueError(f"time_profile 非严格递增，i_t={i_t}, dt={dt}")

        I_now = np.float64(sol["I_profile"][i_t])
        sol["j_a"], sol["j_c"] = _flux_from_current(I_now, params)

        cse_a = np.float64(sol["cs_a"][i_t - 1, -1])
        i0_a = _to_scalar(
            params["i0_a"](
                cse_a,
                sol["ce"],
                params["T"],
                params["alpha_a"],
                params["csanmax"],
                params["R"],
                deg_i0_a,
            )
        )
        Uocp_a = _to_scalar(params["Uocp_a"](cse_a, params["csanmax"]))
        if not LINEARIZE_J:
            sol["phie"][i_t] = (
                -(np.float64(2.0) * params["R"] * params["T"] / params["F"])
                * np.arcsinh(sol["j_a"] * params["F"] / (np.float64(2.0) * i0_a))
                - Uocp_a
            )
        else:
            sol["phie"][i_t] = (
                -sol["j_a"]
                * (params["F"] / i0_a)
                * (params["R"] * params["T"] / params["F"])
                - Uocp_a
            )

        cse_c = np.float64(sol["cs_c"][i_t - 1, -1])
        i0_c = _to_scalar(
            params["i0_c"](
                cse_c,
                sol["ce"],
                params["T"],
                params["alpha_c"],
                params["cscamax"],
                params["R"],
            )
        )
        Uocp_c = _to_scalar(params["Uocp_c"](cse_c, params["cscamax"]))
        if not LINEARIZE_J:
            sol["phis_c"][i_t] = (
                (np.float64(2.0) * params["R"] * params["T"] / params["F"])
                * np.arcsinh(sol["j_c"] * params["F"] / (np.float64(2.0) * i0_c))
                + Uocp_c
                + sol["phie"][i_t]
            )
        else:
            sol["phis_c"][i_t] = (
                sol["j_c"]
                * (params["F"] / i0_c)
                * (params["R"] * params["T"] / params["F"])
                + Uocp_c
                + sol["phie"][i_t]
            )

        if not explicit:
            if solver_backend == "cuda":
                _implicit_step_cuda(
                    sol=sol,
                    i_t=i_t,
                    dt=dt,
                    r_a=r_a,
                    dR_a=dR_a,
                    r_c=r_c,
                    dR_c=dR_c,
                    params=params,
                    deg_ds_c=deg_ds_c,
                    EXACT_GRAD_DS_CS=EXACT_GRAD_DS_CS,
                    GRAD_STEP=GRAD_STEP,
                    device=solver_device,
                )
            else:
                _implicit_step_cpu(
                    sol=sol,
                    i_t=i_t,
                    dt=dt,
                    r_a=r_a,
                    dR_a=dR_a,
                    r_c=r_c,
                    dR_c=dR_c,
                    params=params,
                    deg_ds_c=deg_ds_c,
                    EXACT_GRAD_DS_CS=EXACT_GRAD_DS_CS,
                    GRAD_STEP=GRAD_STEP,
                )
        else:
            # explicit branch kept on CPU for stability / parity
            sol["Ds_a"][:] = _to_numpy(params["D_s_a"](params["T"], params["R"]))
            ddr_csa = np.gradient(sol["cs_a"][i_t - 1, :], r_a, axis=0, edge_order=2)
            ddr_csa[0] = 0.0
            ddr_csa[-1] = -sol["j_a"] / sol["Ds_a"][-1]
            n_r = config["n_r"]
            ddr2_csa = np.zeros(n_r, dtype=np.float64)
            ddr2_csa[1:n_r - 1] = (
                sol["cs_a"][i_t - 1, :n_r - 2]
                - 2 * sol["cs_a"][i_t - 1, 1:n_r - 1]
                + sol["cs_a"][i_t - 1, 2:n_r]
            ) / dR_a**2
            ddr2_csa[0] = (
                sol["cs_a"][i_t - 1, 0]
                - 2 * sol["cs_a"][i_t - 1, 0]
                + sol["cs_a"][i_t - 1, 1]
            ) / dR_a**2
            ddr2_csa[-1] = (
                sol["cs_a"][i_t - 1, -2]
                - 2 * sol["cs_a"][i_t - 1, -1]
                + sol["cs_a"][i_t - 1, -1]
                + ddr_csa[-1] * dR_a
            ) / dR_a**2
            ddr_Ds = np.gradient(sol["Ds_a"], r_a, axis=0, edge_order=2)
            sol["rhs_a"][1:] = (
                sol["Ds_a"][1:] * ddr2_csa[1:]
                + ddr_Ds[1:] * ddr_csa[1:]
                + 2 * sol["Ds_a"][1:] * ddr_csa[1:] / r_a[1:]
            )
            sol["rhs_a"][0] = 3 * sol["Ds_a"][0] * ddr2_csa[0]
            sol["cs_a"][i_t, :] = np.clip(sol["cs_a"][i_t - 1, :] + dt * sol["rhs_a"], a_min=0.0, a_max=None)

            sol["Ds_c"][:] = _to_numpy(
                params["D_s_c"](
                    sol["cs_c"][i_t - 1, :],
                    params["T"],
                    params["R"],
                    params["cscamax"],
                    deg_ds_c * np.ones(sol["cs_c"][i_t - 1, :].shape, dtype=np.float64),
                )
            )
            ddr_csc = np.gradient(sol["cs_c"][i_t - 1, :], r_c, axis=0, edge_order=2)
            ddr_csc[0] = 0.0
            ddr_csc[-1] = -sol["j_c"] / sol["Ds_c"][-1]
            ddr2_csc = np.zeros(n_r, dtype=np.float64)
            ddr2_csc[1:n_r - 1] = (
                sol["cs_c"][i_t - 1, :n_r - 2]
                - 2 * sol["cs_c"][i_t - 1, 1:n_r - 1]
                + sol["cs_c"][i_t - 1, 2:n_r]
            ) / dR_c**2
            ddr2_csc[0] = (
                sol["cs_c"][i_t - 1, 0]
                - 2 * sol["cs_c"][i_t - 1, 0]
                + sol["cs_c"][i_t - 1, 1]
            ) / dR_c**2
            ddr2_csc[-1] = (
                sol["cs_c"][i_t - 1, -2]
                - 2 * sol["cs_c"][i_t - 1, -1]
                + sol["cs_c"][i_t - 1, -1]
                + ddr_csc[-1] * dR_c
            ) / dR_c**2
            ddr_Ds = np.gradient(sol["Ds_c"], r_c, axis=0, edge_order=2)
            sol["rhs_c"][1:] = (
                sol["Ds_c"][1:] * ddr2_csc[1:]
                + ddr_Ds[1:] * ddr_csc[1:]
                + 2 * sol["Ds_c"][1:] * ddr_csc[1:] / r_c[1:]
            )
            sol["rhs_c"][0] = 3 * sol["Ds_c"][0] * ddr2_csc[0]
            sol["cs_c"][i_t, :] = np.clip(sol["cs_c"][i_t - 1, :] + dt * sol["rhs_c"], a_min=0.0, a_max=None)

        if verbose:
            print_progress_bar(i_t, n_t - 1, prefix="Step=", suffix=f"Complete [{solver_backend}]", length=20)


def exec_impl(
    n_r,
    dt,
    params,
    deg_i0_a,
    deg_ds_c,
    verbose=False,
    current_profile=None,
    time_profile=None,
    backend: str = "auto",
    device: str | None = None,
):
    time_s = time.time()
    r_dom = get_r_domain(n_r, params)

    if time_profile is not None:
        time_profile = np.asarray(time_profile, dtype=np.float64).reshape(-1)
        n_t = len(time_profile)
        t_dom = get_t_domain(n_t, params, time_profile=time_profile)
    elif current_profile is not None:
        current_profile = np.asarray(current_profile, dtype=np.float64).reshape(-1)
        n_t = len(current_profile)
        t_dom = get_t_domain(n_t, params)
    else:
        n_t = get_nt_from_dt(dt, params)
        t_dom = get_t_domain(n_t, params)

    config = make_sim_config(t_dom, r_dom)
    sol = init_sol(
        n_t,
        n_r,
        params,
        deg_i0_a,
        deg_ds_c,
        current_profile=current_profile,
    )
    integration(
        sol,
        config,
        params,
        deg_i0_a,
        deg_ds_c,
        explicit=False,
        verbose=verbose,
        LINEARIZE_J=True,
        EXACT_GRAD_DS_CS=False,
        backend=backend,
        device=device,
    )
    time_e = time.time()
    if verbose:
        print(f"n_r: {n_r}, n_t: {n_t}, time = {time_e-time_s:.2f}s, backend={config.get('solver_backend')}")
    return config, sol


def exec_expl(
    n_r,
    params,
    deg_i0_a,
    deg_ds_c,
    verbose=False,
    current_profile=None,
    time_profile=None,
    backend: str = "cpu",
    device: str | None = None,
):
    time_s = time.time()
    r_dom = get_r_domain(n_r, params)
    if time_profile is not None:
        time_profile = np.asarray(time_profile, dtype=np.float64).reshape(-1)
        n_t = len(time_profile)
        t_dom = get_t_domain(n_t, params, time_profile=time_profile)
    else:
        n_t = get_expl_nt(r_dom, params, deg_ds_c)
        t_dom = get_t_domain(n_t, params)

    config = make_sim_config(t_dom, r_dom)
    sol = init_sol(
        n_t,
        n_r,
        params,
        deg_i0_a,
        deg_ds_c,
        current_profile=current_profile,
    )
    integration(
        sol,
        config,
        params,
        deg_i0_a,
        deg_ds_c,
        explicit=True,
        verbose=verbose,
        LINEARIZE_J=True,
        EXACT_GRAD_DS_CS=False,
        backend="cpu",  # explicit branch kept on CPU
        device=device,
    )
    time_e = time.time()
    if verbose:
        print(f"n_r: {n_r}, n_t: {n_t}, time = {time_e-time_s:.2f}s, backend=cpu(explicit)")
    return config, sol
