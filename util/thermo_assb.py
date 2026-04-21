from __future__ import annotations

import numpy as np
import torch

try:
    from .uocp_cs import uocp_c_fun_x, uocp_c_fun_x_numpy
except ImportError:  # pragma: no cover
    from uocp_cs import uocp_c_fun_x, uocp_c_fun_x_numpy


def _to_tensor(x, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float64)
    dtype = torch.float64 if like is None else like.dtype
    device = None if like is None else like.device
    return torch.as_tensor(x, dtype=dtype, device=device)


UOCP_A_LIIN_V = np.float64(0.62)
I0_A_REF = np.float64(0.1)  # A / m^2

DS_C_REF = np.float64(5.0e-15)
DS_C_MIN_FACTOR = np.float64(0.35)
DS_C_PEAK_CENTER = np.float64(0.45)
DS_C_PEAK_WIDTH = np.float64(0.22)

UOCP_C_SCALE = np.float64(0.7624151100416336)
UOCP_C_SHIFT = np.float64(0.44293391056209597)


def cathode_ocp_theta_numpy(theta):
    theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    u_old = uocp_c_fun_x_numpy(theta)
    return UOCP_C_SCALE * u_old + UOCP_C_SHIFT



def cathode_ocp_cs_numpy(cs_c, cscamax):
    theta = np.asarray(cs_c, dtype=np.float64) / np.float64(cscamax)
    return cathode_ocp_theta_numpy(theta)



def infer_theta_c_from_fullcell_voltage(fullcell_voltage_V, uocp_a=UOCP_A_LIIN_V):
    theta_grid = np.linspace(0.0, 1.0, 20001, dtype=np.float64)
    u_grid = cathode_ocp_theta_numpy(theta_grid)
    target_uc = np.float64(fullcell_voltage_V) + np.float64(uocp_a)
    target_uc = np.clip(target_uc, np.min(u_grid), np.max(u_grid))
    theta = np.interp(target_uc, u_grid[::-1], theta_grid[::-1])
    return np.float64(theta)



def infer_cs_c0_from_fullcell_voltage(fullcell_voltage_V, cscamax, uocp_a=UOCP_A_LIIN_V):
    theta_c0 = infer_theta_c_from_fullcell_voltage(fullcell_voltage_V, uocp_a=uocp_a)
    return np.float64(theta_c0 * np.float64(cscamax))



def infer_theta_a0_from_theta_c(theta_c):
    theta_c = np.float64(theta_c)
    theta_a = np.float64(1.30) - theta_c
    return np.float64(np.clip(theta_a, 0.75, 0.99))



def uocp_a_simp(cs_a, csanmax):
    cs_a = _to_tensor(cs_a)
    x = torch.clamp(cs_a / float(csanmax), 0.0, 1.0)
    return np.float64(0.2) - np.float64(0.2) * x


# Li-In negative electrode approximated as a flat plateau.
def uocp_a_fun(cs_a, csanmax):
    cs_a = _to_tensor(cs_a)
    x = torch.clamp(cs_a / float(csanmax), 0.0, 1.0)
    return torch.full_like(x, float(UOCP_A_LIIN_V))



def uocp_c_fun_legacy(cs_c, cscamax):
    cs_c = _to_tensor(cs_c)
    x = torch.clamp(cs_c / float(cscamax), 0.0, 1.0)
    return uocp_c_fun_x(x)



def uocp_c_fun(cs_c, cscamax):
    u_old = uocp_c_fun_legacy(cs_c, cscamax)
    return UOCP_C_SCALE * u_old + UOCP_C_SHIFT



def uocp_c_simp(cs_c, cscamax):
    cs_c = _to_tensor(cs_c)
    x = torch.clamp(cs_c / float(cscamax), 0.0, 1.0)
    return np.float64(4.30) - np.float64(1.50) * x



def i0_a_fun(cs_a_max, ce, T, alpha, csanmax, R):
    cs_a_max = _to_tensor(cs_a_max)
    ce = _to_tensor(ce, cs_a_max)
    T = _to_tensor(T, cs_a_max)
    return (
        np.float64(2.5)
        * np.float64(0.27)
        * torch.exp(
            np.float64((-30.0e6 / float(R)))
            * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
        * torch.clamp(ce, min=np.float64(0.0)) ** float(alpha)
        * torch.clamp(float(csanmax) - cs_a_max, min=np.float64(0.0)) ** float(alpha)
        * torch.clamp(cs_a_max, min=np.float64(0.0)) ** (np.float64(1.0) - float(alpha))
    )


# Effective Li-In kinetic baseline times degradation factor.
def i0_a_degradation_param_fun(cs_a_max, ce, T, alpha, csanmax, R, degradation_param):
    ce_like = torch.ones_like(_to_tensor(ce, _to_tensor(cs_a_max)))
    deg = torch.clamp(_to_tensor(degradation_param, ce_like), min=np.float64(1e-12))
    return np.float64(I0_A_REF) * deg * ce_like



def i0_a_simp(cs_a_max, ce, T, alpha, csanmax, R):
    ce = _to_tensor(ce)
    return np.float64(2.0) * torch.ones_like(ce)



def i0_a_simp_degradation_param(cs_a_max, ce, T, alpha, csanmax, R, degradation_param):
    ce = _to_tensor(ce)
    deg = torch.clamp(_to_tensor(degradation_param, ce), min=np.float64(1e-12))
    return np.float64(2.0) * deg * torch.ones_like(ce)



def i0_c_fun(cs_c_max, ce, T, alpha, cscamax, R):
    cs_c_max = _to_tensor(cs_c_max)
    ce = _to_tensor(ce, cs_c_max)
    T = _to_tensor(T, cs_c_max)
    x = torch.clamp(cs_c_max / float(cscamax), 0.0, 1.0)
    return (
        np.float64(9.0)
        * (
            np.float64(1.650452829641290e01) * x**5
            - np.float64(7.523567141488800e01) * x**4
            + np.float64(1.240524690073040e02) * x**3
            - np.float64(9.416571081287610e01) * x**2
            + np.float64(3.249768821737960e01) * x
            - np.float64(3.585290065824760e00)
        )
        * torch.clamp(ce / np.float64(1.2), min=np.float64(0.0)) ** float(alpha)
        * torch.exp(
            (np.float64(-30.0e6) / float(R)) * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
    )



def i0_c_simp(cs_c_max, ce, T, alpha, cscamax, R):
    ce = _to_tensor(ce)
    return np.float64(3.0) * torch.ones_like(ce)



def ds_a_fun(T, R):
    T = _to_tensor(T)
    return np.float64(3.0e-14) * torch.exp(
        (np.float64(-30.0e6) / float(R)) * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
    )



def grad_ds_a_cs_a(T, R):
    T = _to_tensor(T)
    return torch.zeros_like(T)



def ds_a_fun_simp(T, R):
    T = _to_tensor(T)
    return np.float64(3.0e-14) * torch.ones_like(T)


# NMC811 effective diffusivity with SOC dependence.
def ds_c_fun(cs_c, T, R, cscamax):
    cs_c = _to_tensor(cs_c)
    theta = torch.clamp(cs_c / float(cscamax), np.float64(0.0), np.float64(1.0))
    envelope = np.float64(DS_C_MIN_FACTOR) + (np.float64(1.0) - np.float64(DS_C_MIN_FACTOR)) * torch.exp(
        -((theta - np.float64(DS_C_PEAK_CENTER)) / np.float64(DS_C_PEAK_WIDTH)) ** np.float64(2.0)
    )
    return np.float64(DS_C_REF) * envelope


# Analytical derivative of the Gaussian-envelope diffusivity.
def grad_ds_c_cs_c(cs_c, T, R, cscamax):
    cs_c = _to_tensor(cs_c)
    theta = torch.clamp(cs_c / float(cscamax), np.float64(0.0), np.float64(1.0))
    exp_term = torch.exp(-((theta - np.float64(DS_C_PEAK_CENTER)) / np.float64(DS_C_PEAK_WIDTH)) ** np.float64(2.0))
    d_env_d_theta = (
        (np.float64(1.0) - np.float64(DS_C_MIN_FACTOR))
        * exp_term
        * (
            -np.float64(2.0)
            * (theta - np.float64(DS_C_PEAK_CENTER))
            / (np.float64(DS_C_PEAK_WIDTH) ** np.float64(2.0))
        )
    )
    return np.float64(DS_C_REF) * d_env_d_theta / float(cscamax)


# degradation factor > 1 means slower effective diffusion.
def ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param):
    base = ds_c_fun(cs_c, T, R, cscamax)
    deg = torch.clamp(_to_tensor(degradation_param, base), min=np.float64(1e-12))
    return base / deg



def ds_c_fun_simp(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)



def ds_c_fun_plot(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)



def ds_c_fun_plot_simp(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)



def ds_c_fun_simp_degradation_param(cs_c, T, R, cscamax, degradation_param):
    return ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param)



def phie0_fun(i0_a, j_a, F, R, T, Uocp_a0):
    i0_a = _to_tensor(i0_a)
    return -float(j_a) * (float(F) / i0_a) * (float(R) * float(T) / float(F)) - _to_tensor(Uocp_a0, i0_a)



def phis_c0_fun(i0_a, j_a, F, R, T, Uocp_a0, j_c, i0_c, Uocp_c0):
    phie0 = phie0_fun(i0_a, j_a, F, R, T, Uocp_a0)
    return float(j_c) * (float(F) / _to_tensor(i0_c, phie0)) * (float(R) * float(T) / float(F)) + _to_tensor(Uocp_c0, phie0) + phie0



def setParams(params, deg, bat, an, ca, ic):
    params["deg_i0_a_min"] = deg.bounds[deg.ind_i0_a][0]
    params["deg_i0_a_max"] = deg.bounds[deg.ind_i0_a][1]
    params["deg_ds_c_min"] = deg.bounds[deg.ind_ds_c][0]
    params["deg_ds_c_max"] = deg.bounds[deg.ind_ds_c][1]

    params["param_eff"] = deg.eff
    params["deg_i0_a_ref"] = deg.ref_vals[deg.ind_i0_a]
    params["deg_ds_c_ref"] = deg.ref_vals[deg.ind_ds_c]
    params["deg_i0_a_min_eff"] = params["deg_i0_a_ref"] + (params["deg_i0_a_min"] - params["deg_i0_a_ref"]) * params["param_eff"]
    params["deg_i0_a_max_eff"] = params["deg_i0_a_ref"] + (params["deg_i0_a_max"] - params["deg_i0_a_ref"]) * params["param_eff"]
    params["deg_ds_c_min_eff"] = params["deg_ds_c_ref"] + (params["deg_ds_c_min"] - params["deg_ds_c_ref"]) * params["param_eff"]
    params["deg_ds_c_max_eff"] = params["deg_ds_c_ref"] + (params["deg_ds_c_max"] - params["deg_ds_c_ref"]) * params["param_eff"]

    params["tmin"] = bat.tmin
    params["tmax"] = bat.tmax
    params["rmin"] = bat.rmin

    params["A_a"] = an.A
    params["A_c"] = ca.A
    params["F"] = bat.F
    params["R"] = bat.R
    params["T"] = bat.T
    params["C"] = bat.C
    params["I_discharge"] = bat.I

    params["alpha_a"] = an.alpha
    params["alpha_c"] = ca.alpha

    params["Rs_a"] = an.D50 / np.float64(2.0)
    params["Rs_c"] = ca.D50 / np.float64(2.0)
    params["rescale_R"] = np.float64(max(params["Rs_a"], params["Rs_c"]))
    params["csanmax"] = an.csmax
    params["cscamax"] = ca.csmax
    params["rescale_T"] = np.float64(max(bat.tmax, 1e-16))

    params["mag_cs_a"] = np.float64(25.0)
    params["mag_cs_c"] = np.float64(32.5)
    params["mag_phis_c"] = np.float64(4.25)
    params["mag_phie"] = np.float64(0.15)
    params["mag_ce"] = np.float64(1.2)

    params["Uocp_a"] = an.uocp
    params["Uocp_c"] = ca.uocp
    params["i0_a"] = an.i0
    params["i0_c"] = ca.i0
    params["D_s_a"] = an.ds
    params["D_s_c"] = ca.ds

    params["ce0"] = ic.ce
    params["ce_a0"] = ic.ce
    params["ce_c0"] = ic.ce
    params["cs_a0"] = ic.an.cs
    params["cs_c0"] = ic.ca.cs
    params["eps_s_a"] = an.solids.eps
    params["eps_s_c"] = ca.solids.eps
    params["L_a"] = an.thickness
    params["L_c"] = ca.thickness

    j_a = (-(params["I_discharge"] / params["A_a"]) * params["Rs_a"] / (np.float64(3.0) * params["eps_s_a"] * params["F"] * params["L_a"]))
    j_c = ((params["I_discharge"] / params["A_c"]) * params["Rs_c"] / (np.float64(3.0) * params["eps_s_c"] * params["F"] * params["L_c"]))
    params["j_a"] = j_a
    params["j_c"] = j_c

    cse_a = ic.an.cs
    i0_a = params["i0_a"](
        torch.tensor(cse_a, dtype=torch.float64),
        torch.tensor(params["ce0"], dtype=torch.float64),
        params["T"],
        params["alpha_a"],
        params["csanmax"],
        params["R"],
        torch.tensor(1.0, dtype=torch.float64),
    )
    Uocp_a = params["Uocp_a"](torch.tensor(cse_a, dtype=torch.float64), params["csanmax"])
    params["Uocp_a0"] = float(Uocp_a)
    params["phie0"] = phie0_fun

    cse_c = ic.ca.cs
    i0_c = params["i0_c"](
        torch.tensor(cse_c, dtype=torch.float64),
        torch.tensor(params["ce0"], dtype=torch.float64),
        params["T"],
        params["alpha_c"],
        params["cscamax"],
        params["R"],
    )
    params["i0_c0"] = float(i0_c)
    Uocp_c = params["Uocp_c"](torch.tensor(cse_c, dtype=torch.float64), params["cscamax"])
    params["Uocp_c0"] = float(Uocp_c)
    params["phis_c0"] = phis_c0_fun

    params["rescale_cs_a"] = -ic.an.cs
    params["rescale_cs_c"] = params["cscamax"] - ic.ca.cs
    params["rescale_phis_c"] = abs(np.float64(3.8) - np.float64(4.110916387038547))
    params["rescale_phie"] = abs(np.float64(-0.15) - np.float64(-0.07645356566609385))
    return params
