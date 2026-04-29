from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .uocp_cs import uocp_c_fun_x, uocp_c_fun_x_numpy
except ImportError:  # pragma: no cover
    try:
        from uocp_cs import uocp_c_fun_x, uocp_c_fun_x_numpy  # type: ignore
    except Exception:  # pragma: no cover
        uocp_c_fun_x = None
        uocp_c_fun_x_numpy = None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "util":
        return here.parent.parent
    return here.parent


ROOT = _repo_root()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _candidate_soft_label_summaries() -> list[Path]:
    candidates: list[Path] = []

    env_dir = os.environ.get("ASSB_SOFT_LABEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "soft_label_summary.json")

    env_summary = os.environ.get("ASSB_SOFT_LABEL_SUMMARY")
    if env_summary:
        candidates.append(Path(env_summary))

    candidates.extend(
        [
            ROOT / "Data" / "assb_soft_labels_cycle5_v3" / "soft_label_summary.json",
            ROOT / "Data" / "assb_soft_labels_cycles5plus_v3" / "soft_label_summary.json",
        ]
    )
    return candidates


_SOFT_SUMMARY: dict[str, Any] = {}
for _p in _candidate_soft_label_summaries():
    _SOFT_SUMMARY = _load_json(_p)
    if _SOFT_SUMMARY:
        _SOFT_SUMMARY["_summary_path"] = str(_p)
        break


def _summary_float(names: tuple[str, ...], default: float) -> np.float64:
    """Read a scalar from soft_label_summary.json or ASSB_* environment."""
    for name in names:
        if name in _SOFT_SUMMARY:
            try:
                val = float(_SOFT_SUMMARY[name])
                if np.isfinite(val):
                    return np.float64(val)
            except Exception:
                pass

    for name in names:
        env_name = "ASSB_" + name.upper()
        if env_name in os.environ:
            try:
                val = float(os.environ[env_name])
                if np.isfinite(val):
                    return np.float64(val)
            except Exception:
                pass

    return np.float64(default)


def _to_tensor(x, like: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x
        if like is not None:
            out = out.to(dtype=like.dtype, device=like.device)
        else:
            out = out.to(dtype=torch.float64)
        return out
    dtype = torch.float64 if like is None else like.dtype
    device = None if like is None else like.device
    return torch.as_tensor(x, dtype=dtype, device=device)


# -----------------------------------------------------------------------------
# v3-aligned ASSB constants
# -----------------------------------------------------------------------------

# Li-In/In negative-electrode OCP plateau, relative to Li/Li+.
UOCP_A_LIIN_V = np.float64(0.6246405571428566)

# Effective Li-In/In state scale. This is an equivalent placeholder, not a strict
# material constant; v3 soft labels use 6.0 kmol/m^3 to avoid negative saturation.
CSANMAX_EFF = _summary_float(("csanmax", "csanmax_eff"), 6.0)

# Positive OCP usable-window mapping, fitted in the v3 soft-label generator.
THETA_C_BOTTOM = _summary_float(("theta_c_bottom_v3", "theta_c_bottom"), 0.834)
THETA_C_TOP = _summary_float(("theta_c_top_v3", "theta_c_top"), 0.432)

# Terminal-voltage shift used by v3 labels: V_term = V_interface + I(t)R + offset.
R_OHM_EFF = _summary_float(("R_ohm_eff_v3", "R_ohm_eff"), 105.0)
VOLTAGE_ALIGNMENT_OFFSET_V = _summary_float(
    ("voltage_alignment_offset_V", "voltage_offset_V"), -0.11588681607942332
)

# Kinetic and diffusion priors.
I0_A_REF = np.float64(0.1)       # A / m^2, multiplied by degradation parameter
I0_C_PREFAC = np.float64(2.5)    # ASSB NMC811 positive exchange-current prefactor
DS_A_REF = np.float64(5.0e-13)   # m^2/s, Li-In effective diffusion
DS_C_REF = np.float64(5.0e-15)   # m^2/s, NMC811 effective diffusion baseline
DS_C_MIN_FACTOR = np.float64(0.35)
DS_C_PEAK_CENTER = np.float64(0.45)
DS_C_PEAK_WIDTH = np.float64(0.22)

# Legacy fallback if OCP CSV files are unavailable.
UOCP_C_SCALE = np.float64(0.7624151100416336)
UOCP_C_SHIFT = np.float64(0.44293391056209597)

# Narrow but safe cycle-5 stoichiometric bounds for the bidirectional rescaler.
# These are not data loss. They prevent the PINN from exploring states that are
# far outside the cycle-5 v3 soft-label window while still including the IC.
THETA_A_MIN_RESCALE = _summary_float(("theta_a_min_rescale", "theta_a_min"), 0.40)
THETA_A_MAX_RESCALE = _summary_float(("theta_a_max_rescale", "theta_a_max"), 0.90)
THETA_C_MIN_RESCALE = _summary_float(("theta_c_min_rescale", "theta_c_min"), 0.40)
THETA_C_MAX_RESCALE = _summary_float(("theta_c_max_rescale", "theta_c_max"), 0.92)


# -----------------------------------------------------------------------------
# OCP table loading and interpolation
# -----------------------------------------------------------------------------

def _candidate_ocp_dirs() -> list[Path]:
    dirs: list[Path] = []
    for key in ("ASSB_OCP_DIR", "OCP_DIR"):
        val = os.environ.get(key)
        if val:
            dirs.append(Path(val))

    if _SOFT_SUMMARY.get("ocp_source_dir"):
        dirs.append(Path(str(_SOFT_SUMMARY["ocp_source_dir"])))

    dirs.extend(
        [
            Path(r"C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/ocp_estimation_outputs"),
            Path(r"C:/Users/Tiga_QJW/Desktop/pinn_spm_param/ocp_estimation_outputs"),
            ROOT / "ocp_estimation_outputs",
            ROOT / "Data" / "ocp_estimation_outputs",
        ]
    )

    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _read_two_col_csv(
    path: Path, value_candidates: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
        names = data.dtype.names or ()
        if len(names) < 2:
            return None

        x_name = None
        for name in names:
            if name.lower().replace("_", "") in {"soc0to1", "soc", "x", "theta"}:
                x_name = name
                break
        if x_name is None:
            x_name = names[0]

        y_name = None
        lower_map = {n.lower(): n for n in names}
        for cand in value_candidates:
            if cand.lower() in lower_map:
                y_name = lower_map[cand.lower()]
                break
        if y_name is None:
            y_name = names[1]

        x = np.asarray(data[x_name], dtype=np.float64).reshape(-1)
        y = np.asarray(data[y_name], dtype=np.float64).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 2:
            return None
        order = np.argsort(x)
        x, y = x[order], y[order]

        # Remove duplicate abscissae.
        ux, idx = np.unique(x, return_index=True)
        y = y[idx]
        return ux.astype(np.float64), y.astype(np.float64)
    except Exception:
        return None


def _load_ocp_table(kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    if kind == "positive":
        filename = "positive_ocp_curve.csv"
        y_candidates = ("U_p_ocp_est_V", "U_p", "U", "voltage_V")
    else:
        filename = "negative_ocp_curve.csv"
        y_candidates = ("U_n_ocp_est_V", "U_n", "U", "voltage_V")

    for d in _candidate_ocp_dirs():
        table = _read_two_col_csv(d / filename, y_candidates)
        if table is not None:
            return table
    return None


_POS_TABLE = _load_ocp_table("positive")
_NEG_TABLE = _load_ocp_table("negative")


def _interp1_torch(x: torch.Tensor, xp_np: np.ndarray, fp_np: np.ndarray) -> torch.Tensor:
    x = _to_tensor(x)
    shape = x.shape
    x_flat = x.reshape(-1)
    xp = torch.as_tensor(xp_np, dtype=x.dtype, device=x.device)
    fp = torch.as_tensor(fp_np, dtype=x.dtype, device=x.device)

    xq = torch.clamp(x_flat, min=xp[0], max=xp[-1])
    idx_hi = torch.searchsorted(xp, xq, right=False)
    idx_hi = torch.clamp(idx_hi, 1, xp.numel() - 1)
    idx_lo = idx_hi - 1

    x0 = xp[idx_lo]
    x1 = xp[idx_hi]
    y0 = fp[idx_lo]
    y1 = fp[idx_hi]
    w = (xq - x0) / torch.clamp(x1 - x0, min=np.float64(1.0e-12))
    return (y0 + w * (y1 - y0)).reshape(shape)


def _interp1_numpy(x, xp_np: np.ndarray, fp_np: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(x, dtype=np.float64), xp_np, fp_np)


# -----------------------------------------------------------------------------
# OCP functions
# -----------------------------------------------------------------------------

def cathode_ocp_theta_numpy(theta):
    theta = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    if _POS_TABLE is not None:
        soc = (float(THETA_C_BOTTOM) - theta) / max(float(THETA_C_BOTTOM - THETA_C_TOP), 1.0e-12)
        soc = np.clip(soc, 0.0, 1.0)
        return _interp1_numpy(soc, _POS_TABLE[0], _POS_TABLE[1])

    if uocp_c_fun_x_numpy is None:
        return np.float64(4.30) - np.float64(1.50) * theta
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

    # Use interpolation over a monotonic sorting of U.
    order = np.argsort(u_grid)
    theta = np.interp(target_uc, u_grid[order], theta_grid[order])
    return np.float64(theta)


def infer_cs_c0_from_fullcell_voltage(fullcell_voltage_V, cscamax, uocp_a=UOCP_A_LIIN_V):
    return np.float64(infer_theta_c_from_fullcell_voltage(fullcell_voltage_V, uocp_a) * np.float64(cscamax))


def infer_theta_a0_from_theta_c(theta_c):
    theta_c = np.float64(theta_c)
    theta_a = np.float64(1.30) - theta_c
    return np.float64(np.clip(theta_a, 0.30, 0.95))


def uocp_a_simp(cs_a, csanmax):
    cs_a = _to_tensor(cs_a)
    x = torch.clamp(cs_a / float(csanmax), 0.0, 1.0)
    return np.float64(0.2) - np.float64(0.2) * x


def uocp_a_fun(cs_a, csanmax):
    cs_a = _to_tensor(cs_a)
    x = torch.clamp(cs_a / float(csanmax), 0.0, 1.0)
    if _NEG_TABLE is not None:
        return _interp1_torch(x, _NEG_TABLE[0], _NEG_TABLE[1])
    return torch.full_like(x, float(UOCP_A_LIIN_V))


def uocp_c_fun_legacy(cs_c, cscamax):
    cs_c = _to_tensor(cs_c)
    x = torch.clamp(cs_c / float(cscamax), 0.0, 1.0)
    if uocp_c_fun_x is None:
        return np.float64(4.30) - np.float64(1.50) * x
    return uocp_c_fun_x(x)


def uocp_c_fun(cs_c, cscamax):
    cs_c = _to_tensor(cs_c)
    theta = torch.clamp(cs_c / float(cscamax), 0.0, 1.0)
    if _POS_TABLE is not None:
        soc = (float(THETA_C_BOTTOM) - theta) / max(float(THETA_C_BOTTOM - THETA_C_TOP), 1.0e-12)
        soc = torch.clamp(soc, 0.0, 1.0)
        return _interp1_torch(soc, _POS_TABLE[0], _POS_TABLE[1])

    u_old = uocp_c_fun_legacy(cs_c, cscamax)
    return UOCP_C_SCALE * u_old + UOCP_C_SHIFT


def uocp_c_simp(cs_c, cscamax):
    cs_c = _to_tensor(cs_c)
    x = torch.clamp(cs_c / float(cscamax), 0.0, 1.0)
    return np.float64(4.30) - np.float64(1.50) * x


# -----------------------------------------------------------------------------
# Kinetics and diffusivity
# -----------------------------------------------------------------------------

def i0_a_fun(cs_a_max, ce, T, alpha, csanmax, R):
    cs_a_max = _to_tensor(cs_a_max)
    ce = _to_tensor(ce, cs_a_max)
    T = _to_tensor(T, cs_a_max)
    return (
        np.float64(2.5)
        * np.float64(0.27)
        * torch.exp(
            np.float64((-30.0e6 / float(R))) * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
        )
        * torch.clamp(ce, min=np.float64(0.0)) ** float(alpha)
        * torch.clamp(float(csanmax) - cs_a_max, min=np.float64(0.0)) ** float(alpha)
        * torch.clamp(cs_a_max, min=np.float64(0.0)) ** (np.float64(1.0) - float(alpha))
    )


def i0_a_degradation_param_fun(cs_a_max, ce, T, alpha, csanmax, R, degradation_param):
    ce_like = torch.ones_like(_to_tensor(ce, _to_tensor(cs_a_max)))
    deg = torch.clamp(_to_tensor(degradation_param, ce_like), min=np.float64(1.0e-12))
    return np.float64(I0_A_REF) * deg * ce_like


def i0_a_simp(cs_a_max, ce, T, alpha, csanmax, R):
    ce = _to_tensor(ce)
    return np.float64(2.0) * torch.ones_like(ce)


def i0_a_simp_degradation_param(cs_a_max, ce, T, alpha, csanmax, R, degradation_param):
    ce = _to_tensor(ce)
    deg = torch.clamp(_to_tensor(degradation_param, ce), min=np.float64(1.0e-12))
    return np.float64(2.0) * deg * torch.ones_like(ce)


def i0_c_fun(cs_c_max, ce, T, alpha, cscamax, R):
    cs_c_max = _to_tensor(cs_c_max)
    ce = _to_tensor(ce, cs_c_max)
    T = _to_tensor(T, cs_c_max)
    x = torch.clamp(cs_c_max / float(cscamax), 0.0, 1.0)
    poly = (
        np.float64(1.650452829641290e01) * x**5
        - np.float64(7.523567141488800e01) * x**4
        + np.float64(1.240524690073040e02) * x**3
        - np.float64(9.416571081287610e01) * x**2
        + np.float64(3.249768821737960e01) * x
        - np.float64(3.585290065824760e00)
    )
    poly = torch.clamp(poly, min=np.float64(1.0e-12))
    return (
        I0_C_PREFAC
        * poly
        * torch.clamp(ce / np.float64(1.2), min=np.float64(0.0)) ** float(alpha)
        * torch.exp((np.float64(-30.0e6) / float(R)) * (np.float64(1.0) / T - np.float64(1.0 / 303.15)))
    )


def i0_c_simp(cs_c_max, ce, T, alpha, cscamax, R):
    ce = _to_tensor(ce)
    return I0_C_PREFAC * torch.ones_like(ce)


def ds_a_fun(T, R):
    T = _to_tensor(T)
    return np.float64(DS_A_REF) * torch.exp(
        (np.float64(-30.0e6) / float(R)) * (np.float64(1.0) / T - np.float64(1.0 / 303.15))
    )


def grad_ds_a_cs_a(T, R):
    T = _to_tensor(T)
    return torch.zeros_like(T)


def ds_a_fun_simp(T, R):
    T = _to_tensor(T)
    return np.float64(DS_A_REF) * torch.ones_like(T)


def ds_c_fun(cs_c, T, R, cscamax):
    cs_c = _to_tensor(cs_c)
    theta = torch.clamp(cs_c / float(cscamax), np.float64(0.0), np.float64(1.0))
    envelope = np.float64(DS_C_MIN_FACTOR) + (np.float64(1.0) - np.float64(DS_C_MIN_FACTOR)) * torch.exp(
        -((theta - np.float64(DS_C_PEAK_CENTER)) / np.float64(DS_C_PEAK_WIDTH)) ** np.float64(2.0)
    )
    return np.float64(DS_C_REF) * envelope


def grad_ds_c_cs_c(cs_c, T, R, cscamax):
    cs_c = _to_tensor(cs_c)
    theta = torch.clamp(cs_c / float(cscamax), np.float64(0.0), np.float64(1.0))
    exp_term = torch.exp(
        -((theta - np.float64(DS_C_PEAK_CENTER)) / np.float64(DS_C_PEAK_WIDTH)) ** np.float64(2.0)
    )
    d_env_d_theta = (
        (np.float64(1.0) - np.float64(DS_C_MIN_FACTOR))
        * exp_term
        * (-np.float64(2.0) * (theta - np.float64(DS_C_PEAK_CENTER)) / (np.float64(DS_C_PEAK_WIDTH) ** 2))
    )
    return np.float64(DS_C_REF) * d_env_d_theta / float(cscamax)


def ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param):
    base = ds_c_fun(cs_c, T, R, cscamax)
    deg = torch.clamp(_to_tensor(degradation_param, base), min=np.float64(1.0e-12))
    return base / deg


def ds_c_fun_simp(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)


def ds_c_fun_plot(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)


def ds_c_fun_plot_simp(cs_c, T, R, cscamax):
    return ds_c_fun(cs_c, T, R, cscamax)


def ds_c_fun_simp_degradation_param(cs_c, T, R, cscamax, degradation_param):
    return ds_c_degradation_param_fun(cs_c, T, R, cscamax, degradation_param)


# -----------------------------------------------------------------------------
# Potential closures
# -----------------------------------------------------------------------------

def phie0_fun(i0_a, j_a, F, R, T, Uocp_a0):
    i0_a = _to_tensor(i0_a)
    return -float(j_a) * (float(R) * float(T) / i0_a) - _to_tensor(Uocp_a0, i0_a)


def phis_c0_fun(i0_a, j_a, F, R, T, Uocp_a0, j_c, i0_c, Uocp_c0):
    # Uocp_c0 may already include the t=0 terminal shift I(0)R + offset.
    phie0 = phie0_fun(i0_a, j_a, F, R, T, Uocp_a0)
    return float(j_c) * (float(R) * float(T) / _to_tensor(i0_c, phie0)) + _to_tensor(Uocp_c0, phie0) + phie0


# -----------------------------------------------------------------------------
# Parameter dictionary construction
# -----------------------------------------------------------------------------

def _theta_bounds_with_ic(
    theta_min: float,
    theta_max: float,
    theta0: float,
    floor: float = 0.0,
    ceil: float = 1.0,
) -> tuple[np.float64, np.float64]:
    lo = min(float(theta_min), float(theta0) - 1.0e-6)
    hi = max(float(theta_max), float(theta0) + 1.0e-6)
    lo = max(float(floor), lo)
    hi = min(float(ceil), hi)
    if not lo < hi:
        lo, hi = max(floor, theta0 - 0.05), min(ceil, theta0 + 0.05)
    return np.float64(lo), np.float64(hi)


def setParams(params, deg, bat, an, ca, ic):
    params["deg_i0_a_min"] = deg.bounds[deg.ind_i0_a][0]
    params["deg_i0_a_max"] = deg.bounds[deg.ind_i0_a][1]
    params["deg_ds_c_min"] = deg.bounds[deg.ind_ds_c][0]
    params["deg_ds_c_max"] = deg.bounds[deg.ind_ds_c][1]
    params["param_eff"] = deg.eff
    params["deg_i0_a_ref"] = deg.ref_vals[deg.ind_i0_a]
    params["deg_ds_c_ref"] = deg.ref_vals[deg.ind_ds_c]

    params["deg_i0_a_min_eff"] = params["deg_i0_a_ref"] + (
        params["deg_i0_a_min"] - params["deg_i0_a_ref"]
    ) * params["param_eff"]
    params["deg_i0_a_max_eff"] = params["deg_i0_a_ref"] + (
        params["deg_i0_a_max"] - params["deg_i0_a_ref"]
    ) * params["param_eff"]
    params["deg_ds_c_min_eff"] = params["deg_ds_c_ref"] + (
        params["deg_ds_c_min"] - params["deg_ds_c_ref"]
    ) * params["param_eff"]
    params["deg_ds_c_max_eff"] = params["deg_ds_c_ref"] + (
        params["deg_ds_c_max"] - params["deg_ds_c_ref"]
    ) * params["param_eff"]

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

    # Backward-compatible scalar radius scale, plus new per-electrode scales.
    # The legacy scalar is kept so old code still runs. _losses.py should later
    # prefer rescale_R_a for cs_a and rescale_R_c for cs_c.
    params["rescale_R"] = np.float64(max(params["Rs_a"], params["Rs_c"]))
    params["rescale_R_a"] = np.float64(params["Rs_a"])
    params["rescale_R_c"] = np.float64(params["Rs_c"])
    params["rescale_R_negative"] = params["rescale_R_a"]
    params["rescale_R_positive"] = params["rescale_R_c"]
    params["radial_rescale_mode"] = "per_electrode"
    params["use_per_electrode_rescale_R"] = True

    params["csanmax"] = an.csmax
    params["cscamax"] = ca.csmax
    params["rescale_T"] = np.float64(max(bat.tmax, 1.0e-16))
    params["mag_cs_a"] = np.float64(max(1.0, float(an.csmax)))
    params["mag_cs_c"] = np.float64(51.8)
    params["mag_phis_c"] = np.float64(4.25)
    params["mag_phie"] = np.float64(0.7)
    params["mag_ce"] = np.float64(1.2)

    # Raw OCP functions. Terminal shifts are handled explicitly in _losses.py.
    params["Uocp_a"] = an.uocp
    params["Uocp_c"] = ca.uocp
    params["Uocp_c_raw"] = ca.uocp
    params["i0_a"] = an.i0
    params["i0_c"] = ca.i0
    params["D_s_a"] = an.ds
    params["D_s_c"] = ca.ds

    params["R_ohm_eff"] = np.float64(R_OHM_EFF)
    params["voltage_alignment_offset_V"] = np.float64(VOLTAGE_ALIGNMENT_OFFSET_V)
    params["theta_c_bottom"] = np.float64(THETA_C_BOTTOM)
    params["theta_c_top"] = np.float64(THETA_C_TOP)
    params["ce_eff_physical"] = np.float64(1.0)
    params["ce_ref_code"] = np.float64(1.2)
    params["cs_rescale_mode"] = "cycle"

    params["ce0"] = ic.ce
    params["ce_a0"] = ic.ce
    params["ce_c0"] = ic.ce
    params["cs_a0"] = ic.an.cs
    params["cs_c0"] = ic.ca.cs

    params["eps_s_a"] = an.solids.eps
    params["eps_s_c"] = ca.solids.eps
    params["L_a"] = an.thickness
    params["L_c"] = ca.thickness
    params["V_a"] = params["A_a"] * params["L_a"]
    params["V_c"] = params["A_c"] * params["L_c"]

    j_a = -params["I_discharge"] * params["Rs_a"] / (
        np.float64(3.0) * params["eps_s_a"] * params["F"] * params["V_a"]
    )
    j_c = params["I_discharge"] * params["Rs_c"] / (
        np.float64(3.0) * params["eps_s_c"] * params["F"] * params["V_c"]
    )
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

    Uocp_c_raw = params["Uocp_c_raw"](torch.tensor(cse_c, dtype=torch.float64), params["cscamax"])
    params["Uocp_c0_raw"] = float(Uocp_c_raw)
    initial_terminal_shift = (
        float(params["I_discharge"]) * float(params["R_ohm_eff"])
        + float(params["voltage_alignment_offset_V"])
    )
    params["Uocp_c0"] = float(Uocp_c_raw) + initial_terminal_shift
    params["phis_c0"] = phis_c0_fun

    # Concentration bounds for the bidirectional rescaler.
    theta_a0 = float(params["cs_a0"]) / max(float(params["csanmax"]), 1.0e-12)
    theta_c0 = float(params["cs_c0"]) / max(float(params["cscamax"]), 1.0e-12)
    theta_a_min, theta_a_max = _theta_bounds_with_ic(
        THETA_A_MIN_RESCALE, THETA_A_MAX_RESCALE, theta_a0
    )
    theta_c_min, theta_c_max = _theta_bounds_with_ic(
        THETA_C_MIN_RESCALE, THETA_C_MAX_RESCALE, theta_c0
    )

    params["theta_a_min"] = theta_a_min
    params["theta_a_max"] = theta_a_max
    params["theta_c_min"] = theta_c_min
    params["theta_c_max"] = theta_c_max
    params["cs_a_min"] = np.float64(theta_a_min * params["csanmax"])
    params["cs_a_upper"] = np.float64(theta_a_max * params["csanmax"])
    params["cs_c_min"] = np.float64(theta_c_min * params["cscamax"])
    params["cs_c_upper"] = np.float64(theta_c_max * params["cscamax"])
    params["cs_a_rescale_span"] = np.float64(params["cs_a_upper"] - params["cs_a_min"])
    params["cs_c_rescale_span"] = np.float64(params["cs_c_upper"] - params["cs_c_min"])

    # Keep legacy data-loss scaling unchanged for now. The narrower bounds above
    # are what _rescale.py uses to constrain the predicted concentration states.
    params["rescale_cs_a"] = params["csanmax"]
    params["rescale_cs_c"] = params["cscamax"]
    params["rescale_phis_c"] = np.float64(1.0)
    params["rescale_phie"] = np.float64(1.0)

    # Make provenance visible in smoke tests and ModelFin_*/config.json.
    params["ocp_table_source"] = "csv" if (_POS_TABLE is not None and _NEG_TABLE is not None) else "legacy_fallback"
    params["soft_label_summary_path"] = _SOFT_SUMMARY.get("_summary_path", "")
    params["thermo_patch_note"] = (
        "2026-04 cycle5 debug: added per-electrode radial rescale params "
        "(rescale_R_a/rescale_R_c) and narrower theta bounds. Legacy rescale_R is kept."
    )

    return params
