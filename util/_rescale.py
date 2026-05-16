# -*- coding: utf-8 -*-
"""
ASSB ModelFin_110 aging-fix1 output transforms.

Complete replacement file for ModelFin_110 aging-fix1.
It keeps the 107A-style I(t)-cbar hard baseline and current-aware potential
baseline, and adds explicit aging injection hooks for positive-electrode LAM,
theta-window shrinkage, and optional R_ohm growth.

Important physical convention:
- a / negative side is always Li-In/In.
- c / positive side is always NMC811.
- current sign changes flux direction only; it does not swap material identity.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (str(_ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .assb_aging_injection import current_at_t as _aging_current_at_t, aged_surface_flux, dynamic_theta_window, terminal_shift as aged_terminal_shift
except Exception:  # pragma: no cover
    from assb_aging_injection import current_at_t as _aging_current_at_t, aged_surface_flux, dynamic_theta_window, terminal_shift as aged_terminal_shift


def _to_tensor(x, like: Optional[torch.Tensor] = None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x
        if like is not None:
            out = out.to(dtype=like.dtype, device=like.device)
        else:
            out = out.to(dtype=torch.float64, device=device if device is not None else x.device)
    else:
        dtype = torch.float64 if like is None else like.dtype
        dev = device if like is None else like.device
        out = torch.as_tensor(x, dtype=dtype, device=dev)
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "t", "on"}:
        return True
    if s in {"false", "0", "no", "n", "f", "off", "none", "null", ""}:
        return False
    return bool(default)


def _param_float(params: dict, names, default: float) -> float:
    for name in names:
        if name in params and params[name] is not None:
            try:
                v = params[name]
                if isinstance(v, torch.Tensor):
                    return float(v.detach().reshape(-1)[0].cpu())
                return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])
            except Exception:
                try:
                    return float(params[name])
                except Exception:
                    pass
    return float(default)


def _interp1d_torch(x_grid: torch.Tensor, y_grid: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = _to_tensor(x, like=x).reshape(-1)
    x_grid = _to_tensor(x_grid, like=x).reshape(-1).detach()
    y_grid = _to_tensor(y_grid, like=x).reshape(-1).detach()
    if x_grid.numel() == 0 or y_grid.numel() == 0:
        return torch.zeros_like(x).reshape(-1, 1)
    if x_grid.numel() == 1:
        return y_grid[0].expand_as(x).reshape(-1, 1)
    order = torch.argsort(x_grid)
    xg = x_grid[order]
    yg = y_grid[order]
    q = torch.clamp(x, min=xg[0], max=xg[-1])
    idx_hi = torch.searchsorted(xg, q, right=False)
    idx_hi = torch.clamp(idx_hi, min=1, max=xg.numel() - 1)
    idx_lo = idx_hi - 1
    x0 = xg[idx_lo]
    x1 = xg[idx_hi]
    y0 = yg[idx_lo]
    y1 = yg[idx_hi]
    w = (q - x0) / torch.clamp(x1 - x0, min=torch.as_tensor(1.0e-12, dtype=q.dtype, device=q.device))
    return (y0 + w * (y1 - y0)).reshape(-1, 1)


def _electrode_volume(params: dict, electrode: str) -> float:
    e = electrode.lower()
    if e.startswith(("a", "n")):
        return _param_float(params, ("V_a", "V_n", "volume_a"), _param_float(params, ("A_a",), 1.0) * _param_float(params, ("L_a",), 1.0))
    return _param_float(params, ("V_c", "V_p", "volume_c"), _param_float(params, ("A_c",), 1.0) * _param_float(params, ("L_c",), 1.0))


def _current_profile_tensors(params: dict, like: torch.Tensor):
    # Prefer explicit current_profile tuple; fall back to split arrays.
    prof = params.get("current_profile", None)
    if isinstance(prof, dict):
        t = prof.get("t", prof.get("time_s", prof.get("time", None)))
        i = prof.get("I", prof.get("current_A", prof.get("current", None)))
        if t is not None and i is not None:
            return _to_tensor(t, like=like).reshape(-1), _to_tensor(i, like=like).reshape(-1)
    if isinstance(prof, (tuple, list)) and len(prof) == 2:
        return _to_tensor(prof[0], like=like).reshape(-1), _to_tensor(prof[1], like=like).reshape(-1)
    t = params.get("time_profile", params.get("t_profile", params.get("I_profile_t", None)))
    i = params.get("current_profile_A", params.get("I_profile_A", params.get("I_values", None)))
    if t is not None and i is not None:
        return _to_tensor(t, like=like).reshape(-1), _to_tensor(i, like=like).reshape(-1)
    return None


def _current_at_t(params: dict, t_like: torch.Tensor) -> torch.Tensor:
    try:
        return _aging_current_at_t(params, t_like)
    except Exception:
        pass
    t_like = _to_tensor(t_like, like=t_like)
    prof = _current_profile_tensors(params, t_like)
    if prof is not None:
        return _interp1d_torch(prof[0], prof[1], t_like)
    I = _param_float(params, ("I_discharge", "I", "I_app", "current_A"), 0.0)
    return torch.full_like(t_like, np.float64(I))


def _surface_flux_from_current(params: dict, t_like: torch.Tensor, electrode: str, aging_profiles=None) -> torch.Tensor:
    t_like = _to_tensor(t_like, like=t_like)
    if aging_profiles is not None:
        try:
            return aged_surface_flux(params, t_like, electrode, aging_profiles)
        except Exception:
            pass
    I_t = _current_at_t(params, t_like)
    F = _param_float(params, ("F",), 96485.33212)
    e = electrode.lower()
    if e.startswith(("a", "n")):
        return -I_t * _param_float(params, ("Rs_a", "R_s_a"), 50.0e-6) / (
            np.float64(3.0) * _param_float(params, ("eps_s_a", "epsilon_s_a"), 0.95) * F * _electrode_volume(params, "a")
        )
    return I_t * _param_float(params, ("Rs_c", "R_s_c"), 1.8e-6) / (
        np.float64(3.0) * _param_float(params, ("eps_s_c", "epsilon_s_c"), 0.55) * F * _electrode_volume(params, "c")
    )


def _radial_rescale(params: dict, electrode: str) -> float:
    e = electrode.lower()
    if e.startswith(("a", "n")):
        return _param_float(params, ("rescale_R_a", "Rs_a", "rescale_R"), 50.0e-6)
    return _param_float(params, ("rescale_R_c", "Rs_c", "rescale_R"), 1.8e-6)


def _hard_ic_gate(t: torch.Tensor, timescale: float) -> torch.Tensor:
    return 1.0 - torch.exp(-_to_tensor(t, like=t) / np.float64(max(float(timescale), 1.0e-12)))


def build_cbar_profiles_from_current(params: dict) -> None:
    """Legacy non-aged cbar profiles used for the anode and fallback cathode."""
    if params.get("_assb_cbar_profiles_ready", False):
        return
    try:
        dummy = torch.zeros((1, 1), dtype=torch.float64)
        prof = _current_profile_tensors(params, dummy)
        if prof is None:
            return
        t = prof[0].detach().cpu().numpy().astype(float)
        I = prof[1].detach().cpu().numpy().astype(float)
        if t.size < 2:
            return
        order = np.argsort(t)
        t = t[order]
        I = I[order]
        dt = np.diff(t, prepend=t[0])
        dt[dt < 0.0] = 0.0
        I_mid = np.concatenate([[I[0]], 0.5 * (I[1:] + I[:-1])])
        q_int = np.cumsum(I_mid * dt)
        F = _param_float(params, ("F",), 96485.33212)
        cbar_a0 = _param_float(params, ("cs_a0", "csa0"), 0.0)
        cbar_c0 = _param_float(params, ("cs_c0", "csc0"), 0.0)
        eps_a = _param_float(params, ("eps_s_a",), 0.95)
        eps_c = _param_float(params, ("eps_s_c",), 0.55)
        V_a = _electrode_volume(params, "a")
        V_c = _electrode_volume(params, "c")
        params["cbar_profile_t"] = t
        params["cbar_profile_a"] = cbar_a0 + q_int / max(eps_a * F * V_a, 1.0e-30)
        params["cbar_profile_c"] = cbar_c0 - q_int / max(eps_c * F * V_c, 1.0e-30)
        params["_assb_cbar_profiles_ready"] = True
    except Exception:
        return


def _cbar_from_profile(params: dict, t_like: torch.Tensor, electrode: str) -> Optional[torch.Tensor]:
    build_cbar_profiles_from_current(params)
    key = "cbar_profile_a" if electrode.lower().startswith(("a", "n")) else "cbar_profile_c"
    if "cbar_profile_t" not in params or key not in params:
        return None
    return _interp1d_torch(_to_tensor(params["cbar_profile_t"], like=t_like), _to_tensor(params[key], like=t_like), t_like)


def _get_aging_profiles_for_self(self):
    if not _as_bool(self.params.get("USE_ASSB_AGING_FIX1", self.params.get("USE_ASSB_AGING_MECHANISM", False)), False):
        return None
    if hasattr(self, "get_aging_profiles"):
        try:
            return self.get_aging_profiles()
        except Exception:
            return None
    return None


def _global_current_cumulative_C(params: dict):
    if "_assb_current_cum_ready" in params:
        return params["_assb_current_cum_t"], params["_assb_current_cum_C"]
    dummy = torch.zeros((1, 1), dtype=torch.float64)
    prof = _current_profile_tensors(params, dummy)
    if prof is None:
        return None, None
    t = prof[0].detach().cpu().numpy().astype(float)
    I = prof[1].detach().cpu().numpy().astype(float)
    order = np.argsort(t)
    t = t[order]
    I = I[order]
    if t.size < 2:
        q = np.zeros_like(t)
    else:
        dt = np.diff(t, prepend=t[0])
        dt[dt < 0.0] = 0.0
        I_mid = np.concatenate([[I[0]], 0.5 * (I[1:] + I[:-1])])
        q = np.cumsum(I_mid * dt)
    params["_assb_current_cum_t"] = t
    params["_assb_current_cum_C"] = q
    params["_assb_current_cum_ready"] = True
    return t, q


def _cycle_index_from_t_params(params: dict, t_like: torch.Tensor) -> Optional[torch.Tensor]:
    if "cycle_t_start_s" not in params:
        return None
    starts = _to_tensor(params["cycle_t_start_s"], like=t_like).reshape(-1)
    if starts.numel() == 0:
        return None
    tflat = _to_tensor(t_like, like=t_like).reshape(-1)
    idx = torch.searchsorted(starts.contiguous(), tflat.contiguous(), right=True) - 1
    return torch.clamp(idx, 0, starts.numel() - 1)


def _aged_cbar_c_from_cycle_table(params: dict, t_like: torch.Tensor, aging_profiles) -> Optional[torch.Tensor]:
    """Positive cbar with the same f_LAM_c in flux, cbar and capacity."""
    if aging_profiles is None:
        return None
    try:
        t = _to_tensor(t_like, like=t_like)
        idx = _cycle_index_from_t_params(params, t)
        if idx is None or "q_net_cycle_C" not in params:
            return None
        idx = idx.reshape(-1).to(dtype=torch.long, device=t.device)
        f = torch.as_tensor(aging_profiles.f_LAM_c, dtype=t.dtype, device=t.device).reshape(-1)
        q_cycle = _to_tensor(params["q_net_cycle_C"], like=t).reshape(-1)
        n = min(int(f.numel()), int(q_cycle.numel()))
        if n <= 0:
            return None
        f = torch.clamp(f[:n], min=1.0e-6)
        q_cycle = q_cycle[:n]
        idx = torch.clamp(idx, 0, n - 1)
        denom = (
            _param_float(params, ("eps_s_c",), 0.55)
            * _param_float(params, ("F",), 96485.33212)
            * _electrode_volume(params, "c")
            * f
        )
        per_cycle_dc = q_cycle / torch.clamp(denom, min=1.0e-30)
        prefix = torch.cumsum(per_cycle_dc, dim=0) - per_cycle_dc

        qt, qcum = _global_current_cumulative_C(params)
        if qt is None:
            within = torch.zeros_like(t.reshape(-1))
        else:
            q_at_t = _interp1d_torch(_to_tensor(qt, like=t), _to_tensor(qcum, like=t), t).reshape(-1)
            starts = _to_tensor(params["cycle_t_start_s"], like=t).reshape(-1)[:n]
            q_at_start = _interp1d_torch(_to_tensor(qt, like=t), _to_tensor(qcum, like=t), starts[idx]).reshape(-1)
            within = q_at_t - q_at_start
        cbar0 = _param_float(params, ("cs_c0", "csc0"), 0.0)
        out = np.float64(cbar0) - prefix[idx] - within / torch.clamp(denom[idx], min=1.0e-30)
        return out.reshape(-1, 1)
    except Exception:
        return None


def _dynamic_cathode_bounds(params: dict, t: torch.Tensor, cmax: float, aging_profiles):
    lower = _param_float(params, ("cs_c_min",), 0.0)
    upper = _param_float(params, ("cs_c_upper",), cmax)
    if aging_profiles is None or not _as_bool(params.get("USE_ASSB_AGING_INJECTION_THETA_WINDOW", False), False):
        return lower, upper
    try:
        bottom, top, _scale = dynamic_theta_window(params, t, aging_profiles)
        lo = torch.minimum(bottom, top) * np.float64(cmax)
        hi = torch.maximum(bottom, top) * np.float64(cmax)
        return lo, hi
    except Exception:
        return lower, upper


def _bounded_delta(raw: torch.Tensor, center: torch.Tensor, lower, upper, fraction: float) -> torch.Tensor:
    z = torch.tanh(_to_tensor(raw, like=center))
    center_t = _to_tensor(center, like=raw)
    if not isinstance(lower, torch.Tensor):
        lower_t = torch.as_tensor(float(lower), dtype=center_t.dtype, device=center_t.device)
    else:
        lower_t = lower.to(dtype=center_t.dtype, device=center_t.device)
    if not isinstance(upper, torch.Tensor):
        upper_t = torch.as_tensor(float(upper), dtype=center_t.dtype, device=center_t.device)
    else:
        upper_t = upper.to(dtype=center_t.dtype, device=center_t.device)
    pos = torch.clamp(upper_t - center_t, min=torch.as_tensor(1.0e-12, dtype=center_t.dtype, device=center_t.device))
    neg = torch.clamp(center_t - lower_t, min=torch.as_tensor(1.0e-12, dtype=center_t.dtype, device=center_t.device))
    return np.float64(float(fraction)) * torch.where(z >= 0.0, pos * z, neg * z)


def _radial_basis(params: dict, r: torch.Tensor, electrode: str, zero_mean_flag: bool) -> torch.Tensor:
    e = electrode.lower()
    Rs = _param_float(params, ("Rs_a",), 50e-6) if e.startswith(("a", "n")) else _param_float(params, ("Rs_c",), 1.8e-6)
    s = torch.clamp(_to_tensor(r, like=r) / np.float64(max(Rs, 1.0e-30)), 0.0, 1.0)
    if zero_mean_flag:
        return s.square() - np.float64(0.6)  # spherical average of s^2 is 3/5.
    return s.square()


def _cs_target(self, raw: torch.Tensor, t: torch.Tensor, r: torch.Tensor, electrode: str, clip: bool = True) -> torch.Tensor:
    params = self.params
    is_a = electrode.lower().startswith(("a", "n"))
    start = _param_float(params, ("cs_a0", "csa0"), 0.0) if is_a else _param_float(params, ("cs_c0", "csc0"), 0.0)
    cmax = _param_float(params, ("csanmax",), max(start, 1.0)) if is_a else _param_float(params, ("cscamax",), max(start, 1.0))
    lower = _param_float(params, ("cs_a_min",), 0.0) if is_a else _param_float(params, ("cs_c_min",), 0.0)
    upper = _param_float(params, ("cs_a_upper",), cmax) if is_a else _param_float(params, ("cs_c_upper",), cmax)
    use_cbar = _as_bool(params.get("use_i_cbar_baseline_a" if is_a else "use_i_cbar_baseline_c", False), False)
    frac = _param_float(params, ("cbar_deviation_fraction_a",), 0.15) if is_a else _param_float(params, ("cbar_deviation_fraction_c",), 0.10)
    zero_mean = _as_bool(params.get("use_zero_mean_radial_deviation_a" if is_a else "use_zero_mean_radial_deviation_c", False), False)
    t = _to_tensor(t, like=raw)
    r = _to_tensor(r, like=raw)
    aging_profiles = _get_aging_profiles_for_self(self)
    base = None
    if use_cbar and not is_a and aging_profiles is not None:
        base = _aged_cbar_c_from_cycle_table(params, t, aging_profiles)
    if use_cbar and base is None:
        base = _cbar_from_profile(params, t, "a" if is_a else "c")
    if base is None:
        base = torch.full_like(t, np.float64(start))
    if not is_a:
        lower, upper = _dynamic_cathode_bounds(params, t, cmax, aging_profiles)
    gate = _hard_ic_gate(t, float(getattr(self, "hard_IC_timescale", params.get("HARD_IC_TIMESCALE", 1.0))))
    delta = _bounded_delta(raw, base, lower, upper, frac)
    basis = _radial_basis(params, r, "a" if is_a else "c", zero_mean)
    out = base + gate * delta * basis
    if clip:
        if isinstance(lower, torch.Tensor) or isinstance(upper, torch.Tensor):
            lower_t = lower if isinstance(lower, torch.Tensor) else torch.as_tensor(lower, dtype=out.dtype, device=out.device)
            upper_t = upper if isinstance(upper, torch.Tensor) else torch.as_tensor(upper, dtype=out.dtype, device=out.device)
            out = torch.maximum(torch.minimum(out, upper_t), lower_t)
        else:
            out = torch.clamp(out, min=float(lower), max=float(upper))
    return out


def _terminal_voltage_shift(params: dict, t_like: torch.Tensor, aging_profiles=None) -> torch.Tensor:
    t_like = _to_tensor(t_like, like=t_like)
    offset = _param_float(params, ("voltage_alignment_offset_V", "voltage_offset", "V_OFFSET"), 0.0)
    if aging_profiles is not None and _as_bool(params.get("USE_ASSB_AGING_INJECTION_ROHM", False), False):
        try:
            return aged_terminal_shift(params, t_like, aging_profiles) + np.float64(offset)
        except Exception:
            pass
    I_t = _current_at_t(params, t_like)
    r_ohm = _param_float(params, ("R_ohm_eff", "R_ohm", "AGING_R_OHM0"), 105.0)
    return I_t * np.float64(r_ohm) + np.float64(offset) * torch.ones_like(t_like)


# Public methods bound by myNN -------------------------------------------------

def rescale_param(self, param: torch.Tensor, ind_param: int = 0) -> torch.Tensor:
    return _to_tensor(param, like=param)


def unrescale_param(self, param: torch.Tensor, ind_param: int = 0) -> torch.Tensor:
    return _to_tensor(param, like=param)


def fix_param(self, param: torch.Tensor, ind_param: int = 0) -> torch.Tensor:
    return _to_tensor(param, like=param)


def rescaleCs_a(self, output: torch.Tensor, t: torch.Tensor, r: torch.Tensor, deg_i0_a=None, deg_ds_c=None, clip: bool = True) -> torch.Tensor:
    return _cs_target(self, output, t, r, "a", clip=clip)


def rescaleCs_c(self, output: torch.Tensor, t: torch.Tensor, r: torch.Tensor, deg_i0_a=None, deg_ds_c=None, clip: bool = True) -> torch.Tensor:
    return _cs_target(self, output, t, r, "c", clip=clip)


def rescalePhie(self, output: torch.Tensor, t: torch.Tensor, deg_i0_a=None, deg_ds_c=None) -> torch.Tensor:
    t = _to_tensor(t, like=output)
    start = _param_float(self.params, ("phie0",), 0.0)
    scale = _param_float(self.params, ("rescale_phie", "phie_scale"), 1.0)
    frac = _param_float(self.params, ("potential_baseline_correction_fraction_phie",), 0.20)
    aging_profiles = _get_aging_profiles_for_self(self)
    base = torch.zeros_like(t)
    if _as_bool(self.params.get("use_current_potential_baseline", False), False) or _as_bool(self.params.get("use_current_potential_baseline_phie", False), False):
        base = -_terminal_voltage_shift(self.params, t, aging_profiles)
    gate = _hard_ic_gate(t, float(getattr(self, "hard_IC_timescale", self.params.get("HARD_IC_TIMESCALE", 1.0))))
    return np.float64(start) + base + gate * np.float64(frac * scale) * torch.tanh(_to_tensor(output, like=t))


def rescalePhis_c(self, output: torch.Tensor, t: torch.Tensor, deg_i0_a=None, deg_ds_c=None) -> torch.Tensor:
    t = _to_tensor(t, like=output)
    start = _param_float(self.params, ("phis_c0", "phis0"), 0.0)
    scale = _param_float(self.params, ("rescale_phis_c", "phis_c_scale"), 1.0)
    frac = _param_float(self.params, ("potential_baseline_correction_fraction_phis_c",), 0.20)
    aging_profiles = _get_aging_profiles_for_self(self)
    base = torch.zeros_like(t)
    if _as_bool(self.params.get("use_current_potential_baseline", False), False) or _as_bool(self.params.get("use_current_potential_baseline_phis_c", False), False):
        base = _terminal_voltage_shift(self.params, t, aging_profiles)
    gate = _hard_ic_gate(t, float(getattr(self, "hard_IC_timescale", self.params.get("HARD_IC_TIMESCALE", 1.0))))
    return np.float64(start) + base + gate * np.float64(frac * scale) * torch.tanh(_to_tensor(output, like=t))


# Compatibility HNN helpers. They are intentionally thin because ModelFin_109 is
# delivered as a complete non-indirect-loader file set.
def get_phie0(self, t, deg_i0_a=None, deg_ds_c=None):
    return torch.full_like(_to_tensor(t, device=getattr(self, "device", None)), np.float64(_param_float(self.params, ("phie0",), 0.0)))


def get_phis_c0(self, t, deg_i0_a=None, deg_ds_c=None):
    return torch.full_like(_to_tensor(t, device=getattr(self, "device", None)), np.float64(_param_float(self.params, ("phis_c0", "phis0"), 0.0)))


def get_phie_hnn(*args, **kwargs): return None

def get_phis_c_hnn(*args, **kwargs): return None

def get_cs_a_hnn(*args, **kwargs): return None

def get_cs_c_hnn(*args, **kwargs): return None

def get_phie_hnntime(*args, **kwargs): return None

def get_phis_c_hnntime(*args, **kwargs): return None

def get_cs_a_hnntime(*args, **kwargs): return None

def get_cs_c_hnntime(*args, **kwargs): return None


def capacity_soh_runtime(params: dict, default: float = 1.0) -> float:
    try:
        return float(params.get("capacity_soh_runtime", default))
    except Exception:
        return float(default)


__all__ = [
    "_to_tensor",
    "_current_at_t",
    "_surface_flux_from_current",
    "_radial_rescale",
    "_terminal_voltage_shift",
    "build_cbar_profiles_from_current",
    "rescale_param",
    "unrescale_param",
    "fix_param",
    "rescaleCs_a",
    "rescaleCs_c",
    "rescalePhie",
    "rescalePhis_c",
    "get_phie0",
    "get_phis_c0",
    "get_phie_hnn",
    "get_phis_c_hnn",
    "get_cs_a_hnn",
    "get_cs_c_hnn",
    "get_phie_hnntime",
    "get_phis_c_hnntime",
    "get_cs_a_hnntime",
    "get_cs_c_hnntime",
    "capacity_soh_runtime",
]
