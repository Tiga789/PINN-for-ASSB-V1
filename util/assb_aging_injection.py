# -*- coding: utf-8 -*-
"""Aging injection hooks for ASSB effective-SPM closure.

These helpers are pure functions: they do not train a neural net.  Stage-C
patches to ``_rescale.py`` and ``_losses.py`` should call them so that aging
enters cbar, flux, OCP-window and R_ohm terms consistently.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from util.assb_aging_fix1_config import assert_fixed_material_identity


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for assb_aging_injection")


def _param(params: Dict[str, Any], names, default: float) -> float:
    for name in names:
        if name in params and params[name] is not None:
            try:
                return float(params[name])
            except Exception:
                pass
        upper = str(name).upper()
        if upper in params and params[upper] is not None:
            try:
                return float(params[upper])
            except Exception:
                pass
    return float(default)


def _as_tensor(value: float, like):
    _require_torch()
    if torch.is_tensor(like):
        return torch.as_tensor(float(value), dtype=like.dtype, device=like.device)
    return torch.as_tensor(float(value), dtype=torch.float64)


def _interp1d(x_grid, y_grid, x):
    _require_torch()
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float64)
    x_grid = torch.as_tensor(x_grid, dtype=x.dtype, device=x.device).flatten()
    y_grid = torch.as_tensor(y_grid, dtype=x.dtype, device=x.device).flatten()
    if x_grid.numel() == 0:
        return torch.zeros_like(x)
    if x_grid.numel() == 1:
        return torch.full_like(x, y_grid[0])
    idx = torch.searchsorted(x_grid.contiguous(), x.contiguous(), right=True) - 1
    idx = torch.clamp(idx, 0, x_grid.numel() - 2)
    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    y0 = y_grid[idx]
    y1 = y_grid[idx + 1]
    denom = torch.clamp(x1 - x0, min=torch.as_tensor(1.0e-12, dtype=x.dtype, device=x.device))
    w = torch.clamp((x - x0) / denom, 0.0, 1.0)
    return y0 + w * (y1 - y0)


def current_at_t(params: Dict[str, Any], t):
    """Return I(t) in Ampere using current profile tensors in params."""
    _require_torch()
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    for pair in [("current_t_s", "current_I_A"), ("t_global_s", "I_profile"), ("t_s", "I_A")]:
        if pair[0] in params and pair[1] in params:
            return _interp1d(params[pair[0]], params[pair[1]], t)
    if "I_discharge" in params:
        return torch.full_like(t, float(params["I_discharge"]))
    raise KeyError("No current profile found in params. Expected current_t_s/current_I_A or I_profile.")


def _cycle_index_from_t(params: Dict[str, Any], t):
    _require_torch()
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    if "cycle_t_start_s" in params and "cycle_id_profile" in params:
        starts = torch.as_tensor(params["cycle_t_start_s"], dtype=t.dtype, device=t.device).flatten()
        idx = torch.searchsorted(starts.contiguous(), t.contiguous(), right=True) - 1
        return torch.clamp(idx, 0, starts.numel() - 1)
    if "cycle_id_at_t" in params:
        return params["cycle_id_at_t"](t)
    raise KeyError("No cycle table lookup found in params. Expected cycle_t_start_s/cycle_id_profile.")


def cycle_profile_at_t(params: Dict[str, Any], t, profiles: Any, field: str):
    """Nearest-cycle lookup of an aging profile field at physical time *t*."""
    _require_torch()
    idx = _cycle_index_from_t(params, t)
    values = getattr(profiles, field)
    values = torch.as_tensor(values, dtype=t.dtype if torch.is_tensor(t) else torch.float64, device=idx.device).flatten()
    return values[idx]


def aged_eps_s_c(params: Dict[str, Any], t, profiles: Any, *, enabled: Optional[bool] = None):
    """Effective positive-electrode active volume fraction.

    ``c_s,max`` is intentionally not changed.  LAM changes the effective active
    fraction / usable active amount only.
    """
    _require_torch()
    assert_fixed_material_identity(params)
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    use = bool(params.get("USE_ASSB_AGING_INJECTION_CBAR", False) or params.get("USE_ASSB_AGING_INJECTION_FLUX", False)) if enabled is None else bool(enabled)
    eps0 = _as_tensor(_param(params, ["eps_s_c", "EPS_S_C"], 0.55), t)
    if not use or profiles is None:
        return eps0 + torch.zeros_like(t)
    f_lam = cycle_profile_at_t(params, t, profiles, "f_LAM_c")
    return torch.clamp(eps0 * f_lam, min=torch.as_tensor(1.0e-9, dtype=t.dtype, device=t.device))


def aged_surface_flux(params: Dict[str, Any], t, electrode: str, profiles: Optional[Any] = None):
    """Surface flux J_a or J_c with optional positive-electrode LAM injection.

    Convention is fixed material identity:
    ``a`` = Li-In/In negative electrode, ``c`` = NMC811 positive electrode.
    Current sign changes the sign of J only; it never swaps materials.
    """
    _require_torch()
    assert_fixed_material_identity(params)
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    I = current_at_t(params, t)
    F = _as_tensor(_param(params, ["F"], 96485.3329), t)
    if electrode.lower().startswith("c"):
        Rs = _as_tensor(_param(params, ["Rs_c", "RS_C"], 1.8e-6), t)
        V = _as_tensor(_param(params, ["V_c", "A_c_times_L_c"], _param(params, ["A_c"], 7.853981633974483e-5) * _param(params, ["L_c"], 16e-6)), t)
        eps = aged_eps_s_c(params, t, profiles, enabled=bool(params.get("USE_ASSB_AGING_INJECTION_FLUX", False)))
        return I * Rs / (3.0 * eps * F * V)
    if electrode.lower().startswith("a"):
        Rs = _as_tensor(_param(params, ["Rs_a", "RS_A"], 50e-6), t)
        V = _as_tensor(_param(params, ["V_a", "A_a_times_L_a"], _param(params, ["A_a"], 7.853981633974483e-5) * _param(params, ["L_a"], 100e-6)), t)
        eps = _as_tensor(_param(params, ["eps_s_a", "EPS_S_A"], 0.95), t)
        return -I * Rs / (3.0 * eps * F * V)
    raise ValueError("electrode must be 'a' or 'c'")


def dynamic_theta_window(params: Dict[str, Any], t, profiles: Optional[Any] = None) -> Tuple[Any, Any, Any]:
    """Return aged positive-electrode theta window (bottom, top, scale)."""
    _require_torch()
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    bottom0 = _as_tensor(_param(params, ["theta_c_bottom", "THETA_C_BOTTOM"], 0.834), t)
    top0 = _as_tensor(_param(params, ["theta_c_top", "THETA_C_TOP"], 0.432), t)
    use = bool(params.get("USE_ASSB_AGING_INJECTION_THETA_WINDOW", False))
    if not use or profiles is None:
        scale = torch.ones_like(t)
    else:
        scale = cycle_profile_at_t(params, t, profiles, "theta_window_scale_c")
    center = 0.5 * (bottom0 + top0)
    half = 0.5 * torch.abs(bottom0 - top0) * scale
    # Keep orientation from the base window.
    sign = torch.sign(bottom0 - top0)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    bottom = center + sign * half
    top = center - sign * half
    return bottom, top, scale


def terminal_shift(params: Dict[str, Any], t, profiles: Optional[Any] = None):
    """Return I(t) * (R_ohm_eff(k) - R_ohm_base) if R injection is enabled."""
    _require_torch()
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    if not bool(params.get("USE_ASSB_AGING_INJECTION_ROHM", False)) or profiles is None:
        return torch.zeros_like(t)
    if not bool(params.get("LOCK_COMMON_MODE_GAUGE", True)):
        raise RuntimeError("R_ohm aging injection requires LOCK_COMMON_MODE_GAUGE=True")
    I = current_at_t(params, t)
    r_base = _as_tensor(_param(params, ["R_ohm_eff", "AGING_R_OHM_BASE", "r_ohm_base"], 105.0), t)
    r_eff = cycle_profile_at_t(params, t, profiles, "R_ohm_eff")
    return I * (r_eff - r_base)


__all__ = [
    "current_at_t",
    "cycle_profile_at_t",
    "aged_eps_s_c",
    "aged_surface_flux",
    "dynamic_theta_window",
    "terminal_shift",
]
