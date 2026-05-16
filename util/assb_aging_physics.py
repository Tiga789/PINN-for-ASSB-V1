# -*- coding: utf-8 -*-
"""
Aging-aware effective SPM helper functions for ASSB ModelFin_109.

These functions centralize the places where aging variables enter the existing
107A-style physics closure.  Later patches to _rescale.py and _losses.py should
call these helpers instead of duplicating aging logic.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _as_float_param(params: Dict[str, object], *names: str, default: float) -> float:
    for name in names:
        if name in params and params[name] is not None:
            try:
                return float(params[name])
            except Exception:
                continue
    return float(default)


def _tensor_like(value: float, like):
    if torch is None:  # pragma: no cover
        return value
    if torch.is_tensor(like):
        return torch.as_tensor(float(value), dtype=like.dtype, device=like.device)
    return torch.as_tensor(float(value), dtype=torch.float64)


def _get_profile_at_t(params: Dict[str, object], t, aging_profiles, field: str):
    if torch is None:
        raise RuntimeError("PyTorch is required for aging physics helpers.")
    from util.assb_cycle_table import cycle_at_t

    if aging_profiles is None:
        raise ValueError("aging_profiles must be provided when USE_ASSB_AGING_MECHANISM=True.")
    idx = cycle_at_t(params, t)
    value = getattr(aging_profiles, field)
    return value.to(device=idx.device, dtype=t.dtype if torch.is_tensor(t) else value.dtype)[idx]


def _interp1d_torch(x_grid, y_grid, x):
    """Piecewise-linear interpolation for one-dimensional torch tensors."""
    if torch is None:
        raise RuntimeError("PyTorch is required for _interp1d_torch().")
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float64)
    x_grid = torch.as_tensor(x_grid, dtype=x.dtype, device=x.device)
    y_grid = torch.as_tensor(y_grid, dtype=x.dtype, device=x.device)
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
    w = torch.clamp((x - x0) / torch.clamp(x1 - x0, min=torch.as_tensor(1.0e-12, dtype=x.dtype, device=x.device)), 0.0, 1.0)
    return y0 + w * (y1 - y0)


def current_at_t(params: Dict[str, object], t):
    """Return I(t) in Ampere using the current profile stored in params.

    Supported parameter names are intentionally broad because the local project
    has evolved across many versions.  Later init_pinn.py patches should attach
    `current_t_s` and `current_I_A` tensors explicitly.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for current_at_t().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    if "current_t_s" in params and "current_I_A" in params:
        return _interp1d_torch(params["current_t_s"], params["current_I_A"], t)
    if "t_profile" in params and "I_profile" in params:
        return _interp1d_torch(params["t_profile"], params["I_profile"], t)
    if "t_global_s" in params and "I_profile" in params:
        return _interp1d_torch(params["t_global_s"], params["I_profile"], t)
    return torch.zeros_like(t) + _as_float_param(params, "I_discharge", "I", default=0.0)


def effective_volume_at_t(params: Dict[str, object], t, electrode: str, aging_profiles=None):
    """Return V_a or V_c_eff(t) for aged flux/cbar closure.

    109A modifies only the positive-electrode effective volume via f_LAM_c(k).
    eps_s_c is intentionally left fixed to avoid counting LAM twice.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for effective_volume_at_t().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    e = str(electrode).lower()
    if e in {"a", "anode", "neg", "negative"}:
        return torch.zeros_like(t) + _as_float_param(params, "V_a", default=1.0)
    if e not in {"c", "cathode", "pos", "positive"}:
        raise ValueError(f"Unknown electrode: {electrode}")
    v0 = _as_float_param(params, "V_c", default=1.0)
    use_lam = bool(params.get("AGING_USE_LAM_C", params.get("USE_ASSB_AGING_MECHANISM", False)))
    if not use_lam or aging_profiles is None:
        return torch.zeros_like(t) + v0
    f_lam = _get_profile_at_t(params, t, aging_profiles, "f_lam_c")
    return v0 * torch.clamp(f_lam, min=1.0e-9)


def aged_surface_flux(params: Dict[str, object], t, electrode: str, aging_profiles=None):
    """Return aged SPM surface flux J_a/J_c at time t.

    Convention is inherited from the ASSB effective SPM:
    J_a = -I Rs_a / (3 eps_s_a F V_a)
    J_c = +I Rs_c / (3 eps_s_c F V_c_eff(k))
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for aged_surface_flux().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    I = current_at_t(params, t)
    F = _as_float_param(params, "F", default=96485.33212)
    e = str(electrode).lower()
    if e in {"a", "anode", "neg", "negative"}:
        Rs = _as_float_param(params, "Rs_a", default=50.0e-6)
        eps = _as_float_param(params, "eps_s_a", default=0.95)
        V = effective_volume_at_t(params, t, "a", aging_profiles)
        return -I * Rs / (3.0 * eps * F * V)
    if e in {"c", "cathode", "pos", "positive"}:
        Rs = _as_float_param(params, "Rs_c", default=1.8e-6)
        eps = _as_float_param(params, "eps_s_c", default=0.55)
        V = effective_volume_at_t(params, t, "c", aging_profiles)
        return I * Rs / (3.0 * eps * F * V)
    raise ValueError(f"Unknown electrode: {electrode}")


def aged_terminal_shift(params: Dict[str, object], t, aging_profiles=None):
    """Return I(t) * R_ohm(k) in Volts.

    This is the main slow voltage-drift path in 109A.  Residual common-mode
    gauge should remain small and be diagnosed separately.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for aged_terminal_shift().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    I = current_at_t(params, t)
    if aging_profiles is None or not bool(params.get("AGING_USE_R_OHM", params.get("USE_ASSB_AGING_MECHANISM", False))):
        R = torch.zeros_like(t) + _as_float_param(params, "R_ohm_eff", "R_OHM_EFF", "AGING_R_OHM0", default=105.0)
    else:
        R = _get_profile_at_t(params, t, aging_profiles, "r_ohm")
    return I * R


def aged_theta_window(params: Dict[str, object], t, aging_profiles=None) -> Tuple[object, object]:
    """Return positive-electrode theta window lower/upper values at time t.

    The center is fixed in 109A; only the usable width W_c(k) shrinks.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for aged_theta_window().")
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float64)
    bottom0 = _as_float_param(params, "theta_c_bottom", "THETA_C_BOTTOM", default=0.834)
    top0 = _as_float_param(params, "theta_c_top", "THETA_C_TOP", default=0.432)
    mid0 = _as_float_param(params, "theta_c_window_mid0", default=0.5 * (bottom0 + top0))
    width0 = abs(bottom0 - top0)
    if aging_profiles is None or not bool(params.get("AGING_USE_THETA_WINDOW_C", params.get("USE_ASSB_AGING_MECHANISM", False))):
        W = torch.zeros_like(t) + width0
    else:
        W = _get_profile_at_t(params, t, aging_profiles, "theta_window_c")
    lower = mid0 - 0.5 * W
    upper = mid0 + 0.5 * W
    return lower, upper


def assert_fixed_material_identity(params: Dict[str, object]) -> None:
    """Guard against charge/discharge material swapping.

    ASSB 109 keeps material identity fixed: a is always Li-In/In, c is always
    NMC811.  I(t) changes flux and polarization signs only.
    """
    forbidden = [
        "swap_ocp_on_charge",
        "SWAP_OCP_ON_CHARGE",
        "charge_anode_ocp",
        "charge_cathode_ocp",
        "switch_material_on_current",
        "SWITCH_MATERIAL_ON_CURRENT",
    ]
    present = [k for k in forbidden if k in params and bool(params.get(k))]
    if present:
        raise RuntimeError(
            "Invalid ASSB material-identity configuration. Positive/negative electrode "
            f"OCP/materials must not be swapped with current sign. Forbidden keys set: {present}"
        )


def fixed_material_identity_report(params: Dict[str, object]) -> Dict[str, object]:
    return {
        "negative_electrode_a": "Li-In/In effective pseudo-particle",
        "positive_electrode_c": "NMC811 representative particle",
        "ocp_swap_on_charge": False,
        "current_sign_changes": ["surface_flux", "overpotential", "ohmic_shift"],
        "csmax_aging_enabled": False,
        "R_ohm_dynamic_enabled": bool(params.get("AGING_USE_R_OHM", False)),
        "LAM_c_dynamic_enabled": bool(params.get("AGING_USE_LAM_C", False)),
        "theta_window_dynamic_enabled": bool(params.get("AGING_USE_THETA_WINDOW_C", False)),
    }
