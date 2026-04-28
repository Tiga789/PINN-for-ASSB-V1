from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))


# -----------------------------------------------------------------------------
# Generic tensor helpers
# -----------------------------------------------------------------------------

def _to_tensor(
    x,
    device=None,
    like: torch.Tensor | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert scalar/list/array/tensor inputs to a 2D float64 tensor."""
    if isinstance(x, torch.Tensor):
        out = x
        if like is not None:
            out = out.to(dtype=like.dtype, device=like.device)
        else:
            out = out.to(dtype=torch.float64, device=device)
    else:
        dtype = torch.float64 if like is None else like.dtype
        dev = device if like is None else like.device
        out = torch.as_tensor(x, dtype=dtype, device=dev)

    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(-1, 1)

    if requires_grad:
        if out.requires_grad:
            out = out.clone()
        else:
            out = out.clone().detach().requires_grad_(True)
    return out


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        g = torch.zeros_like(inputs)
    return g


def _mean_square(term: torch.Tensor) -> torch.Tensor:
    term = _to_tensor(term)
    return torch.mean(term.square())


def _zeros(device) -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float64, device=device)


def _safe_abs_float(value, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            return float(torch.max(torch.abs(value.detach())).cpu())
        return float(abs(np.asarray(value, dtype=np.float64)).max())
    except Exception:
        return float(default)


# -----------------------------------------------------------------------------
# ASSB current / surface-flux helpers
# -----------------------------------------------------------------------------

def _get_param_first(params: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in params:
            return params[name]
    return default


def _as_profile_pair(profile):
    """Return (time, current) from several supported profile layouts."""
    if profile is None:
        return None

    if isinstance(profile, dict):
        t = _get_param_first(profile, ("t", "time", "time_s", "times", "t_s"))
        i = _get_param_first(profile, ("I", "current", "current_A", "I_A"))
        if t is not None and i is not None:
            return t, i

    if isinstance(profile, (tuple, list)) and len(profile) == 2:
        return profile[0], profile[1]

    return None


def _profile_tensors(params: dict, like: torch.Tensor):
    """Read a time-current profile from params, if present."""
    # Preferred compact key.
    pair = _as_profile_pair(_get_param_first(params, ("current_profile", "I_profile", "I_app_profile")))
    if pair is None:
        # Separate keys.
        time_values = _get_param_first(
            params,
            (
                "time_profile",
                "t_profile",
                "current_time_profile",
                "I_time_profile",
                "I_profile_t",
                "t_current",
            ),
        )
        current_values = _get_param_first(
            params,
            (
                "current_profile_A",
                "I_profile_A",
                "I_values",
                "current_values",
                "I_app_values",
            ),
        )
        if time_values is not None and current_values is not None:
            pair = (time_values, current_values)

    if pair is None:
        return None

    t_values, i_values = pair
    t_t = _to_tensor(t_values, like=like).reshape(-1).detach()
    i_t = _to_tensor(i_values, like=like).reshape(-1).detach()
    if t_t.numel() < 2 or i_t.numel() != t_t.numel():
        return None

    # Make sure the profile is sorted in time.
    order = torch.argsort(t_t)
    t_t = t_t[order]
    i_t = i_t[order]
    return t_t, i_t


def _interp_current_profile(t: torch.Tensor, t_profile: torch.Tensor, i_profile: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of I(t) for current boundary conditions."""
    t_query = _to_tensor(t, like=t).reshape(-1)
    t_profile = t_profile.to(dtype=t.dtype, device=t.device)
    i_profile = i_profile.to(dtype=t.dtype, device=t.device)

    # Clamp outside the measured profile. This is safer than extrapolating.
    tq = torch.clamp(t_query, min=t_profile[0], max=t_profile[-1])
    idx_hi = torch.searchsorted(t_profile, tq, right=False)
    idx_hi = torch.clamp(idx_hi, 1, t_profile.numel() - 1)
    idx_lo = idx_hi - 1

    t0 = t_profile[idx_lo]
    t1 = t_profile[idx_hi]
    i0 = i_profile[idx_lo]
    i1 = i_profile[idx_hi]
    denom = torch.clamp(t1 - t0, min=np.float64(1.0e-12))
    w = (tq - t0) / denom
    out = i0 + w * (i1 - i0)
    return out.reshape_as(_to_tensor(t, like=t))


def _current_at_t(params: dict, t_like: torch.Tensor) -> torch.Tensor:
    """Return applied cell current I(t) in A.

    Sign convention follows the user's effective SPM prior:
    +I = charge, -I = discharge. If no time-dependent profile is supplied,
    this function falls back to the existing constant-current parameter.
    """
    t_like = _to_tensor(t_like)

    for key in ("I_app_fun", "current_fun", "I_fun"):
        fn = params.get(key)
        if callable(fn):
            return _to_tensor(fn(t_like), like=t_like)

    prof = _profile_tensors(params, t_like)
    if prof is not None:
        return _interp_current_profile(t_like, prof[0], prof[1])

    I_const = _get_param_first(params, ("I_app", "I_discharge", "current_A", "I"), 0.0)
    return torch.full_like(t_like, float(I_const))


def _terminal_voltage_shift(params: dict, t_like: torch.Tensor) -> torch.Tensor:
    """Return dynamic terminal-voltage shift used by the ASSB v3 soft-label generator.

    The network state `phis_c` is interpreted as the measured/terminal positive
    current-collector voltage. Therefore the kinetic cathode overpotential must
    remove the lumped solid-state ohmic term and the v3 voltage alignment offset:

        eta_c = phis_c - phie - U_p(theta_c) - I(t) R_ohm_eff - V_offset.
    """
    t_like = _to_tensor(t_like)
    I_t = _current_at_t(params, t_like)
    r_ohm = float(params.get("R_ohm_eff", params.get("R_ohm", 0.0)))
    v_offset = float(params.get("voltage_alignment_offset_V", 0.0))
    return I_t * np.float64(r_ohm) + np.float64(v_offset) * torch.ones_like(t_like)


def _electrode_volume(params: dict, electrode: str) -> float:
    electrode = electrode.lower()
    if electrode.startswith("a") or electrode.startswith("n"):
        if "V_a" in params:
            return float(params["V_a"])
        return float(params["A_a"]) * float(params["L_a"])
    if "V_c" in params:
        return float(params["V_c"])
    return float(params["A_c"]) * float(params["L_c"])


def _surface_flux_from_current(params: dict, t_like: torch.Tensor, electrode: str) -> torch.Tensor:
    """SPM surface flux J_j(t) from applied current.

    J_a(t) = -I(t) R_a / (3 eps_a F V_a)
    J_c(t) =  I(t) R_c / (3 eps_c F V_c)

    This replaces the discharge-only constant params['j_a']/params['j_c'] logic
    while preserving the old constant-current behavior when no profile is given.
    """
    electrode = electrode.lower()
    I_t = _current_at_t(params, t_like)
    F = float(params["F"])

    if electrode.startswith("a") or electrode.startswith("n"):
        if callable(params.get("j_a_fun")):
            return _to_tensor(params["j_a_fun"](t_like), like=t_like)
        Rs = float(params["Rs_a"])
        eps = float(params["eps_s_a"])
        V = _electrode_volume(params, "a")
        return -I_t * Rs / (np.float64(3.0) * eps * F * V)

    if callable(params.get("j_c_fun")):
        return _to_tensor(params["j_c_fun"](t_like), like=t_like)
    Rs = float(params["Rs_c"])
    eps = float(params["eps_s_c"])
    V = _electrode_volume(params, "c")
    return I_t * Rs / (np.float64(3.0) * eps * F * V)


def _reference_current_abs(params: dict) -> float:
    """Return a robust current scale for loss rescaling."""
    for key in ("current_profile", "I_profile", "I_app_profile"):
        pair = _as_profile_pair(params.get(key))
        if pair is not None:
            return max(_safe_abs_float(pair[1]), 1.0e-30)
    for key in (
        "current_profile_A",
        "I_profile_A",
        "I_values",
        "current_values",
        "I_app_values",
    ):
        if key in params:
            return max(_safe_abs_float(params[key]), 1.0e-30)
    return max(_safe_abs_float(_get_param_first(params, ("I_app", "I_discharge", "current_A", "I"), 0.0)), 1.0e-30)


def _reference_flux_abs(params: dict, electrode: str) -> float:
    """Return a robust surface-flux scale for loss rescaling."""
    I_ref = _reference_current_abs(params)
    F = float(params["F"])
    electrode = electrode.lower()
    if electrode.startswith("a") or electrode.startswith("n"):
        Rs = float(params["Rs_a"])
        eps = float(params["eps_s_a"])
        V = _electrode_volume(params, "a")
        fallback = abs(-I_ref * Rs / (3.0 * eps * F * V))
        return max(_safe_abs_float(params.get("j_a"), fallback), fallback, 1.0e-30)
    Rs = float(params["Rs_c"])
    eps = float(params["eps_s_c"])
    V = _electrode_volume(params, "c")
    fallback = abs(I_ref * Rs / (3.0 * eps * F * V))
    return max(_safe_abs_float(params.get("j_c"), fallback), fallback, 1.0e-30)


# -----------------------------------------------------------------------------
# Loss aggregation helpers
# -----------------------------------------------------------------------------

def loss_fn_lbfgs_SA(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_col_weights,
    bound_col_weights,
    data_col_weights,
    reg_col_weights,
    alpha,
):
    device = interiorTerms[0].device if interiorTerms else torch.device("cpu")
    int_loss = _zeros(device)
    for i_term, term in enumerate(interiorTerms):
        int_loss = int_loss + _mean_square(_to_tensor(int_col_weights[i_term], like=term) * term)
    bound_loss = _zeros(device)
    for i_term, term in enumerate(boundaryTerms):
        bound_loss = bound_loss + _mean_square(_to_tensor(bound_col_weights[i_term], like=term) * term)
    data_loss = _zeros(device)
    for i_term, term in enumerate(dataTerms):
        data_loss = data_loss + _mean_square(_to_tensor(data_col_weights[i_term], like=term) * term)
    reg_loss = _zeros(device)
    for i_term, term in enumerate(regularizationTerms):
        reg_loss = reg_loss + _mean_square(_to_tensor(reg_col_weights[i_term], like=term) * term)
    global_loss = alpha[0] * int_loss + alpha[1] * bound_loss + alpha[2] * data_loss + alpha[3] * reg_loss
    return global_loss, alpha[0] * int_loss, alpha[1] * bound_loss, alpha[2] * data_loss, alpha[3] * reg_loss


def loss_fn_lbfgs(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha):
    device = interiorTerms[0].device if interiorTerms else torch.device("cpu")
    int_loss = _zeros(device)
    for term in interiorTerms:
        int_loss = int_loss + _mean_square(term)
    bound_loss = _zeros(device)
    for term in boundaryTerms:
        bound_loss = bound_loss + _mean_square(term)
    data_loss = _zeros(device)
    for term in dataTerms:
        data_loss = data_loss + _mean_square(term)
    reg_loss = _zeros(device)
    for term in regularizationTerms:
        reg_loss = reg_loss + _mean_square(term)
    global_loss = alpha[0] * int_loss + alpha[1] * bound_loss + alpha[2] * data_loss + alpha[3] * reg_loss
    return global_loss, alpha[0] * int_loss, alpha[1] * bound_loss, alpha[2] * data_loss, alpha[3] * reg_loss


def loss_fn_lbfgs_annealing(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_loss_weights,
    bound_loss_weights,
    data_loss_weights,
    reg_loss_weights,
    alpha,
):
    device = interiorTerms[0].device if interiorTerms else torch.device("cpu")
    int_loss = _zeros(device)
    for i_term, term in enumerate(interiorTerms):
        int_loss = int_loss + _to_tensor(int_loss_weights[i_term], like=term) * _mean_square(term)
    bound_loss = _zeros(device)
    for i_term, term in enumerate(boundaryTerms):
        bound_loss = bound_loss + _to_tensor(bound_loss_weights[i_term], like=term) * _mean_square(term)
    data_loss = _zeros(device)
    for i_term, term in enumerate(dataTerms):
        data_loss = data_loss + _to_tensor(data_loss_weights[i_term], like=term) * _mean_square(term)
    reg_loss = _zeros(device)
    for i_term, term in enumerate(regularizationTerms):
        reg_loss = reg_loss + _to_tensor(reg_loss_weights[i_term], like=term) * _mean_square(term)
    global_loss = alpha[0] * int_loss + alpha[1] * bound_loss + alpha[2] * data_loss + alpha[3] * reg_loss
    return global_loss, alpha[0] * int_loss, alpha[1] * bound_loss, alpha[2] * data_loss, alpha[3] * reg_loss


def loss_fn_dynamicAttention_tensor(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_col_weights,
    bound_col_weights,
    data_col_weights,
    reg_col_weights,
    alpha,
):
    device = interiorTerms.device
    int_loss_unweighted = torch.mean(interiorTerms.square()) * interiorTerms.shape[0]
    int_loss = torch.mean((_to_tensor(int_col_weights, like=interiorTerms) * interiorTerms).square()) * interiorTerms.shape[0]
    int_loss = int_loss * alpha[0]
    int_loss_unweighted = int_loss_unweighted * alpha[0]

    bound_loss_unweighted = torch.mean(boundaryTerms.square()) * boundaryTerms.shape[0]
    bound_loss = torch.mean((_to_tensor(bound_col_weights, like=boundaryTerms) * boundaryTerms).square()) * boundaryTerms.shape[0]
    bound_loss = bound_loss * alpha[1]
    bound_loss_unweighted = bound_loss_unweighted * alpha[1]

    data_loss_unweighted = torch.mean(dataTerms.square()) * dataTerms.shape[0]
    data_loss = torch.mean((_to_tensor(data_col_weights, like=dataTerms) * dataTerms).square()) * dataTerms.shape[0]
    data_loss = data_loss * alpha[2]
    data_loss_unweighted = data_loss_unweighted * alpha[2]

    reg_loss_unweighted = torch.mean(regularizationTerms.square()) * regularizationTerms.shape[0]
    reg_loss = torch.mean((_to_tensor(reg_col_weights, like=regularizationTerms) * regularizationTerms).square()) * regularizationTerms.shape[0]
    reg_loss = reg_loss * alpha[3]
    reg_loss_unweighted = reg_loss_unweighted * alpha[3]

    return (
        int_loss + bound_loss + data_loss + reg_loss,
        int_loss_unweighted + bound_loss_unweighted + data_loss_unweighted + reg_loss_unweighted,
        int_loss,
        bound_loss,
        data_loss,
        reg_loss,
    )


def loss_fn_annealing(
    interiorTerms,
    boundaryTerms,
    dataTerms,
    regularizationTerms,
    int_loss_terms,
    bound_loss_terms,
    data_loss_terms,
    reg_loss_terms,
    int_loss_weights,
    bound_loss_weights,
    data_loss_weights,
    reg_loss_weights,
    alpha,
):
    device = interiorTerms[0].device if interiorTerms else torch.device("cpu")
    int_loss = _zeros(device)
    for i_term, term in enumerate(interiorTerms):
        val = _mean_square(term)
        int_loss_terms[i_term] = val
        int_loss = int_loss + _to_tensor(int_loss_weights[i_term], like=term) * val
    bound_loss = _zeros(device)
    for i_term, term in enumerate(boundaryTerms):
        val = _mean_square(term)
        bound_loss_terms[i_term] = val
        bound_loss = bound_loss + _to_tensor(bound_loss_weights[i_term], like=term) * val
    data_loss = _zeros(device)
    for i_term, term in enumerate(dataTerms):
        val = _mean_square(term)
        data_loss_terms[i_term] = val
        data_loss = data_loss + _to_tensor(data_loss_weights[i_term], like=term) * val
    reg_loss = _zeros(device)
    for i_term, term in enumerate(regularizationTerms):
        val = _mean_square(term)
        reg_loss_terms[i_term] = val
        reg_loss = reg_loss + _to_tensor(reg_loss_weights[i_term], like=term) * val
    return int_loss + bound_loss + data_loss + reg_loss, int_loss, bound_loss, data_loss, reg_loss


def loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha):
    device = interiorTerms[0].device if interiorTerms else torch.device("cpu")
    int_loss = _zeros(device)
    for term in interiorTerms:
        int_loss = int_loss + _mean_square(term)
    int_loss = int_loss * alpha[0]

    bound_loss = _zeros(device)
    for term in boundaryTerms:
        bound_loss = bound_loss + _mean_square(term)
    bound_loss = bound_loss * alpha[1]

    data_loss = _zeros(device)
    for term in dataTerms:
        data_loss = data_loss + _mean_square(term)
    data_loss = data_loss * alpha[2]

    reg_loss = _zeros(device)
    for term in regularizationTerms:
        reg_loss = reg_loss + _mean_square(term)
    reg_loss = reg_loss * alpha[3]

    return int_loss + bound_loss + data_loss + reg_loss, int_loss, bound_loss, data_loss, reg_loss


# -----------------------------------------------------------------------------
# Main PINN loss methods monkey-patched into myNN
# -----------------------------------------------------------------------------

def get_unweighted_loss(
    self,
    int_col_pts,
    int_col_params,
    bound_col_pts,
    bound_col_params,
    reg_col_pts,
    reg_col_params,
    x_trainList,
    x_params_trainList,
    y_trainList,
    n_batch=1,
    tmax=None,
):
    accumulatedLoss = 0.0
    batch_size_int = self.batch_size_int_lbfgs
    batch_size_bound = self.batch_size_bound_lbfgs
    batch_size_data = self.batch_size_data_lbfgs
    batch_size_reg = self.batch_size_reg_lbfgs

    for i_batch in range(n_batch):
        int_col_pts_batch = [pts[i_batch * batch_size_int : (i_batch + 1) * batch_size_int] for pts in int_col_pts]
        int_col_params_batch = [pts[i_batch * batch_size_int : (i_batch + 1) * batch_size_int] for pts in int_col_params]
        bound_col_pts_batch = [pts[i_batch * batch_size_bound : (i_batch + 1) * batch_size_bound] for pts in bound_col_pts]
        bound_col_params_batch = [pts[i_batch * batch_size_bound : (i_batch + 1) * batch_size_bound] for pts in bound_col_params]
        x_trainList_batch = [x[i_batch * batch_size_data : (i_batch + 1) * batch_size_data, :] for x in x_trainList[: self.ind_cs_offset_data]]
        x_cs_trainList_batch = [x[i_batch * batch_size_data : (i_batch + 1) * batch_size_data, :] for x in x_trainList[self.ind_cs_offset_data :]]
        x_params_trainList_batch = [x[i_batch * batch_size_data : (i_batch + 1) * batch_size_data, :] for x in x_params_trainList]
        y_trainList_batch = [y[i_batch * batch_size_data : (i_batch + 1) * batch_size_data, :] for y in y_trainList]
        reg_col_pts_batch = [pts[i_batch * batch_size_reg : (i_batch + 1) * batch_size_reg] for pts in reg_col_pts]

        interiorTerms = self.interior_loss(int_col_pts_batch, int_col_params_batch, tmax)
        boundaryTerms = self.boundary_loss(bound_col_pts_batch, bound_col_params_batch, tmax)
        dataTerms = self.data_loss(x_trainList_batch, x_cs_trainList_batch, x_params_trainList_batch, y_trainList_batch)
        regularizationTerms = self.regularization_loss(reg_col_pts_batch, tmax)

        interiorTerms_rescaled = [interiorTerm[0] * resc for (interiorTerm, resc) in zip(interiorTerms, self.interiorTerms_rescale_unweighted)]
        boundaryTerms_rescaled = [boundaryTerm[0] * resc for (boundaryTerm, resc) in zip(boundaryTerms, self.boundaryTerms_rescale_unweighted)]
        dataTerms_rescaled = [dataTerm[0] * resc for (dataTerm, resc) in zip(dataTerms, self.dataTerms_rescale_unweighted)]
        regularizationTerms_rescaled = [regularizationTerm[0] * resc for (regularizationTerm, resc) in zip(regularizationTerms, self.regTerms_rescale_unweighted)]

        loss_value, _, _, _, _ = loss_fn_lbfgs(
            interiorTerms_rescaled,
            boundaryTerms_rescaled,
            dataTerms_rescaled,
            regularizationTerms_rescaled,
            self.alpha_unweighted,
        )
        accumulatedLoss += float(loss_value.detach().cpu())

    accumulatedLoss /= max(n_batch, 1)
    return accumulatedLoss


# Compatibility stubs for the TensorFlow eager L-BFGS helper.
def get_loss_and_flat_grad_SA(*args, **kwargs):
    raise NotImplementedError("TensorFlow eager L-BFGS helper is not used in the PyTorch port.")


def get_loss_and_flat_grad(*args, **kwargs):
    raise NotImplementedError("TensorFlow eager L-BFGS helper is not used in the PyTorch port.")


def get_loss_and_flat_grad_annealing(*args, **kwargs):
    raise NotImplementedError("TensorFlow eager L-BFGS helper is not used in the PyTorch port.")


def setResidualRescaling(self, weights):
    cs_a = float(self.params["cs_a0"])
    cs_c = float(self.params["cs_c0"])
    cscamax = float(self.params["cscamax"])

    Ds_a = float(self.params["D_s_a"](self.params["T"], self.params["R"]))
    Ds_c = float(self.params["D_s_c"](cs_c, self.params["T"], self.params["R"], cscamax, np.float64(1.0)))

    j_a = _reference_flux_abs(self.params, "a")
    j_c = _reference_flux_abs(self.params, "c")
    C = max(abs(float(self.params.get("C", 1.0))), 1.0e-30)

    self.phie_transp_resc = 1.0 / j_a
    self.phis_c_transp_resc = 1.0 / j_c

    # Bidirectional cycle data can move either away from or toward the initial value.
    cs_a_span = max(cs_a, float(self.params.get("csanmax", cs_a)) - cs_a, 1.0e-12)
    cs_c_span = max(cs_c, cscamax - cs_c, 1.0e-12)
    self.cs_a_transp_resc = (3600.0 / C) / cs_a_span
    self.cs_c_transp_resc = (3600.0 / C) / cs_c_span

    if self.activeInt:
        if self.annealingWeights:
            w_phie_int = w_phis_c_int = w_cs_a_int = w_cs_c_int = 1.0
        elif weights is None:
            w_phie_int = 1.0
            w_phis_c_int = 1.0
            w_cs_a_int = 50.0
            w_cs_c_int = 50.0
        else:
            w_phie_int = weights["phie_int"]
            w_phis_c_int = weights["phis_c_int"]
            w_cs_a_int = weights["cs_a_int"]
            w_cs_c_int = weights["cs_c_int"]

        self.interiorTerms_rescale_unweighted = [
            abs(self.phie_transp_resc),
            abs(self.phis_c_transp_resc),
            abs(self.cs_a_transp_resc),
            abs(self.cs_c_transp_resc),
        ]
        self.interiorTerms_rescale = [
            w_phie_int * self.interiorTerms_rescale_unweighted[0],
            w_phis_c_int * self.interiorTerms_rescale_unweighted[1],
            w_cs_a_int * self.interiorTerms_rescale_unweighted[2],
            w_cs_c_int * self.interiorTerms_rescale_unweighted[3],
        ]
    else:
        self.interiorTerms_rescale_unweighted = [0.0]
        self.interiorTerms_rescale = [0.0]

    self.cs_a_bound_resc = Ds_a / j_a
    self.cs_c_bound_resc = Ds_c / j_c
    self.cs_a_bound_j_resc = Ds_a / j_a
    self.cs_c_bound_j_resc = Ds_c / j_c

    if self.activeBound:
        if self.annealingWeights:
            w_cs_a_rmin_bound = w_cs_c_rmin_bound = w_cs_a_rmax_bound = w_cs_c_rmax_bound = 1.0
        elif weights is None:
            w_cs_a_rmin_bound = 1.0
            w_cs_c_rmin_bound = 1.0
            w_cs_a_rmax_bound = 10.0
            w_cs_c_rmax_bound = 10.0
        else:
            w_cs_a_rmin_bound = weights["cs_a_rmin_bound"]
            w_cs_c_rmin_bound = weights["cs_c_rmin_bound"]
            w_cs_a_rmax_bound = weights["cs_a_rmax_bound"]
            w_cs_c_rmax_bound = weights["cs_c_rmax_bound"]

        self.boundaryTerms_rescale_unweighted = [
            abs(self.cs_a_bound_resc),
            abs(self.cs_c_bound_resc),
            abs(self.cs_a_bound_j_resc),
            abs(self.cs_c_bound_j_resc),
        ]
        self.boundaryTerms_rescale = [
            w_cs_a_rmin_bound * self.boundaryTerms_rescale_unweighted[0],
            w_cs_c_rmin_bound * self.boundaryTerms_rescale_unweighted[1],
            w_cs_a_rmax_bound * self.boundaryTerms_rescale_unweighted[2],
            w_cs_c_rmax_bound * self.boundaryTerms_rescale_unweighted[3],
        ]
    else:
        self.boundaryTerms_rescale_unweighted = [0.0]
        self.boundaryTerms_rescale = [0.0]

    self.n_data_terms = 4
    if self.activeData:
        if self.annealingWeights:
            w_phie_dat = w_phis_c_dat = w_cs_a_dat = w_cs_c_dat = 1.0
        elif weights is None:
            w_phie_dat = w_phis_c_dat = w_cs_a_dat = w_cs_c_dat = 1.0
        else:
            w_phie_dat = weights["phie_dat"]
            w_phis_c_dat = weights["phis_c_dat"]
            w_cs_a_dat = weights["cs_a_dat"]
            w_cs_c_dat = weights["cs_c_dat"]

        self.dataTerms_rescale = [0.0 for _ in range(self.n_data_terms)]
        self.dataTerms_rescale_unweighted = [0.0 for _ in range(self.n_data_terms)]
        self.dataTerms_rescale_unweighted[self.ind_phie_data] = abs(1.0 / float(self.params["rescale_phie"]))
        self.dataTerms_rescale_unweighted[self.ind_phis_c_data] = abs(1.0 / float(self.params["rescale_phis_c"]))
        self.dataTerms_rescale_unweighted[self.ind_cs_a_data] = abs(1.0 / float(self.params["rescale_cs_a"]))
        self.dataTerms_rescale_unweighted[self.ind_cs_c_data] = abs(1.0 / float(self.params["rescale_cs_c"]))
        self.dataTerms_rescale[self.ind_phie_data] = abs(w_phie_dat * self.dataTerms_rescale_unweighted[self.ind_phie_data])
        self.dataTerms_rescale[self.ind_phis_c_data] = abs(w_phis_c_dat * self.dataTerms_rescale_unweighted[self.ind_phis_c_data])
        self.dataTerms_rescale[self.ind_cs_a_data] = abs(w_cs_a_dat * self.dataTerms_rescale_unweighted[self.ind_cs_a_data])
        self.dataTerms_rescale[self.ind_cs_c_data] = abs(w_cs_c_dat * self.dataTerms_rescale_unweighted[self.ind_cs_c_data])
        self.csDataTerms_ind = [self.ind_cs_a_data, self.ind_cs_c_data]
    else:
        self.dataTerms_rescale_unweighted = [0.0]
        self.dataTerms_rescale = [0.0]
        self.csDataTerms_ind = []

    self.regTerms_rescale_unweighted = [0.0]
    self.regTerms_rescale = [0.0]
    return


def data_loss(self, x_batch_trainList, x_cs_batch_trainList, x_params_batch_trainList, y_batch_trainList):
    if not self.activeData:
        return [[torch.tensor(0.0, dtype=torch.float64, device=self.device)]]

    resc_t = float(self.params["rescale_T"])
    resc_r = float(self.params["rescale_R"])

    x_phie = _to_tensor(x_batch_trainList[self.ind_phie_data], device=self.device)
    p_phie = _to_tensor(x_params_batch_trainList[self.ind_phie_data], device=self.device)
    y_phie = _to_tensor(y_batch_trainList[self.ind_phie_data], device=self.device)
    surfR_a = float(self.params["Rs_a"]) * torch.ones_like(x_phie[:, self.ind_t : self.ind_t + 1])
    out_phie = self.model(
        [
            x_phie[:, self.ind_t : self.ind_t + 1] / resc_t,
            surfR_a / resc_r,
            self.rescale_param(p_phie[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1], self.ind_deg_i0_a),
            self.rescale_param(p_phie[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1], self.ind_deg_ds_c),
        ],
        training=True,
    )
    phie_pred_rescaled = self.rescalePhie(
        out_phie[self.ind_phie],
        x_phie[:, self.ind_t : self.ind_t + 1],
        p_phie[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1],
        p_phie[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1],
    )

    x_phis = _to_tensor(x_batch_trainList[self.ind_phis_c_data], device=self.device)
    p_phis = _to_tensor(x_params_batch_trainList[self.ind_phis_c_data], device=self.device)
    y_phis = _to_tensor(y_batch_trainList[self.ind_phis_c_data], device=self.device)
    surfR_a2 = float(self.params["Rs_a"]) * torch.ones_like(x_phis[:, self.ind_t : self.ind_t + 1])
    out_phis = self.model(
        [
            x_phis[:, self.ind_t : self.ind_t + 1] / resc_t,
            surfR_a2 / resc_r,
            self.rescale_param(p_phis[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1], self.ind_deg_i0_a),
            self.rescale_param(p_phis[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1], self.ind_deg_ds_c),
        ],
        training=True,
    )
    phis_c_pred_rescaled = self.rescalePhis_c(
        out_phis[self.ind_phis_c],
        x_phis[:, self.ind_t : self.ind_t + 1],
        p_phis[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1],
        p_phis[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1],
    )

    x_csa = _to_tensor(x_cs_batch_trainList[self.ind_cs_a_data - self.ind_cs_offset_data], device=self.device)
    p_csa = _to_tensor(x_params_batch_trainList[self.ind_cs_a_data], device=self.device)
    y_csa = _to_tensor(y_batch_trainList[self.ind_cs_a_data], device=self.device)
    cs_a_pred_non_rescaled = self.model(
        [
            x_csa[:, self.ind_t : self.ind_t + 1] / resc_t,
            x_csa[:, self.ind_r : self.ind_r + 1] / resc_r,
            self.rescale_param(p_csa[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1], self.ind_deg_i0_a),
            self.rescale_param(p_csa[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1], self.ind_deg_ds_c),
        ],
        training=True,
    )[self.ind_cs_a]
    cs_a_pred_rescaled = self.rescaleCs_a(
        cs_a_pred_non_rescaled,
        x_csa[:, self.ind_t : self.ind_t + 1],
        x_csa[:, self.ind_r : self.ind_r + 1],
        p_csa[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1],
        p_csa[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1],
        clip=False,
    )

    x_csc = _to_tensor(x_cs_batch_trainList[self.ind_cs_c_data - self.ind_cs_offset_data], device=self.device)
    p_csc = _to_tensor(x_params_batch_trainList[self.ind_cs_c_data], device=self.device)
    y_csc = _to_tensor(y_batch_trainList[self.ind_cs_c_data], device=self.device)
    cs_c_pred_non_rescaled = self.model(
        [
            x_csc[:, self.ind_t : self.ind_t + 1] / resc_t,
            x_csc[:, self.ind_r : self.ind_r + 1] / resc_r,
            self.rescale_param(p_csc[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1], self.ind_deg_i0_a),
            self.rescale_param(p_csc[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1], self.ind_deg_ds_c),
        ],
        training=True,
    )[self.ind_cs_c]
    cs_c_pred_rescaled = self.rescaleCs_c(
        cs_c_pred_non_rescaled,
        x_csc[:, self.ind_t : self.ind_t + 1],
        x_csc[:, self.ind_r : self.ind_r + 1],
        p_csc[:, self.ind_deg_i0_a : self.ind_deg_i0_a + 1],
        p_csc[:, self.ind_deg_ds_c : self.ind_deg_ds_c + 1],
        clip=False,
    )

    return [
        [phie_pred_rescaled - y_phie],
        [phis_c_pred_rescaled - y_phis],
        [cs_a_pred_rescaled - y_csa],
        [cs_c_pred_rescaled - y_csc],
    ]


def interior_loss(self, int_col_pts=None, int_col_params=None, tmax=None):
    if not self.activeInt:
        return [[torch.tensor(0.0, dtype=torch.float64, device=self.device)]]

    tmin_int = min(float(self.tmin_int_bound), float(self.tmax))
    if self.collocationMode == "random":
        curr_tmax = float(tmax) if ((self.run_SGD and self.gradualTime_sgd) or (self.run_LBFGS and self.gradualTime_lbfgs)) and tmax is not None else float(self.tmax)
        t = torch.empty((self.batch_size_int, 1), dtype=torch.float64, device=self.device).uniform_(tmin_int, curr_tmax).requires_grad_(True)
        r_a = torch.empty((self.batch_size_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.rmin) + 1e-12, float(self.rmax_a)).requires_grad_(True)
        r_c = torch.empty((self.batch_size_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.rmin) + 1e-12, float(self.rmax_c)).requires_grad_(True)
        rSurf_a = float(self.rmax_a) * torch.ones((self.batch_size_int, 1), dtype=torch.float64, device=self.device)
        rSurf_c = float(self.rmax_c) * torch.ones((self.batch_size_int, 1), dtype=torch.float64, device=self.device)
        deg_i0_a = torch.empty((self.batch_size_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_i0_a_min_eff"]), float(self.params["deg_i0_a_max_eff"]))
        deg_ds_c = torch.empty((self.batch_size_int, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_ds_c_min_eff"]), float(self.params["deg_ds_c_max_eff"]))
    else:
        if (self.run_SGD and self.gradualTime_sgd) or (self.run_LBFGS and self.gradualTime_lbfgs):
            t = self.stretchT(_to_tensor(int_col_pts[self.ind_int_col_t], device=self.device), tmin_int, float(self.firstTime), tmin_int, float(tmax))
        else:
            t = _to_tensor(int_col_pts[self.ind_int_col_t], device=self.device)
        t = t.clone().detach().requires_grad_(True)
        r_a = _to_tensor(int_col_pts[self.ind_int_col_r_a], device=self.device).clone().detach().requires_grad_(True)
        rSurf_a = _to_tensor(int_col_pts[self.ind_int_col_r_maxa], device=self.device)
        r_c = _to_tensor(int_col_pts[self.ind_int_col_r_c], device=self.device).clone().detach().requires_grad_(True)
        rSurf_c = _to_tensor(int_col_pts[self.ind_int_col_r_maxc], device=self.device)
        deg_i0_a = _to_tensor(int_col_params[self.ind_int_col_params_deg_i0_a], device=self.device)
        deg_ds_c = _to_tensor(int_col_params[self.ind_int_col_params_deg_ds_c], device=self.device)

    resc_t = float(self.params["rescale_T"])
    resc_r = float(self.params["rescale_R"])
    ce = float(self.params["ce0"]) * torch.ones_like(t)
    phis_a = torch.zeros_like(t)

    output_a = self.model(
        [
            t / resc_t,
            r_a / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_c = self.model(
        [
            t / resc_t,
            r_c / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_surf_a = self.model(
        [
            t / resc_t,
            rSurf_a / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_surf_c = self.model(
        [
            t / resc_t,
            rSurf_c / resc_r,
            self.rescale_param(deg_i0_a, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c, self.ind_deg_ds_c),
        ],
        training=True,
    )

    cse_a = self.rescaleCs_a(output_surf_a[self.ind_cs_a], t, rSurf_a, deg_i0_a, deg_ds_c)
    i0_a = self.params["i0_a"](cse_a, ce, self.params["T"], self.params["alpha_a"], self.params["csanmax"], self.params["R"], deg_i0_a)
    phie = self.rescalePhie(output_a[self.ind_phie], t, deg_i0_a, deg_ds_c)
    phis_c = self.rescalePhis_c(output_c[self.ind_phis_c], t, deg_i0_a, deg_ds_c)
    cs_a = self.rescaleCs_a(output_a[self.ind_cs_a], t, r_a, deg_i0_a, deg_ds_c)
    cs_c = self.rescaleCs_c(output_c[self.ind_cs_c], t, r_c, deg_i0_a, deg_ds_c)
    cse_c = self.rescaleCs_c(output_surf_c[self.ind_cs_c], t, rSurf_c, deg_i0_a, deg_ds_c)

    eta_a = phis_a - phie - self.params["Uocp_a"](cse_a, self.params["csanmax"])
    if not self.linearizeJ:
        exp1_a = torch.exp((1.0 - float(self.params["alpha_a"])) * float(self.params["F"]) * eta_a / (float(self.params["R"]) * float(self.params["T"])))
        exp2_a = torch.exp(-float(self.params["alpha_a"]) * float(self.params["F"]) * eta_a / (float(self.params["R"]) * float(self.params["T"])))
        j_a = (i0_a / float(self.params["F"])) * (exp1_a - exp2_a)
    else:
        j_a = i0_a * eta_a / (float(self.params["R"]) * float(self.params["T"]))
    j_a_rhs = _surface_flux_from_current(self.params, t, "a")

    cs_a_r = _grad(cs_a, r_a)
    ds_a = self.params["D_s_a"](self.params["T"], self.params["R"]) + 0.0 * r_a

    i0_c = self.params["i0_c"](cse_c, ce, self.params["T"], self.params["alpha_c"], self.params["cscamax"], self.params["R"])
    eta_c = (
        phis_c
        - phie
        - self.params["Uocp_c"](cse_c, self.params["cscamax"])
        - _terminal_voltage_shift(self.params, t)
    )
    if not self.linearizeJ:
        exp1_c = torch.exp((1.0 - float(self.params["alpha_c"])) * float(self.params["F"]) * eta_c / (float(self.params["R"]) * float(self.params["T"])))
        exp2_c = torch.exp(-float(self.params["alpha_c"]) * float(self.params["F"]) * eta_c / (float(self.params["R"]) * float(self.params["T"])))
        j_c = (i0_c / float(self.params["F"])) * (exp1_c - exp2_c)
    else:
        j_c = i0_c * eta_c / (float(self.params["R"]) * float(self.params["T"]))
    j_c_rhs = _surface_flux_from_current(self.params, t, "c")

    cs_c_r = _grad(cs_c, r_c)
    ds_c = self.params["D_s_c"](cs_c, self.params["T"], self.params["R"], self.params["cscamax"], deg_ds_c) + 0.0 * r_c

    cs_a_t = _grad(cs_a, t)
    cs_a_r_r = _grad(cs_a_r, r_a)
    ds_a_r = _grad(ds_a, r_a)
    cs_c_t = _grad(cs_c, t)
    cs_c_r_r = _grad(cs_c_r, r_c)
    ds_c_r = _grad(ds_c, r_c)

    return [
        [j_a - j_a_rhs],
        [j_c - j_c_rhs],
        [cs_a_t - cs_a_r_r * ds_a - 2.0 * ds_a * cs_a_r / r_a - ds_a_r * cs_a_r],
        [cs_c_t - cs_c_r_r * ds_c - 2.0 * ds_c * cs_c_r / r_c - ds_c_r * cs_c_r],
    ]


def boundary_loss(self, bound_col_pts=None, bound_col_params=None, tmax=None):
    if not self.activeBound:
        return [[torch.tensor(0.0, dtype=torch.float64, device=self.device)]]

    tmin_bound = min(float(self.tmin_int_bound), float(self.tmax))
    if self.collocationMode == "random":
        curr_tmax = float(tmax) if ((self.run_SGD and self.gradualTime_sgd) or (self.run_LBFGS and self.gradualTime_lbfgs)) and tmax is not None else float(self.tmax)
        t_bound = torch.empty((self.batch_size_bound, 1), dtype=torch.float64, device=self.device).uniform_(tmin_bound, curr_tmax)
        r_0_bound = torch.zeros((self.batch_size_bound, 1), dtype=torch.float64, device=self.device)
        r_max_a_bound = float(self.rmax_a) * torch.ones((self.batch_size_bound, 1), dtype=torch.float64, device=self.device)
        r_max_c_bound = float(self.rmax_c) * torch.ones((self.batch_size_bound, 1), dtype=torch.float64, device=self.device)
        deg_i0_a_bound = torch.empty((self.batch_size_bound, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_i0_a_min_eff"]), float(self.params["deg_i0_a_max_eff"]))
        deg_ds_c_bound = torch.empty((self.batch_size_bound, 1), dtype=torch.float64, device=self.device).uniform_(float(self.params["deg_ds_c_min_eff"]), float(self.params["deg_ds_c_max_eff"]))
    else:
        if (self.run_SGD and self.gradualTime_sgd) or (self.run_LBFGS and self.gradualTime_lbfgs):
            t_bound = self.stretchT(_to_tensor(bound_col_pts[self.ind_bound_col_t], device=self.device), tmin_bound, float(self.firstTime), tmin_bound, float(tmax))
        else:
            t_bound = _to_tensor(bound_col_pts[self.ind_bound_col_t], device=self.device)
        r_0_bound = _to_tensor(bound_col_pts[self.ind_bound_col_r_min], device=self.device)
        r_max_a_bound = _to_tensor(bound_col_pts[self.ind_bound_col_r_maxa], device=self.device)
        r_max_c_bound = _to_tensor(bound_col_pts[self.ind_bound_col_r_maxc], device=self.device)
        deg_i0_a_bound = _to_tensor(bound_col_params[self.ind_bound_col_params_deg_i0_a], device=self.device)
        deg_ds_c_bound = _to_tensor(bound_col_params[self.ind_bound_col_params_deg_ds_c], device=self.device)

    resc_t = float(self.params["rescale_T"])
    resc_r = float(self.params["rescale_R"])
    r_0_bound = r_0_bound.clone().detach().requires_grad_(True)
    r_max_a_bound = r_max_a_bound.clone().detach().requires_grad_(True)
    r_max_c_bound = r_max_c_bound.clone().detach().requires_grad_(True)

    output_r0_a_bound = self.model(
        [
            t_bound / resc_t,
            r_0_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_r0_c_bound = self.model(
        [
            t_bound / resc_t,
            r_0_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_rmax_a_bound = self.model(
        [
            t_bound / resc_t,
            r_max_a_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )
    output_rmax_c_bound = self.model(
        [
            t_bound / resc_t,
            r_max_c_bound / resc_r,
            self.rescale_param(deg_i0_a_bound, self.ind_deg_i0_a),
            self.rescale_param(deg_ds_c_bound, self.ind_deg_ds_c),
        ],
        training=True,
    )

    cs_r0_a_bound = self.rescaleCs_a(output_r0_a_bound[self.ind_cs_a], t_bound, r_0_bound, deg_i0_a_bound, deg_ds_c_bound)
    cs_r0_c_bound = self.rescaleCs_c(output_r0_c_bound[self.ind_cs_c], t_bound, r_0_bound, deg_i0_a_bound, deg_ds_c_bound)
    cs_rmax_a_bound = self.rescaleCs_a(output_rmax_a_bound[self.ind_cs_a], t_bound, r_max_a_bound, deg_i0_a_bound, deg_ds_c_bound)
    cs_rmax_c_bound = self.rescaleCs_c(output_rmax_c_bound[self.ind_cs_c], t_bound, r_max_c_bound, deg_i0_a_bound, deg_ds_c_bound)

    ds_rmax_a_bound = self.params["D_s_a"](self.params["T"], self.params["R"])
    ds_rmax_c_bound = self.params["D_s_c"](cs_rmax_c_bound, self.params["T"], self.params["R"], self.params["cscamax"], deg_ds_c_bound)

    j_a = _surface_flux_from_current(self.params, t_bound, "a")
    j_c = _surface_flux_from_current(self.params, t_bound, "c")

    cs_r0_a_bound_r = _grad(cs_r0_a_bound, r_0_bound)
    cs_r0_c_bound_r = _grad(cs_r0_c_bound, r_0_bound)
    cs_rmax_a_bound_r = _grad(cs_rmax_a_bound, r_max_a_bound)
    cs_rmax_c_bound_r = _grad(cs_rmax_c_bound, r_max_c_bound)

    time_gate = 1.0 - torch.exp(-t_bound / float(self.hard_IC_timescale))
    return [
        [cs_r0_a_bound_r],
        [cs_r0_c_bound_r * deg_ds_c_bound],
        [time_gate * (cs_rmax_a_bound_r + j_a / ds_rmax_a_bound)],
        [time_gate * (cs_rmax_c_bound_r + j_c / ds_rmax_c_bound)],
    ]


def regularization_loss(self, reg_col_pts=None, tmax=None):
    if not self.activeReg:
        return [[torch.tensor(0.0, dtype=torch.float64, device=self.device)]]
    return [[torch.tensor(0.0, dtype=torch.float64, device=self.device)]]
