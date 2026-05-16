# -*- coding: utf-8 -*-
"""
ASSB ModelFin_110 aging-fix1 loss utilities.

Complete replacement file for ModelFin_110 aging-fix1.
Original pointwise soft-label data loss is deliberately hard-disabled. The aging
regularization constrains mechanism-derived Q/SOH and is not the old data loss.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (str(_ROOT_DIR), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ._rescale import _to_tensor, _surface_flux_from_current, _radial_rescale
    from .assb_aging_capacity import capacity_loss
except Exception:  # pragma: no cover
    from _rescale import _to_tensor, _surface_flux_from_current, _radial_rescale
    from assb_aging_capacity import capacity_loss


def _zeros(device=None) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(0.0, dtype=torch.float64, device=device)


def _mean_square(term: torch.Tensor) -> torch.Tensor:
    term = _to_tensor(term, like=term)
    return torch.mean(term.square())


def _safe_mean(term) -> torch.Tensor:
    if isinstance(term, torch.Tensor):
        return torch.mean(term)
    return torch.as_tensor(float(term), dtype=torch.float64)


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


def _device_from_terms(*term_groups) -> torch.device:
    for group in term_groups:
        for term in group or []:
            if isinstance(term, torch.Tensor):
                return term.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha):
    device = _device_from_terms(interiorTerms, boundaryTerms, dataTerms, regularizationTerms)
    int_loss = _zeros(device)
    for term in interiorTerms or []:
        int_loss = int_loss + _mean_square(term)
    bound_loss = _zeros(device)
    for term in boundaryTerms or []:
        bound_loss = bound_loss + _mean_square(term)
    data_loss = _zeros(device)
    for term in dataTerms or []:
        data_loss = data_loss + _mean_square(term)
    reg_loss = _zeros(device)
    for term in regularizationTerms or []:
        reg_loss = reg_loss + _mean_square(term)
    a = [float(x) for x in list(alpha)[:4]]
    while len(a) < 4:
        a.append(0.0)
    # Hard guard: old data loss is always closed in ModelFin_109.
    a[2] = 0.0
    return a[0] * int_loss + a[1] * bound_loss + a[3] * reg_loss, a[0] * int_loss, a[1] * bound_loss, _zeros(device), a[3] * reg_loss


def loss_fn_lbfgs(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha):
    return loss_fn(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, alpha)


def loss_fn_lbfgs_SA(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, int_col_weights, bound_col_weights, data_col_weights, reg_col_weights, alpha):
    def apply_weights(terms, weights):
        out = []
        for i, term in enumerate(terms or []):
            w = weights[i] if i < len(weights) else 1.0
            out.append(_to_tensor(w, like=term) * term)
        return out
    return loss_fn(
        apply_weights(interiorTerms, int_col_weights),
        apply_weights(boundaryTerms, bound_col_weights),
        [],
        apply_weights(regularizationTerms, reg_col_weights),
        alpha,
    )


def loss_fn_lbfgs_annealing(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, int_loss_weights, bound_loss_weights, data_loss_weights, reg_loss_weights, alpha):
    return loss_fn_lbfgs_SA(interiorTerms, boundaryTerms, [], regularizationTerms, int_loss_weights, bound_loss_weights, [], reg_loss_weights, alpha)


def loss_fn_annealing(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, int_loss_terms, bound_loss_terms, data_loss_terms, reg_loss_terms, int_loss_weights, bound_loss_weights, data_loss_weights, reg_loss_weights, alpha):
    return loss_fn_lbfgs_annealing(interiorTerms, boundaryTerms, [], regularizationTerms, int_loss_weights, bound_loss_weights, [], reg_loss_weights, alpha)


def loss_fn_dynamicAttention_tensor(interiorTerms, boundaryTerms, dataTerms, regularizationTerms, int_col_weights, bound_col_weights, data_col_weights, reg_col_weights, alpha):
    val = loss_fn_lbfgs_SA([interiorTerms], [boundaryTerms], [], [regularizationTerms], [int_col_weights], [bound_col_weights], [], [reg_col_weights], alpha)
    global_loss, int_loss, bound_loss, data_loss, reg_loss = val
    return global_loss, global_loss.detach(), int_loss, bound_loss, data_loss, reg_loss


# -----------------------------------------------------------------------------
# Residual setup
# -----------------------------------------------------------------------------

def setResidualRescaling(self, weights=None):
    weights = weights or getattr(self, "weights", {}) or {}
    def get(k, default):
        try:
            return float(weights.get(k, default))
        except Exception:
            return float(default)
    self.interiorTerms_rescale_unweighted = [1.0, 1.0, 1.0, 1.0]
    self.interiorTerms_rescale = [get("phie_int", 1.0), get("phis_c_int", 1.0), get("cs_a_int", 10.0), get("cs_c_int", 2.0)]
    self.boundaryTerms_rescale_unweighted = [1.0, 1.0, 1.0, 1.0]
    self.boundaryTerms_rescale = [get("cs_a_rmin_bound", 1.0), get("cs_a_rmax_bound", 500.0), get("cs_c_rmin_bound", 1.0), get("cs_c_rmax_bound", 1500.0)]
    self.dataTerms_rescale_unweighted = [1.0, 1.0, 1.0, 1.0]
    self.dataTerms_rescale = [0.0, 0.0, 0.0, 0.0]
    self.regTerms_rescale_unweighted = [1.0, 1.0, 1.0, 1.0]
    self.regTerms_rescale = [
        float(getattr(self, "params", {}).get("AGING_CAP_LOSS_WEIGHT", 1.0)),
        float(getattr(self, "params", {}).get("AGING_MONO_WEIGHT", 0.05)),
        float(getattr(self, "params", {}).get("AGING_SMOOTH_WEIGHT", 0.02)),
        float(getattr(self, "params", {}).get("AGING_PRIOR_WEIGHT", 0.01)),
    ]


def data_loss(self, *args, **kwargs):
    # Hard guard: ModelFin_109 never opens old pointwise cs/phie/phis_c data loss.
    z = _zeros(getattr(self, "device", None))
    return [[z], [z], [z], [z]]


# -----------------------------------------------------------------------------
# Physics-informed residuals
# -----------------------------------------------------------------------------

def _predict_raw(self, t: torch.Tensor, r: torch.Tensor):
    deg_i0 = torch.ones_like(t)
    deg_ds = torch.ones_like(t)
    return self.model([t, r, deg_i0, deg_ds], training=True)


def _sample_t(self, n: int, requires_grad: bool = True) -> torch.Tensor:
    n = max(int(n), 1)
    tmin = float(getattr(self, "tmin", 0.0))
    tmax = float(getattr(self, "tmax", getattr(self, "params", {}).get("tmax", 1.0)))
    if not np.isfinite(tmax) or tmax <= tmin:
        tmax = tmin + 1.0
    t = torch.rand((n, 1), dtype=torch.float64, device=self.device) * np.float64(tmax - tmin) + np.float64(tmin)
    if requires_grad:
        t.requires_grad_(True)
    return t


def _sample_r(self, n: int, electrode: str, surface: bool = False, center: bool = False, requires_grad: bool = True) -> torch.Tensor:
    Rs = float(getattr(self, "params", {}).get("Rs_a" if electrode == "a" else "Rs_c", 1.0))
    if surface:
        r = torch.full((max(int(n), 1), 1), np.float64(Rs), dtype=torch.float64, device=self.device)
    elif center:
        r = torch.zeros((max(int(n), 1), 1), dtype=torch.float64, device=self.device)
    else:
        # Avoid exact r=0 in interior diffusion residual.
        r = (0.02 + 0.98 * torch.rand((max(int(n), 1), 1), dtype=torch.float64, device=self.device)) * np.float64(Rs)
    if requires_grad:
        r.requires_grad_(True)
    return r


def _diffusion_residual(self, electrode: str, n: int) -> torch.Tensor:
    p = self.params
    t = _sample_t(self, n, requires_grad=True)
    r = _sample_r(self, n, electrode, requires_grad=True)
    raw = _predict_raw(self, t, r)
    if electrode == "a":
        cs = self.rescaleCs_a(raw[2], t, r, clip=False)
        Ds = float(p.get("Ds_a", p.get("D_s_a", 1.0e-14)))
    else:
        cs = self.rescaleCs_c(raw[3], t, r, clip=False)
        Ds = float(p.get("Ds_c", p.get("D_s_c", 1.0e-14)))
    dcs_dt = _grad(cs, t)
    dcs_dr = _grad(cs, r)
    flux_r = r.square() * dcs_dr
    dflux_dr = _grad(flux_r, r)
    denom = torch.clamp(r.square(), min=torch.as_tensor(1.0e-30, dtype=r.dtype, device=r.device))
    res = dcs_dt - np.float64(Ds) * dflux_dr / denom
    # Residual normalization: keep magnitudes trainable with high current data.
    scale = torch.clamp(torch.mean(torch.abs(dcs_dt)).detach() + torch.mean(torch.abs(np.float64(Ds) * dflux_dr / denom)).detach(), min=torch.as_tensor(1.0, dtype=r.dtype, device=r.device))
    return res / scale


def _potential_residual(self, n: int, which: str) -> torch.Tensor:
    # Lightweight algebraic regularizer: current-aware potential transform should
    # remain a small residual around its aged baseline.  This avoids adding a
    # free common-mode gauge that competes with R_ohm(k).
    t = _sample_t(self, n, requires_grad=False)
    r = torch.zeros_like(t)
    raw = _predict_raw(self, t, r)
    out = self.rescalePhie(raw[0], t) if which == "phie" else self.rescalePhis_c(raw[1], t)
    # Penalize only the learned correction amplitude, not the baseline level.
    start_key = "phie0" if which == "phie" else "phis_c0"
    start = float(self.params.get(start_key, self.params.get("phis0", 0.0)))
    amp = out - torch.mean(out.detach()) + np.float64(start)
    return amp / np.float64(max(float(self.params.get("rescale_phis_c", 1.0)), 1.0))


def interior_loss(self, *args, **kwargs):
    n = int(getattr(self, "batch_size_int", 0) or kwargs.get("n", 0) or 64)
    if n <= 0:
        z = _zeros(getattr(self, "device", None))
        return [[z], [z], [z], [z]]
    try:
        res_a = _diffusion_residual(self, "a", max(n // 2, 1))
        res_c = _diffusion_residual(self, "c", max(n // 2, 1))
        res_phie = _potential_residual(self, max(n // 4, 1), "phie")
        res_phis = _potential_residual(self, max(n // 4, 1), "phis")
        return [[res_phie], [res_phis], [res_a], [res_c]]
    except Exception as exc:
        if getattr(self, "verbose", False):
            print(f"[ASSB-110] interior_loss fallback zero: {exc}")
        z = _zeros(getattr(self, "device", None))
        return [[z], [z], [z], [z]]


def _surface_gradient_residual(self, electrode: str, n: int) -> torch.Tensor:
    p = self.params
    t = _sample_t(self, n, requires_grad=True)
    r = _sample_r(self, n, electrode, surface=True, requires_grad=True)
    raw = _predict_raw(self, t, r)
    if electrode == "a":
        cs = self.rescaleCs_a(raw[2], t, r, clip=False)
        Ds = float(p.get("Ds_a", p.get("D_s_a", 1.0e-14)))
    else:
        cs = self.rescaleCs_c(raw[3], t, r, clip=False)
        Ds = float(p.get("Ds_c", p.get("D_s_c", 1.0e-14)))
    dcs_dr = _grad(cs, r)
    aging_profiles = self.get_aging_profiles() if hasattr(self, "get_aging_profiles") else None
    J = _surface_flux_from_current(p, t, electrode, aging_profiles=aging_profiles)
    # SPM BC: D_s d c_s / dr |Rs = -J.  The sign convention is already encoded in J.
    res = np.float64(Ds) * dcs_dr + J
    scale = torch.clamp(torch.mean(torch.abs(J)).detach() + torch.mean(torch.abs(np.float64(Ds) * dcs_dr)).detach(), min=torch.as_tensor(1.0e-18, dtype=t.dtype, device=t.device))
    return res / scale


def _center_gradient_residual(self, electrode: str, n: int) -> torch.Tensor:
    t = _sample_t(self, n, requires_grad=True)
    r = _sample_r(self, n, electrode, center=True, requires_grad=True)
    raw = _predict_raw(self, t, r)
    cs = self.rescaleCs_a(raw[2], t, r, clip=False) if electrode == "a" else self.rescaleCs_c(raw[3], t, r, clip=False)
    dcs_dr = _grad(cs, r)
    return dcs_dr


def boundary_loss(self, *args, **kwargs):
    n = int(getattr(self, "batch_size_bound", 0) or kwargs.get("n", 0) or 64)
    if n <= 0:
        z = _zeros(getattr(self, "device", None))
        return [[z], [z], [z], [z]]
    try:
        return [
            [_center_gradient_residual(self, "a", max(n // 4, 1))],
            [_surface_gradient_residual(self, "a", max(n // 4, 1))],
            [_center_gradient_residual(self, "c", max(n // 4, 1))],
            [_surface_gradient_residual(self, "c", max(n // 4, 1))],
        ]
    except Exception as exc:
        if getattr(self, "verbose", False):
            print(f"[ASSB-110] boundary_loss fallback zero: {exc}")
        z = _zeros(getattr(self, "device", None))
        return [[z], [z], [z], [z]]


# -----------------------------------------------------------------------------
# Aging mechanism loss
# -----------------------------------------------------------------------------

def _huber(x: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    beta = float(max(beta, 1.0e-12))
    ax = torch.abs(x)
    return torch.where(ax < beta, 0.5 * x.square() / beta, ax - 0.5 * beta)


def aging_mechanism_loss(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not hasattr(self, "aging_head") or self.aging_head is None:
        z = _zeros(getattr(self, "device", None))
        return z, {"aging_total": z.detach()}
    profiles = self.get_aging_profiles()
    batch = getattr(self, "capacity_target_batch", None)
    if batch is None:
        z = _zeros(getattr(self, "device", None))
        return z, {"aging_total": z.detach(), "cap_mae_mAh": z.detach(), "soh_mae": z.detach()}
    q_obs = batch["Q_obs_Ah"].to(dtype=profiles.Q_pred_Ah.dtype, device=profiles.Q_pred_Ah.device)
    soh_obs = batch["SOH_obs"].to(dtype=profiles.SOH_struct.dtype, device=profiles.SOH_struct.device)
    split = batch.get("split_code", torch.zeros_like(q_obs, dtype=torch.long)).to(device=profiles.Q_pred_Ah.device)
    complete = batch.get("complete", torch.ones_like(q_obs, dtype=torch.bool)).to(device=profiles.Q_pred_Ah.device)
    n_align = min(int(profiles.Q_pred_Ah.numel()), int(q_obs.numel()), int(profiles.SOH_struct.numel()), int(soh_obs.numel()), int(split.numel()))
    if n_align <= 0:
        z = _zeros(getattr(self, "device", None))
        return z, {"aging_total": z.detach()}
    q_pred = profiles.Q_pred_Ah.reshape(-1)[:n_align]
    soh_pred = profiles.SOH_struct.reshape(-1)[:n_align]
    q_obs = q_obs.reshape(-1)[:n_align]
    soh_obs = soh_obs.reshape(-1)[:n_align]
    split = split.reshape(-1)[:n_align]
    complete = complete.reshape(-1)[:n_align]
    # Stage C uses train rows if available, otherwise all valid rows. Stage-B may
    # have been fit to all splits; this term is only to keep injection stable.
    train_mask = split == 0
    if not torch.any(train_mask):
        train_mask = torch.ones_like(split, dtype=torch.bool)
    loss, logs = capacity_loss(
        q_obs,
        q_pred,
        soh_obs,
        soh_pred,
        self.aging_head.cfg,
        train_mask=train_mask,
        complete_mask=complete,
        lam_rate=getattr(profiles, "lam_rate", None),
        window_rate=getattr(profiles, "window_rate", None),
        f_lam=getattr(profiles, "f_LAM_c", None),
        window_scale=getattr(profiles, "theta_window_scale_c", None),
    )
    info = {k: torch.as_tensor(v, dtype=torch.float64, device=profiles.Q_pred_Ah.device).detach() for k, v in logs.items() if isinstance(v, (int, float))}
    info.update({
        "aging_total": loss.detach(),
        "cap_mae_mAh": (torch.mean(torch.abs(q_pred[train_mask] - q_obs[train_mask])) * 1000.0).detach(),
        "soh_mae": torch.mean(torch.abs(soh_pred[train_mask] - soh_obs[train_mask])).detach(),
        "R_ohm_mean": torch.mean(profiles.R_ohm_eff).detach(),
        "f_lam_c_min": torch.min(profiles.f_LAM_c).detach(),
        "theta_window_min": torch.min(profiles.theta_window_scale_c).detach(),
    })
    return loss, info


def regularization_loss(self, *args, **kwargs):
    loss, _ = aging_mechanism_loss(self)
    return [[torch.sqrt(torch.clamp(loss, min=0.0)).reshape(1, 1)]]


def get_unweighted_loss(self, *args, **kwargs):
    loss, _ = aging_mechanism_loss(self)
    return float(loss.detach().cpu())


def get_loss_and_flat_grad_SA(*args, **kwargs):
    raise NotImplementedError("This PyTorch port does not use the TensorFlow L-BFGS flat-gradient helper.")


def get_loss_and_flat_grad(*args, **kwargs):
    raise NotImplementedError("This PyTorch port does not use the TensorFlow L-BFGS flat-gradient helper.")


def get_loss_and_flat_grad_annealing(*args, **kwargs):
    raise NotImplementedError("This PyTorch port does not use the TensorFlow L-BFGS flat-gradient helper.")


__all__ = [
    "loss_fn",
    "loss_fn_lbfgs",
    "loss_fn_lbfgs_SA",
    "loss_fn_lbfgs_annealing",
    "loss_fn_annealing",
    "loss_fn_dynamicAttention_tensor",
    "setResidualRescaling",
    "data_loss",
    "interior_loss",
    "boundary_loss",
    "regularization_loss",
    "aging_mechanism_loss",
    "get_unweighted_loss",
]
