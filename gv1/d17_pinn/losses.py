# -*- coding: utf-8 -*-
"""D17-P2 no-state-label losses."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F

from .torch_ops import finite_diff_time_centered, radial_volume_weights_torch, spherical_laplacian_uniform_r


def _mean_sq(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return torch.mean(x * x)


def voltage_metrics(V_pred: torch.Tensor, V_exp: torch.Tensor) -> Dict[str, float]:
    e = (V_pred - V_exp).detach()
    mae = torch.mean(torch.abs(e)).item()
    rmse = torch.sqrt(torch.mean(e * e)).item()
    bias = torch.mean(e).item()
    if V_pred.numel() > 1 and torch.std(V_pred) > 1e-12 and torch.std(V_exp) > 1e-12:
        corr = torch.corrcoef(torch.stack([V_pred.detach().flatten(), V_exp.detach().flatten()]))[0, 1].item()
    else:
        corr = float("nan")
    return {"voltage_mae_V": mae, "voltage_rmse_V": rmse, "voltage_bias_V": bias, "voltage_corr": corr}


def zero_mean_delta_loss(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    r = batch["r_norm"]
    w = radial_volume_weights_torch(r).to(out["cs_a"].device, out["cs_a"].dtype)
    mean_a = torch.sum((out["cs_a"] - out["cbar_a"].reshape(-1, 1)) * w.reshape(1, -1), dim=1)
    mean_c = torch.sum((out["cs_c"] - out["cbar_c"].reshape(-1, 1)) * w.reshape(1, -1), dim=1)
    # Normalize to concentration scale to keep loss order reasonable.
    sc = torch.clamp(torch.mean(torch.abs(out["cbar_a"])) + torch.mean(torch.abs(out["cbar_c"])), min=1.0)
    return torch.mean((mean_a / sc) ** 2 + (mean_c / sc) ** 2)


def cbar_inventory_loss(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    r = batch["r_norm"]
    w = radial_volume_weights_torch(r).to(out["cs_a"].device, out["cs_a"].dtype)
    mean_a = torch.sum(out["cs_a"] * w.reshape(1, -1), dim=1)
    mean_c = torch.sum(out["cs_c"] * w.reshape(1, -1), dim=1)
    sc_a = torch.clamp(torch.mean(torch.abs(out["cbar_a"])), min=1.0)
    sc_c = torch.clamp(torch.mean(torch.abs(out["cbar_c"])), min=1.0)
    return torch.mean(((mean_a - out["cbar_a"]) / sc_a) ** 2 + ((mean_c - out["cbar_c"]) / sc_c) ** 2)


def state_bounds_loss(out: Mapping[str, torch.Tensor], prior) -> torch.Tensor:
    """Softly keep cs/cbar inside electrode stoichiometry windows.

    This loss is the mechanism-side replacement for post-hoc clamping.
    It preserves zero-volume-mean delta while discouraging unphysical
    theta collapse or inventory drift.
    """
    vals = []
    for cs_key, cbar_key, elec in [
        ("cs_a", "cbar_a", prior.negative),
        ("cs_c", "cbar_c", prior.positive),
    ]:
        cs = out[cs_key]
        cbar = out[cbar_key]
        lo = float(elec.theta_min) * float(elec.csmax_mol_m3)
        hi = float(elec.theta_max) * float(elec.csmax_mol_m3)
        scale = max(hi - lo, 1.0)
        low_violation = torch.relu((lo - cs) / scale)
        high_violation = torch.relu((cs - hi) / scale)
        cbar_low = torch.relu((lo - cbar) / scale)
        cbar_high = torch.relu((cbar - hi) / scale)
        vals.append(torch.mean(low_violation ** 2 + high_violation ** 2))
        vals.append(torch.mean(cbar_low ** 2 + cbar_high ** 2))
    return sum(vals) / len(vals)


def surface_flux_loss(out: Mapping[str, torch.Tensor], prior) -> torch.Tensor:
    losses = []
    for side, cs_key, r_key, J_key, elec in [
        ("a", "cs_a", "r_m_a", "J_a", prior.negative),
        ("c", "cs_c", "r_m_c", "J_c", prior.positive),
    ]:
        cs = out[cs_key]
        r_m = out[r_key]
        if cs.shape[1] < 2:
            continue
        dr = torch.clamp(r_m[-1] - r_m[-2], min=1e-12)
        dcs_dr_surface = (cs[:, -1] - cs[:, -2]) / dr
        Ds = float(elec.Ds_m2_s)
        bc = Ds * dcs_dr_surface + out[J_key]
        # Normalize by typical boundary flux, not by concentration units.
        scale = torch.clamp(torch.mean(torch.abs(out[J_key])) + torch.tensor(1.0e-10, device=cs.device, dtype=cs.dtype), min=1.0e-10)
        losses.append(torch.mean((bc / scale) ** 2))
    if not losses:
        return torch.zeros((), device=out["cs_a"].device, dtype=out["cs_a"].dtype)
    return sum(losses) / len(losses)


def diffusion_pde_loss(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], prior) -> torch.Tensor:
    t = batch["t_s"]
    vals = []
    for cs_key, r_key, elec, ds_key in [
        ("cs_a", "r_m_a", prior.negative, "latent_Ds_scale_a"),
        ("cs_c", "r_m_c", prior.positive, "latent_Ds_scale_c"),
    ]:
        cs = out[cs_key]
        if cs.shape[0] < 3 or cs.shape[1] < 3:
            continue
        D = float(elec.Ds_m2_s) * torch.clamp(out[ds_key].reshape(-1)[0], min=1.0e-6)
        dc_dt = finite_diff_time_centered(cs, t)[:, 1:-1]
        lap = spherical_laplacian_uniform_r(cs, out[r_key])[1:-1, :]
        res = dc_dt - D * lap
        # Normalize by observed concentration-time scale.
        scale = torch.clamp(torch.std(cs) / torch.clamp(t[-1] - t[0], min=1.0) + torch.tensor(1.0e-8, device=cs.device, dtype=cs.dtype), min=1.0e-8)
        vals.append(torch.mean((res / scale) ** 2))
    if not vals:
        return torch.zeros((), device=out["cs_a"].device, dtype=out["cs_a"].dtype)
    return sum(vals) / len(vals)


def ocp_bv_closure_loss(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    # The closure is hard-coded; this loss audits that V_pred equals V_base plus
    # all bounded residual channels. P3.2 adds a smooth voltage-basis residual,
    # which is included here so the audit does not accidentally penalize the
    # explicitly declared inverse residual path.
    inv = out.get("V_residual_inverse", torch.zeros_like(out["V_pred"]))
    basis = out.get("V_residual_basis", torch.zeros_like(out["V_pred"]))
    res = out["V_pred"] - out["V_base"] - out["V_residual_local"] - inv - basis
    return _mean_sq(res)


def gauge_smooth_loss(out: Mapping[str, torch.Tensor]) -> torch.Tensor:
    phie = out["phie"]
    if phie.numel() < 3:
        return torch.zeros((), device=phie.device, dtype=phie.dtype)
    d2 = phie[2:] - 2.0 * phie[1:-1] + phie[:-2]
    return torch.mean(d2 * d2)


def prior_z_loss(out: Mapping[str, torch.Tensor]) -> torch.Tensor:
    raw = out.get("latent_raw")
    if raw is None:
        return torch.zeros((), device=out["cs_a"].device, dtype=out["cs_a"].dtype)
    return torch.mean(raw * raw)


def residual_preservation_loss(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    # If local residual is disabled this is zero. If enabled, make it fade outside the observed low gate.
    res = out["V_residual_local"]
    gate = out.get("low_transition_gate", torch.zeros_like(res))
    non_gate = 1.0 - torch.clamp(gate, 0.0, 1.0)
    return torch.mean((non_gate * res) ** 2)


def inverse_residual_regularization_loss(out: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Regularize bounded inverse voltage residuals without blocking recovery."""
    ref = out["V_pred"]
    inv = out.get("V_residual_inverse", torch.zeros_like(ref))
    basis = out.get("V_residual_basis", torch.zeros_like(ref))
    gate = out.get("voltage_inverse_gate", out.get("low_transition_gate", torch.ones_like(ref)))
    non_gate = 1.0 - torch.clamp(gate, 0.0, 1.0)
    leak = torch.mean((non_gate * inv) ** 2)
    amp = torch.mean(inv ** 2 + basis ** 2)
    smooth_terms = []
    for res in [inv, basis]:
        if res.numel() >= 3:
            d2 = res[2:] - 2.0 * res[1:-1] + res[:-2]
            smooth_terms.append(torch.mean(d2 * d2))
    smooth = sum(smooth_terms) / max(len(smooth_terms), 1) if smooth_terms else torch.zeros((), device=ref.device, dtype=ref.dtype)
    return leak + 0.05 * smooth + 0.02 * amp


def total_d17_loss(
    out: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    prior,
    weights: Mapping[str, float] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weights = weights or {}
    V_exp = batch["voltage_exp"]
    losses: Dict[str, torch.Tensor] = {}
    losses["voltage"] = F.huber_loss(out["V_pred"], V_exp, delta=0.05)
    # P3.4: force the electrochemical forward core itself to explain the
    # voltage, instead of letting a voltage residual dominate the solution.
    # This remains no-state-label because the target is the observed terminal
    # voltage, not cs/theta/phie/phis soft labels.
    if "V_pred_forward" in out:
        losses["forward_voltage"] = F.huber_loss(out["V_pred_forward"], V_exp, delta=0.05)
    else:
        losses["forward_voltage"] = losses["voltage"]
    losses["diffusion_pde"] = diffusion_pde_loss(out, batch, prior)
    losses["surface_flux"] = surface_flux_loss(out, prior)
    losses["cbar_inventory"] = cbar_inventory_loss(out, batch)
    losses["zero_mean_delta"] = zero_mean_delta_loss(out, batch)
    losses["state_bounds"] = state_bounds_loss(out, prior)
    losses["ocp_bv_closure"] = ocp_bv_closure_loss(out, batch)
    losses["gauge_smooth"] = gauge_smooth_loss(out)
    losses["prior_z"] = prior_z_loss(out)
    losses["residual_preservation"] = residual_preservation_loss(out, batch)
    losses["inverse_residual_regularization"] = inverse_residual_regularization_loss(out)
    total = torch.zeros((), device=out["V_pred"].device, dtype=out["V_pred"].dtype)
    for k, v in losses.items():
        total = total + float(weights.get(k, 1.0)) * v
    losses["total"] = total
    return total, losses


def audit_numbers(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> Dict[str, float]:
    r = batch["r_norm"]
    w = radial_volume_weights_torch(r).to(out["cs_a"].device, out["cs_a"].dtype)
    mean_err_a = torch.max(torch.abs(torch.sum((out["cs_a"] - out["cbar_a"].reshape(-1, 1)) * w.reshape(1, -1), dim=1))).item()
    mean_err_c = torch.max(torch.abs(torch.sum((out["cs_c"] - out["cbar_c"].reshape(-1, 1)) * w.reshape(1, -1), dim=1))).item()
    grad_a = out["cs_a"][:, -1] - out["cs_a"][:, 0]
    grad_c = out["cs_c"][:, -1] - out["cs_c"][:, 0]
    d: Dict[str, float] = {}
    d.update(voltage_metrics(out["V_pred"], batch["voltage_exp"]))
    if "V_pred_forward" in out:
        fm = voltage_metrics(out["V_pred_forward"], batch["voltage_exp"])
        d.update({f"forward_{k}": v for k, v in fm.items()})
    for rk in ["V_residual_local", "V_residual_inverse", "V_residual_basis", "V_residual_total"]:
        if rk in out:
            rv = out[rk].detach()
            d[f"{rk}_abs_mean_V"] = float(torch.mean(torch.abs(rv)).cpu())
            d[f"{rk}_abs_max_V"] = float(torch.max(torch.abs(rv)).cpu())
    theta_a_min_t = torch.min(out["theta_a"]).detach()
    theta_a_max_t = torch.max(out["theta_a"]).detach()
    theta_c_min_t = torch.min(out["theta_c"]).detach()
    theta_c_max_t = torch.max(out["theta_c"]).detach()
    pa = getattr(batch.get("_prior", None), "negative", None) if isinstance(batch, dict) else None
    # The trainer does not pass prior through batch; report generic theta range.
    d.update({
        "zero_mean_max_abs_a_mol_m3": float(mean_err_a),
        "zero_mean_max_abs_c_mol_m3": float(mean_err_c),
        "radial_grad_abs_mean_a_mol_m3": float(torch.mean(torch.abs(grad_a)).detach().cpu()),
        "radial_grad_abs_mean_c_mol_m3": float(torch.mean(torch.abs(grad_c)).detach().cpu()),
        "V_pred_min": float(torch.min(out["V_pred"]).detach().cpu()),
        "V_pred_max": float(torch.max(out["V_pred"]).detach().cpu()),
        "theta_a_min": float(theta_a_min_t.cpu()),
        "theta_a_max": float(theta_a_max_t.cpu()),
        "theta_c_min": float(theta_c_min_t.cpu()),
        "theta_c_max": float(theta_c_max_t.cpu()),
    })
    return d
