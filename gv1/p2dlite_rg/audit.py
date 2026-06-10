from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .io_utils import (
    get_cbar_field,
    get_cs_or_theta,
    get_time_and_current,
    infer_profile_id,
    npz_to_dict,
    volume_weights_for_nr,
    weighted_cbar,
)
from .radial_solver import expected_surface_center_sign


@dataclass
class ElectrodeAuditConfig:
    csmax_mol_m3: float
    active_current_abs_A_min: float = 0.02
    gradient_nonzero_abs_norm_min: float = 0.002
    p95_active_abs_gradient_norm_warn_below: float = 0.008
    mean_active_abs_gradient_norm_warn_below: float = 0.003
    direction_match_fraction_pass_min: float = 0.70
    direction_match_fraction_warn_min: float = 0.50
    mass_cbar_mae_norm_pass_max: float = 5e-4
    mass_cbar_mae_norm_warn_max: float = 2e-3
    physical_theta_min: float = -0.02
    physical_theta_max: float = 1.02


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float('nan')
    return float(np.nanmean(x))


def _safe_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float('nan')
    return float(np.nanpercentile(x, q))


def audit_electrode(cs: np.ndarray, cbar_field: Optional[np.ndarray], current_A: np.ndarray, electrode: str, cfg: ElectrodeAuditConfig) -> Dict[str, Any]:
    nr = cs.shape[1]
    weights = volume_weights_for_nr(nr)
    cbar_calc = weighted_cbar(cs, weights)
    surface_center = cs[:, -1] - cs[:, 0]
    surface_mean = cs[:, -1] - cbar_calc
    center_mean = cs[:, 0] - cbar_calc
    grad_norm = surface_center / float(cfg.csmax_mol_m3)
    abs_grad_norm = np.abs(grad_norm)
    active = np.abs(current_A) >= float(cfg.active_current_abs_A_min)
    rest = ~active
    expected_sign = expected_surface_center_sign(current_A, electrode)
    strong = active & (abs_grad_norm >= cfg.gradient_nonzero_abs_norm_min)
    direction_match = np.full(current_A.shape, np.nan, dtype=float)
    if np.any(strong):
        direction_match[strong] = (np.sign(surface_center[strong]) == expected_sign[strong]).astype(float)
    strong_fraction = float(np.mean(strong[active])) if np.any(active) else float('nan')
    direction_match_fraction = float(np.nanmean(direction_match[strong])) if np.any(strong) else float('nan')

    theta = cs / float(cfg.csmax_mol_m3)
    theta_min = float(np.nanmin(theta)) if theta.size else float('nan')
    theta_max = float(np.nanmax(theta)) if theta.size else float('nan')
    theta_outside_fraction = float(np.mean((theta < cfg.physical_theta_min) | (theta > cfg.physical_theta_max))) if theta.size else float('nan')

    mass_cbar_mae_norm = float('nan')
    mass_cbar_rmse_norm = float('nan')
    if cbar_field is not None and cbar_field.shape == cbar_calc.shape:
        diff = (cbar_calc - cbar_field) / float(cfg.csmax_mol_m3)
        mass_cbar_mae_norm = float(np.mean(np.abs(diff)))
        mass_cbar_rmse_norm = float(np.sqrt(np.mean(diff * diff)))

    p95_active = _safe_percentile(abs_grad_norm[active], 95) if np.any(active) else float('nan')
    mean_active = _safe_mean(abs_grad_norm[active]) if np.any(active) else float('nan')
    p95_rest = _safe_percentile(abs_grad_norm[rest], 95) if np.any(rest) else float('nan')

    # Flag logic: designed to diagnose source v1 as weak and RG as improved.
    reasons: List[str] = []
    if np.isfinite(p95_active) and p95_active < cfg.p95_active_abs_gradient_norm_warn_below:
        reasons.append(f'p95_active_abs_gradient_norm={p95_active:.6g} below warn threshold {cfg.p95_active_abs_gradient_norm_warn_below}')
    if np.isfinite(mean_active) and mean_active < cfg.mean_active_abs_gradient_norm_warn_below:
        reasons.append(f'mean_active_abs_gradient_norm={mean_active:.6g} below warn threshold {cfg.mean_active_abs_gradient_norm_warn_below}')
    if np.isfinite(direction_match_fraction) and direction_match_fraction < cfg.direction_match_fraction_warn_min:
        reasons.append(f'direction_match_fraction={direction_match_fraction:.4g} below warn threshold {cfg.direction_match_fraction_warn_min}')
    if np.isfinite(mass_cbar_mae_norm) and mass_cbar_mae_norm > cfg.mass_cbar_mae_norm_warn_max:
        reasons.append(f'mass_cbar_mae_norm={mass_cbar_mae_norm:.6g} above warn threshold {cfg.mass_cbar_mae_norm_warn_max}')
    if theta_outside_fraction > 0:
        reasons.append(f'theta outside relaxed physical range fraction={theta_outside_fraction:.6g}')

    if reasons:
        flag = 'WARN'
        if (np.isfinite(p95_active) and p95_active < 0.5 * cfg.p95_active_abs_gradient_norm_warn_below) or theta_outside_fraction > 0.01:
            flag = 'FAIL'
    else:
        flag = 'PASS'

    # Promote PASS only if direction is also good when enough strong points exist.
    if flag == 'PASS' and np.isfinite(direction_match_fraction) and direction_match_fraction < cfg.direction_match_fraction_pass_min:
        flag = 'WARN'
        reasons.append(f'direction_match_fraction={direction_match_fraction:.4g} below pass threshold {cfg.direction_match_fraction_pass_min}')

    return {
        'flag': flag,
        'reasons': reasons,
        'nr': int(nr),
        'time_points': int(cs.shape[0]),
        'active_points': int(np.sum(active)),
        'rest_points': int(np.sum(rest)),
        'csmax_mol_m3': float(cfg.csmax_mol_m3),
        'surface_center_mean_mol_m3': float(np.nanmean(surface_center)),
        'surface_center_abs_mean_mol_m3': float(np.nanmean(np.abs(surface_center))),
        'surface_center_abs_p95_mol_m3': _safe_percentile(np.abs(surface_center), 95),
        'active_abs_gradient_norm_mean': mean_active,
        'active_abs_gradient_norm_p50': _safe_percentile(abs_grad_norm[active], 50) if np.any(active) else float('nan'),
        'active_abs_gradient_norm_p95': p95_active,
        'rest_abs_gradient_norm_p95': p95_rest,
        'strong_active_fraction': strong_fraction,
        'direction_match_fraction': direction_match_fraction,
        'theta_min': theta_min,
        'theta_max': theta_max,
        'theta_outside_fraction': theta_outside_fraction,
        'mass_cbar_mae_norm': mass_cbar_mae_norm,
        'mass_cbar_rmse_norm': mass_cbar_rmse_norm,
        'surface_mean_abs_p95_norm': _safe_percentile(np.abs(surface_mean) / float(cfg.csmax_mol_m3), 95),
        'center_mean_abs_p95_norm': _safe_percentile(np.abs(center_mean) / float(cfg.csmax_mol_m3), 95),
    }


def audit_npz_file(npz_path: Path, source_root: Path, prior: Dict[str, Any]) -> Dict[str, Any]:
    d = npz_to_dict(npz_path)
    t, I = get_time_and_current(d)
    thresholds = prior.get('audit_thresholds', {})
    pos = prior['electrodes']['positive']
    neg = prior['electrodes']['negative']
    cfg_c = ElectrodeAuditConfig(csmax_mol_m3=float(pos['csmax_mol_m3']), **{k: thresholds[k] for k in thresholds if k in ElectrodeAuditConfig.__dataclass_fields__})
    cfg_a = ElectrodeAuditConfig(csmax_mol_m3=float(neg['csmax_mol_m3']), **{k: thresholds[k] for k in thresholds if k in ElectrodeAuditConfig.__dataclass_fields__})
    cs_a, src_a = get_cs_or_theta(d, 'a', len(t), cfg_a.csmax_mol_m3)
    cs_c, src_c = get_cs_or_theta(d, 'c', len(t), cfg_c.csmax_mol_m3)
    cbar_a = get_cbar_field(d, 'a', len(t))
    cbar_c = get_cbar_field(d, 'c', len(t))
    audit_a = audit_electrode(cs_a, cbar_a, I, 'a', cfg_a)
    audit_c = audit_electrode(cs_c, cbar_c, I, 'c', cfg_c)
    flags = [audit_a['flag'], audit_c['flag']]
    overall = 'PASS' if all(f == 'PASS' for f in flags) else ('FAIL' if any(f == 'FAIL' for f in flags) else 'WARN')
    return {
        'profile_id': infer_profile_id(npz_path, source_root),
        'npz_path': str(npz_path),
        'overall_flag': overall,
        'source_key_a': src_a,
        'source_key_c': src_c,
        'time_points': int(len(t)),
        'time_min_s': float(np.nanmin(t)),
        'time_max_s': float(np.nanmax(t)),
        'current_abs_max_A': float(np.nanmax(np.abs(I))) if I.size else 0.0,
        'a': audit_a,
        'c': audit_c,
    }


def flatten_audit_for_csv(result: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        'profile_id': result.get('profile_id'),
        'npz_path': result.get('npz_path'),
        'overall_flag': result.get('overall_flag'),
        'time_points': result.get('time_points'),
        'time_min_s': result.get('time_min_s'),
        'time_max_s': result.get('time_max_s'),
        'current_abs_max_A': result.get('current_abs_max_A'),
        'source_key_a': result.get('source_key_a'),
        'source_key_c': result.get('source_key_c'),
    }
    for el in ['a', 'c']:
        sub = result.get(el, {})
        for k, v in sub.items():
            if k == 'reasons':
                row[f'{el}_reasons'] = ' | '.join(v) if isinstance(v, list) else str(v)
            else:
                row[f'{el}_{k}'] = v
    return row
