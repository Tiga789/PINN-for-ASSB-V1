# -*- coding: utf-8 -*-
"""D17-PINN重构 package.

P0/P1 provide protocol and observed-only data audit. P2 adds an executable
mechanistic voltage-informed inverse PINN smoke core. P3 adds a 6-profile mechanism smoke trainer. P3.1 adds a 12-profile voltage-recovery smoke. P3.2 fixes voltage gate wiring and adds aggressive smooth voltage recovery. P3.3 separates forward core from residual-corrected voltage and audits D12 formula migration.
"""

from .dataset import D17ProfileDataset, load_observed_profile
from .spec_resolver import load_resolved_spec, audit_resolved_spec
from .cbar_core import integrate_cbar_from_current, current_to_surface_flux
from .radial_fv_core import radial_volume_weights, zero_volume_mean_project, radial_gradient_audit

# P2 torch modules
from .p2dlite_prior import D17P2DlitePrior, load_p2dlite_prior, prior_to_jsonable
from .model import D17MechanisticPINN, make_batch_from_profile
from .losses import total_d17_loss, audit_numbers
from .trainer import train_smoke
from .p3_trainer import train_p3_mechanism_smoke
from .p31_trainer import train_p31_mechanism_smoke
from .p32_trainer import train_p32_mechanism_smoke
from .p33_trainer import train_p33_forward_core_reliability

__all__ = [
    "D17ProfileDataset", "load_observed_profile", "load_resolved_spec", "audit_resolved_spec",
    "integrate_cbar_from_current", "current_to_surface_flux", "radial_volume_weights",
    "zero_volume_mean_project", "radial_gradient_audit",
    "D17P2DlitePrior", "load_p2dlite_prior", "prior_to_jsonable", "D17MechanisticPINN",
    "make_batch_from_profile", "total_d17_loss", "audit_numbers", "train_smoke", "train_p3_mechanism_smoke", "train_p31_mechanism_smoke", "train_p32_mechanism_smoke", "train_p33_forward_core_reliability",
]
