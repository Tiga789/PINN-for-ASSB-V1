"""Measured-current replay utilities for GV1.

This package turns a standardized battery time-series table into the exact
external forcing objects used by GV1: t, I(t), V(t), cycle/step metadata,
capacity/energy integrals, and replay audits.
"""

from .profile_builder import ReplayBuildOptions, ReplayProfile, build_replay_profile, profile_to_dataframe, save_replay_profile_npz
from .current_interpolator import CurrentInterpolator, build_current_interpolator
from .capacity_integrator import (
    integrate_trapezoid,
    cumulative_integral,
    cumulative_charge_discharge_Ah,
    cumulative_energy_Wh,
    build_cycle_integrals,
)
from .step_classifier import StepClassificationOptions, classify_step_types, assign_step_ids
from .replay_audit import ReplayAuditResult, audit_replay_profile, audit_standard_table

__all__ = [
    'ReplayBuildOptions', 'ReplayProfile', 'build_replay_profile', 'profile_to_dataframe', 'save_replay_profile_npz',
    'CurrentInterpolator', 'build_current_interpolator',
    'integrate_trapezoid', 'cumulative_integral', 'cumulative_charge_discharge_Ah', 'cumulative_energy_Wh', 'build_cycle_integrals',
    'StepClassificationOptions', 'classify_step_types', 'assign_step_ids',
    'ReplayAuditResult', 'audit_replay_profile', 'audit_standard_table',
]
