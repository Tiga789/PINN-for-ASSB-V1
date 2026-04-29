# ASSB cycle5 Experiment A no secondary conservation

Purpose: test whether cathode `theta_c/cs_c` returns to a near-constant solution when secondary conservation is disabled.

Files:
- `input_assb_cycle5_A_no_seccons`: cycle5-only, physics-only, no data loss, no secondary conservation.
- `assb_cycle5_integration_mass_audit.py`: imports `integration_spm.spm_int_assb_cycle.surface_flux_from_current` and checks the soft-label spherical-average sign.
- `run_assb_cycle5_experiment_A.ps1`: optional wrapper for audit, training, evaluation.

Expected comparison baseline: previous ID=80 had `theta_c std_ratio = 0.522` and `theta_c corr = -0.636` with secondary conservation on.

Key judgment:
- If Experiment A gives `theta_c std_ratio < 0.05`, the secondary conservation term is the reason cathode dynamics appeared.
- If Experiment A still gives `theta_c std_ratio > 0.2` but corr stays negative, the cathode direction problem is likely boundary/current/OCP mapping rather than the secondary conservation term alone.
- If Experiment A improves corr, the secondary conservation term is harmful or has a sign/weight issue.
