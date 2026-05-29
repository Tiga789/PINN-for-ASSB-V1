GV1 D9.6.2 rollback-style targeted repair
=========================================

Purpose
-------
D9.6.2 is a rollback-style repair after D9.6.1 failed on B1_2C battery-8 200 ks.
D9.6.1 used a soft voltage guard. The observed failure was not solved by that
guard: the direct voltage head ran to ~6.6 V and the guard clipped a large
fraction of points at ~4.27 V, reducing correlation.

D9.6.2 therefore returns to the D9.6 / D9.5.1 trend-first transform:
- no soft output guard;
- no hard output clamp by default;
- weaker late-2C event/tail channels;
- a loss-level component guardrail that discourages voltage_direct_head and
  voltage_base_branch runaway while preserving gradient flow.

Recommended first step
----------------------
After manually adding/overwriting these files in the project root, run only the
single borderline profile first:

  cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

  powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_borderline_B1_2C_battery8_200ks_d962.ps1

Then inspect:

  Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d962\metrics_borderline_200ks_d962.json" -Raw

Suggested acceptance signals for the single profile:
- MAE should improve versus D9.6.1_v2 (~0.178 V) and preferably approach or beat original D9.6 (~0.101 V).
- corr should improve versus D9.6.1_v2 (~0.814) and preferably approach or beat original D9.6 (~0.893).
- pred_upper_frac_ge_4p269 should be far below D9.6.1_v2 (~0.36), preferably <0.03.
- voltage_direct_head_max should no longer be near 6+ V.

Do not directly run 24-profile 200ks until the single borderline check passes.

Files
-----
gv1/model.py
gv1/output_transform.py
gv1/profile_adaptive.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
scripts/gv1_prediction_metrics.py
scripts/gv1_d96_profile_inventory.py
scripts/gv1_multicell_scorecard_d962.py
scripts/gv1_run_borderline_B1_2C_battery8_200ks_d962.ps1
scripts/gv1_run_multicell_verify_d962.ps1
scripts/gv1_run_multicell_6x40ks_d962.ps1
scripts/gv1_run_multicell_6x200ks_d962.ps1
scripts/gv1_run_multicell_24x40ks_d962.ps1
scripts/gv1_run_multicell_24x200ks_d962.ps1

Boundary
--------
D9.6.2 is still a verification/training-layer repair, not a final 24-profile
joint training model and not a completed SOH/generalization model.
