GV1 D9.4.1 profile-adaptive hybrid repair

Purpose
-------
D9.4 routed low-rate 2C profiles to a smooth branch and R2.5/R3 profiles to an event-aware branch.  The branch routing worked, but the B1/2C 200ks result regressed: B1_2C_200ks had MAE around 0.105 V and corr around 0.824, while B3/R2.5 and B4/R3 remained good.  D9.4.1 therefore fixes only the low-rate smooth branch and leaves the event/high-rate branch unchanged.

Main change
-----------
For ordinary low-rate / low-temperature 2C profiles, auto mode now uses a D9.2-locked smooth preset:

- profile_event_gate = 0
- profile_dynamic_event_gate = 0
- low-tail/event/temperature correction scales = 0
- event / asymmetric / quantile loss terms disabled
- event-weighted sampling disabled

For R2.5/R3 or high-temperature/high-current profiles, auto mode keeps the D9.3-like event branch.

Files
-----
Overwrite these files in the project root:

- gv1/model.py
- gv1/output_transform.py
- gv1/profile_adaptive.py
- gv1/losses.py
- gv1/trainer.py
- scripts/gv1_train_conditioned_pinn.py
- scripts/gv1_prediction_metrics.py

New run scripts:

- scripts/gv1_run_profile_compare_d941.ps1
- scripts/gv1_run_profile_compare_40ks_d941.ps1
- scripts/gv1_run_profile_compare_200ks_d941.ps1
- scripts/gv1_run_profile_compare_500ks_d941.ps1

Manual deployment note
----------------------
The user manually extracts zip packages and adds/overwrites files in the local project.  Do not provide unzip commands in later ASSB-D9 instructions unless explicitly requested.

Recommended first run
---------------------
Run the 40ks diagnostic first:

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_40ks_d941.ps1

Expected branch behavior
------------------------
The console JSON should show:

- B1_2C: selected_mode smooth_2c, transform_config.profile_adaptive_mode smooth_2c_d92locked
- B3_R25: selected_mode event_highrate, transform_config.profile_adaptive_mode event_highrate_d93like
- B4_R3: selected_mode event_highrate, transform_config.profile_adaptive_mode event_highrate_d93like

Proceed only if 40ks is acceptable; then run 200ks.  Do not jump directly to 500ks or 24-profile training.
