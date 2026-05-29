GV1 D9.6.1 targeted repair package
===================================

Purpose
-------
D9.6 verified the D9.5.1 core on multi-profile XJTU measured-current replay:
- 6x40ks: pass
- 6x200ks: pass
- 24x40ks: 23 pass + 1 borderline

The borderline case was B1_2C battery-8 at 40ks. A separate 200ks check showed
that battery-8 remained weak, with lower correlation and rare high-voltage
overshoot. D9.6.1 therefore adds a targeted repair preset for late 2C Batch-1
profiles, especially battery-6/7/8. It does not change the intended D9.5.1
behavior for R2.5/R3 and ordinary early 2C profiles.

User workflow rule
------------------
The user manually extracts downloaded zip packages and manually adds/overwrites
files in the local project. Do not provide unzip/Expand-Archive instructions in
future D9 responses unless the user explicitly asks.

Files updated / added
---------------------
gv1/model.py                                      unchanged architecture
gv1/output_transform.py                           adds optional differentiable soft voltage guard
gv1/profile_adaptive.py                           adds late_2c_guarded_trend_warmup preset
gv1/losses.py                                     D9.5.1 losses retained
gv1/trainer.py                                    saves soft-guard diagnostic arrays
scripts/gv1_train_conditioned_pinn.py             adds D9.6.1 preset and soft-guard CLI options
scripts/gv1_prediction_metrics.py                 adds soft-guard diagnostics to metrics
scripts/gv1_multicell_scorecard_d961.py           D9.6.1 scorecard wrapper
scripts/gv1_run_borderline_B1_2C_battery8_200ks_d961.ps1
scripts/gv1_run_multicell_2c_tail_3x200ks_d961.ps1
scripts/gv1_run_multicell_verify_d961.ps1
scripts/gv1_run_multicell_24x40ks_d961.ps1
scripts/gv1_run_multicell_24x200ks_d961.ps1

First recommended run
---------------------
After manually adding/overwriting files, run only the targeted borderline check:

cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_borderline_B1_2C_battery8_200ks_d961.ps1

Then inspect:

E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d961\metrics_borderline_200ks_d961.json

Pass target
-----------
Prefer:
- corr >= 0.90
- MAE <= 0.10 V, ideally below the previous 0.1008 V
- pred_overshoot_frac_gt_4p35 <= 0.001
- pred_upper_frac_ge_4p269 <= 0.005

If battery-8 passes, run the 2C tail group:

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_multicell_2c_tail_3x200ks_d961.ps1

Do not run 24-profile 200ks until the battery-8 and 2C-tail repair checks pass.
