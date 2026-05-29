GV1 D9.6 multi-cell / multi-profile verification package
=========================================================

Purpose
-------
D9.5.1 is the current D9.x mainline after passing 40ks/200ks/500ks three-profile checks.
D9.6 changes the validation scope from single-profile voltage fitting to cell-level and
protocol-level stability checks.

This package intentionally reuses the D9.5.1 model core and adds D9.6 verification scripts.
It is NOT a 24-profile joint training package and it does NOT modify old ASSB mainline files
(main.py, util/*, integration_spm/*).

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
scripts/gv1_multicell_scorecard_d96.py
scripts/gv1_run_multicell_verify_d96.ps1
scripts/gv1_run_multicell_6x40ks_d96.ps1
scripts/gv1_run_multicell_6x200ks_d96.ps1
scripts/gv1_run_multicell_24x40ks_d96.ps1
scripts/gv1_run_multicell_24x200ks_d96.ps1
ASSB-D9_GV1_D9_6_multicell_verification_record.txt

Recommended execution order
---------------------------
1. 6-profile 40ks verification:
   powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_multicell_6x40ks_d96.ps1

2. If 6x40ks passes or is only mildly borderline, run 6-profile 200ks:
   powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_multicell_6x200ks_d96.ps1

3. Only after 6x40ks and 6x200ks pass, run 24-profile 40ks:
   powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_multicell_24x40ks_d96.ps1

4. Only after 24x40ks passes, consider 24-profile 200ks:
   powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_multicell_24x200ks_d96.ps1

Primary outputs
---------------
For 6x40ks:
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_6x40ks_d96\selected_profiles_d96_40ks.json
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_6x40ks_d96\metrics_summary_d96_40ks.json
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_6x40ks_d96\scorecard_d96_40ks.json

The file to paste back first is scorecard_d96_40ks.json.

Pass/fail interpretation
------------------------
D9.6 is a verification stage, so thresholds are intentionally not as strict as a single-profile
calibration benchmark.

Preferred pass:
- corr >= 0.90 for 40ks/200ks;
- MAE <= 0.12 V;
- pred_upper_frac_ge_4p269 <= 0.02;
- no meaningful >4.35 V overshoot or <2.35 V undershoot.

Fail / pause:
- corr < 0.85;
- MAE > 0.18 V;
- 4.269 V saturation > 5%;
- large voltage guardrail violations.

Known boundary
--------------
D9.5.1 solved B1 well and improved B3/B4 globally, but low-voltage tail remains a known weak
regime in high-rate profiles. D9.6 should not be judged only by rare low-voltage points; the
main question is whether the model is stable across cells/protocols without reintroducing high
voltage saturation or global trend failure.
