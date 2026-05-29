GV1 D9.2 voltage transform correction package
=============================================

Fixed ASSB-D9 project operation rule
-----------------------------------
The user manually extracts every zip package and manually adds/overwrites files
inside the project.  Future instructions should provide only post-overwrite
checks and run commands, not extraction commands.

Why D9.2 exists
---------------
D9.1 passed the 40/200/500 ks shape diagnostics: three XJTU profiles kept
corr around 0.92--0.94 and MAE around 0.24--0.25 V at 500 ks.  However, the
500 ks diagnostics also exposed the voltage-transform bottleneck:

1. 22--25% of predictions were pinned at the previous ~4.27 V ceiling.
2. Low-voltage coverage was weak; predicted voltage rarely entered <=2.75 V.
3. A negative voltage bias around -0.17 to -0.20 V remained.
4. Bias correction helped only partly, so this was not just a scalar offset.

D9.2 changes
------------
1. output_transform.py
   - Removes the forward hard clamp on phis_c by default.
   - Adds an unclamped affine direct voltage head scaled by the selected window's voltage span.
   - Keeps a weak OCV-like cbar baseline and ohmic term so the head is not purely black-box.
   - Uses soft guardrails around roughly 2.30--4.40 V through the loss, not through hard clipping.

2. losses.py
   - Adds tail-aware voltage MSE.
   - Adds voltage bias loss.
   - Adds voltage dynamic-range loss.
   - Adds soft voltage guardrail loss instead of hard clipping in the forward pass.

3. trainer.py
   - Saves additional voltage component arrays in prediction.npz for diagnosis.

4. scripts/gv1_prediction_metrics.py
   - Reports raw and bias-corrected metrics.
   - Reports low-voltage and upper-envelope fractions.
   - Reports low/mid/high target-region errors.

5. scripts/gv1_run_profile_compare_*_d92.ps1
   - Provides 40ks, 200ks, and 500ks three-profile comparison launchers.

Recommended order after manually adding files
---------------------------------------------
1. Syntax check:

D:\Anaconda\envs\torchgpu\python.exe -m py_compile `
  .\gv1\model.py `
  .\gv1\output_transform.py `
  .\gv1\losses.py `
  .\gv1\trainer.py `
  .\scripts\gv1_train_conditioned_pinn.py `
  .\scripts\gv1_prediction_metrics.py

2. Run 40ks D9.2 profile comparison first:

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_40ks_d92.ps1

3. If 40ks is stable, run 200ks:

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_200ks_d92.ps1

4. Run 500ks only after 40ks and 200ks look stable:

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_500ks_d92.ps1

Output metric files
-------------------
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_40ks_d92\metrics_summary_d92_40ks.json
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_200ks_d92\metrics_summary_d92_200ks.json
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_500ks_d92\metrics_summary_d92_500ks.json

Pass/fail focus
---------------
D9.2 should be judged mainly by:

- corr remains > 0.90;
- MAE does not worsen materially versus D9.1;
- bias magnitude decreases versus D9.1;
- pred_upper_frac_ge_4p269 drops clearly from D9.1's ~22--25%;
- pred_low_voltage_frac_le_2p75 increases from near-zero;
- pred_overshoot_frac_gt_4p35 stays small.

Still do not run all 24 profiles until the three-profile 40/200/500 ks D9.2
comparison is reviewed.
