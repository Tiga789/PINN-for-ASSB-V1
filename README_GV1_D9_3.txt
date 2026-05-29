GV1 D9.3 regime-aware voltage/event correction package
========================================================

Purpose
-------
D9.2 solved the large D9.1 voltage-range compression and high-voltage hard
saturation.  The remaining failure was not merely a single low-voltage-tail
bug.  The broader issue is regime imbalance: rare deep-discharge voltage tails,
high-current R2.5/R3 segments, current transitions, rest transitions and
large temperature deviations are underrepresented by uniform random batches.

D9.3 therefore makes the next step a general regime-aware correction package:

1. Event-aware sampling in gv1/trainer.py
   - mixes uniform sampling with weighted sampling;
   - boosts low/high voltage tails, high-current points, current transitions and
     temperature-extreme points;
   - records event_sampling diagnostics in training_summary.json.

2. Regime-aware voltage transform in gv1/output_transform.py
   - keeps weak OCV-like baseline + direct affine voltage head;
   - adds raw_voltage_low and raw_voltage_event channels;
   - low-voltage gate is based on the model-side OCV-like baseline, not the
     target voltage at inference;
   - adds current/temperature/event corrections without hard terminal-voltage
     clipping.

3. Broader voltage losses in gv1/losses.py
   - robust weighted voltage loss;
   - low/high tail loss;
   - asymmetric tail penalty for under-covered extremes;
   - quantile/range alignment;
   - event-aware loss.

4. Stronger diagnostics in scripts/gv1_prediction_metrics.py
   - low/high coverage gaps;
   - q05/q10/q90/q95 regime metrics;
   - high-current and temperature-event subset metrics;
   - voltage component summaries for the new correction channels.

User workflow note
------------------
ASSB-D9 record: the user manually extracts zip packages and adds/overwrites the
files in the project.  Do not give automatic Expand-Archive instructions unless
the user explicitly asks for them.

Package contents
----------------
gv1/model.py
gv1/output_transform.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
scripts/gv1_prediction_metrics.py
scripts/gv1_run_profile_compare_d93.ps1
scripts/gv1_run_profile_compare_40ks_d93.ps1
scripts/gv1_run_profile_compare_200ks_d93.ps1
scripts/gv1_run_profile_compare_500ks_d93.ps1
README_GV1_D9_3.txt
ASSB-D9_GV1_D9_3_regime_aware_record.txt

After manual overwrite
----------------------
Run syntax check:

D:\Anaconda\envs\torchgpu\python.exe -m py_compile `
  .\gv1\model.py `
  .\gv1\output_transform.py `
  .\gv1\losses.py `
  .\gv1\trainer.py `
  .\scripts\gv1_train_conditioned_pinn.py `
  .\scripts\gv1_prediction_metrics.py

Run in order:

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_40ks_d93.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_200ks_d93.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_500ks_d93.ps1

Decision rules
--------------
D9.3 is considered useful if it preserves the D9.2 global MAE/corr while
reducing low-tail metrics and coverage gaps.  Do not judge it only by global
MAE, because the whole point of this package is to avoid hiding rare-regime
errors behind good average metrics.

For 40ks:
- all three prediction.npz files must exist;
- corr should remain roughly >= 0.88;
- global MAE should not regress above ~0.12 V;
- pred_upper_frac_ge_4p269 should stay near zero;
- low_target_le_2p75 MAE should improve versus D9.2 where it was ~0.7-1.0 V;
- low_coverage_gap_le_2p75_pred_minus_target should move closer to zero.

For 200ks and 500ks:
- B1_2C should stay strong;
- B3_R25/B4_R3 should not lose high-current subset performance;
- improvement should be visible in q05/q10 low-target metrics and not only at
  the fixed 2.75 V threshold.

Still do not run 24 profiles directly after this package.  Finish 40ks -> 200ks
-> 500ks three-profile diagnostics first.
