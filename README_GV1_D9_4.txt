GV1 D9.4 profile-adaptive hybrid voltage transform
====================================================

Purpose
-------
D9.4 is the next diagnostic stage after D9.2 and D9.3:

- D9.2 fixed voltage dynamic range and worked especially well on the low-rate 2C profile.
- D9.3 added regime/event-aware channels and improved R2.5/R3 high-rate profiles, but B1/2C regressed relative to D9.2.
- D9.4 keeps one code path but selects an auditable profile-adaptive preset before training:
  - smooth_2c: D9.2-like smooth voltage branch for low-rate/low-temperature 2C.
  - event_highrate: D9.3-like event branch for R2.5/R3, high current, or high temperature.

This is still a one-profile diagnostic trainer. It is not a 24-profile final training script.
Do not modify old ASSB main.py / util/* / integration_spm/* with this package.

New or updated files
--------------------

gv1/model.py
gv1/output_transform.py
gv1/profile_adaptive.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
scripts/gv1_prediction_metrics.py
scripts/gv1_run_profile_compare_d94.ps1
scripts/gv1_run_profile_compare_40ks_d94.ps1
scripts/gv1_run_profile_compare_200ks_d94.ps1
scripts/gv1_run_profile_compare_500ks_d94.ps1
ASSB-D9_GV1_D9_4_profile_adaptive_record.txt

Manual deployment reminder
--------------------------
The user manually extracts zip packages and adds/overwrites files in the project.
Do not provide decompression instructions in later steps.

First run
---------

cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe -m py_compile `
  .\gv1\model.py `
  .\gv1\output_transform.py `
  .\gv1\profile_adaptive.py `
  .\gv1\losses.py `
  .\gv1\trainer.py `
  .\scripts\gv1_train_conditioned_pinn.py `
  .\scripts\gv1_prediction_metrics.py

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_40ks_d94.ps1

After it finishes, inspect:

Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_40ks_d94\metrics_summary_d94_40ks.json" -Raw

Expected selection behavior
---------------------------
The training log prints a JSON line named d9_4_profile_adaptive. For the standard
three-profile comparison it should normally select:

B1_2C   -> smooth_2c
B3_R25  -> event_highrate
B4_R3   -> event_highrate

Proceed only if 40ks diagnostics are acceptable. Do not jump straight to 24 profiles.
