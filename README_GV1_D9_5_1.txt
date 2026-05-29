# GV1 D9.5.1: trend-first warmup rare-regime correction

This package repairs D9.5 after the 40 ks diagnostics.

Observed D9.5 issue:
- B1/2C remained usable, but B3/R2.5 and B4/R3 were weaker than D9.3.
- Low-voltage coverage improved only slightly, while global MAE/correlation degraded.
- The main cause is that rare-regime objectives were too strong from epoch 1.

D9.5.1 decision:
- Keep the D9.3 event-aware output transform for all profiles.
- Keep ordinary voltage, range and correlation losses active from epoch 1.
- Warm up explicit rare-regime losses gradually: tail, event, asymmetry, ultra-quantile, coverage and tail-balance.
- Use softer event sampling and weaker sample weights than D9.5.
- Do not route B1/2C to a D9.2-like smooth branch.
- Do not modify old ASSB main.py / util/* / integration_spm/*.

Files:

gv1/model.py
gv1/output_transform.py
gv1/profile_adaptive.py
gv1/losses.py
gv1/trainer.py
scripts/gv1_train_conditioned_pinn.py
scripts/gv1_prediction_metrics.py
scripts/gv1_run_profile_compare_d951.ps1
scripts/gv1_run_profile_compare_40ks_d951.ps1
scripts/gv1_run_profile_compare_200ks_d951.ps1
scripts/gv1_run_profile_compare_500ks_d951.ps1
ASSB-D9_GV1_D9_5_1_trend_first_warmup_record.txt

Recommended validation order:

1. Syntax check:

D:\Anaconda\envs\torchgpu\python.exe -m py_compile `
  .\gv1\model.py `
  .\gv1\output_transform.py `
  .\gv1\profile_adaptive.py `
  .\gv1\losses.py `
  .\gv1\trainer.py `
  .\scripts\gv1_train_conditioned_pinn.py `
  .\scripts\gv1_prediction_metrics.py

2. First run only 40 ks:

powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_40ks_d951.ps1

3. Read metrics:

Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_40ks_d951\metrics_summary_d951_40ks.json" -Raw

Pass criteria for 40 ks:
- all three prediction.npz files exist;
- t_end_s is close to 40000;
- B1 corr should recover relative to D9.4/D9.4.1 and preferably be >= 0.90;
- B3/B4 should not be worse than D9.5; ideally MAE < 0.13 V and corr > 0.92;
- pred_upper_frac_ge_4p269 should remain 0;
- low-voltage tail improvement is secondary in this step; trend preservation is primary.

Only after 40 ks passes should 200 ks be run. Do not directly run 500 ks or 24 profiles.
