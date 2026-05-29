# GV1 D9.5: trend-preserving rare-regime voltage correction

## Purpose

D9.5 follows the D9.3 -> D9.4 -> D9.4.1 diagnostics:

- D9.3 preserved global voltage trend and worked especially well for B3/R2.5 and B4/R3, but low-voltage tail coverage remained weak.
- D9.4 / D9.4.1 opened some B1 low-voltage coverage, but B1_2C 200ks correlation regressed to about 0.83.
- D9.5 therefore does not simply route B1/2C back to a D9.2-like smooth branch.  Instead, it keeps the D9.3 event/low-tail transform channels and adds trend-preserving rare-regime losses.

The goal is broader than "fix the low-voltage tail": D9.5 targets rare regimes in general, including low-voltage tails, high-voltage tails, high-rate/current-event periods and temperature-event periods.

## Main changes

1. `gv1/losses.py`
   - Adds centered voltage-shape correlation loss: `voltage_corr`.
   - Adds ultra-low/high quantile matching: `voltage_ultra_quantile`.
   - Adds differentiable low/high voltage coverage matching: `voltage_low_coverage`.
   - Adds low/high tail mean balance: `voltage_tail_balance`.
   - Keeps D9.3 robust voltage, tail, range, event and physics regularizers.

2. `gv1/profile_adaptive.py`
   - Replaces the D9.4.1 low-rate smooth-branch idea with two auditable D9.5 modes:
     - `lowrate_trend_tail` for low-rate 2C profiles.
     - `event_highrate_trend_tail` for R2.5/R3/high-current/high-temperature profiles.
   - Auto mode chooses from protocol/current/temperature only; it does not inspect metrics.

3. `gv1/output_transform.py`
   - Keeps D9.3 event/low-tail voltage correction channels available.
   - Keeps branch/component diagnostics in `prediction.npz`.

4. `scripts/gv1_prediction_metrics.py`
   - Keeps D9.4/D9.4.1 branch diagnostics plus D9.3 rare-regime metrics.

## Manual deployment note

The user manually extracts zip files and adds/overwrites files in the project.  Do not provide automatic unzip instructions unless the user explicitly asks.

## Recommended validation order

After manual overwrite, run syntax check:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

D:\Anaconda\envs\torchgpu\python.exe -m py_compile `
  .\gv1\model.py `
  .\gv1\output_transform.py `
  .\gv1\profile_adaptive.py `
  .\gv1\losses.py `
  .\gv1\trainer.py `
  .\scripts\gv1_train_conditioned_pinn.py `
  .\scripts\gv1_prediction_metrics.py
```

Then run 40ks first:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_profile_compare_40ks_d95.ps1
```

Inspect:

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_40ks_d95\metrics_summary_d95_40ks.json" -Raw
```

Only after 40ks passes, run 200ks.  Do not directly run 24 profiles.

## First-pass pass/fail guide

40ks can continue if:

- all three `prediction.npz` files are generated;
- `t_end_s` is about 40000;
- no high-voltage saturation: `pred_upper_frac_ge_4p269` remains near 0;
- B1 correlation is not allowed to collapse as in D9.4/D9.4.1; target is preferably `corr > 0.90`, minimally better than D9.4.1;
- MAE should remain in the D9.2/D9.3 range or improve;
- low-voltage coverage should improve versus D9.3 without sacrificing overall trend.

