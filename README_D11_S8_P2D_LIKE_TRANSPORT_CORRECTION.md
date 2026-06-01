# D11-S8 P2D-like low-voltage transport deficit correction

This package is a **post-transform diagnostic correction** for the D11 GV1/XJTU voltage surrogate.
It does not train and does not overwrite the D9.6/D9.5.1 mainline.

## Purpose

D11-S7 showed that the low-voltage target region remains too high even after escape-head attempts.
D11-S8 tests a P2D-like transport-deficit term:

```text
V_corr = V_base - G_low * G_discharge * G_transport * DeltaV_transport
```

where `DeltaV_transport >= 0` and is derived only from existing prediction components and measured-current replay features.  Target voltage is used only for scoring, not to build the correction.

## Expected scope

- Uses existing baseline prediction.npz files from D11-S7/S5C/S5A/S4 or D12-S3.
- 6 profiles, battery-8 excluded.
- Modes:
  - baseline_copy
  - p2dlike_transport_mild
  - p2dlike_transport_medium
  - p2dlike_transport_strong_guarded
  - p2dlike_transport_discharge_only
- Expected outputs: 6 x 5 = 30 prediction.npz files.

## Run order

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
D:\Anaconda\envs\torchgpu\python.exe -m compileall gv1 scripts
.\scripts\run_gv1_d11_s8_preflight_check.ps1
.\scripts\run_gv1_d11_s8_prepare_commands.ps1
.\scripts\run_gv1_d11_s8_preflight_check.ps1 -AfterPrepare
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s8_p2dlike_transport_correction_commands\run_d11_s8_all_modes.generated.ps1"
.\scripts\run_gv1_d11_s8_collect_scorecard.ps1
```

## Files to send back

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s8_p2dlike_transport_correction_scorecard\D11_S8_scorecard_summary.json
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s8_p2dlike_transport_correction_scorecard\D11_S8_RECOMMENDATION.md
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s8_p2dlike_transport_correction_scorecard\D11_S8_global_vs_lowtarget_tradeoff.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s8_p2dlike_transport_correction_scorecard\D11_S8_mode_segment_summary.csv
```

## Promotion rule

Promote a candidate only if:

```text
low_target MAE decreases by at least 20 mV,
low_target_le_2p75 MAE decreases by at least 20 mV,
global MAE does not increase materially,
corr does not drop materially,
rest/high-target segments remain stable,
no high-voltage overshoot is introduced.
```
