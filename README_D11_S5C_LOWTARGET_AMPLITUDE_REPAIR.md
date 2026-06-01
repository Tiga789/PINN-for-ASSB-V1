# D11-S5C low-target correction amplitude repair

Purpose: D11-S5A showed that the low-voltage gate is active, but the low-tail
correction became weaker under previous candidates. D11-S5C tests distinct
amplitude-repair modes before any 200 ks confirmation.

This package does not modify the D9.6/D9.5.1 mainline. It only adds scripts.
Battery-8 remains excluded. metadata_on is not used. Hard clamp must remain disabled.

## Files

```text
scripts/gv1_d11_s5c_prepare_lowtarget_amplitude_repair_commands.py
scripts/gv1_d11_s5c_scorecard_from_predictions.py
scripts/run_gv1_d11_s5c_preflight_check.ps1
scripts/run_gv1_d11_s5c_prepare_commands.ps1
scripts/run_gv1_d11_s5c_collect_scorecard.ps1
README_D11_S5C_LOWTARGET_AMPLITUDE_REPAIR.md
RUN_ORDER_D11_S5C.txt
```

## Run order

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
D:\Anaconda\envs\torchgpu\python.exe -m compileall gv1 scripts

.\scripts\run_gv1_d11_s5c_preflight_check.ps1
.\scripts\run_gv1_d11_s5c_prepare_commands.ps1
.\scripts\run_gv1_d11_s5c_preflight_check.ps1 -AfterPrepare
```

Then run the generated scripts under:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_commands
```

Recommended first run:

```powershell
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_commands\run_d11_s5c_baseline_d951.generated.ps1"
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_commands\run_d11_s5c_lowtarget_amplify_down_1p25.generated.ps1"
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_commands\run_d11_s5c_lowtarget_amplify_down_1p50.generated.ps1"
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_commands\run_d11_s5c_lowtarget_amplify_down_1p75_guarded.generated.ps1"
```

Or all modes:

```powershell
& "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_commands\run_d11_s5c_all_modes.generated.ps1"
```

Collect the scorecard:

```powershell
.\scripts\run_gv1_d11_s5c_collect_scorecard.ps1
```

Output:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5c_lowtarget_amplitude_repair_scorecard
```

## Promotion rule

Do not expand to 200 ks unless a candidate improves both:

```text
low_target
low_target_le_2p75
```

while preserving global MAE/corr, rest_I_zero, and high-target/overshoot behavior.
