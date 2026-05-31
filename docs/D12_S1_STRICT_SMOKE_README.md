# D12-S1 strict metadata ablation smoke runner

This package replaces the earlier D12 runtime smoke workflow with a stricter and safer D12-S1 workflow.

## What this package fixes

1. It never generates 40000-epoch or 200ks commands.
2. It uses a separate D12-S1 output prefix so old broken D12 runtime outputs are not mixed with new runs.
3. It computes metrics directly from `prediction.npz`, so scorecard collection no longer depends on a missing `d10_voltage_metrics.json`.
4. It keeps B1_2C battery-8 excluded from the normal 23-profile metadata ablation scope.
5. It does not modify D9.6/D9.5.1 mainline source files.

## Default D12-S1 smoke parameters

```text
epochs = 100
time_window_s = 40000
max_time_points = 1024
batch_size = 512
profile_limit = 3
```

## Recommended manual workflow

Run from the project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
```

Prepare commands:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_prepare_strict_smoke_commands.ps1"
```

Preflight. Do not continue unless it prints `D12-S1 preflight PASS`:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_preflight.ps1"
```

Run the three modes:

```powershell
$CmdRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1_metadata_ablation_commands"
powershell -ExecutionPolicy Bypass -File "$CmdRoot\run_d12_s1_metadata_off_3profile.generated.ps1"
powershell -ExecutionPolicy Bypass -File "$CmdRoot\run_d12_s1_metadata_zero_3profile.generated.ps1"
powershell -ExecutionPolicy Bypass -File "$CmdRoot\run_d12_s1_metadata_on_3profile.generated.ps1"
```

Collect scorecard directly from `prediction.npz`:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_collect_scorecard.ps1"
```

View outputs:

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1_metadata_ablation_scorecard\d12_s1_scorecard_summary.json" -Raw

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1_metadata_ablation_scorecard\d12_s1_scorecard.csv" |
  Sort-Object mode, metadata_profile_id |
  Format-Table mode,status,mae_V,rmse_V,corr,bias_V,metadata_dim,metadata_profile_id,metrics_source -AutoSize

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1_metadata_ablation_scorecard\d12_s1_mode_summary.csv" |
  Format-Table mode,n,ok,mean_mae_V,mean_corr,mean_bias_V -AutoSize
```

## Do not do

Do not run any scripts under:

```text
xjtu_batch134_d12_runtime_metadata_ablation_commands
```

Those were the earlier mistaken long-run commands.
