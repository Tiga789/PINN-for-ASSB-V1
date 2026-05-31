# D12-S2 Balanced 6-profile strict smoke

D12-S2 expands the validated D12-S1 result to a balanced 6-profile metadata ablation.

## Scope

- Selects 2 non-target profiles from each protocol group: `2C`, `R2.5`, `R3`.
- Runs 3 modes: `metadata_off`, `metadata_zero`, `metadata_on`.
- Total short runs: 18 = 6 profiles × 3 modes.
- B1_2C battery-8 remains excluded.
- No D9.6/D9.5.1 source file is modified.

## Fixed strict-smoke parameters

```text
epochs = 100
time_window_s = 40000
max_time_points = 1024
batch_size = 512
prediction_time_points = 1024
```

The generator refuses non-smoke settings such as `40000 epochs`, `200ks`, `8192 time points`, or `2048 batch size`.

## Recommended commands

Run from project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s2_prepare_balanced_strict_smoke_commands.ps1"

powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s2_preflight.ps1"
```

Only after seeing:

```text
D12-S2 preflight PASS: generated scripts are balanced strict smoke commands.
```

run:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s2_run_triplet.ps1"
```

Collect scorecard:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s2_collect_scorecard.ps1"
```

View outputs:

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s2_metadata_ablation_scorecard\d12_s2_scorecard_summary.json" -Raw

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s2_metadata_ablation_scorecard\d12_s2_scorecard.csv" |
  Sort-Object mode,metadata_profile_id |
  Format-Table mode,status,mae_V,rmse_V,corr,bias_V,metadata_dim,metadata_profile_id,metrics_source -AutoSize

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s2_metadata_ablation_scorecard\d12_s2_mode_summary.csv" |
  Format-Table mode,n,ok,mean_mae_V,mean_corr,mean_bias_V -AutoSize
```

## Decision rule

- `18/18 smoke_completed_metrics_ok`: runtime is valid.
- `metadata_on` lower mean MAE than both `off` and `zero`: metadata signal is worth expanding to D12-S3.
- `metadata_on` lower MAE than `off` but not better than `zero`: likely architecture/control effect; be cautious.
- `metadata_on` worse than `off`/`zero`: pause metadata expansion.

Do not run 23-profile or 24-profile training from this package.
