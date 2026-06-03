# D12-S1C balanced P2D-like localized correction

This package follows D12-S1 and D12-S1B.

Observed failure pattern:

- D12-S1: low_target improved by roughly 35--48 mV, but normal/global leaked downward and global_ok failed.
- D12-S1B: normal/global preservation succeeded, but the low-voltage correction was over-suppressed and low_ok failed.

D12-S1C changes the mechanism rather than simply increasing or decreasing the same knobs:

1. It disables the over-strong S1B prediction-side normal suppression in the supplied modes.
2. It adds asymmetric normal-region down-shift guards in the training loss.
3. It keeps low_target / deep-low terms active, because low samples are excluded from the normal preservation mask.
4. It keeps global and normal promotion thresholds strict: <= 5 mV regression.

Files are additive to the GV1/D12 experimental path and do not overwrite the D9.6/D9.5.1 mainline trainer.

## Smoke

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$py = "D:\Anaconda\envs\torchgpu\python.exe"

.\scripts\gv1_run_d12_s1c_6profile_40ks.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PythonExe $py `
  -Epochs 2 `
  -BatchSize 512 `
  -MaxTimePoints 512 `
  -PredictionTimePoints 256 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

Smoke pass criterion:

```text
prediction_count = 24
metrics_ok_count = 24
read_error_count = 0
```

## Formal 40ks run

```powershell
.\scripts\gv1_run_d12_s1c_6profile_40ks.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PythonExe $py `
  -Epochs 1200 `
  -BatchSize 2048 `
  -MaxTimePoints 4096 `
  -PredictionTimePoints 2048 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

Scorecard:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1c_p2d_balanced_6x40ks_scorecard
```

Promotion requires:

```text
low_ok=True
deep_ok=True
global_ok=True
normal_ok=True
corr_ok=True
rest_ok=True
high_ok=True
promote_to_200ks=True
```

Practical target:

```text
delta_low_target_MAE_V <= -0.020
delta_low_le_2p75_MAE_V <= -0.020
delta_all_MAE_V <= 0.005
delta_normal_MAE_V <= 0.005
delta_corr >= -0.005
```

Do not run 200ks unless the 40ks scorecard promotes at least one candidate.
