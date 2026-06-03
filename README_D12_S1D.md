# GV1 D12-S1D: train-inside P2D-like correction with normal-regret guards

D12-S1D is a **small-window diagnostic / promotion candidate** for the GV1 low-voltage bottleneck.  It does not replace the D9.6/D9.5.1 mainline and does not modify ASSB legacy files.

## Why S1D exists

Observed sequence:

- D12-S1: low_target improved by about 35–48 mV, but global/normal MAE regressed.
- D12-S1B: normal/global preservation was safe, but low_target improvement disappeared.
- D12-S1C: low_target just crossed the 20 mV threshold for mid/guarded, but normal_target_gt_3p20 still regressed by about 8.6–10.1 mV and global MAE by about 8.1–9.5 mV.

S1D therefore adds **target-aware normal/non-low regret guards** inside the training loss:

```text
regret = max(|pred - target| - |baseline_without_p2d - target| - allowed, 0)^2
```

This is different from S1B's broad preservation: it allows the P2D branch when it improves a point, but blocks it when it worsens normal/non-low regions.

## Files

```text
gv1/d12_s1_p2d_model.py
gv1/d12_s1_p2d_transform.py
gv1/d12_s1_p2d_losses.py
gv1/d12_s1_p2d_trainer.py
scripts/gv1_train_d12_s1_p2d_local.py
scripts/gv1_scorecard_d12_s1d.py
scripts/gv1_run_d12_s1d_6profile_40ks.ps1
scripts/gv1_run_d12_s1d_6profile_200ks.ps1
README_D12_S1D.md
install_manifest.json
```

## Run smoke first

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$py = "D:\Anaconda\envs\torchgpu\python.exe"

.\scripts\gv1_run_d12_s1d_6profile_40ks.ps1 `
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

Expected smoke integrity:

```text
prediction_count = 24
metrics_ok_count = 24
read_error_count = 0
```

## Formal 40ks run

```powershell
.\scripts\gv1_run_d12_s1d_6profile_40ks.ps1 `
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
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1d_p2d_regret_guarded_6x40ks_scorecard
```

Promotion requires:

```text
promote_to_200ks = True
low_ok = True
deep_ok = True
global_ok = True
normal_ok = True
corr_ok = True
rest_ok = True
high_ok = True
```

Numerical rule:

```text
delta_low_target_MAE_V <= -0.020
delta_low_le_2p75_MAE_V <= -0.020
delta_all_MAE_V <= 0.005
delta_normal_MAE_V <= 0.005
delta_corr >= -0.005
```

Do not run 200ks unless a 40ks candidate is promoted.
