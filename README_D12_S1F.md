# GV1 D12-S1F: low-anchor + high-voltage lock, fast 6-profile screen

D12-S1F is a focused follow-up to D12-S1E.  S1E proved that the low-voltage branch is strong enough: the `soft` candidate passed low/deep/global/normal/corr/rest, but failed only `high_ok` because `high_target_ge_4p10` regressed and `pred_high_overshoot_gt_4p35` appeared.

S1F therefore does **not** further strengthen low correction.  It keeps the S1E low residual anchor and adds high-voltage protection:

1. target-free high-voltage suppression gate in the P2D branch;
2. no upward P2D correction by default, preventing the S1E overshoot path;
3. high-region regret guard relative to the no-P2D baseline;
4. high-region correction budget;
5. explicit pred > 4.35 V overshoot penalty.

It remains an isolated diagnostic package.  It does **not** overwrite the D9.6/D9.5.1 GV1 mainline, does **not** unflag battery-8, does **not** enable metadata, and does **not** use hard voltage clamp.

## Faster/GPU-friendlier defaults

Compared with S1E, this package is configured to be faster by default:

```text
S1E default: 6 profiles × 4 modes = 24 runs, BatchSize 2048, Epochs 1200
S1F default: 6 profiles × 3 modes = 18 runs, BatchSize 4096, Epochs 800
```

The default modes are:

```text
baseline_d951
d12s1f_p2d_low_anchor_highsafe_soft
d12s1f_p2d_low_anchor_highsafe_mid
```

`guarded` is intentionally removed from the default sweep because S1E showed it was slower and worse for high-voltage preservation.

## Files

```text
gv1/d12_s1_p2d_model.py
gv1/d12_s1_p2d_transform.py
gv1/d12_s1_p2d_losses.py
gv1/d12_s1_p2d_trainer.py
scripts/gv1_train_d12_s1_p2d_local.py
scripts/gv1_scorecard_d12_s1f.py
scripts/gv1_run_d12_s1f_6profile_40ks.ps1
scripts/gv1_run_d12_s1f_6profile_200ks.ps1
README_D12_S1F.md
install_manifest.json
```

## Run order

First run a 2-epoch smoke.  Run the `.ps1` directly in PowerShell; do not prefix it with Python.

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$py = "D:\Anaconda\envs\torchgpu\python.exe"
$cache = "E:\XJTU battery dataset\_gv1_cache"
$proj = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

.\scripts\gv1_run_d12_s1f_6profile_40ks.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Epochs 2 `
  -BatchSize 1024 `
  -MaxTimePoints 512 `
  -PredictionTimePoints 256 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

Smoke pass standard:

```text
prediction_count = 18
metrics_ok_count = 18
read_error_count = 0
```

Then run the faster formal 40ks screen:

```powershell
.\scripts\gv1_run_d12_s1f_6profile_40ks.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Epochs 800 `
  -BatchSize 4096 `
  -MaxTimePoints 4096 `
  -PredictionTimePoints 2048 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

Scorecard directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1f_p2d_highsafe_fast_6x40ks_scorecard
```

Do **not** run 200ks unless `D12_S1F_candidate_decisions.csv` contains `promote_to_200ks=True`.

## Promotion rules

S1F keeps the strict S1E promotion gates:

```text
delta_low_target_MAE_V <= -0.020
delta_low_le_2p75_MAE_V <= -0.020
delta_all_MAE_V <= 0.005
delta_normal_MAE_V <= 0.005
delta_corr >= -0.005
rest/high not degraded beyond 20 mV
```

The expected S1F success pattern is:

```text
low_ok=True
deep_ok=True
global_ok=True
normal_ok=True
high_ok=True
promote_to_200ks=True
```

If no candidate promotes, stop and inspect segment metrics. Do not expand to 200ks or 23-profile.
