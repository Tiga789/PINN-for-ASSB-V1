# GV1 D12-S1E: low-residual anchor + normal correction budget

D12-S1E is a small-window diagnostic / promotion candidate for the GV1 low-voltage bottleneck. It does **not** replace the D9.6/D9.5.1 mainline, does **not** unflag battery-8, and does **not** modify old ASSB `main.py`, `util/*`, or `integration_spm/*` files.

## Why S1E exists

Earlier runs showed a clear tradeoff:

- **D12-S1C**: low_target / low<=2.75 improved by about 20 mV, but normal/global MAE regressed too much.
- **D12-S1D**: normal-regret guards reduced the branch leakage pressure, but low_target improvement became too weak.

S1E changes the mechanism rather than simply increasing or decreasing a weight:

1. **Low residual anchor**: on low_target/deep-low samples, the P2D correction is trained toward the no-P2D residual `baseline_without_p2d - target`.
2. **Normal correction budget**: outside the low segment, the positive downward correction is constrained by a small mV budget.
3. **Regret guard remains**: if a correction worsens normal/non-low error relative to the no-P2D baseline, it is penalized.

This is intended to keep the S1C low-voltage benefit while preventing the normal/global leakage that blocked promotion.

## Files

```text
gv1/d12_s1_p2d_model.py
gv1/d12_s1_p2d_transform.py
gv1/d12_s1_p2d_losses.py
gv1/d12_s1_p2d_trainer.py
scripts/gv1_train_d12_s1_p2d_local.py
scripts/gv1_scorecard_d12_s1e.py
scripts/gv1_run_d12_s1e_6profile_40ks.ps1
scripts/gv1_run_d12_s1e_6profile_200ks.ps1
README_D12_S1E.md
install_manifest.json
```

## Run order

First run a 2-epoch smoke:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$py = "D:\Anaconda\envs\torchgpu\python.exe"
$cache = "E:\XJTU battery dataset\_gv1_cache"
$proj = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

.\scripts\gv1_run_d12_s1e_6profile_40ks.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Epochs 2 `
  -BatchSize 512 `
  -MaxTimePoints 512 `
  -PredictionTimePoints 256 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

Smoke pass standard:

```text
prediction_count = 24
metrics_ok_count = 24
read_error_count = 0
```

Then run formal 6-profile 40ks:

```powershell
.\scripts\gv1_run_d12_s1e_6profile_40ks.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Epochs 1200 `
  -BatchSize 2048 `
  -MaxTimePoints 4096 `
  -PredictionTimePoints 2048 `
  -Seed 42 `
  -Device "auto" `
  -Clean
```

Scorecard directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1e_p2d_anchor_budget_6x40ks_scorecard
```

Do **not** run 200ks unless `D12_S1E_candidate_decisions.csv` contains `promote_to_200ks=True`.

## Promotion rules

S1E keeps the strict S1C/S1D promotion gates:

```text
delta_low_target_MAE_V <= -0.020
delta_low_le_2p75_MAE_V <= -0.020
delta_all_MAE_V <= 0.005
delta_normal_MAE_V <= 0.005
delta_corr >= -0.005
rest/high not degraded beyond 20 mV
```

If no candidate promotes, stop and inspect segment metrics. Do not expand to 200ks or 23-profile.
