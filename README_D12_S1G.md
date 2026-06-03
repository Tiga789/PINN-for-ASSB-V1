# D12-S1G: S1E-soft + local high-only limiter (fast parallel)

## Why this package exists

D12-S1E was the best recent direction: the soft candidate fixed the low-voltage target and preserved global/normal/corr/rest, but failed only `high_ok` because `high_target_ge_4p10` degraded and `pred_high_overshoot_gt_4p35` appeared.

D12-S1F tried a broad prediction-side high-safe suppression. It failed because the whole voltage curve was pulled downward: low/deep still improved, but global, normal, high, rest, and corr all failed.

D12-S1G therefore rolls back to S1E-soft as the mother candidate and adds only **local target-aware high-voltage loss-side protection**:

- keep S1E low residual anchor;
- keep S1E normal correction budget;
- disable prediction-side high suppression by default;
- keep a small upward correction allowance to avoid S1F-style global downward bias;
- add local high regret, high correction budget, and pred>4.35 V overshoot penalties.

## Main files

```text
gv1/d12_s1_p2d_model.py
gv1/d12_s1_p2d_transform.py
gv1/d12_s1_p2d_losses.py
gv1/d12_s1_p2d_trainer.py
scripts/gv1_train_d12_s1_p2d_local.py
scripts/gv1_scorecard_d12_s1g.py
scripts/gv1_run_d12_s1g_6profile_40ks.ps1
scripts/gv1_run_d12_s1g_6profile_40ks_fast_parallel.ps1
scripts/gv1_run_d12_s1g_6profile_200ks.ps1
```

## Recommended smoke

```powershell
.\scripts\gv1_run_d12_s1g_6profile_40ks_fast_parallel.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Epochs 2 `
  -BatchSize 1024 `
  -MaxTimePoints 512 `
  -PredictionTimePoints 256 `
  -Seed 42 `
  -Device "auto" `
  -MaxParallel 2 `
  -Clean
```

Smoke success criteria:

```text
prediction_count = 18
metrics_ok_count = 18
read_error_count = 0
```

## Recommended formal 40ks

```powershell
.\scripts\gv1_run_d12_s1g_6profile_40ks_fast_parallel.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Epochs 800 `
  -BatchSize 4096 `
  -MaxTimePoints 4096 `
  -PredictionTimePoints 2048 `
  -Seed 42 `
  -Device "auto" `
  -MaxParallel 2 `
  -Clean
```

Only run 200ks if `D12_S1G_candidate_decisions.csv` contains `promote_to_200ks=True`.
