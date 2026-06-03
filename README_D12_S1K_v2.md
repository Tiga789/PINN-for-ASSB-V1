# D12-S1K v2 source-generation + two-candidate confirmation package

## Why this v2 package exists

The first S1K package assumed this source directory already existed:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1e_p2d_anchor_budget_23x40ks
```

That assumption was wrong for your local project. S1K is a wrapper confirmation stage, so it **requires** paired source predictions first:

- `baseline_d951`
- `d12s1e_p2d_low_anchor_soft`

This v2 package adds the missing source-generation runner.

## What it does

1. `gv1_run_d12_s1k_generate_s1e_source_23profile_40ks_fast_parallel.ps1`
   - Generates baseline + S1E-soft predictions for 23 profiles.
   - Excludes only `Batch-1_2C_battery-8`.
   - Uses a fast parallel queue.
   - Expected prediction count: `23 profiles x 2 modes = 46`.

2. `gv1_run_d12_s1k_apply_23profile_40ks.ps1`
   - Applies two S1J-promoted wrapper candidates:
     - `d12s1k_low_only_revert_nonlow_to_baseline`
     - `d12s1k_low_plus_transition_fade_to_baseline`

No mainline GV1 D9.6/D9.5.1 files are modified.

## Run source prediction generation

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

$py = "D:\Anaconda\envs\torchgpu\python.exe"
$proj = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$cache = "E:\XJTU battery dataset\_gv1_cache"

.\scripts\gv1_run_d12_s1k_generate_s1e_source_23profile_40ks_fast_parallel.ps1 `
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

If CUDA OOM occurs:

```powershell
-MaxParallel 1
```

or:

```powershell
-BatchSize 2048
```

Check source prediction count:

```powershell
Get-ChildItem "$cache\xjtu_batch134_d12_s1e_p2d_anchor_budget_23x40ks" -Recurse -Filter prediction.npz |
  Measure-Object
```

Expected:

```text
46
```

## Run S1K two-candidate wrapper

```powershell
.\scripts\gv1_run_d12_s1k_apply_23profile_40ks.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -SourceRunsRoot "$cache\xjtu_batch134_d12_s1e_p2d_anchor_budget_23x40ks" `
  -Clean
```

Output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1k_two_candidate_23x40ks_scorecard
```

Send these files for analysis:

```text
D12_S1K_scorecard_summary.json
D12_S1K_candidate_decisions.csv
D12_S1K_mode_summary.csv
D12_S1K_segment_metrics.csv
D12_S1K_run_metrics.csv
D12_S1K_source_leakage_overview.csv
D12_S1K_RECOMMENDATION.md
```
