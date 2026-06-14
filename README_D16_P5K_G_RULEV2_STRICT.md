# D16-P5K-G rule_v2 strict-gate formal training package

This package is the formal training step after the G0-G4 audit chain.

Boundary:
- Training uses only observed time series: `t_global_s`, `I_profile`, `voltage_exp`.
- It does not use `theta/cs/phie/phis_c` soft-label arrays as data loss.
- Soft labels are used only in baseline preflight/evaluation.

Core change vs P5K-F:
- Reverts to the P5K-C hard baseline as the default normal-profile baseline.
- Adds the G4-validated strict metadata theta0 adapter:
  - Batch-5 battery-8: shift_a=-0.4536, shift_c=+0.4186
  - Batch-1 battery-8: shift_a=-0.4320, shift_c=+0.3970
  - Batch-6 battery-6: shift_a=-0.2916, shift_c=+0.2566
  - Batch-2 battery-2: shift_a=-0.2700, shift_c=+0.2350
- Keeps the residual NN bounded; the NN should learn residuals, not the mean inventory trajectory.

Important evidence:
- G4 exact array audit PASS: normal eval does not regress vs P5K-C baseline, hard_probe is repaired.
- P5K-G training must still run full evaluation to check `phis_c/phie/theta/cs/gradient` together.

## Commands

### Build manifest only
```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg_train_fast.ps1 `
  -AllowOverwrite `
  -BuildManifestOnly `
  -TrainSet "G_train12_rulev2_strict"
```

### 50 epoch smoke training
```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg_train_fast.ps1 `
  -AllowOverwrite `
  -SkipBaselineOnlyAudit `
  -TrainOnly `
  -TrainSet "G_train12_rulev2_strict" `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Check:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kg_outputs.ps1 `
  -TrainSet "G_train12_rulev2_strict"
```

For smoke, eval files may be MISSING because `-TrainOnly` was used. Manifest, training summary, train input audit, and best checkpoint must be FOUND.

### Formal training + ALL55 evaluation
```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg_train_fast.ps1 `
  -AllowOverwrite `
  -SkipBaselineOnlyAudit `
  -TrainSet "G_train12_rulev2_strict" `
  -Epochs 1200 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 32768 `
  -ChunkSize 100000 `
  -ValEvery 10 `
  -MmapCacheRoot "E:\XJTU battery dataset\_gv1_cache\_p5kg_eval_mmap_cache_clean"
```

If E: disk space is tight, use another disk for `-MmapCacheRoot`.

### Result view
```powershell
$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg_rulev2_strict_gate_FAST\G_train12_rulev2_strict"
$EvalDir = "$RunDir\eval_all55_vs_softlabels"
$Score = "$EvalDir\D16_P5KG_FINAL_SCORECARD.json"
$j = Get-Content $Score -Raw | ConvertFrom-Json
$j | Select-Object stage, operational_status, profile_count_requested, profile_count_evaluated, failure_count
$j.global_metrics_weighted | Format-List
Import-Csv "$EvalDir\D16_P5KG_SPLIT_METRICS.csv" | Format-List
```

Promotion gate:
- eval theta_a_mean_mae <= 0.139017
- eval theta_a_mean_r2 >= 0.474238
- eval theta_c_mean_mae <= 0.123569
- eval theta_c_mean_r2 >= 0.391913
- hard_probe theta_a_mean_mae <= 0.10
- hard_probe theta_c_mean_mae <= 0.10
- phis_c_r2 > 0.999

