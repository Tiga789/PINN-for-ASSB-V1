# D16-P5K train6to10 hard-cbar/OCP residual package v1

## Purpose

D16-P5K is a structural rollback/rewrite after the P5H exact-R2 audit. It is not another P5G-style heuristic gauge-loss patch.

The package restores the ASSB success principle:

- hard `I(t)` / cbar-style inventory baseline,
- OCP/observed-voltage endpoint style phase anchor,
- bounded residual learning,
- asymmetric radial residual,
- exact R2 in every evaluation scorecard.

It also keeps the project-level direction: ASSB and XJTU should remain one upper-level Aether/PINN workflow, but with separate electrochemical branches/experts. P5K is the XJTU/P2Dlite-RG branch rewrite, designed to stay compatible with the successful ASSB hard-baseline logic.

## Hard constraints

Training reads only:

```text
t_global_s / time
I_profile / current
voltage_exp / voltage
```

Training does not read these soft-label internal targets:

```text
theta_a, theta_c, cs_a, cs_c, phie, phis_c, phis_c_soft
```

Soft labels are used only by `gv1_d16_p5k_eval55_vs_softlabels.py` for evaluation/audit.

## Train sets

The package supports three splits:

```text
A_train6  = original 6-cell representative training set
B_train8  = train6 + random_walk hard case + GEO long profile
C_train10 = train8 + flagged 2C boundary + 3C stress hard case
```

Default is `C_train10`, because the previous P5B-P5G results showed persistent failure on random_walk, flagged/late 2C, GEO long profile and Batch-2 stress cases.

## Files

```text
configs/d16_p5k_hard_cbar_ocp_residual_config.json
scripts/gv1_d16_p5k_build_manifest.py
scripts/gv1_d16_p5k_train10_hard_cbar_ocp_residual_fast.py
scripts/gv1_d16_p5k_eval55_vs_softlabels.py
scripts/gv1_run_d16_p5k_train6to10_fast.ps1
scripts/gv1_check_d16_p5k_outputs.ps1
```

## Recommended execution

### 1. Build manifest only

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5k_train6to10_fast.ps1 `
  -AllowOverwrite `
  -BuildManifestOnly `
  -TrainSet "C_train10"
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5k_outputs.ps1 `
  -TrainSet "C_train10"
```

You should see:

```text
Manifest rows: 55; train=10; eval=45
```

### 2. Smoke training

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5k_train6to10_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -TrainSet "C_train10" `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5k_outputs.ps1 `
  -TrainSet "C_train10"
```

You should see:

```text
FOUND: ...\model_hard_cbar_ocp_residual\model\best_with_state.pt
FOUND: ...\D16_P5K_TRAIN_INPUT_AUDIT.json
```

### 3. Formal P5K-C train10 training and evaluation

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5k_train6to10_fast.ps1 `
  -AllowOverwrite `
  -TrainSet "C_train10" `
  -Epochs 1200 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

If GPU memory is tight:

```powershell
-BatchSize 65536 -EvalBatchSize 32768 -ChunkSize 100000
```

## Output locations

Default run directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST\C_train10
```

Training summary:

```text
...\model_hard_cbar_ocp_residual\D16_P5K_TRAINING_SUMMARY.json
```

Training input audit:

```text
...\model_hard_cbar_ocp_residual\D16_P5K_TRAIN_INPUT_AUDIT.json
```

Final scorecard:

```text
...\eval_all55_vs_softlabels\D16_P5K_FINAL_SCORECARD.json
```

Metrics:

```text
...\eval_all55_vs_softlabels\D16_P5K_METRICS_BY_PROFILE.csv
...\eval_all55_vs_softlabels\D16_P5K_SPLIT_METRICS.csv
...\eval_all55_vs_softlabels\D16_P5K_BATCH_METRICS.csv
...\eval_all55_vs_softlabels\D16_P5K_PROTOCOL_METRICS.csv
```

## How to inspect results

```powershell
$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST\C_train10"
$EvalDir = "$RunDir\eval_all55_vs_softlabels"
$Score = "$EvalDir\D16_P5K_FINAL_SCORECARD.json"

$j = Get-Content $Score -Raw | ConvertFrom-Json
$j | Select-Object stage, operational_status, profile_count_evaluated, failure_count
$j.global_metrics_weighted | Format-List
Import-Csv "$EvalDir\D16_P5K_SPLIT_METRICS.csv" | Format-List
```

Key promotion metrics:

```text
eval theta_a_mean_mae
eval theta_a_mean_r2
eval theta_c_mean_mae
eval theta_c_mean_r2
eval cs_a_mean_r2 / cs_c_mean_r2 if cs fields are available
phis_c_r2
phie_r2
```

## Promotion target

P5K is only meaningful if it corrects the P5H-discovered absolute inventory/gauge failure. Minimum target:

```text
eval theta_a_mean_mae < 0.15
eval theta_c_mean_mae < 0.15
eval theta_a_mean_r2 > 0.85
eval theta_c_mean_r2 > 0.85
phis_c_r2 > 0.99
```

If P5K does not improve exact R2, do not continue adding epochs. The next step would be to inspect the hard-baseline initialization and OCP/cbar phase anchor, not to add heuristic gap losses.
