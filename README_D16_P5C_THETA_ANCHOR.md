# D16-P5C theta-anchor physics training package

## Purpose

D16-P5C continues the D16-P5B train6/eval49 experiment but strengthens the internal-state physics constraints. It keeps the same scientific boundary:

- train cells: Batch-1 battery-3, Batch-2 battery-8, Batch-3 battery-6, Batch-4 battery-7, Batch-5 battery-7, Batch-6 battery-3;
- eval cells: the remaining 49 cells;
- training-visible time-series keys: `t_global_s`, `I_profile`, `voltage_exp` only;
- no internal-state data loss during training;
- `theta_a/theta_c/cs_a/cs_c/phie/phis_c/phis_c_soft` are not loaded by the trainer;
- P2Dlite-RG soft labels are used only in the separate evaluation script.

P5C is designed because D16-P5B-500epochs gave good potential performance but poor theta-state performance. P5C adds observation-derived theta-gauge constraints:

1. voltage/OCP-inverse theta anchor derived from `voltage_exp`;
2. Coulomb-integral anchor derived from `I_profile` integral;
3. two-electrode mass-gauge prior `theta_a + theta_c ~= 1`;
4. current-direction gradient anchor;
5. rest-relaxation penalty.

These are not soft-label data losses. They are physics/observation constraints derived from allowed I/V time series and fixed modeling priors.

## Files

```text
configs/d16_p5c_theta_anchor_config.json
scripts/gv1_d16_p5c_build_manifest.py
scripts/gv1_d16_p5c_train6_theta_anchor_fast.py
scripts/gv1_d16_p5c_eval55_vs_softlabels.py
scripts/gv1_run_d16_p5c_train6_eval49_fast.ps1
scripts/gv1_check_d16_p5c_outputs.ps1
```

## Run commands

Copy this package into the project root:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

### 1. Build manifest only

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5c_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -BuildManifestOnly
```

Expected manifest:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\D16_P5C_TRAIN6_EVAL49_MANIFEST.csv
```

### 2. Fast smoke training

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5c_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5c_outputs.ps1
```

Expected checkpoint:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\model_train6_theta_anchor_observation_physics\model\best_with_state.pt
```

### 3. Formal training and all55 evaluation

Suggested first run uses 500 epochs, matching your D16-P5B comparison point:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5c_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -Epochs 500 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

If GPU memory is stable but utilization is low, you may try:

```powershell
-BatchSize 262144
```

If CUDA OOM occurs, use:

```powershell
-BatchSize 65536
```

### 4. Eval only

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5c_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -EvalOnly `
  -Device "cuda:0" `
  -EvalBatchSize 65536 `
  -ChunkSize 200000
```

## Result files

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\model_train6_theta_anchor_observation_physics\D16_P5C_TRAINING_SUMMARY.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\model_train6_theta_anchor_observation_physics\D16_P5C_TRAIN_INPUT_AUDIT.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\eval_all55_vs_softlabels\D16_P5C_FINAL_SCORECARD.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\eval_all55_vs_softlabels\D16_P5C_METRICS_BY_PROFILE.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\eval_all55_vs_softlabels\D16_P5C_SPLIT_METRICS.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\eval_all55_vs_softlabels\D16_P5C_BATCH_METRICS.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST\eval_all55_vs_softlabels\D16_P5C_PROTOCOL_METRICS.csv
```

## Key comparison targets

D16-P5B 500epochs baseline:

```text
phis_c_mae ~= 0.0061 V
phie_mae ~= 0.0285 V
theta_a_mean_mae ~= 0.252
theta_c_mean_mae ~= 0.257
```

P5C should preserve phis_c/phie while reducing theta_mean MAE, especially on Batch-5/random_walk, Batch-6/GEO, and Batch-1 battery-8-like regimes.
