# D16-P5B train6 / eval49 observation-physics training package

This package implements the next training stage after D16-P5A.

## Goal

Train a new 55-cell XJTU model using only **6 representative train cells** and evaluate on all 55 cells:

```text
train:  one cell per Batch = 6 cells
eval:   remaining 49 cells + train6 reported separately
all:    all 55 cells scorecard
```

The selected train cells are:

```text
Batch-1 battery-3
Batch-2 battery-8
Batch-3 battery-6
Batch-4 battery-7
Batch-5 battery-7
Batch-6 battery-3
```

## Critical boundary

Training does **not** use internal soft-label data loss.

The training script reads only these time-series keys from each `solution_softlabels.npz` container:

```text
t_global_s / time
I_profile / current
voltage_exp / voltage
```

The training script does **not** load:

```text
theta_a, theta_c, cs_a, cs_c, phie, phis_c, phis_c_soft
```

Those internal soft labels are used only by the evaluation script, after training, to compute NN-vs-P2Dlite-RG metrics.

This matches the intended ASSB-like strategy: no direct internal-state data loss, fixed priors / hard physical structure, and observable I/V consistency.

## Files

```text
configs/d16_p5b_train6_eval49_config.json
scripts/gv1_d16_p5b_build_manifest.py
scripts/gv1_d16_p5b_train6_observation_physics.py
scripts/gv1_d16_p5b_eval55_vs_softlabels.py
scripts/gv1_run_d16_p5b_train6_eval49.ps1
scripts/gv1_check_d16_p5b_outputs.ps1
README_D16_P5B_TRAIN6_EVAL49.md
PACKAGE_MANIFEST.json
```

## How to run

Copy / overwrite this package into:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

### 1. Build manifest only

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5b_train6_eval49.ps1 `
  -AllowOverwrite `
  -BuildManifestOnly
```

Expected manifest:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5b_train6_eval49_observation_physics\D16_P5B_TRAIN6_EVAL49_MANIFEST.csv
```

It must contain 55 rows, with train=6 and eval=49.

### 2. Short smoke training

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5b_train6_eval49.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -Epochs 20 `
  -Device "cuda:0" `
  -BatchSize 32768
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5b_outputs.ps1
```

Checkpoint should exist:

```text
...\model_train6_observation_physics\model\best_with_state.pt
```

### 3. Full train + full eval

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5b_train6_eval49.ps1 `
  -AllowOverwrite `
  -Device "cuda:0" `
  -BatchSize 65536 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000
```

If GPU memory is tight:

```powershell
-BatchSize 32768 -EvalBatchSize 32768 -ChunkSize 100000
```

## Results

Final scorecard:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5b_train6_eval49_observation_physics\eval_all55_vs_softlabels\D16_P5B_FINAL_SCORECARD.json
```

Profile-level metrics:

```text
...\eval_all55_vs_softlabels\D16_P5B_METRICS_BY_PROFILE.csv
```

Split metrics:

```text
...\eval_all55_vs_softlabels\D16_P5B_SPLIT_METRICS.csv
```

Batch metrics:

```text
...\eval_all55_vs_softlabels\D16_P5B_BATCH_METRICS.csv
```

Protocol metrics:

```text
...\eval_all55_vs_softlabels\D16_P5B_PROTOCOL_METRICS.csv
```

Training input audit:

```text
...\model_train6_observation_physics\D16_P5B_TRAIN_INPUT_AUDIT.json
```

This audit explicitly lists the keys used during training and confirms that internal soft-label arrays are not read by the training script.

## Interpretation

Important: this is an observation-physics train6/eval49 baseline.

A successful run means:

```text
operational_status = PASS
profile_count_evaluated = 55
failure_count = 0
```

Scientific success should be judged by heldout49 and all55 metrics, not by train6 metrics alone.
