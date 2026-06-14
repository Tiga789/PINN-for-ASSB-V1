# D16-P5E train6/eval49 cathode-side offset-safe package

## Purpose

D16-P5E continues from the current D16-P5D-800 candidate. P5D improved `theta_a` and preserved excellent `phis_c`, but `theta_c_mean_mae` and `theta_c_mean_bias` remained worse than desired. P5E therefore adds a *weak cathode-side offset-safe gauge* on top of P5D.

This package keeps the same scientific boundary:

```text
train = 6 cells
eval = 49 held-out cells
training-visible time-series = t_global_s / I_profile / voltage_exp only
NO theta/cs/phie/phis_c soft-label data loss during training
soft labels are used only by the evaluation script after training
```

## What changed from P5D

P5E keeps P5D's delta-gauge/correlation terms but adds weak fixed-prior guards:

```text
cathode_floor_guard        -> discourages theta_c collapse
anode_ceiling_guard        -> discourages theta_a over-lithiation
theta_cathode_center_prior -> weak bounded voltage-shape center prior
theta_pair_offset_compensation -> slack pair-sum guard, not hard theta_a+theta_c=1
```

These terms are deliberately weak. They use only the observed I/V-derived feature tensor and fixed prior ranges. They do not read soft-label internal states.

## Files

```text
configs/d16_p5e_cathode_gauge_config.json
scripts/gv1_d16_p5e_build_manifest.py
scripts/gv1_d16_p5e_train6_cathode_gauge_fast.py
scripts/gv1_d16_p5e_eval55_vs_softlabels.py
scripts/gv1_run_d16_p5e_train6_eval49_fast.ps1
scripts/gv1_check_d16_p5e_outputs.ps1
```

## Default directories

```text
Soft labels:
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL

Run dir:
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5e_train6_eval49_cathode_gauge_FAST

Default warm start:
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5d_train6_eval49_delta_gauge_FAST\model_train6_delta_gauge_observation_physics
```

## Recommended commands

### 1. 50 epoch smoke

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5e_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5e_outputs.ps1
```

### 2. Main P5E run

If smoke passes, run 700 epochs first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5e_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -Epochs 700 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

If GPU OOM occurs, reduce `-BatchSize` to `65536`. If evaluation memory is tight, reduce `-ChunkSize` to `100000`.

### 3. Eval only

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5e_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -EvalOnly `
  -Device "cuda:0" `
  -EvalBatchSize 65536 `
  -ChunkSize 200000
```

## Key outputs

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5e_train6_eval49_cathode_gauge_FAST\model_train6_cathode_gauge_observation_physics\D16_P5E_TRAIN_INPUT_AUDIT.json

E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5e_train6_eval49_cathode_gauge_FAST\eval_all55_vs_softlabels\D16_P5E_FINAL_SCORECARD.json

E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5e_train6_eval49_cathode_gauge_FAST\eval_all55_vs_softlabels\D16_P5E_SPLIT_METRICS.csv
```

## Decision criteria

Compare P5E against P5D-800:

```text
P5D eval49 phis_c_mae          ≈ 0.00102 V
P5D eval49 phie_mae            ≈ 0.02775 V
P5D eval49 theta_a_mean_mae    ≈ 0.23839
P5D eval49 theta_c_mean_mae    ≈ 0.26660
P5D eval49 theta_a_mean_bias   ≈ +0.20007
P5D eval49 theta_c_mean_bias   ≈ -0.25631
```

P5E is useful if it lowers `theta_c_mean_mae` and `theta_c_mean_bias` without destroying `theta_a_mean_mae` or `phis_c_mae`. If `theta_c` improves but `theta_a` degrades severely, keep P5D as the main candidate and treat P5E as a diagnostic ablation.
