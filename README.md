# PINN-for-ASSB-V1 · ASSB-D6 current status

This repository adapts the PINNSTRIPES-style battery PINN workflow to the user's NMC811||Li-In all-solid-state battery (ASSB). The current D6 milestone focuses on protecting the `ModelFin_107A` four-state benchmark while adding a strict 30%/70% SOH prediction branch.

## Current milestone

**Current engineering candidate:** `ModelFin_111_seed42locked_repro_c00`

This package is a five-target engineering wrapper:

```text
cs_a, cs_c, phie, phis_c  <- frozen ModelFin_107A state benchmark
SOH                       <- ASSB-111 seed42-locked saturating_v2 SOH head
```

It is **not yet a fully end-to-end single neural network** sharing one 107A core for all five targets. It is also not a multi-seed robust SOH predictor. The current SOH branch is a seed42-locked strict30 engineering baseline.

## Important methodological boundary

D5's `ModelFin_110_stageB` SOH result used all cycle labels through its default `fit_splits=train,val,test` workflow. It is therefore treated as a full-cycle calibration upper bound, not as a held-out forecasting result.

D6's ASSB-111 protocol uses:

```text
train: cycle 5-139      135 complete cycles
val:   cycle 140-159     20 complete cycles
test:  cycle 160-521    362 complete cycles
partial/incomplete: cycle 522
```

SOH training and checkpoint selection are allowed to use only train/val. Test is used only after the model is fixed.

## Current key results

### Five-target scorecard for `EvalFin_111_seed42locked_repro_c00`

| variable | source | n | MAE | RMSE | R2 | corr |
|---|---|---:|---:|---:|---:|---:|
| cs_a | frozen ModelFin_107A state eval NPZ | 1,280,000 | 0.02349 | 0.03333 | 0.99526 | 0.99763 |
| cs_c | frozen ModelFin_107A state eval NPZ | 1,280,000 | 0.39229 | 0.54137 | 0.98920 | 0.99623 |
| phie | frozen ModelFin_107A state eval NPZ | 373,235 | 0.00617 | 0.00725 | 0.99856 | 0.99944 |
| phis_c | frozen ModelFin_107A state eval NPZ | 373,235 | 0.00946 | 0.01117 | 0.99847 | 0.99967 |
| SOH | seed42-locked saturating_v2, test cycles 160-521 | 362 | 0.00414 | 0.00509 | 0.97934 | 0.99173 |

### Seed42 recovery training audit

```text
model_dir: ModelFin_111_seed42locked_repro_c00
best_epoch: 720
best_selection_status: visible_guarded
train_mae: 0.001874
train_r2: 0.995733
val_mae: 0.000984
val_r2: 0.906739
leakage_ok: true
protocol_audit_ok: true
selected_checkpoint_audit_ok: true
no_test_metrics_in_training_history: true
test_metrics_used_for_selection: false
```

## What succeeded in D6

1. StageB was correctly reclassified as full-cycle calibration instead of strict SOH forecasting.
2. The strict30 SOH split and leakage audit workflow were built.
3. Frozen 107A state protection succeeded: `cs_a`, `cs_c`, `phie`, and `phis_c` remain at the 107A benchmark level.
4. The first accumulative SOH head failure was diagnosed: it over-decayed and hit the 0.4 lower bound on late test cycles.
5. `saturating_v2` fixed the lower-bound collapse and recovered an SOH test R2 of about 0.97934 for seed42.
6. Bad checkpoint fallback was fixed: unguarded checkpoints are no longer allowed to be packaged silently.

## What failed or remains unresolved

1. `saturating_v2` is not multi-seed stable.
   - seed42: SOH test R2 about 0.97934
   - seed7: SOH test R2 about 0.88711
   - seed2026: SOH test R2 about 0.85303
2. `saturating_v3`, `v3_floorfix`, and `v2stable` over-stabilized the SOH curve and produced worse held-out results.
3. The current SOH head still behaves partly like a curve extrapolator. Train/val metrics are not sufficient to determine the late-cycle platform shape.
4. The current package is a five-target wrapper, not a true end-to-end single model.
5. SOH R2 is close to but still below the requested 0.98 threshold.

## Current important directories

```text
ModelFin_107A
EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only
Data\\assb_capacity_soh_targets\\capacity_soh_targets.csv
Data\\assb111_seed42locked_repro_c00
ModelFin_111_seed42locked_repro_c00
EvalFin_111_seed42locked_repro_c00
input_assb111_strict30_saturating_v2_seed42locked
```

Failed/diagnostic directories such as `EvalFin_111_smoke`, `EvalFin_111_saturating_v3_smoke`, `EvalFin_111_saturating_v3_floorfix_smoke`, `EvalFin_111_saturating_v2stable_smoke`, and the first `EvalFin_111_seed42_locked_smoke` can be archived or deleted after backup.

## Reproduce the current D6 recovery baseline

```powershell
cd "C:/Users/Tiga_QJW/Desktop/ASSB_Scheme_V1/PINN-for-ASSB-V1"
$py = "D:\Anaconda\envs\torchgpu\python.exe"

Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
Remove-Item Env:ASSB_SOFT_LABEL_DIR -ErrorAction SilentlyContinue
Remove-Item Env:ASSB_OCP_DIR -ErrorAction SilentlyContinue

.\scripts\run_ModelFin111_strict30.ps1 `
  -PythonExe $py `
  -ProjectRoot "." `
  -InputFile "input_assb111_strict30_saturating_v2_seed42locked" `
  -WorkDir "Data\\assb111_seed42locked_repro_c00" `
  -ModelDir "ModelFin_111_seed42locked_repro_c00" `
  -EvalDir "EvalFin_111_seed42locked_repro_c00" `
  -Epochs 5000 `
  -Device "cuda" `
  -Seed 42 `
  -SOHModelVariant "saturating_v2" `
  -SOHFloorPrior 0.72 `
  -SOHNumericMin 0.60 `
  -LR 2e-3 `
  -WeightDecay 1e-5 `
  -Patience 800 `
  -MinTrainR2ForBest 0.990 `
  -MaxTrainMAEForBest 0.0030 `
  -MaxValMAEForBest 0.00150 `
  -CandidateTag "c00_repro_lr2e3_e5000" `
  -ProtocolTag "seed42_locked_recovery_repro_trainval_only" `
  -SelectionMode "visible_train_val_only" `
  -RunOverdecayDiagnostics
```

Check results:

```powershell
Import-Csv .\EvalFin_111_seed42locked_repro_c00\five_state_scorecard.csv |
  Format-Table variable,source,n,MAE,RMSE,NMAE,NRMSE,R2,corr -AutoSize

Get-Content .\ModelFin_111_seed42locked_repro_c00\train_summary.json
Get-Content .\EvalFin_111_seed42locked_repro_c00\soh_overdecay_diagnostic.json
```

## Recommended D7 route

1. Freeze and back up `ModelFin_111_seed42locked_repro_c00` and `EvalFin_111_seed42locked_repro_c00`.
2. Do not continue the v3/floorfix/v2stable route without a new reason.
3. Build a feature audit comparing:
   - cycle/throughput only,
   - 107A physical summaries only,
   - voltage-derived health features only,
   - combined feature groups.
4. Add voltage-health features such as fixed-capacity voltage, dV/dQ approximations, platform voltage, rest relaxation amplitude/slope, and polarization growth.
5. Keep the no-test-selection rule. Test cycles 160-521 must not be used for hyperparameter or checkpoint choice.
6. Only after SOH prediction is stable should D7 revisit true single-model integration.

## Reporting language

Use this wording:

```text
ASSB-111 seed42-locked strict30 engineering benchmark: four states are protected by frozen ModelFin_107A; SOH is predicted by a saturating_v2 head selected using train/val only. The held-out test SOH R2 is about 0.97934. This is close to but below the 0.98 target and is not yet a multi-seed robust or end-to-end single-model result.
```

Do not write:

```text
ModelFin_111 is a final unified end-to-end five-state model with SOH R2 >= 0.98.
```
