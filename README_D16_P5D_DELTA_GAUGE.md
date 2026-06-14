# D16-P5D train6/eval49 delta-gauge observation-physics package

## Purpose

D16-P5D follows P5B/P5C boundaries:

- train = 6 representative cells, eval = 49 held-out cells plus ALL55 reporting;
- training reads only `t_global_s`, `I_profile`, and `voltage_exp` from each `solution_softlabels.npz`;
- no `theta/cs/phie/phis_c` soft-label arrays are used as training data loss;
- P2Dlite-RG soft labels are used only in the evaluation script for NN-vs-soft-label scorecards.

## Why P5D differs from P5C-v1

P5C-v1 improved `phis_c` but pushed the internal theta gauge in the wrong direction. P5D removes the hard absolute OCP/theta target and uses only weak relative/integral constraints:

- q-integral correlation: `theta_a` should increase with cumulative charge, `theta_c` should decrease;
- weak voltage-trend correlation, not absolute theta supervision;
- centered two-electrode mass coupling, without forcing a fixed theta sum;
- weak midrange guard;
- current-driven radial-gradient direction and rest relaxation;
- phie gauge regularization;
- voltage observation anchor for `phis_c` from observed `V(t)`.

## Files

```text
configs/d16_p5d_delta_gauge_config.json
scripts/gv1_d16_p5d_build_manifest.py
scripts/gv1_d16_p5d_train6_delta_gauge_fast.py
scripts/gv1_d16_p5d_eval55_vs_softlabels.py
scripts/gv1_run_d16_p5d_train6_eval49_fast.ps1
scripts/gv1_check_d16_p5d_outputs.ps1
```

## Default paths

```text
SoftlabelRoot = E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
RunDir        = E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5d_train6_eval49_delta_gauge_FAST
WarmStart     = E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5b_train6_eval49_observation_physics_FAST\model_train6_observation_physics
```

The warm start is intentional: P5B-500 was the better theta baseline. P5D starts from it if available and then trains the weaker delta-gauge constraints. To disable warm start, pass `-NoWarmStart`.

## Smoke run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5d_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5d_outputs.ps1
```

## Recommended full run

Run 800 epochs first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5d_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -Epochs 800 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

If GPU memory is stable and utilization is still low, you may try `-BatchSize 262144`. If OOM occurs, use `-BatchSize 65536`.

## Result files

```text
<RunDir>\model_train6_delta_gauge_observation_physics\D16_P5D_TRAINING_SUMMARY.json
<RunDir>\model_train6_delta_gauge_observation_physics\D16_P5D_TRAIN_INPUT_AUDIT.json
<RunDir>\model_train6_delta_gauge_observation_physics\model\best_with_state.pt
<RunDir>\eval_all55_vs_softlabels\D16_P5D_FINAL_SCORECARD.json
<RunDir>\eval_all55_vs_softlabels\D16_P5D_METRICS_BY_PROFILE.csv
<RunDir>\eval_all55_vs_softlabels\D16_P5D_SPLIT_METRICS.csv
<RunDir>\eval_all55_vs_softlabels\D16_P5D_BATCH_METRICS.csv
<RunDir>\eval_all55_vs_softlabels\D16_P5D_PROTOCOL_METRICS.csv
```

## Key comparison targets

Compare D16-P5D eval49 against:

- P5B-500: `phis_c_mae≈0.00593 V`, `phie_mae≈0.02807 V`, `theta_a_mean_mae≈0.25249`, `theta_c_mean_mae≈0.25814`.
- P5C-1000: `phis_c_mae≈0.00136 V`, `phie_mae≈0.02777 V`, `theta_a_mean_mae≈0.27863`, `theta_c_mean_mae≈0.31029`.

P5D is useful only if it keeps voltage errors competitive while reducing theta mean MAE/bias relative to P5B/P5C.
