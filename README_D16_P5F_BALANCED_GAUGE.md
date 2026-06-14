# D16-P5F train6/eval49 balanced anode-cathode gauge package

## Purpose

P5F is a follow-up to P5D/P5E for the XJTU ALL55 P2Dlite-RG workflow.

P5D improved `theta_a` and kept excellent voltage fitting, while P5E improved `theta_c` but degraded `theta_a`.
P5F is designed to balance both electrodes:

- keep P5D's anode advantage through P5D warm-start teacher preservation;
- use weaker mean-level cathode/anode gauge guards, not hard absolute theta anchors;
- keep the same train6/eval49 split and the same observed-data-only training boundary.

## Training boundary

The training script only reads observed time-series keys:

```text
t_global_s
I_profile
voltage_exp
```

The training script does **not** read or use these internal soft-label keys as data loss:

```text
theta_a, theta_c, cs_a, cs_c, phie, phis_c, phis_c_soft
```

Soft labels are used only in the evaluation script to compute NN-vs-P2Dlite-RG scorecards.

## Default paths

```text
SoftlabelRoot = E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
RunDir        = E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST
WarmStart     = E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5d_train6_eval49_delta_gauge_FAST\model_train6_delta_gauge_observation_physics
```

## Smoke test

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5f_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5f_outputs.ps1
```

## First formal run

Recommended first formal run is 800-900 epochs.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5f_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -Epochs 900 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

If GPU memory is tight, lower batch size:

```powershell
-BatchSize 65536
```

## Evaluation only

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5f_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -EvalOnly `
  -Device "cuda:0" `
  -EvalBatchSize 65536 `
  -ChunkSize 200000
```

## Output files

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST\eval_all55_vs_softlabels\D16_P5F_FINAL_SCORECARD.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST\eval_all55_vs_softlabels\D16_P5F_SPLIT_METRICS.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST\eval_all55_vs_softlabels\D16_P5F_BATCH_METRICS.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST\eval_all55_vs_softlabels\D16_P5F_PROTOCOL_METRICS.csv
```

## Success criteria vs P5D/P5E

P5F should be judged against the previous candidates:

```text
P5D eval49 theta_a_mean_mae ≈ 0.23839
P5D eval49 theta_c_mean_mae ≈ 0.26660
P5E eval49 theta_a_mean_mae ≈ 0.25500
P5E eval49 theta_c_mean_mae ≈ 0.25592
```

P5F is successful only if it improves the balance, ideally:

```text
eval49 theta_a_mean_mae <= 0.24
eval49 theta_c_mean_mae <= 0.256
eval49 phis_c_mae remains around mV level
eval49 phie_mae remains around 0.028 V
```

If P5F only improves one electrode while degrading the other, keep P5D as the current balanced candidate and treat P5F as an ablation.
