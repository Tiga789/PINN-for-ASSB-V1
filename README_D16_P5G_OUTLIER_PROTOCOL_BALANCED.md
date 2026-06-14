# D16-P5G train6/eval49 outlier/protocol-balanced theta-gauge refinement

## Purpose

P5G continues the D16 train6/eval49 line after P5F-900. It keeps the same training boundary:

- train = 6 representative cells;
- eval = remaining 49 held-out cells plus all55 summary;
- training reads only `t_global_s`, `I_profile`, and `voltage_exp` from the soft-label containers;
- training does **not** load `theta_a`, `theta_c`, `cs_a`, `cs_c`, `phie`, `phis_c`, or `phis_c_soft` as data loss;
- P2Dlite-RG soft labels are used only by the evaluation script.

P5G specifically targets the P5F residual failure mode: random_walk, GEO, and flagged-like profiles still show a paired `+theta_a / -theta_c` gauge offset. P5G adds an observed-only theta-gap refinement and stress-balanced gauge loss based on I/V-derived features. It also adds exact global R² metrics to evaluation outputs.

## Files

```text
configs/d16_p5g_outlier_protocol_balanced_gauge_config.json
scripts/gv1_d16_p5g_build_manifest.py
scripts/gv1_d16_p5g_train6_outlier_protocol_balanced_gauge_fast.py
scripts/gv1_d16_p5g_eval55_vs_softlabels.py
scripts/gv1_run_d16_p5g_train6_eval49_fast.ps1
scripts/gv1_check_d16_p5g_outputs.ps1
```

## Default paths

```text
ProjectRoot = C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
SoftlabelRoot = E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL
RunDir = E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5g_train6_eval49_outlier_protocol_balanced_gauge_FAST
```

Default warm start is P5F:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST\model_train6_balanced_gauge_observation_physics
```

If P5F is missing, the runner falls back to P5D if present.

## Smoke run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5g_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5g_outputs.ps1
```

## Formal run

Suggested first formal run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5g_train6_eval49_fast.ps1 `
  -AllowOverwrite `
  -Epochs 1000 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

If GPU memory is tight, lower `BatchSize` to `65536`. If evaluation memory is tight, lower `ChunkSize` to `100000`.

## Result files

```text
<RunDir>\eval_all55_vs_softlabels\D16_P5G_FINAL_SCORECARD.json
<RunDir>\eval_all55_vs_softlabels\D16_P5G_METRICS_BY_PROFILE.csv
<RunDir>\eval_all55_vs_softlabels\D16_P5G_SPLIT_METRICS.csv
<RunDir>\eval_all55_vs_softlabels\D16_P5G_BATCH_METRICS.csv
<RunDir>\eval_all55_vs_softlabels\D16_P5G_PROTOCOL_METRICS.csv
```

## R² metrics

P5G evaluation outputs exact R² using chunked sufficient statistics:

```text
phis_c_r2
phie_r2
theta_a_r2
theta_c_r2
theta_a_mean_r2
theta_c_mean_r2
grad_a_surface_center_r2
grad_c_surface_center_r2
```

These are exact group-level `1 - SSE/SST` values, not `corr_mean^2`.

## Success target

P5G should be compared to P5F-900:

```text
P5F eval49:
phis_c_mae        ≈ 0.000902 V
phie_mae          ≈ 0.027688 V
theta_a_mean_mae  ≈ 0.238153
theta_c_mean_mae  ≈ 0.216421
```

P5G should improve theta without destroying voltage:

```text
eval49 phis_c_mae <= 0.002 V
eval49 phie_mae ~= 0.028 V
eval49 theta_a_mean_mae < 0.238
eval49 theta_c_mean_mae < 0.216
bias terms reduced relative to P5F
```

A stricter high-precision target remains:

```text
eval49 theta_a_mean_mae < 0.15
eval49 theta_c_mean_mae < 0.15
```

If P5G does not reduce both theta errors, keep P5F as the current best and use P5G only as an ablation.
