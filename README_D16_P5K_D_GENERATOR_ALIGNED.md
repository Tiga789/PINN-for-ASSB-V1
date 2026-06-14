# D16-P5K-D generator-aligned hard-cbar / OCP residual package

This package is the correction after P5H/P5K-C: it keeps the D16 train/eval protocol but integrates the XJTU D15 P2Dlite-RG soft-label generator structure and prior-consistency rules into the training-side hard baseline.

## What changed

- No direct `theta/cs/phie/phis_c` soft-label data loss during training.
- Training input boundary remains `t_global_s / I_profile / voltage_exp`.
- The model reads generator sidecar metadata (`soft_label_summary.json`, `soft_label_audit.json`) only for audit/prior consistency, not as targets.
- The hard state baseline is now based on P2Dlite-RG prior windows, measured-current Coulomb replay, absolute voltage/OCP phase, and bounded residuals.
- Evaluation uses exact streaming R² and robust v3 mmap extraction.
- Default train split is `D_train10_prior_balanced`, which avoids putting the most extreme hard-probe cells directly into the train split; those remain marked as held-out hard probes for diagnosis.

## Recommended commands

Smoke training:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kd_train_fast.ps1 `
  -AllowOverwrite `
  -TrainOnly `
  -TrainSet "D_train10_prior_balanced" `
  -Epochs 50 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -ValEvery 5
```

Formal training + ALL55 evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kd_train_fast.ps1 `
  -AllowOverwrite `
  -TrainSet "D_train10_prior_balanced" `
  -Epochs 1400 `
  -Device "cuda:0" `
  -BatchSize 131072 `
  -EvalBatchSize 65536 `
  -ChunkSize 200000 `
  -ValEvery 10
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kd_outputs.ps1 `
  -TrainSet "D_train10_prior_balanced"
```

## Promotion target

P5K-D should be compared against P5K-C eval45 and P5F/P5H exact-R² audit. Minimum stage target:

- eval theta_a_mean_mae < 0.15
- eval theta_c_mean_mae < 0.15
- eval theta_a_mean_r2 > 0.50 as first correction step, ultimately > 0.85
- eval theta_c_mean_r2 > 0.50 as first correction step, ultimately > 0.85
- phis_c_r2 > 0.99

If exact R² fails to improve over P5K-C, stop training and inspect OCP/window/capacity prior alignment before any further loss tuning.
