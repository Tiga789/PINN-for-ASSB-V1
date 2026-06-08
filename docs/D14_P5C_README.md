# D14-P5C 8-cell Closed-set Precision Audit

## Position

D14-P5B-v2 already completed the actual closed-set training. P5C does **not**
train again. It freezes and audits the P5B-v2 result so the project has a clean
archive-level precision report.

## What P5C checks

P5C reads the existing P5B-v2 output directory and checks:

```text
training_summary.json
loss_history.csv
D14_P5B_EVAL_REPORT.json
metrics_by_profile.csv
metrics_global.json
D14_P5B_VERIFY_REPORT.json
```

It then generates:

```text
D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.json
D14_P5C_CLOSEDSET_PRECISION_AUDIT_REPORT.md
D14_P5C_PROFILE_METRICS_COMPACT.csv
D14_P5C_CHECKS.csv
D14_P5C_LOSS_SUMMARY.json
D14_P5C_BATCH_METRICS.json
D14_P5C_PROTOCOL_METRICS.json
D14_P5C_VERIFY_REPORT.json
```

## Precision thresholds

The default closed-set PASS targets are:

```text
mean phis_c MAE <= 0.010 V
max  phis_c MAE <= 0.015 V
mean phie MAE   <= 0.010
max  phie MAE   <= 0.015
mean theta_mean MAE <= 0.010
max  theta_mean MAE <= 0.015
min phis_c corr >= 0.999
min theta corr  >= 0.999
```

These are closed-set calibration thresholds, not held-out generalization
thresholds.

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p5c_closedset_precision_audit.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -P5BOutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5b_8cell_closedset_precision_v2" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p5c_8cell_closedset_precision_audit" `
  -AllowWarn
```

## Boundary

- P5C does not train.
- P5C does not generate soft labels.
- P5C does not generate SOH.
- P5C does not modify GV1 mainline code.
- P5C is a closed-set calibration audit, not a new generalization result.
