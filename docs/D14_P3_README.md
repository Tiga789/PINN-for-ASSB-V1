# D14-P3 Batch-5/6 Feasibility Audit

## Position

D14-P3 is not a training step. It checks whether XJTU Batch-5 and Batch-6 can
enter the GV1 measured-current replay pipeline after D14-P0/P1/P2 have frozen
the current mainline.

## What this package does

It generates a feasibility report for:

- raw file discovery for Batch-5 and Batch-6;
- schema availability: time/current/voltage/temperature/capacity-like fields;
- time-axis recoverability;
- cycle/step recoverability;
- complete-discharge / capacity-check SOH eligibility;
- partial-discharge replay-only policy;
- replay-readiness;
- no-regression against D14-P0/P1/P2 outputs.

## What this package does not do

It does not train a model.  
It does not modify `gv1/model.py`, `gv1/output_transform.py`, `gv1/losses.py`,
`gv1/trainer.py`, or `scripts/gv1_train_conditioned_pinn.py`.  
It does not generate SOH in the voltage soft-label generator.  
It does not generate P2D internal-state labels.  
It does not unflag Batch-1_2C_battery-8.

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p3_batch56_feasibility.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -DataRoot "E:\XJTU battery dataset" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -P0Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2" `
  -P1Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p1_evidence_boundary_v2" `
  -P2Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p2_generalization_scorecard" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3_batch56_feasibility_audit" `
  -AllowWarn
```

## Expected interpretation

- `PASS`: Batch-5/6 can proceed to a controlled replay-profile build package.
- `WARN`: Data are mostly usable, but a policy or raw-data caveat needs review.
- `FAIL`: Do not proceed to profile generation; inspect the listed failures first.
