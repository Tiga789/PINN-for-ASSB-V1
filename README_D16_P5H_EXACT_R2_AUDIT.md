# D16-P5H Exact-R2 Audit Package

Purpose: no training. Recompute exact global and split-level R² for P5B/P5D/P5E/P5F/P5G with one unified evaluator and one final Markdown report file.

This audit is designed to correct the earlier mistake of relying on `corr_mean` or MAE alone for theta. Exact R² is computed as:

```text
R² = 1 - SSE / SST
```

The script streams every profile chunk, recomputes predictions from each checkpoint, and accumulates sufficient statistics for:

```text
phis_c, phie, theta_a, theta_c, theta_a_mean, theta_c_mean,
grad_a_surface_center, grad_c_surface_center
```

It preserves the project boundary:

```text
No training.
No soft-label data loss.
Soft labels are read only for evaluation/audit.
No modification of P5B/P5D/P5E/P5F/P5G training scripts.
```

## Default output

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5h_exact_r2_audit\D16_P5H_EXACT_R2_AUDIT_REPORT.md
```

This is the single file to send back for inspection.

## Smoke

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5h_exact_r2_audit.ps1 `
  -Models "P5F,P5G" `
  -LimitProfiles 2 `
  -Device "cuda:0" `
  -BatchSize 65536 `
  -ChunkSize 200000
```

## Full audit

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5h_exact_r2_audit.ps1 `
  -Models "P5B,P5D,P5E,P5F,P5G" `
  -Device "cuda:0" `
  -BatchSize 65536 `
  -ChunkSize 200000
```

If GPU memory is tight:

```powershell
-BatchSize 32768 -ChunkSize 100000
```

## Check the single output file

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5h_outputs.ps1
```

Then upload or paste:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5h_exact_r2_audit\D16_P5H_EXACT_R2_AUDIT_REPORT.md
```
