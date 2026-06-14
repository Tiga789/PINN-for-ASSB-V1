# D16-P5K-E diagnostic-first audit

This package does **not** train a model. It produces one primary Markdown report:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke_diagnostic_first_audit\D16_P5K_E_DIAGNOSTIC_REPORT.md
```

The report is designed to be pasted back directly.

## Purpose

P5K-C improved exact R² from negative to positive, while P5K-D regressed badly. P5K-E diagnoses why before any new training:

- P5K-C vs P5K-D metrics and exact R² comparison.
- Training input audit comparison.
- Checkpoint/config high-risk diff.
- Soft-label generator sidecar and prior/hash audit.
- Selected-profile Coulomb / q-integral diagnostic.

## Run smoke/normal diagnostic

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5ke_diagnostic.ps1 `
  -MaxDeepProfiles 6 `
  -SamplePointsPerProfile 8000
```

If you want a faster no-array smoke:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5ke_diagnostic.ps1 `
  -SkipDeepSoftlabelAudit
```

## Check output

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5ke_outputs.ps1
```

Then paste:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke_diagnostic_first_audit\D16_P5K_E_DIAGNOSTIC_REPORT.md
```

## Notes

The deep soft-label audit extracts selected `.npy` members from `.npz` into a short SHA1 cache directory:

```text
E:\XJTU battery dataset\_gv1_cache\_p5ke_diag_mmap_cache
```

This avoids Windows long-path and repeated full-array loads. It is diagnostic cache only and can be deleted after use.
