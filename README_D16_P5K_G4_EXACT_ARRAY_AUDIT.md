# D16-P5K-G4 exact array-level audit

This package performs the final required no-training audit before deciding whether to enter P5K-G training.

It materializes the G3 candidate:

```text
P5K-C hard baseline + strict metadata gate + rule_v2_strict_aggressive theta0 shift
```

and recomputes exact streaming MAE/RMSE/Bias/R² against ALL55 P2Dlite-RG soft-label arrays.

## Boundaries

- No training.
- No checkpoint loading.
- No model mutation.
- Evaluates theta/cs/gradient state arrays only.
- Does not recompute phis_c/phie in this no-network baseline audit.
- If G4 fails, stop the no-state-label G-audit loop and make a route decision.

## Run smoke

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg4_exact_array_audit.ps1 `
  -AllowOverwrite `
  -LimitProfiles 6 `
  -ChunkSize 100000
```

## Run full audit

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg4_exact_array_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 200000
```

If disk space is tight:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg4_exact_array_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 100000 `
  -MmapCacheRoot "F:\_p5kg4_exact_array_mmap_cache"
```

## Check outputs

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kg4_outputs.ps1
```

## File to paste back

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg4_exact_array_audit\D16_P5KG4_EXACT_ARRAY_AUDIT_REPORT.md
```
