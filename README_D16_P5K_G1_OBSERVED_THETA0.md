# D16-P5K-G1 observed-only theta0 estimator audit

This package is a **diagnostic audit only**. It does not train, does not load checkpoints, and does not modify models.

Purpose:

1. Use P5K-C hard baseline as the reference because P5K-G0 showed it is the strongest baseline-only candidate on normal eval.
2. Compare:
   - `P5K-C-baseline`
   - `G1-theta0_oracle` diagnostic upper bound
   - `G1-rule_v1` observable heuristic, no soft-label shift labels
   - `G1-ridge_core_fit` diagnostic estimator fitted from core_train oracle shifts
   - `G1-ridge_core_plus_hard_fit` diagnostic estimator fitted from core_train+hard_probe oracle shifts
3. Output a single Markdown report for review.

Important boundary:

- `theta0_oracle` uses soft-label initial internal states and is **not deployable**.
- `ridge_*` candidates fit oracle shifts from soft labels and are **diagnostic only** unless the experiment protocol explicitly allows such calibration.
- `rule_v1` is the only fully observed-only heuristic candidate, but it may be weak.

Default output:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg1_observed_theta0_audit\D16_P5KG1_OBSERVED_THETA0_AUDIT_REPORT.md
```

## Smoke

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg1_observed_theta0_audit.ps1 `
  -AllowOverwrite `
  -LimitProfiles 12 `
  -ChunkSize 200000
```

Check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_check_d16_p5kg1_outputs.ps1
```

## Full audit

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg1_observed_theta0_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 200000
```

If disk space is tight:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5kg1_observed_theta0_audit.ps1 `
  -AllowOverwrite `
  -ChunkSize 100000 `
  -MmapCacheRoot "F:\_p5kg1_observed_theta0_mmap_cache"
```

Paste the Markdown report to the assistant after the full audit.
