# D14-P4 XJTU P2Dlite Soft-label Generator Smoke

## Key design requirement

All physical parameters are stored in one standalone, user-editable prior file:

```text
configs/P2Dlite_prior_xjtu_lr18650la_v0.json
```

The soft-label generator, auditor, and future model prediction/training code should read this same file. Do not duplicate P2Dlite physical parameters inside scripts.

## What this package does

- Reads XJTU replay-profile NPZ files.
- Reads `configs/P2Dlite_prior_xjtu_lr18650la_v0.json`.
- Generates model-consistent P2Dlite soft labels:
  - `cs_a`, `cs_c`
  - `theta_a`, `theta_c`
  - `phie`, `phis_c`
  - `phis_c_base`, `phis_c_soft`
- Uses `n_r=17` by default.
- Writes all generated data under `E:/XJTU battery dataset/_gv1_cache`.

## What this package does not do

- No training.
- No modification of GV1 mainline code.
- No SOH generation inside the voltage soft-label generator.
- No full-P2D internal-state ground-truth claim.
- No change to Batch-1_2C_battery-8 outlier policy.

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p4_xjtu_p2dlite_softlabel_smoke.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PriorFile "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\configs\P2Dlite_prior_xjtu_lr18650la_v0.json" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_smoke" `
  -MaxProfilesTotal 2 `
  -MaxPointsPerProfile 100000 `
  -NR 17 `
  -AllowWarn
```
