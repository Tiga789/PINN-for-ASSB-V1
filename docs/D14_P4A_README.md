# D14-P4A XJTU P2Dlite Soft-label Bounds/Metadata Patch

## Why P4A is needed

D14-P4 proved that the standalone-prior P2Dlite generator can produce
`cs_a/cs_c/phie/phis_c` model-consistent soft labels. The P4 smoke output also
revealed two issues that must be fixed before any wider generation:

1. Some Batch-1/3/4 replay profiles did not carry scalar `batch` and `protocol`
   fields. This breaks later grouping by batch/protocol/cell and can poison
   scorecards and split manifests.
2. The D12-S1K-style residual correction could produce `phis_c_soft` above the
   nominal 4.2 V upper limit. This is unacceptable for a soft label that the
   PINN will later learn as a target.

P4A fixes both without changing the GV1 training mainline.

## What this package actually changes

This package overwrites the real P4 modules:

```text
gv1/softlabels/p2dlite_prior.py
gv1/softlabels/xjtu_p2dlite_voltage.py
gv1/softlabels/xjtu_p2dlite_solver.py
gv1/softlabels/xjtu_softlabel_audit.py
scripts/gv1_generate_xjtu_p2dlite_softlabels.py
scripts/gv1_audit_xjtu_p2dlite_softlabels.py
```

It also adds:

```text
configs/d14_p4a_xjtu_softlabel_patch_config.json
scripts/gv1_d14_p4a_verify_outputs.py
scripts/run_gv1_d14_p4a_xjtu_p2dlite_softlabel_patch.ps1
```

## Single prior file rule

All physical parameters are still read from:

```text
configs/P2Dlite_prior_xjtu_lr18650la_v0.json
```

P4A adds this section to the prior file:

```json
"soft_voltage_bounds": {
  "enabled": true,
  "upper_margin_V": 0.02,
  "lower_margin_V": 0.02,
  "upper_warn_V": 4.25,
  "upper_fail_V": 4.35,
  "lower_warn_V": 2.45,
  "lower_fail_V": 2.35
}
```

This means the generated `phis_c_soft` is clipped to roughly:

```text
2.48 V <= phis_c_soft <= 4.22 V
```

while the unbounded value is still stored as:

```text
phis_c_soft_raw
```

and the applied correction is stored as:

```text
voltage_bound_correction = phis_c_soft - phis_c_soft_raw
```

## Metadata inference

If a replay profile lacks scalar `batch`/`protocol`, P4A infers them from the
source path and parent directory. For example:

```text
0003_battery-3_2C_battery-3 -> Batch-1 / 2C
R2.5 -> Batch-3 / R2.5
R3 -> Batch-4 / R3
Batch-5 -> Batch-5 / random_walk
Batch-6 -> Batch-6 / GEO
```

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p4a_xjtu_p2dlite_softlabel_patch.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PriorFile "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\configs\P2Dlite_prior_xjtu_lr18650la_v0.json" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_smoke_p4a" `
  -MaxProfilesTotal 2 `
  -MaxPointsPerProfile 100000 `
  -NR 17 `
  -AllowWarn
```

## Expected result

The output should have no FAIL checks. `phis_c_soft_max_V` should not exceed
the P4A soft bound. If `phis_c_soft_raw_max_V` is higher, this is a useful
diagnostic showing that P4A prevented high-voltage leakage.
