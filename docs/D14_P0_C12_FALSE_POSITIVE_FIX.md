# D14-P0 C12 false-positive fix

This patch replaces only:

`scripts/gv1_d14_p0_freeze_mainline_audit.py`

Reason:
The first corrected D14-P0 audit package falsely failed C12 because it matched generic
`battery-8` text in output folder names such as `exclude_battery8` and in normal
`Batch-3_battery-8` / `Batch-4_battery-8` rows. The D14 battery-8 policy is only for
the known flagged profile `Batch-1 / 2C / battery-8`.

After overwriting the script, rerun:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p0_freeze_audit.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2" `
  -StrictCache `
  -StrictASSB
```

Expected result:
- C12 should become PASS unless actual `Batch-1_2C_battery-8` rows are present without a flagged/excluded status.
- The previous C06 WARN can remain acceptable if it only reports disabled `enable_voltage_hard_clamp=False` markers.
