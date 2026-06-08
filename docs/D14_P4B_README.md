# D14-P4B v3 Batch/Protocol Final Fix + Source Voltage Audit

## Why v3 is needed

P4B-v2 generated valid NPZ files, but the discovery table misclassified R2.5
and R3 profiles as Batch-1 because it accidentally read `xjtu_batch134` as
`Batch-1`.

P4B-v3 fixes this by:

1. refusing to match `batch134` as `Batch-1`;
2. forcing `protocol=R2.5` to `Batch-3`;
3. forcing `protocol=R3` to `Batch-4`;
4. requiring Batch-1/3/4 coverage in the selected set;
5. adding source `voltage_exp` bound audit;
6. fixing `D14_P4B_OUTPUT_INDEX.json` write order.

## Recommended command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p4b_multicell_softlabels.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -PriorFile "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1\configs\P2Dlite_prior_xjtu_lr18650la_v0.json" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_p4b_multicell_v3" `
  -MaxPointsPerProfile 100000 `
  -NR 17 `
  -AllowWarn
```

## Expected interpretation

If there is a single isolated source voltage spike in `voltage_exp`, the run may
finish as `WARN`, not `FAIL`. That warning is a source-data audit finding, not a
P2Dlite generator failure.
