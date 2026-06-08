# D14-P3 FAST Feasibility Patch v3

This patch replaces the slow Batch-5/6 feasibility audit with a shallow audit
that avoids full `.mat` loading.

## Run command

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p3_batch56_fast.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -DataRoot "E:\XJTU battery dataset" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -P0Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_v2" `
  -P1Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p1_evidence_boundary_v2" `
  -P2Dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p2_generalization_scorecard" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p3_batch56_feasibility_audit_fast" `
  -AllowWarn
```

Expected runtime should be short because it only uses raw file discovery and shallow metadata inspection.
