# D15-P4B SSD staging runner

This clean add-on contains only two scripts and no `gv1/` files. It avoids overwriting `gv1/__init__.py`.

Purpose: move the replay-profile I/O and generated soft-label output from the HDD `E:` cache to an SSD staging folder, then run the existing P4B generator with higher workers. This is meant for machines where CPU usage stays low because the HDD is the bottleneck.

Default staging root:

```text
C:\XJTU_gv1_cache_staging\d15_p4b_ready18
```

Run:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p4b_run_all_ssd_stage.ps1 -AllowOverwrite -Workers 8 -SaveMode uncompressed
```

If your C drive has insufficient space, use another SSD path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4b_run_all_ssd_stage.ps1 -AllowOverwrite -Workers 8 -SaveMode uncompressed -StagingRoot "D:\XJTU_gv1_cache_staging\d15_p4b_ready18"
```

It copies only the review zip back to:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4b_results_for_review.zip
```

To also mirror full generated outputs back to E, add:

```powershell
-MirrorFullOutputsToE
```
