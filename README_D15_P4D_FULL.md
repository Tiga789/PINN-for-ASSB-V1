# D15-P4D full Batch-5/6 remaining14 P2Dlite-RG generation

This package completes P2Dlite-RG soft-label generation for the remaining 14 XJTU Batch-5/6 cells whose replay profiles were completed in D15-P4C.

It is a CPU/NumPy fanout pipeline. It does **not** make the RG generator use GPU. D15-P4D-smoke showed that 4 independent cell processes can use CPU effectively without memory saturation. CUDA is expected to remain idle during label generation unless the generator backend is rewritten.

## Scope

Target cells:

- Batch-5_battery-1/2/3/4/5/6/8
- Batch-6_battery-1/2/4/5/6/7/8

Default output:

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_d15p4d_batch56_remaining14
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_batch56_remaining14_radial_audit
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_batch56_remaining14_scorecard
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_results_for_review.zip
```

## Run

Recommended full run:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\d15_p4d_full_run_all.ps1 `
  -Workers 4 `
  -SaveMode uncompressed
```

Do not start with 8 workers on a 32GB machine. Batch-6 profiles are large and can trigger paging. If memory available stays comfortably above 6GB for the first 15 minutes, you may stop and restart with `-Workers 5` or `-Workers 6`; otherwise keep `-Workers 4`.

## Resume behavior

The runner resumes by default. If a profile already has both:

```text
profiles/<cell>/solution_softlabels.npz
profiles/<cell>/soft_label_summary.json
```

it is skipped and counted as complete. This avoids losing completed work.

To force regeneration of completed profiles, add:

```powershell
-RegenerateCompleted
```

Do not use that unless intentional.

## Monitoring

Check progress:

```powershell
$SoftDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p4d_batch56_remaining14"
Get-ChildItem "$SoftDir\profiles" -Directory -ErrorAction SilentlyContinue |
  Where-Object { Test-Path "$($_.FullName)\solution_softlabels.npz" } |
  Select-Object Name
```

Check resource monitor:

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4d_batch56_remaining14_generation_run\D15_P4D_FULL_RESOURCE_MONITOR.csv" -Tail 10
```

## Send for review

Upload:

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_results_for_review.zip
```

Do not upload full `solution_softlabels.npz` files.
