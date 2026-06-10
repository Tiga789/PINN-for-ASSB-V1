# D15-P4D-smoke: Batch-5/6 remaining14 soft-label generation resource smoke

This package does **not** generate the final 14-cell soft labels. It runs a controlled resource smoke before D15-P4D full generation.

It checks:

1. Whether CUDA is available and can be used by PyTorch operations.
2. Whether partial P2Dlite-RG generation for selected Batch-5/6 cells can run with explicit fanout.
3. CPU utilization, GPU monitor, memory availability, runtime, and output size.
4. Whether full D15-P4D should use chunked/resume CPU generation or requires a torch/CuPy backend redesign.

Default selected cells:

- Batch-5_battery-1
- Batch-5_battery-2
- Batch-6_battery-5
- Batch-6_battery-2

Default generation is partial prefix-only, capped by `max_time_points_per_cell = 300000`, and output is for smoke only.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\d15_p4d_run_smoke.ps1 `
  -AllowOverwrite `
  -Workers 4 `
  -MaxTimePoints 300000 `
  -SaveMode uncompressed
```

If CPU utilization is low, do not blindly increase workers. Upload the review zip and inspect the scorecard first.

## Outputs

Run directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4d_smoke_resource_test
```

Review zip:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4d_smoke_results_for_review.zip
```

Upload the review zip for inspection.

## Boundary

This smoke does not prove remaining14 final soft labels are complete. It also does not make NumPy P2Dlite-RG generation use GPU. CUDA is tested separately.
