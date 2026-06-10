# D15-P3B — Batch-2 NN theta boundary / projection repair

## Purpose

D15-P3 produced valid Batch-2 P2Dlite-RG soft labels and passed radial-gradient audit, but the 3-cell NN smoke ended in `REVIEW` because raw NN theta predictions had a high outside-range fraction.

D15-P3B fixes only the NN inference boundary behavior:

- It does **not** regenerate Batch-2 P2Dlite-RG soft labels.
- It does **not** retrain the NN.
- It evaluates raw predictions and projected predictions on full Batch-2 3-cell profiles.
- It applies an explicit inference-time projection to theta channels only: `theta_a/theta_c -> [1e-4, 0.9999]`.
- It leaves `phis_c` and `phie` unchanged.

This is a repair/audit stage. If it passes, the correct statement is:

> Batch-2 3-cell NN smoke can be promoted from REVIEW to projection-repaired PASS, with the projection explicitly reported.

It is still not a Batch-2 15-cell benchmark and not held-out generalization.

## Default inputs

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_d15p3_batch2_3cell
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p3_batch2_3cell_rg_nn_smoke\model\best_with_state.pt
```

## Default outputs

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p3b_batch2_boundary_projection_repair
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p3b_results_for_review.zip
```

## GPU use

The evaluation uses the existing PyTorch model on `cuda` when available. The default batch size is large:

```text
batch_size = 262144
```

This is intentional to use GPU throughput for the full-profile predictions. If VRAM is insufficient, rerun with `-Quick` or manually lower batch size inside the script call.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p3b_run_all.ps1
```

If the output directory already exists and you want to deliberately overwrite only D15-P3B outputs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3b_run_all.ps1 -AllowOverwrite
```

Quick debug only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3b_run_all.ps1 -AllowOverwrite -Quick
```

Do not use `-Quick` for final review.

## Files to upload for review

Upload the automatically generated zip:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p3b_results_for_review.zip
```

It contains only JSON/CSV audit files, not models or large prediction arrays.

The most important files inside are:

```text
D15_P3B_FINAL_SCORECARD.json
D15_P3B_BOUNDARY_REPAIR_SUMMARY.json
D15_P3B_BOUNDARY_REPAIR_BY_PROFILE.csv
D15_P3B_TOP_RAW_THETA_OUTSIDE_POINTS.csv
D15_P3B_TOP_PROJECTED_THETA_ERROR_POINTS.csv
```

## What counts as success?

Expected PASS condition:

```text
projected_scorecard = PASS
projected pred_theta_outside_fraction <= 0.001
projection nonregression = PASS
```

If projection fixes outside fraction but strongly worsens theta/gradient MAE, the result remains `REVIEW`.
