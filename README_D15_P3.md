# D15-P3 · XJTU Batch-2 P2Dlite-RG applicability validation

This package extends the D15 P2Dlite-RG chain to **XJTU Batch-2**, which was not part of the earlier 8-cell D15-P0/P1/P2 closed-set benchmark.

Batch-2 is treated as a **3C charge / 1C discharge fixed full-depth stress test**. The goal is applicability validation, not held-out generalization and not experimental proof of true radial internal states.

## Default outputs

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch2_replay_profiles_d15p3
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_d15p3_batch2_3cell
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3_batch2_3cell_radial_audit
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3_batch2_3cell_rg_nn_smoke
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3_batch2_applicability_scorecard
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p3_results_for_review.zip
```

## One-command run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p3_run_all.ps1
```

Rerun:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3_run_all.ps1 -AllowOverwrite
```

Fast debug:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3_run_all.ps1 -AllowOverwrite -Quick
```

Skip NN smoke and only check raw/replay/soft-label/radial audit:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p3_run_all.ps1 -AllowOverwrite -SkipNN
```

## What it does

1. Preflight checks raw Batch-2 `.mat` files and D15 dependencies.
2. Discovers Batch-2 files under `E:/XJTU battery dataset/Batch-2` or equivalent names.
3. Builds best-effort measured-current replay profiles.
4. Selects 3 representatives: preferred battery-1 / battery-8 / battery-15, or first/middle/last fallback.
5. Generates provisional P2Dlite-RG soft labels for the 3 selected cells.
6. Runs D15-P0 radial-gradient audit on generated labels.
7. Runs optional D15-P1-style closed-set NN smoke on the 3 generated Batch-2 profiles.
8. Packs a lightweight review zip.

## Important boundary

D15-P3 generated Batch-2 labels are **provisional model-consistent P2Dlite-RG soft labels**. They should not be described as experimentally measured internal states. If D15-P3 passes, the correct next step is expanding Batch-2 to 5-cell / 15-cell closed-set validation.
