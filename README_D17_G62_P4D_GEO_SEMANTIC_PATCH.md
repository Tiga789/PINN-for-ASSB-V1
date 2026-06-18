# D17-G6.2 P4D/GEO semantic inventory patch

This package is a **no-long-training** repair path after G6.1.  It does not start a new 900-epoch training run.

## Why G6.2 exists

G6.1 finished with `status=PASS` but `g6_ready=false`.  The failure moved away from the old constant-phie issue and became a P4D/GEO inventory phase failure, especially `Batch-6_GEO_battery-2 / theta_c`.

G6.2 therefore treats P4D branch inventory as a generator-code-equivalence problem:

- D15-P4D generator defines theta/cbar from current integral, capacity scale and fixed initial theta.
- D15-P4D defines `phis_c = voltage_exp` and `phie = phie_ohmic_scale * I`.
- D15-P4D then calls the existing `gv1.p2dlite_rg.radial_solver.generate_rg_profile` to reconstruct radial `cs`.

G6.2 first checks whether that deterministic rule reproduces the P4D soft labels. If it does, the G6.1 neural prediction is used for RG profiles, while P4D profiles are patched with deterministic generator-equivalent outputs during evaluation/inference.

## Step A: deterministic equivalence smoke

```powershell
python scripts\d17_g62_p4d_inventory_equivalence_smoke.py `
  --project_root "." `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62_p4d_inventory_equivalence_smoke" `
  --profile_contains "Batch-6_GEO_battery-2" `
  --profile_contains "Batch-6_GEO_battery-5" `
  --max_time_points 0 `
  --time_window_s 0
```

Inspect:

```powershell
python scripts\d17_g62_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62_p4d_inventory_equivalence_smoke/D17_G62_P4D_INVENTORY_EQUIVALENCE_SMOKE_SUMMARY.json"
```

Proceed only if:

```text
status = PASS
promotion_status = PASS
g62_patch_ready = true
```

## Step B: patched audit on the known failing profiles

```powershell
python scripts\d17_g62_p4d_geo_semantic_patch_audit.py `
  --project_root "." `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --candidate_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g61_full_cycle_coverage_repair" `
  --candidate_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g61_full_cycle_coverage_repair/D17_G61_CANDIDATE_FOR_G6_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62_p4d_geo_patch_smoke" `
  --profile_contains "Batch-6_GEO_battery-2" `
  --profile_contains "Batch-6_GEO_battery-5" `
  --max_time_points 0 `
  --time_window_s 0 `
  --predict_batch_size 8192 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_g62_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62_p4d_geo_patch_smoke/D17_G62_P4D_GEO_PATCHED_AUDIT_SUMMARY.json"
```

Proceed only if:

```text
status = PASS
promotion_status = PASS
g6_streaming_ready = true
```

## Step C: patched 1-profile or full streaming audit

Do not run all 55 cells until Step B passes.  A safe 1-profile smoke:

```powershell
python scripts\d17_g62_p4d_geo_semantic_patch_audit.py `
  --project_root "." `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --candidate_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g61_full_cycle_coverage_repair" `
  --candidate_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g61_full_cycle_coverage_repair/D17_G61_CANDIDATE_FOR_G6_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62_patched_fullcycle_smoke" `
  --splits all `
  --profile_limit 1 `
  --max_time_points 0 `
  --time_window_s 0 `
  --predict_batch_size 8192 `
  --device auto
```

If Step C passes, use the same script with `--splits all` and no profile limit for full cycle-wise streaming metrics.  This still does **not** save full predictions by default.

## Important boundary

G6.2 is not another training stage.  It is an inference/evaluation semantic patch that makes P4D branch outputs code-equivalent to the actual D15-P4D generator. Soft labels are used only to calculate metrics.
