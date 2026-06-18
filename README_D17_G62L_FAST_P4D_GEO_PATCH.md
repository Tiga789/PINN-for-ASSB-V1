# D17-G6.2L fast P4D/GEO inventory patch package

This package replaces the slow G6.2 Step A with a lightweight, no-training, no-radial-solver diagnostic and model-patch smoke.

It is designed to answer only one question first: whether the G6.1 failure is caused by P4D/GEO current-integral inventory phase mismatch and whether a deterministic mean-inventory patch can repair the selected bad profiles.

## Files

```text
gv1/d17_g/g62_lite_patch.py
scripts/d17_g62l_formula_only_inventory_check.py
scripts/d17_g62l_model_patch_smoke.py
scripts/d17_g62l_inspect_summary.py
configs/d17_g62l_fast_p4d_geo_patch.json
docs/D17_G62L_FILE_LIST_ACTUAL.txt
README_D17_G62L_FAST_P4D_GEO_PATCH.md
```

## What this package does not do

- It does not train.
- It does not run radial FVM solver.
- It does not run 55-cell full audit.
- It does not save large prediction files by default.

## Step 1: formula-only check, expected runtime minutes or less

```powershell
python scripts\d17_g62l_formula_only_inventory_check.py `
  --config configs/d17_g62l_fast_p4d_geo_patch.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62l_formula_only_inventory_check" `
  --profile_contains "Batch-6_GEO_battery-2" `
  --profile_contains "Batch-6_GEO_battery-5" `
  --max_time_points 4096 `
  --time_window_s 0
```

Inspect:

```powershell
python scripts\d17_g62l_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62l_formula_only_inventory_check/D17_G62L_FORMULA_ONLY_SUMMARY.json"
```

Only continue if `promotion_status = PASS` and `g62_patch_formula_ready = true`.

## Step 2: model patch smoke, still no training

```powershell
python scripts\d17_g62l_model_patch_smoke.py `
  --config configs/d17_g62l_fast_p4d_geo_patch.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --candidate_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g61_full_cycle_coverage_repair" `
  --candidate_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g61_full_cycle_coverage_repair/D17_G61_CANDIDATE_FOR_G6_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62l_model_patch_smoke" `
  --profile_contains "Batch-6_GEO_battery-2" `
  --profile_contains "Batch-6_GEO_battery-5" `
  --max_time_points 4096 `
  --time_window_s 0 `
  --predict_batch_size 8192 `
  --device auto
```

Inspect:

```powershell
python scripts\d17_g62l_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g62l_model_patch_smoke/D17_G62L_MODEL_PATCH_SMOKE_SUMMARY.json"
```

Only if `promotion_status = PASS` should you proceed to a selected-cycle or cycle-wise streaming smoke. Do not run long training.
