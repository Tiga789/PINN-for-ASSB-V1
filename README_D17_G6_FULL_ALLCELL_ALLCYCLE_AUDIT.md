# D17-G6 full all-cell all-cycle audit

D17-G6 is a frozen, report-only full-time-grid audit of the D17-G generator surrogate. It evaluates all selected profiles on the full P2Dlite-RG soft-label time grid, not just the earlier 512-point / 40ks sampled-window audit.

G6 does not train, select checkpoints, change the split, or update the model. It reads soft labels only for report-only metrics.

## Smoke first

```powershell
python scripts\d17_g6_full_allcell_allcycle_audit.py `
  --config configs/d17_g6_full_allcell_allcycle_audit.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --candidate_g21_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair" `
  --candidate_g21_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair/D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g6_full_allcell_allcycle_audit_smoke" `
  --profile_limit 1 `
  --max_time_points 0 `
  --time_window_s 0 `
  --predict_batch_size 8192 `
  --device auto
```

## Full audit

```powershell
python scripts\d17_g6_full_allcell_allcycle_audit.py `
  --config configs/d17_g6_full_allcell_allcycle_audit.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --candidate_g21_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair" `
  --candidate_g21_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair/D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g6_full_allcell_allcycle_audit" `
  --splits all `
  --include_flagged_probe `
  --max_time_points 0 `
  --time_window_s 0 `
  --predict_batch_size 8192 `
  --device auto
```

## Inspect

```powershell
python scripts\d17_g6_inspect_scorecard.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g6_full_allcell_allcycle_audit/D17_G6_SCORECARD.json"
```

## Output files

```text
D17_G6_FULL_ALLCELL_ALLCYCLE_AUDIT_SUMMARY.json
D17_G6_SCORECARD.json
D17_G6_PROFILE_TARGET_METRICS.csv
D17_G6_CYCLE_TARGET_METRICS.csv
D17_G6_RAW_WEIGHTED_GROUP_TARGET_METRICS.csv
D17_G6_PROFILE_R2_AGGREGATES.csv
D17_G6_FEATURE_AND_CYCLE_AUDIT.csv
D17_G6_LOAD_FAILURES.csv
D17_G6_PREDICTION_MANIFEST.csv
```

By default, G6 is metrics-only. Saving all full-cycle predictions can require very large disk space. Use `--save_predictions compressed_npz` only for small selected subsets or if you intentionally want full prediction NPZs.
