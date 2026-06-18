# D17-P1 split manifest patch v2

用途：修复 D17-P1 split manifest 中 `Batch-1_battery-8` 没有被标记为 `flagged_probe`，以及 Batch-1/3/4 softlabel 命名与 replay 命名不一致导致 replay 匹配缺失的问题。

覆盖文件：

```text
scripts/d17_make_split_manifest.py
```

修复点：

```text
Batch-1_battery-k -> Batch-1_2C_battery-k
Batch-2_battery-k -> Batch-2_3C_battery-k
Batch-3_battery-k -> Batch-3_R2.5_battery-k
Batch-4_battery-k -> Batch-4_R3_battery-k
Batch-5_battery-k -> Batch-5_random_walk_battery-k
Batch-6_battery-k -> Batch-6_GEO_battery-k
```

并强制：

```text
Batch-1 battery-8 -> split=flagged_probe, is_flagged_probe=true
```

重新运行 D17-P1：

```powershell
python scripts\d17_make_split_manifest.py `
  --softlabel_root "E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL" `
  --replay_root "E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_replay_profiles" `
  --replay_root "E:/XJTU battery dataset/_gv1_cache/xjtu_batch2_replay_profiles_d15p3" `
  --replay_root "E:/XJTU battery dataset/_gv1_cache/xjtu_batch56_remaining14_replay_profiles_d15p4c" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split" `
  --seed 20260615 `
  --flag_cell "Batch-1_2C_battery-8" `
  --force
```

通过标准：

```text
d17_split_audit.json:
  pass=true
  battery8_flagged=true
  flagged_cells contains Batch-1_2C_battery-8
  missing_replay_count_for_normal_splits=0
```

然后重新运行 no-state-label 输入审计：

```powershell
python scripts\d17_audit_no_state_label_inputs.py `
  --config configs/d17_pinn_rebuild_smoke.yaml `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --out_json "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/audit/no_state_label_audit.json" `
  --project_root "."
```
