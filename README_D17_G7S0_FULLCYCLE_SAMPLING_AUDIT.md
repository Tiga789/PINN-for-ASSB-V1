# D17-G7-S0 full-cycle sampling / data coverage audit

## Purpose

S0 is a no-training stage. It checks full-cycle time/cycle coverage and creates a deterministic cycle-stratified sampling plan for the next small full-cycle training smoke.

It is designed to answer only this question:

> Can we build a full-cycle sample plan that covers early/middle/late cycles and all protocol/branch groups without loading huge state arrays?

It does **not** train a model, choose a checkpoint, run G6, or generate prediction files.

## Files

```text
scripts/d17_g7s0_fullcycle_sampling_audit.py
scripts/d17_g7s0_inspect_summary.py
configs/d17_g7s0_fullcycle_sampling_audit.json
README_D17_G7S0_FULLCYCLE_SAMPLING_AUDIT.md
docs/D17_G7S0_FILE_LIST_ACTUAL.txt
gv1/d17_g/__init__.py
```

## Smoke command: first 3 profiles

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g7s0_fullcycle_sampling_audit.py `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s0_fullcycle_sampling_audit_smoke" `
  --splits train,validation `
  --profile_limit 3 `
  --max_time_points_per_profile 4096 `
  --write_sample_points `
  --seed 20260615
```

Inspect:

```powershell
python scripts\d17_g7s0_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s0_fullcycle_sampling_audit_smoke/D17_G7S0_FULLCYCLE_SAMPLING_AUDIT_SUMMARY.json"
```

## Formal S0 command: train + validation

```powershell
python scripts\d17_g7s0_fullcycle_sampling_audit.py `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s0_fullcycle_sampling_audit" `
  --splits train,validation `
  --max_time_points_per_profile 4096 `
  --write_sample_points `
  --seed 20260615
```

Inspect:

```powershell
python scripts\d17_g7s0_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s0_fullcycle_sampling_audit/D17_G7S0_FULLCYCLE_SAMPLING_AUDIT_SUMMARY.json"
```

## Pass criteria

```text
status = PASS
s1_ready = true
recommendation = ENTER_G7_S1_SMALL_FULLCYCLE_SMOKE
```

If `s1_ready=false`, do not train S1 yet. Inspect:

```text
D17_G7S0_PROFILE_COVERAGE.csv
D17_G7S0_CYCLE_COVERAGE.csv
D17_G7S0_LOAD_FAILURES.csv
```

## Outputs

```text
D17_G7S0_FULLCYCLE_SAMPLING_AUDIT_SUMMARY.json
D17_G7S0_PROFILE_COVERAGE.csv
D17_G7S0_CYCLE_COVERAGE.csv
D17_G7S0_SAMPLE_POINTS.csv
D17_G7S0_SPLIT_PROTOCOL_BRANCH_COUNTS.csv
D17_G7S0_LOAD_FAILURES.csv
D17_G7S0_RECOMMENDED_S1_CONFIG.json
```

`D17_G7S0_SAMPLE_POINTS.csv` is written only when `--write_sample_points` is used. This file is intended for S1.
