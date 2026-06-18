# D17-G6.3 P4D/GEO Generator Formula Forensics

Purpose: fast, no-training, no-radial-solver diagnosis for the G6.1/G6.2 failure mode. It tests whether D15-P4D `theta_a/theta_c` inventory labels can be reconstructed from replay `I(t)` by the expected current-integral formula, or whether sign/theta0/capacity/cycle-reset assumptions are wrong.

This package is intentionally small. It does **not** train a model. It does **not** run `generate_rg_profile`. It does **not** do 55-cell full audit.

## Run

```powershell
python scripts\d17_g63_p4d_generator_forensics.py `
  --config configs/d17_g63_p4d_generator_formula_forensics.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g63_p4d_generator_forensics" `
  --profile_contains "Batch-6_GEO_battery-2" `
  --profile_contains "Batch-6_GEO_battery-5" `
  --max_time_points 4096
```

Inspect:

```powershell
python scripts\d17_g63_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g63_p4d_generator_forensics/D17_G63_P4D_FORMULA_FORENSICS_SUMMARY.json"
```

## Outputs

- `D17_G63_P4D_FORMULA_FORENSICS_SUMMARY.json`
- `D17_G63_PROFILE_FORMULA_SUMMARIES.json`
- `D17_G63_FORMULA_CANDIDATE_METRICS.csv`

## Decision

- `patch_ready=true`: a current-integral formula family matches; patch only that identified formula/feature path.
- `patch_ready=false`: do not train, do not run G6. Find the real D15-P4D generation semantics first.
