# D17-G3 frozen-test report-only audit

G3 is the first frozen-test audit for the D17-G supervised generator-surrogate route.
It must be treated as a one-time report-only evaluation of the G2.1 candidate.

## What this package does

- Loads the frozen G2.1 candidate checkpoint.
- Evaluates all normal `frozen_test` profiles from the locked D17 split.
- Optionally evaluates `flagged_probe` separately.
- Computes `theta_a / theta_c / cs_a / cs_c / phie / phis_c` R²/MAE/RMSE against D15 P2Dlite-RG soft labels.
- Writes predictions and scorecards.

## What it does not do

- No training.
- No checkpoint selection.
- No frozen-test feedback to model, split, gate, or rule.
- No use of flagged probe in normal promotion metrics.

## Run

```powershell
python scripts\d17_g3_frozen_test_report_only_audit.py `
  --config configs/d17_g3_frozen_test_report_only_audit.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --candidate_g21_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair" `
  --candidate_g21_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair/D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit" `
  --max_time_points 512 `
  --time_window_s 40000 `
  --device auto
```

## Inspect

```powershell
python scripts\d17_g3_inspect_scorecard.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit/D17_G3_SCORECARD.json"
```

## Decision fields

- `status = PASS`: G3 workflow ran correctly.
- `promotion_status = PASS` and `g4_ready = true`: frozen-test state audit passed the configured gate.
- `promotion_status = REVIEW`: do not enter G4; inspect `worst_frozen_test_target_profile` and per-target CSV.
