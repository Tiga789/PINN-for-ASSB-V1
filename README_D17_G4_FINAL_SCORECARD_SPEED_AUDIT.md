# D17-G4 final scorecard + speed audit

G4 is a final report/export stage after G3 frozen-test report-only audit has passed. It does **not** train, choose checkpoints, edit splits, or use frozen-test feedback.

## Purpose

- Verify G0/G2.1/G3 prerequisite states.
- Freeze the G2.1/G3 candidate as a D17-G generator-surrogate candidate.
- Hash and inventory critical artifacts.
- Produce final scorecard and model card-style report.
- Run an optional synthetic forward-only speed audit on the frozen checkpoint.

## Required upstream state

- G0: `status=PASS`, generator semantics known.
- G2.1: `status=PASS`, `g3_ready=true`.
- G3: `status=PASS`, `promotion_status=PASS`, `g4_ready=true`.

## Run

```powershell
python scripts\d17_g4_final_scorecard_speed_audit.py `
  --config configs/d17_g4_final_scorecard_speed_audit.json `
  --g0_audit "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --g21_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair/D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json" `
  --g21_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair" `
  --g3_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit/D17_G3_FROZEN_TEST_REPORT_ONLY_AUDIT_SUMMARY.json" `
  --g3_scorecard "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit/D17_G3_SCORECARD.json" `
  --g3_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g4_final_scorecard_speed_audit" `
  --device auto `
  --speed_trials 200 `
  --speed_batch_size 8192
```

## Inspect

```powershell
python scripts\d17_g4_inspect_final.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g4_final_scorecard_speed_audit/D17_G4_FINAL_SCORECARD.json"
```

## Decision fields

- `status=PASS` and `final_candidate_ready=true`: candidate can be frozen/exported.
- `status=REVIEW`: do not freeze; inspect blockers.

## Outputs

- `D17_G4_FINAL_SCORECARD.json`
- `D17_G4_FINAL_REPORT.md`
- `D17_G4_SPEED_AUDIT.json`
- `D17_G4_ARTIFACT_MANIFEST.csv`
- `D17_G4_COPIED_ARTIFACTS.csv`
- `D17_G4_FROZEN_CANDIDATE_MANIFEST.json`
