# D17-G5 final release / reproducibility freeze

G5 is the final release/handoff stage after **D17-G4 final scorecard + speed audit** has passed.

It performs:

- prerequisite verification for G0, G2.1, G3, and G4;
- artifact hashing and small-artifact copying;
- model card / final report generation;
- reproducibility notes generation.

It does **not** train, choose checkpoints, change the split, or use frozen-test feedback to modify the candidate.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g5_final_release.py `
  --config configs/d17_g5_final_release.json `
  --project_root "." `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_audit "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_GENERATOR_EQUIVALENCE_AUDIT.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --g21_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair/D17_G21_P4D_BRANCH_REPAIR_SUMMARY.json" `
  --g21_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g21_p4d_branch_repair" `
  --g3_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit/D17_G3_FROZEN_TEST_REPORT_ONLY_AUDIT_SUMMARY.json" `
  --g3_scorecard "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit/D17_G3_SCORECARD.json" `
  --g3_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g3_frozen_test_report_only_audit" `
  --g4_scorecard "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g4_final_scorecard_speed_audit/D17_G4_FINAL_SCORECARD.json" `
  --g4_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g4_final_scorecard_speed_audit" `
  --no_state_label_audit "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/audit/no_state_label_audit.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g5_final_release"
```

## Inspect

```powershell
python scripts\d17_g5_inspect_release.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g5_final_release/D17_G5_FINAL_RELEASE_MANIFEST.json"
```

## PASS criteria

```text
status = PASS
final_release_ready = true
candidate_id = D17-G4_GENERATOR_SURROGATE_CANDIDATE__G5_RELEASE
```

If `status=REVIEW`, do not publish the candidate. Inspect `reasons` in `D17_G5_FINAL_RELEASE_MANIFEST.json`.

## Main outputs

```text
D17_G5_FINAL_RELEASE_MANIFEST.json
D17_G5_FINAL_RELEASE_REPORT.md
D17_G5_MODEL_CARD.md
D17_G5_REPRODUCIBILITY_NOTES.md
D17_G5_ARTIFACT_HASHES.csv
D17_G5_COPIED_SMALL_ARTIFACTS.csv
frozen_small_artifacts/
```

## Boundary statement

This candidate is a **P2Dlite-RG generator surrogate**. It is trained using train-cell soft labels and evaluated with validation/frozen-test soft labels only as report-only audits. It should not be described as direct experimental verification of internal states.
