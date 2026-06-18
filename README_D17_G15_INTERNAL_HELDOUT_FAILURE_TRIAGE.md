# D17-G1.5 internal-heldout failure triage

This package diagnoses the D17-G1.4 result before any G2 expansion.
It does **not** train a model and does **not** modify G1.4/G1.3/G1.2 code.

G1.4 repaired validation phie, but its train-internal heldout gate failed. G1.5 answers:

1. Which internal-heldout profile and target caused the gate failure?
2. Is the failure phie-only or a broader generator-choice/state issue?
3. Does the bad heldout profile appear outside the fit-train observed feature coverage?
4. What should the next repair run change, without using validation/frozen-test labels for training?

## Run

```powershell
python scripts\d17_g15_internal_heldout_failure_triage.py `
  --config configs/d17_g15_internal_heldout_failure_triage.json `
  --g14_out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g14_phie_validation_robustness" `
  --g13_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g13_validation_aware_surrogate/D17_G13_VALIDATION_AWARE_SURROGATE_SUMMARY.json" `
  --g12_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g12_phie_gauge_closedset_repair/D17_G12_PHIE_GAUGE_CLOSEDSET_REPAIR_SUMMARY.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g15_internal_heldout_failure_triage"
```

Inspect:

```powershell
python scripts\d17_g15_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g15_internal_heldout_failure_triage/D17_G15_INTERNAL_HELDOUT_TRIAGE_SUMMARY.json"
```

## Outputs

```text
D17_G15_INTERNAL_HELDOUT_TRIAGE_SUMMARY.json
D17_G15_DECISION_REPORT.md
D17_G15_INTERNAL_HELDOUT_FAILURE_RANKING.csv
D17_G15_VALIDATION_RANKING.csv
D17_G15_FIT_TRAIN_RANKING.csv
D17_G15_PROFILE_COVERAGE_AUDIT.csv
D17_G15_PHIE_PROFILE_AUDIT_COPY.csv
D17_G15_RECOMMENDED_G15R_CONFIG.json
```

## Interpretation

`status=PASS` means the diagnostic ran and produced rankings. It does **not** mean G2 is ready.

`recommendation=DO_NOT_ENTER_G2_RUN_G15R_STRATIFIED_HELDOUT_OR_COVERAGE_REPAIR` means the next step should be a targeted G1.5R repair, not G2.

