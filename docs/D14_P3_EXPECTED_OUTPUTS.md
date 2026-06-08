# D14-P3 Expected Outputs

The runner should create:

```text
D14_P3_BATCH56_FEASIBILITY_REPORT.json
D14_P3_BATCH56_FEASIBILITY_REPORT.md
D14_P3_RAW_FILE_INDEX.csv
D14_P3_FILE_SCHEMA_AUDIT.csv
D14_P3_CYCLE_ELIGIBILITY_SUMMARY.csv
D14_P3_REPLAY_READINESS.csv
D14_P3_BATCH_SUMMARY.csv
D14_P3_SOH_POLICY.csv
D14_P3_OUTPUT_INDEX.json
D14_P3_RUN_SUMMARY.txt
README_D14_P3_PATCH.md
D14_P3_BATCH56_FEASIBILITY_AUDIT_console.log
D14_P3_VERIFY_console.log
```

The key files are:

- `D14_P3_BATCH56_FEASIBILITY_REPORT.json`: machine-readable audit result.
- `D14_P3_BATCH56_FEASIBILITY_REPORT.md`: human-readable recommendation.
- `D14_P3_RAW_FILE_INDEX.csv`: raw Batch-5/6 file discovery table.
- `D14_P3_FILE_SCHEMA_AUDIT.csv`: per-file schema/time/replay audit.
- `D14_P3_CYCLE_ELIGIBILITY_SUMMARY.csv`: per-cycle/subrecord eligibility summary.
- `D14_P3_REPLAY_READINESS.csv`: file-level replay-readiness table.
- `D14_P3_SOH_POLICY.csv`: explicit policy that SOH comes from original cycle/capacity data, not the voltage soft-label generator.
