# D14-P3B Expected Outputs

```text
D14_P3B_REPLAY_SMOKE_REPORT.json
D14_P3B_REPLAY_SMOKE_REPORT.md
D14_P3B_SELECTED_FILES.csv
D14_P3B_PROFILE_MANIFEST.csv
D14_P3B_PROFILE_SMOKE_SUMMARY.csv
D14_P3B_REPLAY_VALIDATION.csv
D14_P3B_SOH_POLICY.csv
D14_P3B_OUTPUT_INDEX.json
D14_P3B_RUN_SUMMARY.txt
README_D14_P3B_PATCH.md
D14_P3B_REPLAY_SMOKE_console.log
D14_P3B_AUDIT_stdout.log
D14_P3B_AUDIT_stderr.log
D14_P3B_VERIFY_stdout.log
D14_P3B_VERIFY_stderr.log
profiles/
  <batch>_<cell>/
    solution_replay_profile.npz
    profile_summary.json
```

A successful smoke should have no FAIL checks. WARN is acceptable if it only
records P3 FAST inherited WARN or optional temperature/capacity fields.
