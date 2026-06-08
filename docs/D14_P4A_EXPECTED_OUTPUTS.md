# D14-P4A Expected Outputs

The runner creates an output directory such as:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_v1_smoke_p4a
```

Expected files:

```text
D14_P4A_SOFTLABEL_SMOKE_REPORT.json
D14_P4A_SOFTLABEL_SMOKE_REPORT.md
D14_P4A_SELECTED_PROFILES.csv
D14_P4A_SOFTLABEL_MANIFEST.csv
D14_P4A_SOFTLABEL_AUDIT.csv
D14_P4A_SOFTLABEL_AUDIT_RERUN.csv
D14_P4A_PRIOR_RESOLVED.json
D14_P4A_PRIOR_HASH.txt
D14_P4A_OUTPUT_INDEX.json
D14_P4A_RUN_SUMMARY.txt
README_D14_P4A_PATCH.md
D14_P4A_SOFTLABEL_SMOKE_console.log
D14_P4A_GENERATE_stdout.log
D14_P4A_GENERATE_stderr.log
D14_P4A_AUDIT_stdout.log
D14_P4A_AUDIT_stderr.log
D14_P4A_VERIFY_stdout.log
D14_P4A_VERIFY_stderr.log
profiles/<cell_uid>/
  solution_softlabels.npz
  soft_label_summary.json
  soft_label_audit.json
```

New required NPZ fields compared with P4:

```text
phis_c_soft_raw
voltage_bound_correction
batch
protocol
cell_uid
metadata_inferred_from_path
```

Key audit fields:

```text
metadata_ok
phis_c_soft_min_V
phis_c_soft_max_V
phis_c_soft_raw_min_V
phis_c_soft_raw_max_V
voltage_upper_warn_count
voltage_upper_fail_count
voltage_lower_warn_count
voltage_lower_fail_count
max_abs_voltage_bound_correction_V
```
