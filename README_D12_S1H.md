# D12-S1H diagnostic-only package

## Purpose

D12-S1H does not train a model. It reads the existing D12-S1E 6-profile 40ks
prediction outputs and diagnoses why `d12s1e_p2d_low_anchor_soft` failed `high_ok`
while passing low/deep/global/normal/corr/rest more closely than later S1F/S1G.

The goal is to avoid wasting another 40ks training round before knowing whether
S1E-soft can be repaired by a high-only local limiter, a scalar recenter, or a
correction budget.

## Added files

```text
scripts/gv1_d12_s1h_diagnose_s1e_soft.py
scripts/gv1_run_d12_s1h_diagnose_s1e_soft.ps1
README_D12_S1H.md
install_manifest.json
```

## Output files

```text
D12_S1H_diagnostic_summary.json
D12_S1H_variant_decisions.csv
D12_S1H_variant_segment_metrics.csv
D12_S1H_profile_high_diagnostics.csv
D12_S1H_RECOMMENDATION.md
```

## Default input

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1e_p2d_anchor_budget_6x40ks
```

## Important

This package does not modify the D9.6/D9.5.1 mainline and does not run 200ks.
