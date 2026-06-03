# D12-S1J — 200ks normal/rest leakage diagnostic wrapper

## Purpose

D12-S1I proved that the 40ks high-region failure of S1E-soft can be fixed by high-region fallback/budget.  However, in 200ks the S1E-soft source prediction no longer only fails in high voltage; normal/rest/global also regress.

D12-S1J does **not** train a new neural network.  It reads the already generated D12-S1E 6-profile 200ks predictions and creates diagnostic wrapper variants to answer:

- Is S1E correction only useful in `low_target` / `low_target_le_2p75`?
- Does 200ks fail because the S1E correction leaks into normal/rest/high regions?
- Can a low-only or non-low-budget wrapper pass 6-profile 200ks?

## Files

```text
scripts/gv1_d12_s1j_diagnose_200ks_leakage.py
scripts/gv1_run_d12_s1j_diagnose_200ks_leakage.ps1
README_D12_S1J.md
install_manifest.json
```

## Default input

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1e_p2d_anchor_budget_6x200ks
```

It must contain `baseline_d951` and `d12s1e_p2d_low_anchor_soft` prediction folders.

## Default outputs

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1j_200ks_normal_rest_leakage_wrapper
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1j_200ks_normal_rest_leakage_scorecard
```

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

$py = "D:\Anaconda\envs\torchgpu\python.exe"
$proj = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$cache = "E:\XJTU battery dataset\_gv1_cache"

.\scripts\gv1_run_d12_s1j_diagnose_200ks_leakage.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Clean
```

## Main diagnostic variants

- `d12s1j_low_only_revert_nonlow_to_baseline`
- `d12s1j_low_plus_transition_fade_to_baseline`
- `d12s1j_low_full_nonlow_budget_5mV_rest_revert`
- `d12s1j_low_full_nonlow_budget_10mV_rest_revert`
- `d12s1j_low_full_nonlow_budget_20mV_rest_revert`
- `d12s1j_low_preserve_normal_rest_high_revert`
- `d12s1j_high_and_rest_revert`

## Send back these files

```text
D12_S1J_scorecard_summary.json
D12_S1J_candidate_decisions.csv
D12_S1J_mode_summary.csv
D12_S1J_segment_metrics.csv
D12_S1J_leakage_decomposition.csv
D12_S1J_RECOMMENDATION.md
```

## Decision rule

A variant promotes only if:

```text
low_ok = True
deep_ok = True
global_ok = True
normal_ok = True
high_ok = True
rest_ok = True
corr_ok = True
```

If no variant promotes, stop using post-wrapper fixes and return to source S1E training loss design.
