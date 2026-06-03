# D12-S1I: S1E-soft high-region local fallback/budget wrapper

## Purpose

D12-S1I is not another full training sweep. It formalizes the D12-S1H diagnostic finding:

- S1E-soft already passes low/deep/global/normal/rest/corr.
- S1E-soft fails only because P2D correction leaks into high-voltage regions.
- The best diagnostic repair was high-region fallback to baseline.

Therefore S1I reads existing S1E prediction files and writes corrected prediction files using high-region local logic.

## Inputs

Default source root:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_s1e_p2d_anchor_budget_6x40ks
```

Required modes inside the source root:

```text
baseline_d951
d12s1e_p2d_low_anchor_soft
```

## Output modes

```text
baseline_d951
d12s1i_high_region_revert_to_baseline
d12s1i_high_region_delta_budget_20mV
d12s1i_clip_4p35_plus_high_budget_20mV
```

The first candidate is the strictest implementation of the S1H best variant.

## What is changed

Only new scripts are added. The D9.6/D9.5.1 mainline is not modified.

## Primary run command

```powershell
.\scripts\gv1_run_d12_s1i_apply_s1e_soft_40ks.ps1 `
  -ProjectRoot $proj `
  -CacheRoot $cache `
  -PythonExe $py `
  -Clean
```

## Decision

Use the S1I scorecard. Only if `promote_to_200ks=True` should you plan a longer 200ks confirmation.
