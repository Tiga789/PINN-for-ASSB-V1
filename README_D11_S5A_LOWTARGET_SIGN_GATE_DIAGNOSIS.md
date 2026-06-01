# D11-S5A low-target sign/gate diagnosis

This package is for **D11-S5A**, not D11-S5 confirmation.

D11-S4 showed that the global MAE improved for lowtail modes, but the actual
`low_target` and `low_target_le_2p75` segments became worse. Therefore, the next
step is a small 6-profile / 40 ks sign-and-gate diagnosis, not a 200 ks expansion.

## Purpose

Diagnose why low-voltage target segments remain over-predicted:

- Is the low-voltage gate active on `target <= 2.75 V`?
- Is the correction sign wrong or too weak?
- Is global MAE improvement hiding low-target degradation?
- Does any mode improve both `low_target` and `low_target_le_2p75` without damaging global trend?

## Files

```text
scripts/gv1_d11_s5a_prepare_lowtarget_diagnosis_commands.py
scripts/gv1_d11_s5a_scorecard_from_predictions.py
scripts/run_gv1_d11_s5a_preflight_check.ps1
scripts/run_gv1_d11_s5a_prepare_commands.ps1
scripts/run_gv1_d11_s5a_collect_scorecard.ps1
README_D11_S5A_LOWTARGET_SIGN_GATE_DIAGNOSIS.md
RUN_ORDER_D11_S5A.txt
```

## Scope

```text
profile scope  = 6 profiles
window         = 40 ks
modes          = baseline_d951 / lowtarget_gate_probe / lowtarget_downward_mild / lowtarget_downward_strict
battery-8      = excluded
metadata_on    = not used
hard clamp     = disabled
expected runs  = 24
```

## Output directories

Commands:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5a_lowtarget_sign_gate_commands
```

Predictions:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5a_lowtarget_sign_gate_diagnosis
```

Scorecard:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_s5a_lowtarget_sign_gate_scorecard
```

## Success criterion

A candidate can only proceed to a later 6-profile 200ks confirmation if:

```text
low_target MAE decreases versus baseline
low_target_le_2p75 MAE decreases versus baseline
global MAE does not materially increase
global corr does not materially decrease
rest_I_zero does not materially worsen
no high-voltage overshoot is introduced
```

If no candidate satisfies this, do not expand to 200 ks. Redesign the low-target gate or correction sign.
