# D11-S7 low-voltage escape redesign

## Purpose

D11-S6 showed a structural low-voltage floor/capacity barrier: low-target true voltage is around 2.5-2.9 V, but current predictions stay near 3.4 V or higher. D11-S7 adds a **diagnostic low-voltage escape branch** to the GV1 output transform, disabled by default and enabled only through D11-S7 CLI modes.

This package is for a **6-profile / 40 ks smoke test only**. It must not be promoted to mainline unless low-target segments actually improve.

## Included scripts

```text
scripts/gv1_d11_s7_apply_lowvoltage_escape_patch.py
scripts/gv1_d11_s7_prepare_lowvoltage_escape_commands.py
scripts/gv1_d11_s7_scorecard_from_predictions.py
scripts/run_gv1_d11_s7_preflight_check.ps1
scripts/run_gv1_d11_s7_apply_patch.ps1
scripts/run_gv1_d11_s7_prepare_commands.ps1
scripts/run_gv1_d11_s7_collect_scorecard.ps1
RUN_ORDER_D11_S7.txt
```

## Experiment modes

```text
baseline_d951
lowvoltage_escape_mild
lowvoltage_escape_medium
lowvoltage_escape_strong_guarded
```

## Scope

```text
profiles = 6
window = 40 ks
battery-8 = excluded
metadata_on = not used
hard clamp = disabled
expected runs = 24
```

## Promotion rule

A candidate is only eligible for later 200 ks confirmation if:

```text
low_target MAE decreases vs baseline
low_target_le_2p75 MAE decreases vs baseline
global MAE does not materially increase
corr does not materially drop
rest_I_zero does not materially worsen
no new high-voltage overshoot appears
```
