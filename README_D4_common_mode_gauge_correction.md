# D4 common-mode potential gauge correction for ModelFin_105

## Purpose

ModelFin_105 kept the good concentration behavior of ModelFin_104/105, but `phie` and `phis_c` still share a large negative common-mode offset. The differential potential `phis_c - phie` is already much better than the absolute potentials, so this package performs a post-hoc potential gauge calibration.

The script estimates a shared offset from a calibration cycle slice, by default **cycle 5-20**, and applies the same correction to both:

```text
phie_pred_corrected   = phie_pred   + offset
phis_c_pred_corrected = phis_c_pred + offset
```

It does not change `theta_a`, `theta_c`, `cs_a`, or `cs_c`.

## Where to put files

Unzip this package into:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

After unzipping, these files should exist:

```text
PINN-for-ASSB-V1\calibrate_apply_common_mode_potential_offset.py
PINN-for-ASSB-V1\scripts\run_common_mode_gauge_ModelFin105_cycle5_100.ps1
PINN-for-ASSB-V1\scripts\run_common_mode_gauge_method_compare_ModelFin105_cycle5_100.ps1
PINN-for-ASSB-V1\README_D4_common_mode_gauge_correction.md
```

## Main command

Run from project root:

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
.\scripts\run_common_mode_gauge_ModelFin105_cycle5_100.ps1
```

This reads:

```text
EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only\eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz
```

and writes:

```text
EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_commonModeCorrected\
  gauge_calibration_summary.json
  metrics_global_before.json
  metrics_global_corrected.json
  metrics_by_cycle_before.csv
  metrics_by_cycle_corrected.csv
  potential_common_mode_diagnostic_before_after.json
  potential_common_mode_by_cycle_before_after.csv
  eval_sampled_arrays_common_mode_corrected.npz
  plots_common_mode_corrected\
```

## Optional method comparison

To compare constant mean, constant median, and linear cycle-bias extrapolation:

```powershell
.\scripts\run_common_mode_gauge_method_compare_ModelFin105_cycle5_100.ps1
```

This writes:

```text
EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_commonModeCorrected_compare\
  gauge_method_comparison.json
  gauge_method_comparison.csv
  constant_mean\...
  constant_median\...
  linear_cycle_mean\...
```

## Interpretation

A successful gauge correction should:

```text
1. strongly reduce phie and phis_c MAE / bias;
2. leave phis_c - phie differential error nearly unchanged;
3. leave theta_c / cs_c unchanged;
4. preserve ModelFin_105's good concentration metrics.
```

If `constant_mean` over-corrects later cycles, inspect `linear_cycle_mean`, but treat linear extrapolation cautiously because it uses only cycle5-20 for calibration.

## Important

This is not a new trained model. It is a calibrated evaluation/post-processing result. If this solves the potential branch, the correction can later be embedded into the model output map or evaluation script as a formal `ModelFin_106` gauge parameter.
