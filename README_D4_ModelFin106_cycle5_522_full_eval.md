# D4 / ModelFin_106 cycle5-522 full evaluation package

This package extends the current ModelFin_106 evaluation from cycle5-100 to the full continuous cycle5-522 range.

## What this evaluates

- Model directory: `ModelFin_106`
- Base weights: `ModelFin_105/best.pt`, materialized in `ModelFin_106/best.pt`
- Gauge: `ModelFin_106/gauge_config.json`
- Soft labels: `C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate`
- Cycle range: `5-522`
- Reference: soft labels only; `voltage_exp` is ignored.

The raw evaluation first computes ModelFin_106 outputs without applying the gauge offset. Then `apply_ModelFin106_linear_cycle_gauge_cycle5_522.py` applies:

```text
offset_to_add_V = -(linear_bias_slope_V_per_cycle * cycle_id + linear_bias_intercept_V)
```

to both `phie_pred` and `phis_c_pred`.

## Files and placement

Unzip this package into:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

Expected added files:

```text
PINN-for-ASSB-V1\
  evaluate_assb_pinn_cycles5_522_v2_massclosed_softlabels.py
  apply_ModelFin106_linear_cycle_gauge_cycle5_522.py
  build_ModelFin106_from_ModelFin105_linearCycleGauge.py
  README_D4_ModelFin106_cycle5_522_full_eval.md

  scripts\
    run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1
    show_ModelFin106_cycle5_522_worst_cycles.ps1
    check_ModelFin106_cycle5_522_package.ps1
```

## Run

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
.\scripts\check_ModelFin106_cycle5_522_package.ps1
.\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1
```

Optional: after completion, show worst per-cycle rows:

```powershell
.\scripts\show_ModelFin106_cycle5_522_worst_cycles.ps1
```

## Outputs

Raw outputs:

```text
EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only\
  metrics_global.json
  metrics_by_cycle.csv
  debug_model_and_data.json
  eval_sampled_arrays_cycles5_522_v2_massclosed_softlabel_only.npz
  plots_softlabel_only\
```

Corrected outputs:

```text
EvalFin_106_cycles5_522_v2_massclosed_candidate_linearCycleGauge_softlabel_only\
  metrics_global_corrected.json
  metrics_by_cycle_corrected.csv
  metrics_global_raw.json
  metrics_by_cycle_raw.csv
  potential_common_mode_diagnostic_before_after.json
  potential_common_mode_by_cycle_before_after.csv
  model106_linear_cycle_gauge_summary.json
  plots_linearCycleGauge\
```

## Sampling note

For the full cycle5-522 range, the script uses all potential time points but samples concentration rows by default:

```text
max_time_points = 0      # all potential rows
max_cs_time_points = 20000
```

This keeps GPU/memory and output size manageable. To increase concentration sampling density:

```powershell
$env:ASSB_EVAL_MAX_CS_ROWS="30000"
.\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1
```

Setting `ASSB_EVAL_MAX_CS_ROWS=0` requests all concentration rows and may create very large arrays.

## How to judge

Compare full-cycle corrected metrics against the cycle5-100 benchmark:

```text
cycle5-100 corrected:
phis_c MAE ≈ 0.00725 V
phie   MAE ≈ 0.00151 V
theta_c MAE ≈ 0.00566
cs_c    MAE ≈ 0.293
```

For cycle5-522, the key question is whether the linear-cycle common-mode gauge extrapolates beyond cycle100. Watch:

```text
phis_c/phie corrected MAE and common_mode_mae_after
per-cycle drift in potential_common_mode_by_cycle_before_after.csv
theta_c/cs_c per-cycle stability
```
