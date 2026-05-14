# ModelFin_106: linear-cycle common-mode gauge wrapper

ModelFin_106 is defined as **ModelFin_105/best.pt + a linear cycle-dependent common-mode potential offset**.

It is not a fresh network training run. The concentration branches and the differential potential `phis_c - phie` remain inherited from ModelFin_105.

Correction convention:

```text
common_mode_error = 0.5*((phie_pred - phie_true) + (phis_c_pred - phis_c_true))
fitted_bias(cycle) = slope * cycle_id + intercept
offset_to_add(cycle) = -fitted_bias(cycle)
phie_corrected   = phie_pred   + offset_to_add(cycle)
phis_c_corrected = phis_c_pred + offset_to_add(cycle)
```

Fitted coefficients:

```text
slope_V_per_cycle = 0.0002924856517888649
intercept_V       = -0.09888082346895946
calibration cycles = 5-20
application cycles = 5-100
```

See `gauge_config.json` for the full provenance.
