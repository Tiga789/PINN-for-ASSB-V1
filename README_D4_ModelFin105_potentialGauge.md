# D4 / ModelFin_105 potential-gauge package

## Why ModelFin_105

ModelFin_104 fixed the positive concentration branch but introduced a large nearly constant negative bias in both potential outputs:

```text
ModelFin_104:
phis_c MAE ≈ 0.112 V, bias ≈ -0.12 V, corr ≈ 0.9996
phie   MAE ≈ 0.113 V, bias ≈ -0.13 V, corr ≈ 0.9986
theta_c/cs_c very good
```

This indicates that the current/time shape and concentration state are good, but the absolute potential level is not anchored. ModelFin_105 therefore keeps the ModelFin_104 concentration structure and fine-tunes from `ModelFin_104/best.pt`, while adding a compact potential-only soft-label data anchor.

## Files and locations

Unzip this package into:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

Expected files:

```text
PINN-for-ASSB-V1\
  input_assb_cycles5to522_v2_massclosed_ID105_potentialGauge
  build_assb_potential_gauge_data_cycle5_20_v2_massclosed.py
  evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py
  diagnose_eval_potential_common_mode.py
  README_D4_ModelFin105_potentialGauge.md

  scripts\
    check_input105_parser_format.ps1
    run_train_ModelFin105_v2_massclosed_potentialGauge.ps1
    check_ModelFin105_config.ps1
    run_eval_ModelFin105_v2_massclosed_cycle5_100.ps1
    run_diagnose_ModelFin104_potential_bias.ps1
```

## Main changes from ModelFin_104

```text
ID = 105
LOAD_MODEL = .\ModelFin_104\best.pt
alpha = 1.0 1.0 1.0 0.0
MAX_BATCH_SIZE_DATA = 2048
w_phie_dat = 50.0
w_phis_c_dat = 50.0
w_cs_a_dat = 0.0
w_cs_c_dat = 0.0
LEARNING_RATE_MODEL = 2e-4
LEARNING_RATE_MODEL_FINAL = 5e-5
EPOCHS = 80
CBAR_BASELINE_DEVIATION_FRACTION_C = 0.10
```

The concentration settings are intentionally kept from ModelFin_104 because they worked well. Only potential anchoring is added.

## Run order

```powershell
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

.\scripts\run_diagnose_ModelFin104_potential_bias.ps1
.\scripts\run_train_ModelFin105_v2_massclosed_potentialGauge.ps1
.\scripts\check_ModelFin105_config.ps1
.\scripts\run_eval_ModelFin105_v2_massclosed_cycle5_100.ps1
```

The train script automatically builds:

```text
DataFin_105_v2_massclosed_cycle5_20_potentialGauge\
  data_phie.npz
  data_phis_c.npz
  data_cs_a.npz
  data_cs_c.npz
  data_build_summary.json
```

and then runs:

```powershell
main.py -i input_assb_cycles5to522_v2_massclosed_ID105_potentialGauge -df DataFin_105_v2_massclosed_cycle5_20_potentialGauge
```

## Success criteria

Compare against ModelFin_104:

```text
104 phis_c MAE ≈ 0.112 V
104 phie   MAE ≈ 0.113 V
104 theta_c MAE ≈ 0.00556
104 cs_c    MAE ≈ 0.288
```

ModelFin_105 should primarily reduce:

```text
phis_c bias_mean
phie bias_mean
phis_c MAE
phie MAE
common_mode_error
```

without substantially hurting:

```text
theta_c / cs_c
```

If `theta_c/cs_c` stays good but `phis_c/phie` returns near the ModelFin_103-v2 level or better, ModelFin_105 is a successful fix.
