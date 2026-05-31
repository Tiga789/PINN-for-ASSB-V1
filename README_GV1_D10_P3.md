# GV1 D10-P3 battery-8 lightweight correction package

This package tests whether the flagged `B1_2C battery-8` case can be handled by a lightweight post-hoc correction without modifying the D9.6 / D9.5.1 mainline.

It only adds scripts. It does not overwrite `gv1/model.py`, `gv1/output_transform.py`, `gv1/losses.py`, `gv1/trainer.py`, or `scripts/gv1_train_conditioned_pinn.py`.

## Files

```text
scripts/gv1_d10_p3_battery8_lightweight_correction.py
scripts/run_gv1_d10_p3_battery8_lightweight_correction.ps1
manifests/d10_p3_expected_outputs.json
README_GV1_D10_P3.md
```

## Default input

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96/prediction.npz
```

## Default output

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_batch134_d10_p3_battery8_lightweight_correction
```

## Run

From project root:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d10_p3_battery8_lightweight_correction.ps1"
```

With plots:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d10_p3_battery8_lightweight_correction.ps1" -MakePlots
```

## Read outputs

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p3_battery8_lightweight_correction\D10_P3_RECOMMENDATION.md" -Raw

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p3_battery8_lightweight_correction\d10_p3_candidate_metrics_fullfit.csv" |
  Sort-Object {[double]$_.score_rank} |
  Format-Table score_rank,candidate,recommendation_class,mae_V,corr,discharge_mae_V,charge_mae_V,pred_max_V,pred_overshoot_frac_gt_4p35 -AutoSize

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d10_p3_battery8_lightweight_correction\d10_p3_candidate_metrics_holdout.csv" |
  Sort-Object {[double]$_.score_rank} |
  Format-Table score_rank,candidate,recommendation_class,mae_V,corr,discharge_mae_V,charge_mae_V,pred_max_V,pred_overshoot_frac_gt_4p35 -AutoSize
```

## Interpret verdict

```text
safe_lightweight_correction_supported_by_fullfit_and_holdout
```

Use the selected correction only as a flagged battery-8 wrapper and proceed to a D10-P4 corrected battery-8 report.

```text
fullfit_correction_good_but_holdout_not_confirmed_calibration_only
```

Record it as a calibration benchmark, but keep battery-8 flagged for generalization claims.

```text
no_safe_lightweight_correction_keep_battery8_flagged
weak_correction_only_keep_battery8_flagged
```

Do not force battery-8 into normal 24-profile mainline. Keep 23-profile D10-P1 as the non-outlier generalization result.
