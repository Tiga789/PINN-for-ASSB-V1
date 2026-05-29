GV1 D9.7 battery-8 outlier/regime diagnosis plot patch

Purpose:
- Adds the missing plotting command files mentioned in the D9.7 diagnostic step.
- Generates voltage overlay, residual, pred-vs-exp, current/temperature regime plots,
  plus a CSV metric table and d97_plot_manifest.json.
- Diagnostic only: does not train, modify checkpoints, or change GV1 model code.

Files:
- scripts/gv1_plot_battery8_regime_d97.py
- scripts/gv1_plot_battery8_regime_d97.ps1

Run from project root:
  powershell -ExecutionPolicy Bypass -File .\scripts\gv1_plot_battery8_regime_d97.ps1

Expected outputs:
  E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\diagnosis_plots\d97_plot_manifest.json
  E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\diagnosis_plots\d97_candidate_metrics_table.csv
  E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\diagnosis_plots\*.png
