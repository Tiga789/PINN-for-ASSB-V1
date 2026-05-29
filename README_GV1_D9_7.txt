GV1 D9.7 battery-8 outlier / regime diagnosis package
=======================================================

This package is diagnostic-only. It does not modify model.py, output_transform.py, losses.py, or trainer.py.

Reason for D9.7:
- D9.6 remains the current mainline.
- D9.6.1 / D9.6.1_v2 produced high-voltage upper-tail instability.
- D9.6.2 collapsed the voltage range near 3.8 V.
- D9.6.3 training-strategy sweep did not clear conservative replacement checks.

First command after manual add/overwrite:
cd C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_battery8_outlier_diagnosis_d97.ps1

Main output to send back:
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis\d97_battery8_diagnosis_summary.json

Additional outputs:
- prediction_summary_d97.csv
- component_health_d97.csv
- time_bins_all_predictions_d97.csv
- scorecard_B1_2C_worst_first_d97.csv
- plots/*.png

Do not run 24-profile 200ks before reviewing D9.7 diagnosis.
