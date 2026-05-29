GV1 D9.6.3 conservative training-strategy repair
=================================================
D9.6.3 is a cautious rollback-style repair after D9.6.1 and D9.6.2 both failed
on B1_2C battery-8 200ks.

D9.6.1 failed by high-voltage saturation/overshoot. D9.6.2 failed by collapsing
the voltage range near 3.8 V. Therefore this package does not add new clamps or
component guards. It restores D9.6 core behavior and tests only lower learning
rate / seed candidates with conservative score-based selection.

Recommended first command:
powershell -ExecutionPolicy Bypass -File .\scripts\gv1_run_borderline_B1_2C_battery8_200ks_d963_probe.ps1

Paste this result after running:
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d963_probe\scorecard_borderline_200ks_d963_probe.json

Do not run 24-profile 200ks until this scorecard is reviewed.
