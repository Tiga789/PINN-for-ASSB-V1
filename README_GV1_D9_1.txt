GV1 D9.1 voltage-range update

Purpose:
- D9 v1 already passed forward/loss/backward smoke and 40 ks profile comparison.
- The 40 ks comparison showed high corr (~0.927-0.937) but compressed voltage range.
- D9.1 widens the voltage output transform so predictions can cover measured 2.5--4.2 V more easily.

Files:
- gv1/output_transform.py: wider voltage map; default profile_minmax voltage range; phis_c correction scale 0.60 V.
- gv1/trainer.py: computes transform voltage range from selected profile window using the new strategy.
- scripts/gv1_train_conditioned_pinn.py: adds CLI args for voltage range/correction controls.
- scripts/gv1_prediction_metrics.py: computes MAE/RMSE/bias/corr and voltage min/max from prediction.npz.
- scripts/gv1_run_profile_compare_200ks.ps1: optional PowerShell helper for 2C/R2.5/R3 200 ks comparison.

Recommended next action:
1. Back up the old D9 files if desired.
2. Copy this package into project root, overwriting files.
3. Run the 200 ks three-profile comparison.
4. Check metrics_summary.json. Do not start 24-profile training yet.
