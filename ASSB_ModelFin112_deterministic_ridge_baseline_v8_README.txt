ASSB ModelFin112 deterministic ridge/SOH baseline v8

Purpose
-------
This package replaces the unstable neural SOH-head route with a deterministic ridge baseline.
It does not use random seeds, Start-Job, Start-Process, or multi-window PowerShell.
It fits StandardScaler on train cycles only, selects ridge alpha on train/val only, and writes held-out test metrics after selection is fixed.

Files
-----
scripts/train_assb112_deterministic_soh_baseline.py
  Main deterministic ridge trainer/evaluator. Writes ridge_model.json, soh_pred_by_cycle.csv, train_summary.json,
  selected_checkpoint_audit.json, metrics_soh_by_split_final_report.json, alpha_selection_visible_only.csv,
  feature_importance.csv, deterministic_soh_scorecard.csv.

scripts/run_assb112_deterministic_ridge_baseline.ps1
  One foreground command. No pop-up windows. Uses CUDA by default and attempts to reserve configurable GPU memory.

scripts/run_assb112_deterministic_ridge_cpu_fallback.ps1
  CPU fallback wrapper.

scripts/summarize_assb112_deterministic_baseline.py
  Creates EvalFin_112_deterministicSOH_ridge_g4 summary and scorecard.

util/assb_soh_feature_schema.py
  Strict SOH feature schema. Keeps capacity/SOH-equivalent columns out of strict feature mode.

Default run
-----------
Set-Location "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
.\scripts\run_assb112_deterministic_ridge_baseline.ps1 -Clean -GpuReserveGB 2.0 -GpuWorkRepeats 4

If CUDA OOM occurs
------------------
.\scripts\run_assb112_deterministic_ridge_baseline.ps1 -Clean -GpuReserveGB 0.5 -GpuWorkRepeats 1

If CUDA is unavailable
----------------------
.\scripts\run_assb112_deterministic_ridge_cpu_fallback.ps1 -Clean

Audit points
------------
1. train_summary.json: no_test_metrics_in_training_history=true
2. train_summary.json: test_metrics_used_for_selection=false
3. selected_checkpoint_audit.json: ok=true
4. metrics_soh_by_split_final_report.json: inspect held-out test SOH_R2 / SOH_MAE / SOH_BIAS

Notes
-----
Ridge fitting itself is tiny, so GPU utilization may not stay high for long. The -GpuReserveGB parameter is an explicit, auditable memory reservation knob. It does not change the model or metrics.
