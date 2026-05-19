ASSB ModelFin112 guard v5 patch

Purpose
- Fix premature hard failure before the first guarded checkpoint.
- Keep hard guard selection strict: selected_model.pt is saved only after train/val guards pass.
- Replace the broken PowerShell Job parallel runner with safer foreground and Start-Process runners.

Files
- scripts/train_assb111_soh_head.py
  Modified. Adds soft-score patience tracking and min_epochs_before_patience.
- scripts/run_assb112_guarded_single_seed.ps1
  New. Foreground one-seed run for debugging.
- scripts/run_assb112_guarded_seed_sweep_startprocess.ps1
  New. Starts up to MaxParallel independent Python processes; no Start-Job / Receive-Job.
- scripts/summarize_assb112_guarded_seed_sweep.py
  Included for convenience.

Recommended verification after manual overwrite
Select-String -Path ".\scripts\train_assb111_soh_head.py" -Pattern "best_soft_score","min_epochs_before_patience","allow_patience_before_first_guard"
Select-String -Path ".\scripts\run_assb112_guarded_seed_sweep_startprocess.ps1" -Pattern "Start-Process","RedirectStandardOutput","Receive-Job"

Recommended first run
.\scripts\run_assb112_guarded_single_seed.ps1 -Seed 7 -Clean

If seed7 no longer exits at epoch 800, run the 4-process sweep
.\scripts\run_assb112_guarded_seed_sweep_startprocess.ps1 -MaxParallel 4 -Clean

Summarize
$py = "D:\Anaconda\envs\torchgpu\python.exe"
& $py ".\scripts\summarize_assb112_guarded_seed_sweep.py" --model_prefix ".\ModelFin_112_guardedSOH_seed" --seeds "7,42,2026,3407,7890" --output_dir ".\EvalFin_112_guarded_soh_sweep_v5"

Pass/fail
- ok=true for selected_checkpoint_audit.json in every seed.
- no_test_metrics_in_training_history=true.
- test_metrics_used_for_selection=false.
- mean test_R2 >= 0.98 and worst test_R2 >= 0.96.

If a seed ends with failed_no_visible_guarded_checkpoint after the full/warmup run, that is an honest fail of the guard criteria, not a leakage or path bug.
