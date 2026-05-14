# Optional: compare constant_mean, constant_median, and linear_cycle_mean gauge corrections.
# This does not retrain or alter ModelFin_105. It only post-processes the evaluation npz.

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-Location $ProjectRoot

$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
$EvalDir = ".\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only"
$OutDir = ".\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_commonModeCorrected_compare"

& $PythonExe .\calibrate_apply_common_mode_potential_offset.py `
  --eval_dir $EvalDir `
  --output_dir $OutDir `
  --calib_cycle_from 5 `
  --calib_cycle_to 20 `
  --apply_cycle_from 5 `
  --apply_cycle_to 100 `
  --method all `
  --save_npz

Write-Host ""
Write-Host "Done. Method comparison outputs:" -ForegroundColor Green
Write-Host "$OutDir\gauge_method_comparison.json"
Write-Host "$OutDir\gauge_method_comparison.csv"
Write-Host "$OutDir\constant_mean\metrics_global_corrected.json"
Write-Host "$OutDir\constant_median\metrics_global_corrected.json"
Write-Host "$OutDir\linear_cycle_mean\metrics_global_corrected.json"
