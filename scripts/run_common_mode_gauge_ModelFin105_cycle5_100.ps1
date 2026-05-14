# D4: ModelFin_105 potential common-mode / gauge correction.
# Place this script under PINN-for-ASSB-V1\scripts and run from project root.

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-Location $ProjectRoot

$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
$EvalDir = ".\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only"
$OutDir = ".\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_commonModeCorrected"

& $PythonExe .\calibrate_apply_common_mode_potential_offset.py `
  --eval_dir $EvalDir `
  --output_dir $OutDir `
  --calib_cycle_from 5 `
  --calib_cycle_to 20 `
  --apply_cycle_from 5 `
  --apply_cycle_to 100 `
  --method constant_mean `
  --save_npz

Write-Host ""
Write-Host "Done. Key outputs:" -ForegroundColor Green
Write-Host "$OutDir\gauge_calibration_summary.json"
Write-Host "$OutDir\metrics_global_corrected.json"
Write-Host "$OutDir\metrics_by_cycle_corrected.csv"
Write-Host "$OutDir\potential_common_mode_diagnostic_before_after.json"
