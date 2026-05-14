$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE="False"
$env:ASSB_EVAL_REFERENCE="soft_labels_only"

if (!(Test-Path ".\ModelFin_106\gauge_config.json")) {
  Write-Host "ModelFin_106\gauge_config.json not found. Building ModelFin_106 first..."
  .\scripts\run_build_ModelFin106_linearCycleGauge.ps1
}

$RawDir = ".\EvalFin_106_cycles5_100_v2_massclosed_candidate_linearGauge_raw_softlabel_only"
$OutDir = ".\EvalFin_106_cycles5_100_v2_massclosed_candidate_linearCycleGauge_softlabel_only"

D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py `
  --model_dir ModelFin_106 `
  --soft_label_dir $env:ASSB_SOFT_LABEL_DIR `
  --ocp_dir $env:ASSB_OCP_DIR `
  --cycle_from 5 `
  --cycle_to 100 `
  --output_dir $RawDir `
  --debug_print_first_batch

D:\Anaconda\envs\torchgpu\python.exe .\apply_ModelFin106_linear_cycle_gauge.py `
  --eval_dir $RawDir `
  --model_dir .\ModelFin_106 `
  --output_dir $OutDir `
  --cycle_from 5 `
  --cycle_to 100 `
  --save_npz

Write-Host ""
Write-Host "Corrected ModelFin_106 outputs:"
Write-Host "$OutDir\metrics_global_corrected.json"
Write-Host "$OutDir\metrics_by_cycle_corrected.csv"
Write-Host "$OutDir\potential_common_mode_diagnostic_before_after.json"

$mg = Get-Content "$OutDir\metrics_global_corrected.json" | ConvertFrom-Json
foreach ($v in "phis_c","phie","theta_a","theta_c","cs_a","cs_c") {
  $m = $mg.$v
  if ($null -ne $m) {
    Write-Host ("{0,-8} MAE={1} RMSE={2} R2={3} corr={4} NMAE={5}" -f $v,$m.mae,$m.rmse,$m.r2,$m.corr,$m.nmae)
  }
}
$diag = Get-Content "$OutDir\potential_common_mode_diagnostic_before_after.json" | ConvertFrom-Json
Write-Host "common_mode_mae_before =" $diag.common_mode_error_before.mae
Write-Host "common_mode_mae_after  =" $diag.common_mode_error_after.mae
