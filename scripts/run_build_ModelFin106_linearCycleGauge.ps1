$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE="False"
$env:ASSB_EVAL_REFERENCE="soft_labels_only"

if (!(Test-Path ".\ModelFin_105\best.pt")) { throw "ModelFin_105\best.pt not found. Finish ModelFin_105 training first." }
if (!(Test-Path ".\ModelFin_105\config.json")) { throw "ModelFin_105\config.json not found." }
if (!(Test-Path ".\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only\eval_sampled_arrays_cycles5_100_v2_massclosed_softlabel_only.npz")) {
  throw "Missing ModelFin_105 eval npz. Run .\scripts\run_eval_ModelFin105_v2_massclosed_cycle5_100.ps1 first."
}

D:\Anaconda\envs\torchgpu\python.exe .\build_ModelFin106_from_ModelFin105_linearCycleGauge.py `
  --base_model_dir .\ModelFin_105 `
  --source_eval_dir .\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only `
  --output_model_dir .\ModelFin_106 `
  --calib_cycle_from 5 `
  --calib_cycle_to 20 `
  --apply_cycle_from 5 `
  --apply_cycle_to 100 `
  --overwrite

Write-Host ""
Write-Host "ModelFin_106 created:"
Write-Host ".\ModelFin_106\best.pt"
Write-Host ".\ModelFin_106\config.json"
Write-Host ".\ModelFin_106\gauge_config.json"
