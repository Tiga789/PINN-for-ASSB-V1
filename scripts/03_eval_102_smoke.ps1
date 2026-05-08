$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:ASSB_SOFT_LABEL_DIR = (Resolve-Path ".\Data\assb_soft_lable_cycle5-20_v1_smoke").Path
$env:ASSB_OCP_DIR = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE = "False"

D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_pinn_vs_softlabels.py `
  --model_dir ModelFin_102 `
  --soft_label_dir Data\assb_soft_lable_cycle5-20_v1_smoke `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --output_dir EvalFin_102_smoke_cycle5_20_v1 `
  --max_time_points 50000 `
  --max_cs_time_points 5000 `
  --debug_print_first_batch
