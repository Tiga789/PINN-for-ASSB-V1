$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:ASSB_SOFT_LABEL_DIR = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1"
$env:ASSB_OCP_DIR = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
$env:ASSB_CYCLE_FROM = "5"
$env:ASSB_CYCLE_TO = "20"
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE = "False"

D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_pinn_vs_softlabels.py `
  --model_dir ModelFin_102 `
  --soft_label_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --cycle_from 5 `
  --cycle_to 20 `
  --output_dir EvalFin_102_smoke_cycle5_20_v1_existing_softlabels `
  --max_time_points 50000 `
  --max_cs_time_points 5000 `
  --debug_print_first_batch
