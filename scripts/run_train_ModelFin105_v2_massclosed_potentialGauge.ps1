$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE="False"
$env:ASSB_EVAL_REFERENCE="soft_labels_only"

if (!(Test-Path ".\ModelFin_104\best.pt")) {
  throw "ModelFin_104\best.pt was not found. Train/evaluate ModelFin_104 first, or edit LOAD_MODEL in the ID105 input."
}

.\scripts\check_input105_parser_format.ps1

$DataDir = ".\DataFin_105_v2_massclosed_cycle5_20_potentialGauge"
D:\Anaconda\envs\torchgpu\python.exe .\build_assb_potential_gauge_data_cycle5_20_v2_massclosed.py `
  --soft_label_dir $env:ASSB_SOFT_LABEL_DIR `
  --output_dir $DataDir `
  --cycle_from 5 `
  --cycle_to 20 `
  --n_data 16384 `
  --seed 105

D:\Anaconda\envs\torchgpu\python.exe .\main.py `
  -i .\input_assb_cycles5to522_v2_massclosed_ID105_potentialGauge `
  -df $DataDir
