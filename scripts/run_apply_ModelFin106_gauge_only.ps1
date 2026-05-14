$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$RawDir = ".\EvalFin_106_cycles5_100_v2_massclosed_candidate_linearGauge_raw_softlabel_only"
$OutDir = ".\EvalFin_106_cycles5_100_v2_massclosed_candidate_linearCycleGauge_softlabel_only"

if (!(Get-ChildItem $RawDir -Filter "eval_sampled_arrays*.npz" -ErrorAction SilentlyContinue)) {
  throw "Raw eval npz not found in $RawDir. Run .\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_100_linearGauge.ps1 first."
}

D:\Anaconda\envs\torchgpu\python.exe .\apply_ModelFin106_linear_cycle_gauge.py `
  --eval_dir $RawDir `
  --model_dir .\ModelFin_106 `
  --output_dir $OutDir `
  --cycle_from 5 `
  --cycle_to 100 `
  --save_npz
