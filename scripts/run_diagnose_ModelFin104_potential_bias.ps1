$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

D:\Anaconda\envs\torchgpu\python.exe .\diagnose_eval_potential_common_mode.py `
  --eval_dir .\EvalFin_104_cycles5_100_v2_massclosed_candidate_cRadial010_softlabel_only
