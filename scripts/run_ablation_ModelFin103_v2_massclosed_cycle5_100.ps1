$ErrorActionPreference = "Stop"

# This script can be launched from project root or from scripts\.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_radial_ablation_from_eval_npz.py `
  --eval_dir .\EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only `
  --scales 0 0.05 0.10 0.25 0.50 1.0
