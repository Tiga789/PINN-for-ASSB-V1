$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\torchgpu\python.exe"

$RAW_EVAL = "EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only"
$OUT = "EvalFin_106_cycles5_522_v2_massclosed_candidate_csA_diagnostic"

if (!(Test-Path ".\$RAW_EVAL")) {
  throw "Missing raw ModelFin_106 full-cycle eval directory: .\$RAW_EVAL. Run ModelFin_106 cycle5-522 evaluation first."
}

& $PY .\diagnose_ModelFin106_csA_cbar_radial_fullcycle.py `
  --raw_eval_dir $RAW_EVAL `
  --output_dir $OUT `
  --cycle_from 5 `
  --cycle_to 522

Write-Host "`n[OK] cs_a diagnostic finished." -ForegroundColor Green
Write-Host ".\$OUT\cs_a_cbar_radial_diagnostic_global.json"
Write-Host ".\$OUT\cs_a_cbar_radial_diagnostic_by_cycle.csv"
