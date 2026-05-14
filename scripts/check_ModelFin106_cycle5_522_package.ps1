$ErrorActionPreference = "Stop"
Write-Host "Checking required files for ModelFin_106 cycle5-522 evaluation..." -ForegroundColor Cyan
$files = @(
  ".\evaluate_assb_pinn_cycles5_522_v2_massclosed_softlabels.py",
  ".\apply_ModelFin106_linear_cycle_gauge_cycle5_522.py",
  ".\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1",
  ".\scripts\show_ModelFin106_cycle5_522_worst_cycles.ps1",
  ".\ModelFin_106\best.pt",
  ".\ModelFin_106\gauge_config.json"
)
foreach ($f in $files) {
  if (Test-Path $f) { Write-Host "OK: $f" -ForegroundColor Green }
  else { Write-Host "MISSING: $f" -ForegroundColor Red }
}
