$ErrorActionPreference = "Stop"
$files = @(
  ".\diagnose_ModelFin106_csA_cbar_radial_fullcycle.py",
  ".\fit_apply_ModelFin107A_anode_state_correction.py",
  ".\scripts\run_diagnose_ModelFin106_csA_fullcycle.ps1",
  ".\scripts\run_all_ModelFin107A_csA_calib5_522_eval5_522.ps1",
  ".\scripts\run_all_ModelFin107A_csA_calib5_100_eval5_522.ps1",
  ".\scripts\show_ModelFin107A_cycle5_522_worst_cycles.ps1"
)
foreach ($f in $files) {
  if (Test-Path $f) { Write-Host "OK: $f" -ForegroundColor Green } else { throw "Missing: $f" }
}
if (Test-Path ".\EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only") {
  Write-Host "OK: raw ModelFin_106 full-cycle eval directory exists." -ForegroundColor Green
} else {
  Write-Host "WARN: raw ModelFin_106 full-cycle eval directory is missing. Run ModelFin_106 full-cycle eval first." -ForegroundColor Yellow
}
