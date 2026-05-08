# Optional cleanup if an earlier wrong Model102 wrapper was copied.
# Run from project root only if check_model102_no_wrapper.ps1 shows __pre102/_load_legacy_module traces.
$ErrorActionPreference = "Stop"
Write-Host "[CLEAN] Removing old Model102 wrapper backup files..." -ForegroundColor Cyan
Remove-Item .\util\*__pre102.py -Force -ErrorAction SilentlyContinue
Remove-Item .\evaluate_assb_pinn_vs_softlabels__pre102.py -Force -ErrorAction SilentlyContinue

Write-Host "[CLEAN] Restoring util/init_pinn.py, util/_losses.py, util/_rescale.py from origin/main if git is available..." -ForegroundColor Cyan
if (Test-Path .\.git) {
  git fetch origin main | Out-Null
  git restore --source=origin/main -- .\util\init_pinn.py .\util\_losses.py .\util\_rescale.py
  Write-Host "[CLEAN] git restore completed." -ForegroundColor Green
} else {
  Write-Host "[CLEAN] No .git folder found. Please manually replace util/init_pinn.py, util/_losses.py, util/_rescale.py with your clean ID101/origin-main versions." -ForegroundColor Yellow
}

Write-Host "[CLEAN] Done. Now re-extract/copy this overwrite package into project root, then run scripts/check_model102_no_wrapper.ps1." -ForegroundColor Green
