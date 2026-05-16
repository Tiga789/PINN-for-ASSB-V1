param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"
Push-Location $ProjectRoot
try {
  Write-Host "[check_ModelFin110_aging_fix1_package] ProjectRoot=$(Get-Location)"
  $required = @(
    "util\assb_aging_fix1_config.py",
    "util\assb_aging_mechanism.py",
    "util\assb_aging_capacity.py",
    "util\assb_aging_injection.py",
    "util\assb_model_integrity.py",
    "scripts\prepare_assb_aging_fix1_cycle_table.py",
    "scripts\train_assb_aging_stageB.py",
    "scripts\check_ModelFin110_aging_fix1_package.ps1",
    "scripts\run_ModelFin110_stageB.ps1",
    "scripts\run_ModelFin110_stageC_smoke.ps1",
    "compare_assb_107A_core_integrity.py",
    "evaluate_assb_aging_fix1.py",
    "input_assb_ModelFin110_agingStageC"
  )
  $missing = @()
  foreach ($f in $required) {
    if (-not (Test-Path $f)) { $missing += $f }
  }
  if ($missing.Count -gt 0) {
    Write-Host "Missing files:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 2
  }
  Write-Host "required new files: OK" -ForegroundColor Green

  $pyFiles = $required | Where-Object { $_.EndsWith(".py") }
  & $PythonExe -m py_compile @pyFiles
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host "py_compile: OK" -ForegroundColor Green

  # Do not scan the guard/check files themselves, because they must contain the
  # forbidden strings as literals.  Scan production modules and scripts only.
  $patterns = @("_base.py", "107A_base", "SourceFileLoader", "spec_from_file_location")
  $scanFiles = $required | Where-Object {
    $_ -notin @("util\assb_model_integrity.py", "scripts\check_ModelFin110_aging_fix1_package.ps1", "compare_assb_107A_core_integrity.py")
  }
  $hits = @()
  foreach ($f in $scanFiles) {
    if (Test-Path $f) {
      $text = Get-Content $f -Raw -ErrorAction SilentlyContinue
      foreach ($p in $patterns) {
        if ($text -match [regex]::Escape($p)) { $hits += "$f :: $p" }
      }
    }
  }
  if ($hits.Count -gt 0) {
    Write-Host "Forbidden overlay/base dependency patterns found:" -ForegroundColor Red
    $hits | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 3
  }
  Write-Host "no backup/base-file dependency pattern detected: OK" -ForegroundColor Green
  Write-Host "Package check passed." -ForegroundColor Green
}
finally {
  Pop-Location
}
