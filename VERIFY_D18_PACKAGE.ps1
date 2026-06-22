param(
    [switch]$SkipSynthetic
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$python = Get-Command python -ErrorAction Stop

Write-Host "[VERIFY] 1/4 package manifest"
& $python.Source ".\scripts\d18_verify_package_manifest.py" --root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Package manifest verification failed." }

Write-Host "[VERIFY] 2/4 Python compile checks"
$compileFiles = @(
    Get-ChildItem ".\gv1\d18_cycleaware\*.py" -File | ForEach-Object { $_.FullName }
) + @(
    Get-ChildItem ".\scripts\d18_*.py" -File | ForEach-Object { $_.FullName }
) + @(
    (Resolve-Path ".\tests\test_d18_core.py").Path
)
& $python.Source -m py_compile @compileFiles
if ($LASTEXITCODE -ne 0) { throw "D18 Python compile checks failed." }

Write-Host "[VERIFY] 3/4 unit tests"
& $python.Source -m unittest discover -s ".\tests" -p "test_d18_core.py" -v
if ($LASTEXITCODE -ne 0) { throw "D18 unit tests failed." }

if (-not $SkipSynthetic) {
    Write-Host "[VERIFY] 4/4 synthetic end-to-end dry-run"
    & $python.Source ".\scripts\d18_package_selftest.py" --package-root $ProjectRoot
    if ($LASTEXITCODE -ne 0) { throw "Synthetic D18 package self-test failed." }
} else {
    Write-Host "[VERIFY] 4/4 synthetic dry-run skipped by user"
}

Write-Host "PASS: D18-P0_S0_S1 package verification completed."
exit 0
