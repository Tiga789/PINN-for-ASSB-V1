param(
    [string]$ProjectRoot = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
Set-Location $ProjectRoot
$env:PYTHONUTF8 = "1"

Write-Host "[1/4] Verifying installed file sizes and SHA256 hashes..."
python ".\scripts\d18_verify_s2_package_manifest.py" --project-root "$ProjectRoot"
if ($LASTEXITCODE -ne 0) { throw "D18-S2 package manifest verification failed." }

Write-Host "[2/4] Compiling D18-S2 Python files..."
python -m compileall -q ".\gv1\d18_s2" ".\scripts\d18_run_s2_preflight_micro_smoke.py" ".\scripts\d18_s2_preflight.py" ".\scripts\d18_s2_package_selftest.py" ".\scripts\d18_verify_s2_package_manifest.py" ".\tests\test_d18_s2_preflight_micro_smoke_v1.py"
if ($LASTEXITCODE -ne 0) { throw "D18-S2 Python compile check failed." }

Write-Host "[3/4] Running the dedicated D18-S2 unit tests..."
python ".\tests\test_d18_s2_preflight_micro_smoke_v1.py" -v
if ($LASTEXITCODE -ne 0) { throw "D18-S2 unit tests failed." }

Write-Host "[4/4] Running synthetic exact-UID and end-to-end micro-smoke selftest..."
python ".\scripts\d18_s2_package_selftest.py" --project-root "$ProjectRoot"
if ($LASTEXITCODE -ne 0) { throw "D18-S2 synthetic end-to-end selftest failed." }

Write-Host "PASS: D18-S2 package verification completed."
