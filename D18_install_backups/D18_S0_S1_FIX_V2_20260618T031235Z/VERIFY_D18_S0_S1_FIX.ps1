param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

& $PythonExe ".\scripts\d18_verify_package_manifest.py" --root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw "Package manifest verification failed." }

& $PythonExe -m compileall -q ".\gv1\d18_cycleaware" ".\scripts" ".\tests"
if ($LASTEXITCODE -ne 0) { throw "Python compileall failed." }

& $PythonExe -m unittest discover -s ".\tests" -v
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed." }

& $PythonExe ".\scripts\d18_package_selftest.py"
if ($LASTEXITCODE -ne 0) { throw "Synthetic end-to-end self-test failed." }

Write-Host "PASS: D18-S0/S1-FIX package verification completed."
