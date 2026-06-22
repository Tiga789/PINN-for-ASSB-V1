param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

& $PythonExe ".\scripts\d18_verify_package_manifest.py" --root $ProjectRoot --manifest "D18_S0_S1_FIX_PACKAGE_MANIFEST.json"
if ($LASTEXITCODE -ne 0) { throw "D18-S0/S1-FIX V2 installed-manifest verification failed." }

$CompileTargets = @(
    ".\gv1\d18_cycleaware",
    ".\scripts\d18_package_selftest.py",
    ".\scripts\d18_run_s0_s1_fix.py",
    ".\scripts\d18_s0_validate_architecture.py",
    ".\scripts\d18_s1_array_latent_diagnostic.py",
    ".\scripts\d18_s1_build_dense_casepack.py",
    ".\scripts\d18_verify_package_manifest.py",
    ".\tests\test_d18_s0_s1_fix_v2.py"
)
& $PythonExe -m compileall -q @CompileTargets
if ($LASTEXITCODE -ne 0) { throw "D18-S0/S1-FIX V2 Python compileall failed." }

& $PythonExe ".\tests\test_d18_s0_s1_fix_v2.py" -v
if ($LASTEXITCODE -ne 0) { throw "D18-S0/S1-FIX V2 package-specific tests failed." }

& $PythonExe ".\scripts\d18_package_selftest.py"
if ($LASTEXITCODE -ne 0) { throw "D18-S0/S1-FIX V2 synthetic end-to-end self-test failed." }

Write-Host "PASS: D18-S0/S1-FIX V2 verification completed."
