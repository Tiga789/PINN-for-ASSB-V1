param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$ConfigPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
Set-Location $ProjectRoot
$env:PYTHONUTF8 = "1"

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $ProjectRoot "configs\d18_s2_preflight_micro_smoke.json"
}
$ConfigPath = (Resolve-Path $ConfigPath).Path

Write-Host "Verifying installed D18-S2 files before execution..."
python ".\scripts\d18_verify_s2_package_manifest.py" --project-root "$ProjectRoot"
if ($LASTEXITCODE -ne 0) { throw "Installed D18-S2 files do not match the package manifest." }

Write-Host "Starting D18-S2 preflight + bounded micro-smoke."
Write-Host "This command cannot start formal S2 training."
python ".\scripts\d18_run_s2_preflight_micro_smoke.py" --project-root "$ProjectRoot" --config "$ConfigPath"
if ($LASTEXITCODE -ne 0) {
    throw "D18-S2 preflight/micro-smoke stopped. Inspect D18_S2_PREFLIGHT_MICRO_SMOKE_OVERALL_SUMMARY.json."
}

Write-Host "Completed. Default output directory:"
Write-Host "E:\XJTU battery dataset\_gv1_cache\xjtu_d18_s2_preflight_micro_smoke"
