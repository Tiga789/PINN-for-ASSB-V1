param(
    [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
    [string]$Manifest = "",
    [string]$Models = "P5B,P5D,P5E,P5F,P5G",
    [string]$OutputFile = "",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 65536,
    [int]$ChunkSize = 200000,
    [int]$LimitProfiles = 0,
    [switch]$KeepCache
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not $OutputFile -or $OutputFile.Trim() -eq "") {
    $OutputFile = Join-Path $CacheRoot "xjtu_d16_p5h_exact_r2_audit\D16_P5H_EXACT_R2_AUDIT_REPORT.md"
}

Write-Host "[D16-P5H] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5H] CacheRoot=$CacheRoot"
Write-Host "[D16-P5H] Models=$Models"
Write-Host "[D16-P5H] OutputFile=$OutputFile"
Write-Host "[D16-P5H] Device=$Device BatchSize=$BatchSize ChunkSize=$ChunkSize LimitProfiles=$LimitProfiles"

python -m py_compile scripts\gv1_d16_p5h_exact_r2_audit.py

$argsList = @(
    "scripts\gv1_d16_p5h_exact_r2_audit.py",
    "--cache-root", $CacheRoot,
    "--model", $Models,
    "--output-file", $OutputFile,
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--chunk-size", "$ChunkSize"
)
if ($Manifest -and $Manifest.Trim() -ne "") { $argsList += @("--manifest", $Manifest) }
if ($LimitProfiles -gt 0) { $argsList += @("--limit-profiles", "$LimitProfiles") }
if ($KeepCache) { $argsList += @("--keep-cache") }

& python @argsList
if ($LASTEXITCODE -ne 0) { throw "P5H exact-R2 audit failed with exit code $LASTEXITCODE" }

Write-Host "[D16-P5H] DONE"
Write-Host "Report: $OutputFile"
