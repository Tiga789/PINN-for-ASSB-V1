param(
  [switch]$AllowOverwrite,
  [string]$ByProfile = "",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg4_exact_array_audit",
  [string]$MmapCacheRoot = "E:\XJTU battery dataset\_gv1_cache\_p5kg4_exact_array_mmap_cache",
  [int]$ChunkSize = 200000,
  [int]$LimitProfiles = 0
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
Write-Host "[D16-P5K-G4] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-G4] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K-G4] OutDir=$OutDir"
Write-Host "[D16-P5K-G4] MmapCacheRoot=$MmapCacheRoot"
Write-Host "[D16-P5K-G4] ByProfile=$ByProfile"
Write-Host "[D16-P5K-G4] ChunkSize=$ChunkSize LimitProfiles=$LimitProfiles"

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }

$argsList = @(
  "scripts\gv1_d16_p5kg4_exact_array_audit.py",
  "--softlabel-root", $SoftlabelRoot,
  "--out-dir", $OutDir,
  "--mmap-cache-root", $MmapCacheRoot,
  "--chunk-size", "$ChunkSize"
)
if ($ByProfile -ne "") { $argsList += @("--by-profile", $ByProfile) }
if ($LimitProfiles -gt 0) { $argsList += @("--limit-profiles", "$LimitProfiles") }
if ($AllowOverwrite) { $argsList += "--allow-overwrite" }

python @argsList
if ($LASTEXITCODE -ne 0) { throw "P5K-G4 exact array audit failed with exit code $LASTEXITCODE" }
Write-Host "[D16-P5K-G4] DONE"
Write-Host "Report: $OutDir\D16_P5KG4_EXACT_ARRAY_AUDIT_REPORT.md"
