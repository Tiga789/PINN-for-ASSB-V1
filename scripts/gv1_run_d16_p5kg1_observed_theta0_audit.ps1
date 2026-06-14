param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$Manifest = "",
  [string]$OutDir = "",
  [string]$MmapCacheRoot = "",
  [int]$LimitProfiles = 0,
  [int]$ChunkSize = 200000,
  [double]$RidgeAlpha = 0.01,
  [double]$ShiftClip = 0.55,
  [switch]$AllowOverwrite
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Manifest)) {
  $Manifest = Join-Path $CacheRoot "xjtu_d16_p5kf_train12_profile_theta0_hard_cbar_FAST\F_train12_profile_theta0\D16_P5KF_F_train12_profile_theta0_MANIFEST.csv"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = Join-Path $CacheRoot "xjtu_d16_p5kg1_observed_theta0_audit"
}
if ([string]::IsNullOrWhiteSpace($MmapCacheRoot)) {
  $MmapCacheRoot = Join-Path $CacheRoot "_p5kg1_observed_theta0_mmap_cache"
}

Write-Host "[D16-P5K-G1 observed theta0 audit] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-G1 observed theta0 audit] CacheRoot=$CacheRoot"
Write-Host "[D16-P5K-G1 observed theta0 audit] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K-G1 observed theta0 audit] Manifest=$Manifest"
Write-Host "[D16-P5K-G1 observed theta0 audit] OutDir=$OutDir"
Write-Host "[D16-P5K-G1 observed theta0 audit] MmapCacheRoot=$MmapCacheRoot"
Write-Host "[D16-P5K-G1 observed theta0 audit] LimitProfiles=$LimitProfiles ChunkSize=$ChunkSize RidgeAlpha=$RidgeAlpha"

if (-not (Test-Path $ProjectRoot)) { throw "ProjectRoot not found: $ProjectRoot" }
if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path $Manifest)) { throw "Manifest not found: $Manifest. Restore or regenerate the P5K-F manifest first." }

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force $OutDir | Out-Null
New-Item -ItemType Directory -Force $MmapCacheRoot | Out-Null

$pyArgs = @(
  "scripts\gv1_d16_p5kg1_observed_theta0_estimator_audit.py",
  "--project-root", $ProjectRoot,
  "--manifest", $Manifest,
  "--softlabel-root", $SoftlabelRoot,
  "--out-dir", $OutDir,
  "--mmap-cache-root", $MmapCacheRoot,
  "--chunk-size", "$ChunkSize",
  "--ridge-alpha", "$RidgeAlpha",
  "--shift-clip", "$ShiftClip",
  "--cleanup-profile-cache"
)
if ($LimitProfiles -gt 0) { $pyArgs += @("--limit-profiles", "$LimitProfiles") }
if ($AllowOverwrite) { $pyArgs += @("--allow-overwrite") }

python @pyArgs
$ec = $LASTEXITCODE
if ($ec -ne 0) {
  Write-Host "[D16-P5K-G1 observed theta0 audit] Python exited with code $ec"
  exit $ec
}

Write-Host "[D16-P5K-G1 observed theta0 audit] DONE"
Write-Host "Report: $(Join-Path $OutDir 'D16_P5KG1_OBSERVED_THETA0_AUDIT_REPORT.md')"
