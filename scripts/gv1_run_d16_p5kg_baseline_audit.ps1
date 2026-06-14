param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$TrainSet = "F_train12_profile_theta0",
  [string]$P5KFStageRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kf_train12_profile_theta0_hard_cbar_FAST",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg_baseline_noregression_audit",
  [string]$MmapCacheRoot = "E:\XJTU battery dataset\_gv1_cache\_p5kg_baseline_audit_mmap_cache",
  [string]$Models = "P5K-C,P5K-F",
  [int]$LimitProfiles = 0,
  [int]$ChunkSize = 200000,
  [switch]$AllowOverwrite,
  [switch]$KeepMmapCache
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

$StageRunDir = Join-Path $P5KFStageRoot $TrainSet
$Manifest = Join-Path $StageRunDir ("D16_P5KF_{0}_MANIFEST.csv" -f $TrainSet)

Write-Host "[D16-P5K-G baseline audit] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-G baseline audit] CacheRoot=$CacheRoot"
Write-Host "[D16-P5K-G baseline audit] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K-G baseline audit] Manifest=$Manifest"
Write-Host "[D16-P5K-G baseline audit] OutDir=$OutDir"
Write-Host "[D16-P5K-G baseline audit] MmapCacheRoot=$MmapCacheRoot"
Write-Host "[D16-P5K-G baseline audit] Models=$Models LimitProfiles=$LimitProfiles ChunkSize=$ChunkSize"

if (-not (Test-Path $Manifest)) { throw "Manifest not found: $Manifest" }
if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path "scripts\gv1_d16_p5kg_baseline_noregression_audit.py")) { throw "Missing script: scripts\gv1_d16_p5kg_baseline_noregression_audit.py" }

if ($AllowOverwrite -and (Test-Path $OutDir)) {
  Write-Host "[D16-P5K-G baseline audit] Removing old OutDir=$OutDir"
  Remove-Item $OutDir -Recurse -Force -ErrorAction SilentlyContinue
}

$cleanupFlag = "--cleanup-profile-cache"
if ($KeepMmapCache) { $cleanupFlag = "" }

$argsList = @(
  "scripts\gv1_d16_p5kg_baseline_noregression_audit.py",
  "--project-root", $ProjectRoot,
  "--manifest", $Manifest,
  "--softlabel-root", $SoftlabelRoot,
  "--out-dir", $OutDir,
  "--mmap-cache-root", $MmapCacheRoot,
  "--models", $Models,
  "--chunk-size", "$ChunkSize"
)
if ($LimitProfiles -gt 0) { $argsList += @("--limit-profiles", "$LimitProfiles") }
if ($AllowOverwrite) { $argsList += "--allow-overwrite" }
if ($cleanupFlag -ne "") { $argsList += $cleanupFlag }

Write-Host "[D16-P5K-G baseline audit] Running Python audit..."
python @argsList
$code = $LASTEXITCODE
Write-Host "[D16-P5K-G baseline audit] Python exit code=$code"

Write-Host "[D16-P5K-G baseline audit] Report: $OutDir\D16_P5KG_BASELINE_NOREGRESSION_AUDIT_REPORT.md"
exit $code
