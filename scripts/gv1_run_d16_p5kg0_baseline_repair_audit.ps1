param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$TrainSet = "F_train12_profile_theta0",
  [string]$P5KFStageRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kf_train12_profile_theta0_hard_cbar_FAST",
  [string]$MiniEvidenceDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kf_1300epochs_MINI_EVIDENCE",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg0_baseline_repair_audit",
  [string]$MmapCacheRoot = "E:\XJTU battery dataset\_gv1_cache\_p5kg0_baseline_repair_mmap_cache",
  [string]$Models = "P5K-C,P5K-F",
  [int]$LimitProfiles = 0,
  [int]$ChunkSize = 200000,
  [switch]$AllowOverwrite,
  [switch]$KeepMmapCache,
  [switch]$NoTheta0Oracle
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$StageRunDir = Join-Path $P5KFStageRoot $TrainSet
$Manifest = Join-Path $StageRunDir ("D16_P5KF_{0}_MANIFEST.csv" -f $TrainSet)
$MiniManifest = Join-Path $MiniEvidenceDir ("D16_P5KF_{0}_MANIFEST.csv" -f $TrainSet)
$MiniSummary = Join-Path $MiniEvidenceDir ("D16_P5KF_{0}_MANIFEST_SUMMARY.json" -f $TrainSet)
$ManifestSummary = Join-Path $StageRunDir ("D16_P5KF_{0}_MANIFEST_SUMMARY.json" -f $TrainSet)

Write-Host "[D16-P5K-G0 baseline repair audit] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-G0 baseline repair audit] CacheRoot=$CacheRoot"
Write-Host "[D16-P5K-G0 baseline repair audit] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K-G0 baseline repair audit] StageRunDir=$StageRunDir"
Write-Host "[D16-P5K-G0 baseline repair audit] OutDir=$OutDir"
Write-Host "[D16-P5K-G0 baseline repair audit] MmapCacheRoot=$MmapCacheRoot"
Write-Host "[D16-P5K-G0 baseline repair audit] Models=$Models LimitProfiles=$LimitProfiles ChunkSize=$ChunkSize"

if (-not (Test-Path $Manifest)) {
  if (Test-Path $MiniManifest) {
    Write-Host "[D16-P5K-G0 baseline repair audit] Manifest missing in StageRunDir; restoring from MiniEvidenceDir."
    New-Item -ItemType Directory -Force $StageRunDir | Out-Null
    Copy-Item $MiniManifest $Manifest -Force
    if (Test-Path $MiniSummary) { Copy-Item $MiniSummary $ManifestSummary -Force }
  }
}

if (-not (Test-Path $Manifest)) {
  throw "Manifest not found: $Manifest. Rebuild with scripts\gv1_run_d16_p5kf_train_fast.ps1 -BuildManifestOnly -TrainSet $TrainSet or restore MINI_EVIDENCE."
}
if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path "scripts\gv1_d16_p5kg0_baseline_repair_audit.py")) { throw "Missing script: scripts\gv1_d16_p5kg0_baseline_repair_audit.py" }

if ($AllowOverwrite -and (Test-Path $OutDir)) {
  Write-Host "[D16-P5K-G0 baseline repair audit] Removing old OutDir=$OutDir"
  Remove-Item $OutDir -Recurse -Force -ErrorAction SilentlyContinue
}

$argsList = @(
  "scripts\gv1_d16_p5kg0_baseline_repair_audit.py",
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
if (-not $KeepMmapCache) { $argsList += "--cleanup-profile-cache" }
if ($NoTheta0Oracle) { $argsList += "--no-theta0-oracle" }

Write-Host "[D16-P5K-G0 baseline repair audit] Running Python audit..."
python @argsList
$code = $LASTEXITCODE
Write-Host "[D16-P5K-G0 baseline repair audit] Python exit code=$code"
Write-Host "[D16-P5K-G0 baseline repair audit] Report: $OutDir\D16_P5KG0_BASELINE_REPAIR_AUDIT_REPORT.md"
exit $code
