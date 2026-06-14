param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "",
  [string]$RunDir = "",
  [string]$Config = "configs\d16_p5b_train6_eval49_config.json",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 65536,
  [int]$EvalBatchSize = 65536,
  [int]$ChunkSize = 200000,
  [int]$Epochs = 0,
  [switch]$AllowOverwrite,
  [switch]$BuildManifestOnly,
  [switch]$TrainOnly,
  [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($SoftlabelRoot)) {
  $SoftlabelRoot = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL"
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $CacheRoot "xjtu_d16_p5b_train6_eval49_observation_physics"
}
$ManifestCsv = Join-Path $RunDir "D16_P5B_TRAIN6_EVAL49_MANIFEST.csv"
$ManifestJson = Join-Path $RunDir "D16_P5B_TRAIN6_EVAL49_MANIFEST_SUMMARY.json"
$TrainDir = Join-Path $RunDir "model_train6_observation_physics"
$EvalDir = Join-Path $RunDir "eval_all55_vs_softlabels"
$MmapCache = Join-Path $RunDir "mmap_cache"

Write-Host "[D16-P5B] ProjectRoot=$ProjectRoot" -ForegroundColor Cyan
Write-Host "[D16-P5B] SoftlabelRoot=$SoftlabelRoot" -ForegroundColor Cyan
Write-Host "[D16-P5B] RunDir=$RunDir" -ForegroundColor Cyan
Write-Host "[D16-P5B] Config=$Config" -ForegroundColor Cyan
Write-Host "[D16-P5B] Device=$Device BatchSize=$BatchSize EvalBatchSize=$EvalBatchSize ChunkSize=$ChunkSize" -ForegroundColor Cyan

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
New-Item -ItemType Directory -Force $RunDir | Out-Null

Write-Host "[D16-P5B] Compile package scripts" -ForegroundColor Green
python -m py_compile scripts\gv1_d16_p5b_build_manifest.py scripts\gv1_d16_p5b_train6_observation_physics.py scripts\gv1_d16_p5b_eval55_vs_softlabels.py

Write-Host "[D16-P5B] Build train6/eval49 manifest" -ForegroundColor Green
$manifestArgs = @(
  "scripts\gv1_d16_p5b_build_manifest.py",
  "--softlabel-root", $SoftlabelRoot,
  "--config", $Config,
  "--out-csv", $ManifestCsv,
  "--out-json", $ManifestJson,
  "--allow-overwrite"
)
python @manifestArgs

if ($BuildManifestOnly) {
  Write-Host "[D16-P5B] BuildManifestOnly set; stopping after manifest." -ForegroundColor Yellow
  exit 0
}

if (-not $EvalOnly) {
  Write-Host "[D16-P5B] Train observation-physics model on 6 selected cells" -ForegroundColor Green
  $trainArgs = @(
    "scripts\gv1_d16_p5b_train6_observation_physics.py",
    "--manifest", $ManifestCsv,
    "--out-dir", $TrainDir,
    "--config", $Config,
    "--device", $Device,
    "--batch-size", "$BatchSize"
  )
  if ($AllowOverwrite) { $trainArgs += "--allow-overwrite" }
  if ($Epochs -gt 0) { $trainArgs += @("--epochs", "$Epochs") }
  python @trainArgs
}

if ($TrainOnly) {
  Write-Host "[D16-P5B] TrainOnly set; stopping after training." -ForegroundColor Yellow
  exit 0
}

Write-Host "[D16-P5B] Evaluate all55 vs P2Dlite-RG soft labels" -ForegroundColor Green
$evalArgs = @(
  "scripts\gv1_d16_p5b_eval55_vs_softlabels.py",
  "--manifest", $ManifestCsv,
  "--model-dir", $TrainDir,
  "--out-dir", $EvalDir,
  "--config", $Config,
  "--device", $Device,
  "--batch-size", "$EvalBatchSize",
  "--chunk-size", "$ChunkSize",
  "--mmap-cache-root", $MmapCache
)
if ($AllowOverwrite) { $evalArgs += "--allow-overwrite" }
python @evalArgs

Write-Host "[D16-P5B] DONE" -ForegroundColor Cyan
Write-Host "Manifest: $ManifestCsv" -ForegroundColor Cyan
Write-Host "Training summary: $TrainDir\D16_P5B_TRAINING_SUMMARY.json" -ForegroundColor Cyan
Write-Host "Final scorecard: $EvalDir\D16_P5B_FINAL_SCORECARD.json" -ForegroundColor Cyan
