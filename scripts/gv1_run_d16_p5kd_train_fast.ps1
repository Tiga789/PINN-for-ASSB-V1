param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kd_train10_generator_aligned_hard_cbar_ocp_FAST",
  [string]$Config = "configs\d16_p5kd_generator_aligned_hard_cbar_ocp_config.json",
  [string]$TrainSet = "D_train10_prior_balanced",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 131072,
  [int]$EvalBatchSize = 65536,
  [int]$ChunkSize = 200000,
  [int]$Epochs = 0,
  [int]$ValEvery = 10,
  [int]$StepsPerEpoch = 0,
  [string]$WarmStartModelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST\model_train6_balanced_gauge_observation_physics",
  [string]$MmapCacheRoot = "E:\XJTU battery dataset\_gv1_cache\_p5kd_eval_mmap_cache",
  [switch]$NoWarmStart,
  [switch]$AllowOverwrite,
  [switch]$BuildManifestOnly,
  [switch]$TrainOnly,
  [switch]$EvalOnly,
  [int]$LimitProfiles = 0
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[D16-P5K-D FAST] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-D FAST] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K-D FAST] RunDir=$RunDir"
Write-Host "[D16-P5K-D FAST] Config=$Config"
Write-Host "[D16-P5K-D FAST] TrainSet=$TrainSet"
Write-Host "[D16-P5K-D FAST] Device=$Device BatchSize=$BatchSize EvalBatchSize=$EvalBatchSize ChunkSize=$ChunkSize ValEvery=$ValEvery StepsPerEpoch=$StepsPerEpoch LimitProfiles=$LimitProfiles"
Write-Host "[D16-P5K-D FAST] WarmStartModelDir=$WarmStartModelDir NoWarmStart=$NoWarmStart"

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

Write-Host "[D16-P5K-D FAST] Compile package scripts"
python -m py_compile scripts\gv1_d16_p5kd_build_manifest.py scripts\gv1_d16_p5kd_train_generator_aligned_fast.py scripts\gv1_d16_p5kd_eval55_vs_softlabels_v3.py

$StageRunDir = Join-Path $RunDir $TrainSet
$Manifest = Join-Path $StageRunDir "D16_P5KD_${TrainSet}_MANIFEST.csv"
$ManifestSummary = Join-Path $StageRunDir "D16_P5KD_${TrainSet}_MANIFEST_SUMMARY.json"
$ModelDir = Join-Path $StageRunDir "model_generator_aligned_hard_cbar_ocp_residual"
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"

Write-Host "[D16-P5K-D FAST] Build manifest"
$manifestArgs = @(
  "scripts\gv1_d16_p5kd_build_manifest.py",
  "--softlabel-root", $SoftlabelRoot,
  "--out-csv", $Manifest,
  "--out-json", $ManifestSummary,
  "--config", $Config,
  "--train-set", $TrainSet
)
if ($AllowOverwrite) { $manifestArgs += "--allow-overwrite" }
python @manifestArgs

if ($BuildManifestOnly) {
  Write-Host "[D16-P5K-D FAST] BuildManifestOnly set; stopping after manifest."
  exit 0
}

if (-not $EvalOnly) {
  Write-Host "[D16-P5K-D FAST] Train generator-aligned hard-cbar/OCP residual model"
  $trainArgs = @(
    "scripts\gv1_d16_p5kd_train_generator_aligned_fast.py",
    "--manifest", $Manifest,
    "--out-dir", $ModelDir,
    "--config", $Config,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--val-every", $ValEvery,
    "--steps-per-epoch", $StepsPerEpoch
  )
  if (-not $NoWarmStart -and -not [string]::IsNullOrWhiteSpace($WarmStartModelDir)) { $trainArgs += @("--warm-start-model-dir", $WarmStartModelDir) }
  if ($NoWarmStart) { $trainArgs += "--no-warm-start" }
  if ($Epochs -gt 0) { $trainArgs += @("--epochs", $Epochs) }
  if ($AllowOverwrite) { $trainArgs += "--allow-overwrite" }
  python @trainArgs
}

if (-not $TrainOnly) {
  Write-Host "[D16-P5K-D FAST] Evaluate all55 vs P2Dlite-RG soft labels with exact R2"
  $evalArgs = @(
    "scripts\gv1_d16_p5kd_eval55_vs_softlabels_v3.py",
    "--manifest", $Manifest,
    "--model-dir", $ModelDir,
    "--out-dir", $EvalDir,
    "--config", $Config,
    "--device", $Device,
    "--batch-size", $EvalBatchSize,
    "--chunk-size", $ChunkSize,
    "--mmap-cache-root", $MmapCacheRoot,
    "--softlabel-root", $SoftlabelRoot
  )
  if ($LimitProfiles -gt 0) { $evalArgs += @("--limit-profiles", $LimitProfiles) }
  if ($AllowOverwrite) { $evalArgs += "--allow-overwrite" }
  python @evalArgs
}

Write-Host "[D16-P5K-D FAST] DONE"
Write-Host "Manifest: $Manifest"
Write-Host "Training summary: $(Join-Path $ModelDir 'D16_P5KD_TRAINING_SUMMARY.json')"
Write-Host "Final scorecard: $(Join-Path $EvalDir 'D16_P5KD_FINAL_SCORECARD.json')"
