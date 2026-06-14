param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST",
  [string]$Config = "configs\d16_p5k_hard_cbar_ocp_residual_config.json",
  [ValidateSet("A_train6", "B_train8", "C_train10")]
  [string]$TrainSet = "C_train10",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 131072,
  [int]$EvalBatchSize = 65536,
  [int]$ChunkSize = 200000,
  [int]$Epochs = 0,
  [int]$ValEvery = 10,
  [int]$StepsPerEpoch = 0,
  [string]$WarmStartModelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_train6_eval49_balanced_gauge_FAST\model_train6_balanced_gauge_observation_physics",
  [switch]$NoWarmStart,
  [switch]$AllowOverwrite,
  [switch]$BuildManifestOnly,
  [switch]$TrainOnly,
  [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[D16-P5K FAST] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K FAST] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K FAST] RunDir=$RunDir"
Write-Host "[D16-P5K FAST] Config=$Config"
Write-Host "[D16-P5K FAST] TrainSet=$TrainSet"
Write-Host "[D16-P5K FAST] Device=$Device BatchSize=$BatchSize EvalBatchSize=$EvalBatchSize ChunkSize=$ChunkSize ValEvery=$ValEvery StepsPerEpoch=$StepsPerEpoch"
Write-Host "[D16-P5K FAST] WarmStartModelDir=$WarmStartModelDir NoWarmStart=$NoWarmStart"

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

Write-Host "[D16-P5K FAST] Compile package scripts"
python -m py_compile scripts\gv1_d16_p5k_build_manifest.py scripts\gv1_d16_p5k_train10_hard_cbar_ocp_residual_fast.py scripts\gv1_d16_p5k_eval55_vs_softlabels.py

$StageRunDir = Join-Path $RunDir $TrainSet
$Manifest = Join-Path $StageRunDir "D16_P5K_${TrainSet}_MANIFEST.csv"
$ManifestSummary = Join-Path $StageRunDir "D16_P5K_${TrainSet}_MANIFEST_SUMMARY.json"
$ModelDir = Join-Path $StageRunDir "model_hard_cbar_ocp_residual"
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"

Write-Host "[D16-P5K FAST] Build manifest"
$manifestArgs = @(
  "scripts\gv1_d16_p5k_build_manifest.py",
  "--softlabel-root", $SoftlabelRoot,
  "--out-csv", $Manifest,
  "--out-json", $ManifestSummary,
  "--config", $Config,
  "--train-set", $TrainSet
)
if ($AllowOverwrite) { $manifestArgs += "--allow-overwrite" }
python @manifestArgs

if ($BuildManifestOnly) {
  Write-Host "[D16-P5K FAST] BuildManifestOnly set; stopping after manifest."
  exit 0
}

if (-not $EvalOnly) {
  Write-Host "[D16-P5K FAST] Train hard-cbar/OCP residual model"
  $trainArgs = @(
    "scripts\gv1_d16_p5k_train10_hard_cbar_ocp_residual_fast.py",
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
  Write-Host "[D16-P5K FAST] Evaluate all55 vs P2Dlite-RG soft labels"
  $evalArgs = @(
    "scripts\gv1_d16_p5k_eval55_vs_softlabels.py",
    "--manifest", $Manifest,
    "--model-dir", $ModelDir,
    "--out-dir", $EvalDir,
    "--config", $Config,
    "--device", $Device,
    "--batch-size", $EvalBatchSize,
    "--chunk-size", $ChunkSize
  )
  if ($AllowOverwrite) { $evalArgs += "--allow-overwrite" }
  python @evalArgs
}

Write-Host "[D16-P5K FAST] DONE"
Write-Host "Manifest: $Manifest"
Write-Host "Training summary: $(Join-Path $ModelDir 'D16_P5K_TRAINING_SUMMARY.json')"
Write-Host "Final scorecard: $(Join-Path $EvalDir 'D16_P5K_FINAL_SCORECARD.json')"
