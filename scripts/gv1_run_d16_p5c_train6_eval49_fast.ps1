param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5c_train6_eval49_theta_anchor_FAST",
  [string]$Config = "configs\d16_p5c_theta_anchor_config.json",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 131072,
  [int]$EvalBatchSize = 65536,
  [int]$ChunkSize = 200000,
  [int]$Epochs = 0,
  [int]$ValEvery = 10,
  [int]$StepsPerEpoch = 0,
  [switch]$AllowOverwrite,
  [switch]$BuildManifestOnly,
  [switch]$TrainOnly,
  [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[D16-P5C FAST] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5C FAST] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5C FAST] RunDir=$RunDir"
Write-Host "[D16-P5C FAST] Config=$Config"
Write-Host "[D16-P5C FAST] Device=$Device BatchSize=$BatchSize EvalBatchSize=$EvalBatchSize ChunkSize=$ChunkSize ValEvery=$ValEvery StepsPerEpoch=$StepsPerEpoch"

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

Write-Host "[D16-P5C FAST] Compile package scripts"
python -m py_compile scripts\gv1_d16_p5c_build_manifest.py scripts\gv1_d16_p5c_train6_theta_anchor_fast.py scripts\gv1_d16_p5c_eval55_vs_softlabels.py

$Manifest = Join-Path $RunDir "D16_P5C_TRAIN6_EVAL49_MANIFEST.csv"
$ManifestSummary = Join-Path $RunDir "D16_P5C_TRAIN6_EVAL49_MANIFEST_SUMMARY.json"
$ModelDir = Join-Path $RunDir "model_train6_theta_anchor_observation_physics"
$EvalDir = Join-Path $RunDir "eval_all55_vs_softlabels"

Write-Host "[D16-P5C FAST] Build train6/eval49 manifest"
$manifestArgs = @(
  "scripts\gv1_d16_p5c_build_manifest.py",
  "--softlabel-root", $SoftlabelRoot,
  "--out-csv", $Manifest,
  "--out-json", $ManifestSummary,
  "--config", $Config
)
if ($AllowOverwrite) { $manifestArgs += "--allow-overwrite" }
python @manifestArgs

if ($BuildManifestOnly) {
  Write-Host "[D16-P5C FAST] BuildManifestOnly set; stopping after manifest."
  exit 0
}

if (-not $EvalOnly) {
  Write-Host "[D16-P5C FAST] Train 6-cell theta-anchor observation-physics model"
  $trainArgs = @(
    "scripts\gv1_d16_p5c_train6_theta_anchor_fast.py",
    "--manifest", $Manifest,
    "--out-dir", $ModelDir,
    "--config", $Config,
    "--device", $Device,
    "--batch-size", $BatchSize,
    "--val-every", $ValEvery,
    "--steps-per-epoch", $StepsPerEpoch
  )
  if ($Epochs -gt 0) { $trainArgs += @("--epochs", $Epochs) }
  if ($AllowOverwrite) { $trainArgs += "--allow-overwrite" }
  python @trainArgs
}

if (-not $TrainOnly) {
  Write-Host "[D16-P5C FAST] Evaluate all55 vs P2Dlite-RG soft labels"
  $evalArgs = @(
    "scripts\gv1_d16_p5c_eval55_vs_softlabels.py",
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

Write-Host "[D16-P5C FAST] DONE"
Write-Host "Manifest: $Manifest"
Write-Host "Training summary: $(Join-Path $ModelDir 'D16_P5C_TRAINING_SUMMARY.json')"
Write-Host "Final scorecard: $(Join-Path $EvalDir 'D16_P5C_FINAL_SCORECARD.json')"
