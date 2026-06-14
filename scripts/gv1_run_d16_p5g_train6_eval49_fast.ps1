param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5g_train6_eval49_outlier_protocol_balanced_gauge_FAST",
  [string]$Config = "configs\d16_p5g_outlier_protocol_balanced_gauge_config.json",
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

Write-Host "[D16-P5G FAST] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5G FAST] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5G FAST] RunDir=$RunDir"
Write-Host "[D16-P5G FAST] Config=$Config"
Write-Host "[D16-P5G FAST] Device=$Device BatchSize=$BatchSize EvalBatchSize=$EvalBatchSize ChunkSize=$ChunkSize ValEvery=$ValEvery StepsPerEpoch=$StepsPerEpoch"
Write-Host "[D16-P5G FAST] WarmStartModelDir=$WarmStartModelDir NoWarmStart=$NoWarmStart"

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

if (-not $NoWarmStart -and -not [string]::IsNullOrWhiteSpace($WarmStartModelDir)) {
  $candidates = @(
    $WarmStartModelDir,
    "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5f_900epochs_LOCKED_current_best",
    "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5d_train6_eval49_delta_gauge_FAST\model_train6_delta_gauge_observation_physics",
    "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5d_800epochs_LOCKED_current_candidate"
  )
  $selectedWarm = ""
  foreach ($cand in $candidates) {
    if ([string]::IsNullOrWhiteSpace($cand)) { continue }
    $ck1 = Join-Path $cand "model\best_with_state.pt"
    $ck2 = Join-Path $cand "best_with_state.pt"
    if ((Test-Path $ck1) -or (Test-Path $ck2)) {
      $selectedWarm = $cand
      break
    }
  }
  if (-not [string]::IsNullOrWhiteSpace($selectedWarm)) {
    if ($selectedWarm -ne $WarmStartModelDir) {
      Write-Host "[D16-P5G FAST] Warm-start default missing; selected fallback: $selectedWarm" -ForegroundColor Yellow
    }
    $WarmStartModelDir = $selectedWarm
  } else {
    Write-Host "[D16-P5G FAST] WARNING: no P5F/P5D warm-start checkpoint found; training will start from random init unless -NoWarmStart is set." -ForegroundColor Yellow
  }
}

Write-Host "[D16-P5G FAST] Compile package scripts"
python -m py_compile scripts\gv1_d16_p5g_build_manifest.py scripts\gv1_d16_p5g_train6_outlier_protocol_balanced_gauge_fast.py scripts\gv1_d16_p5g_eval55_vs_softlabels.py

$Manifest = Join-Path $RunDir "D16_P5G_TRAIN6_EVAL49_MANIFEST.csv"
$ManifestSummary = Join-Path $RunDir "D16_P5G_TRAIN6_EVAL49_MANIFEST_SUMMARY.json"
$ModelDir = Join-Path $RunDir "model_train6_outlier_protocol_balanced_gauge_observation_physics"
$EvalDir = Join-Path $RunDir "eval_all55_vs_softlabels"

Write-Host "[D16-P5G FAST] Build train6/eval49 manifest"
$manifestArgs = @(
  "scripts\gv1_d16_p5g_build_manifest.py",
  "--softlabel-root", $SoftlabelRoot,
  "--out-csv", $Manifest,
  "--out-json", $ManifestSummary,
  "--config", $Config
)
if ($AllowOverwrite) { $manifestArgs += "--allow-overwrite" }
python @manifestArgs

if ($BuildManifestOnly) {
  Write-Host "[D16-P5G FAST] BuildManifestOnly set; stopping after manifest."
  exit 0
}

if (-not $EvalOnly) {
  Write-Host "[D16-P5G FAST] Train 6-cell outlier-protocol-balanced-gauge observation-physics model"
  $trainArgs = @(
    "scripts\gv1_d16_p5g_train6_outlier_protocol_balanced_gauge_fast.py",
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
  Write-Host "[D16-P5G FAST] Evaluate all55 vs P2Dlite-RG soft labels"
  $evalArgs = @(
    "scripts\gv1_d16_p5g_eval55_vs_softlabels.py",
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

Write-Host "[D16-P5G FAST] DONE"
Write-Host "Manifest: $Manifest"
Write-Host "Training summary: $(Join-Path $ModelDir 'D16_P5G_TRAINING_SUMMARY.json')"
Write-Host "Final scorecard: $(Join-Path $EvalDir 'D16_P5G_FINAL_SCORECARD.json')"
