param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg_rulev2_strict_gate_FAST",
  [string]$Config = "configs\d16_p5kg_rulev2_strict_gate_config.json",
  [ValidateSet("G_train12_rulev2_strict")]
  [string]$TrainSet = "G_train12_rulev2_strict",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 131072,
  [int]$EvalBatchSize = 65536,
  [int]$ChunkSize = 200000,
  [int]$Epochs = 0,
  [int]$ValEvery = 10,
  [int]$StepsPerEpoch = 0,
  [int]$LimitProfiles = 0,
  [string]$WarmStartModelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST\C_train10\model_hard_cbar_ocp_residual",
  [string]$MmapCacheRoot = "E:\XJTU battery dataset\_gv1_cache\_p5kg_eval_mmap_cache",
  [switch]$NoWarmStart,
  [switch]$AllowOverwrite,
  [switch]$BuildManifestOnly,
  [switch]$BaselineOnlyAuditOnly,
  [switch]$SkipBaselineOnlyAudit,
  [switch]$TrainOnly,
  [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[D16-P5K-G FAST] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K-G FAST] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K-G FAST] RunDir=$RunDir"
Write-Host "[D16-P5K-G FAST] Config=$Config"
Write-Host "[D16-P5K-G FAST] TrainSet=$TrainSet"
Write-Host "[D16-P5K-G FAST] Device=$Device BatchSize=$BatchSize EvalBatchSize=$EvalBatchSize ChunkSize=$ChunkSize ValEvery=$ValEvery StepsPerEpoch=$StepsPerEpoch LimitProfiles=$LimitProfiles"
Write-Host "[D16-P5K-G FAST] WarmStartModelDir=$WarmStartModelDir NoWarmStart=$NoWarmStart"

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

Write-Host "[D16-P5K-G FAST] Compile package scripts"
python -m py_compile scripts\gv1_d16_p5kg_build_manifest.py scripts\gv1_d16_p5kg_train_rulev2_strict_fast.py scripts\gv1_d16_p5kg_eval55_vs_softlabels_v3.py scripts\gv1_d16_p5kg_baseline_only_audit.py

$StageRunDir = Join-Path $RunDir $TrainSet
$Manifest = Join-Path $StageRunDir "D16_P5KG_${TrainSet}_MANIFEST.csv"
$ManifestSummary = Join-Path $StageRunDir "D16_P5KG_${TrainSet}_MANIFEST_SUMMARY.json"
$ModelDir = Join-Path $StageRunDir "model_rulev2_strict_gate_hard_cbar_ocp_residual"
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"
$BaselineDir = Join-Path $StageRunDir "baseline_only_preflight"

Write-Host "[D16-P5K-G FAST] Build manifest"
$manifestArgs = @(
  "scripts\gv1_d16_p5kg_build_manifest.py",
  "--softlabel-root", $SoftlabelRoot,
  "--out-csv", $Manifest,
  "--out-json", $ManifestSummary,
  "--config", $Config,
  "--train-set", $TrainSet
)
if ($AllowOverwrite) { $manifestArgs += "--allow-overwrite" }
python @manifestArgs

if ($BuildManifestOnly) {
  Write-Host "[D16-P5K-G FAST] BuildManifestOnly set; stopping after manifest."
  exit 0
}

if (-not $SkipBaselineOnlyAudit -and -not $EvalOnly) {
  Write-Host "[D16-P5K-G FAST] Baseline-only exact-R2 preflight"
  $baseArgs = @(
    "scripts\gv1_d16_p5kg_baseline_only_audit.py",
    "--manifest", $Manifest,
    "--out-dir", $BaselineDir,
    "--config", $Config,
    "--softlabel-root", $SoftlabelRoot,
    "--mmap-cache-root", (Join-Path $CacheRoot "_p5kg_baseline_mmap_cache"),
    "--chunk-size", $ChunkSize
  )
  if ($LimitProfiles -gt 0) { $baseArgs += @("--limit-profiles", $LimitProfiles) }
  if ($AllowOverwrite) { $baseArgs += "--allow-overwrite" }
  python @baseArgs
}

if ($BaselineOnlyAuditOnly) {
  Write-Host "[D16-P5K-G FAST] BaselineOnlyAuditOnly set; stopping after baseline preflight."
  exit 0
}

if (-not $EvalOnly) {
  Write-Host "[D16-P5K-G FAST] Train profile-theta0 hard-cbar/OCP residual model"
  $trainArgs = @(
    "scripts\gv1_d16_p5kg_train_rulev2_strict_fast.py",
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
  Write-Host "[D16-P5K-G FAST] Evaluate all55 vs P2Dlite-RG soft labels with explicit v3-safe evaluator"
  $evalArgs = @(
    "scripts\gv1_d16_p5kg_eval55_vs_softlabels_v3.py",
    "--manifest", $Manifest,
    "--model-dir", $ModelDir,
    "--out-dir", $EvalDir,
    "--config", $Config,
    "--device", $Device,
    "--batch-size", $EvalBatchSize,
    "--chunk-size", $ChunkSize,
    "--softlabel-root", $SoftlabelRoot,
    "--mmap-cache-root", $MmapCacheRoot
  )
  if ($LimitProfiles -gt 0) { $evalArgs += @("--limit-profiles", $LimitProfiles) }
  if ($AllowOverwrite) { $evalArgs += "--allow-overwrite" }
  python @evalArgs
}

Write-Host "[D16-P5K-G FAST] DONE"
Write-Host "Manifest: $Manifest"
Write-Host "Baseline preflight: $(Join-Path $BaselineDir 'D16_P5KG_BASELINE_ONLY_SCORECARD.json')"
Write-Host "Training summary: $(Join-Path $ModelDir 'D16_P5KG_TRAINING_SUMMARY.json')"
Write-Host "Final scorecard: $(Join-Path $EvalDir 'D16_P5KG_FINAL_SCORECARD.json')"
