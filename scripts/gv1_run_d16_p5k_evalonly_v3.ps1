param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST",
  [string]$Config = "configs\d16_p5k_hard_cbar_ocp_residual_config.json",
  [ValidateSet("A_train6", "B_train8", "C_train10")]
  [string]$TrainSet = "C_train10",
  [string]$Device = "cuda:0",
  [int]$EvalBatchSize = 65536,
  [int]$ChunkSize = 200000,
  [int]$LimitProfiles = 0,
  [string]$MmapCacheRoot = "E:\XJTU battery dataset\_gv1_cache\_p5k_eval_mmap_cache_v3",
  [switch]$AllowOverwrite
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "[D16-P5K eval v3] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5K eval v3] SoftlabelRoot=$SoftlabelRoot"
Write-Host "[D16-P5K eval v3] RunDir=$RunDir"
Write-Host "[D16-P5K eval v3] TrainSet=$TrainSet"
Write-Host "[D16-P5K eval v3] Device=$Device EvalBatchSize=$EvalBatchSize ChunkSize=$ChunkSize LimitProfiles=$LimitProfiles"
Write-Host "[D16-P5K eval v3] MmapCacheRoot=$MmapCacheRoot"

if (-not (Test-Path $SoftlabelRoot)) { throw "SoftlabelRoot not found: $SoftlabelRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

python -m py_compile scripts\gv1_d16_p5k_eval55_vs_softlabels_v3.py

$StageRunDir = Join-Path $RunDir $TrainSet
$Manifest = Join-Path $StageRunDir "D16_P5K_${TrainSet}_MANIFEST.csv"
$ManifestSummary = Join-Path $StageRunDir "D16_P5K_${TrainSet}_MANIFEST_SUMMARY.json"
$ModelDir = Join-Path $StageRunDir "model_hard_cbar_ocp_residual"
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"

if (-not (Test-Path (Join-Path $ModelDir "model\best_with_state.pt"))) {
  throw "P5K checkpoint missing. Do not eval before training: $(Join-Path $ModelDir 'model\best_with_state.pt')"
}

if (-not (Test-Path $Manifest)) {
  Write-Host "[D16-P5K eval v3] Manifest missing; rebuilding manifest only."
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
}

if ($AllowOverwrite -and (Test-Path $EvalDir)) {
  Write-Host "[D16-P5K eval v3] Removing old EvalDir=$EvalDir"
  Remove-Item $EvalDir -Recurse -Force -ErrorAction SilentlyContinue
}

$evalArgs = @(
  "scripts\gv1_d16_p5k_eval55_vs_softlabels_v3.py",
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

Write-Host "[D16-P5K eval v3] Running eval script explicitly: scripts\gv1_d16_p5k_eval55_vs_softlabels_v3.py"
python @evalArgs

Write-Host "[D16-P5K eval v3] DONE"
Write-Host "Scorecard: $(Join-Path $EvalDir 'D16_P5K_FINAL_SCORECARD.json')"
