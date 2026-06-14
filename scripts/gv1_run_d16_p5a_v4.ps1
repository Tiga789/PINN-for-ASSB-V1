
param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelDir = "",
  [string]$RunDir = "",
  [string]$ModelDir = "auto",
  [string]$Config = "configs\d15_p2_precision_benchmark_config.json",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 65536,
  [int]$LimitCells = 0,
  [ValidateSet("raw", "projected")]
  [string]$PrimaryMode = "projected",
  [switch]$AllowOverwrite,
  [switch]$NoAudit,
  [switch]$RebuildExistingP2IfMissing
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($SoftlabelDir)) {
  $SoftlabelDir = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL"
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $CacheRoot "xjtu_d16_p5a_V4_D15_existing_on_ALL55"
}

$P2Dir = Join-Path $CacheRoot "xjtu_d15_p2_rg_precision_benchmark"
$P2Checkpoint = Join-Path $P2Dir "model\best_with_state.pt"
$P1Dir = Join-Path $CacheRoot "xjtu_d15_p1_rg_closedset_nn_smoke"
$P1Checkpoint = Join-Path $P1Dir "model\best_with_state.pt"

Write-Host "[D16-P5A v4] ProjectRoot=$ProjectRoot" -ForegroundColor Cyan
Write-Host "[D16-P5A v4] SoftlabelDir=$SoftlabelDir" -ForegroundColor Cyan
Write-Host "[D16-P5A v4] RunDir=$RunDir" -ForegroundColor Cyan
Write-Host "[D16-P5A v4] ModelDir=$ModelDir" -ForegroundColor Cyan
Write-Host "[D16-P5A v4] PrimaryMode=$PrimaryMode" -ForegroundColor Cyan

if (-not (Test-Path $SoftlabelDir)) {
  throw "SoftlabelDir not found: $SoftlabelDir"
}

if ($ModelDir -eq "auto") {
  if (Test-Path $P2Checkpoint) {
    Write-Host "[D16-P5A v4] Found D15-P2 checkpoint: $P2Checkpoint" -ForegroundColor Green
  } elseif (Test-Path $P1Checkpoint) {
    Write-Host "[D16-P5A v4] D15-P2 missing; found D15-P1 checkpoint: $P1Checkpoint" -ForegroundColor Yellow
  } elseif ($RebuildExistingP2IfMissing) {
    $P0SoftlabelDir = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_d15p0_8cell"
    if (-not (Test-Path $P0SoftlabelDir)) {
      throw "Cannot rebuild D15-P2 because D15-P0 8-cell softlabel dir is missing: $P0SoftlabelDir"
    }
    Write-Host "[D16-P5A v4] Rebuilding D15-P2 existing benchmark from D15-P0 8-cell labels..." -ForegroundColor Yellow
    python scripts\d15_p2_train_rg_precision_benchmark.py `
      --softlabel-dir "$P0SoftlabelDir" `
      --out-dir "$P2Dir" `
      --config "$Config" `
      --device "$Device" `
      --allow-overwrite
    if (-not (Test-Path $P2Checkpoint)) {
      throw "D15-P2 rebuild finished but checkpoint still missing: $P2Checkpoint"
    }
  } else {
    Write-Host "[D16-P5A v4] No canonical D15-P2/P1 checkpoint found; Python will search recursively for best_with_state.pt only." -ForegroundColor Yellow
  }
}

$argsPy = @(
  "scripts\gv1_d16_p5a_existing_transfer_eval_v4.py",
  "--softlabel-dir", $SoftlabelDir,
  "--run-dir", $RunDir,
  "--model-dir", $ModelDir,
  "--cache-root", $CacheRoot,
  "--config", $Config,
  "--device", $Device,
  "--batch-size", "$BatchSize",
  "--primary-mode", $PrimaryMode
)
if ($AllowOverwrite) { $argsPy += "--allow-overwrite" }
if ($NoAudit) { $argsPy += "--no-audit" }
if ($LimitCells -gt 0) { $argsPy += @("--limit-cells", "$LimitCells") }

Write-Host "[D16-P5A v4] Running Python evaluator..." -ForegroundColor Green
python @argsPy

$PredRoot = Join-Path $RunDir "eval_full_profiles\predictions"
$Scorecard = Join-Path $RunDir "D16_P5A_FINAL_SCORECARD.json"
Write-Host "[D16-P5A v4] DONE" -ForegroundColor Cyan
Write-Host "Predictions: $PredRoot" -ForegroundColor Cyan
Write-Host "Final scorecard: $Scorecard" -ForegroundColor Cyan
