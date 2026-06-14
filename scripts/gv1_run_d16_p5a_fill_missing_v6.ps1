param(
    [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
    [string]$SoftlabelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
    [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55",
    [string]$ModelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_rg_precision_benchmark",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 32768,
    [int]$ChunkSize = 200000,
    [int]$LimitMissing = 0,
    [switch]$NoCompress,
    [switch]$RecomputeExistingMetrics
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

Write-Host "[D16-P5A v6] ProjectRoot=$ProjectRoot" -ForegroundColor Cyan
Write-Host "[D16-P5A v6] SoftlabelDir=$SoftlabelDir" -ForegroundColor Cyan
Write-Host "[D16-P5A v6] RunDir=$RunDir" -ForegroundColor Cyan
Write-Host "[D16-P5A v6] ModelDir=$ModelDir" -ForegroundColor Cyan
Write-Host "[D16-P5A v6] Device=$Device BatchSize=$BatchSize ChunkSize=$ChunkSize" -ForegroundColor Cyan

$ckpt1 = Join-Path $ModelDir "model\best_with_state.pt"
$ckpt2 = Join-Path $ModelDir "best_with_state.pt"
if (-not ((Test-Path $ckpt1) -or (Test-Path $ckpt2))) {
    throw "D15 compatible checkpoint not found. Expected $ckpt1 or $ckpt2"
}

$PredDir = Join-Path $RunDir "eval_full_profiles\predictions"
New-Item -ItemType Directory -Force $PredDir | Out-Null

$argsList = @(
    "scripts\gv1_d16_p5a_fill_missing_fullprofile_v6.py",
    "--softlabel-dir", $SoftlabelDir,
    "--run-dir", $RunDir,
    "--model-dir", $ModelDir,
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--chunk-size", "$ChunkSize"
)

if ($LimitMissing -gt 0) {
    $argsList += @("--limit-missing", "$LimitMissing")
}
if ($NoCompress) {
    $argsList += "--no-compress"
}
if ($RecomputeExistingMetrics) {
    $argsList += "--recompute-existing-metrics"
}

python @argsList

Write-Host "[D16-P5A v6] DONE" -ForegroundColor Green
Write-Host "Predictions: $PredDir" -ForegroundColor Yellow
Write-Host "Scorecard: $(Join-Path $RunDir 'D16_P5A_V6_FINAL_SCORECARD.json')" -ForegroundColor Yellow
