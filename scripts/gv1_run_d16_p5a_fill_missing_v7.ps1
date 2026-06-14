param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$SoftlabelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55",
  [string]$ModelDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p2_rg_precision_benchmark",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 32768,
  [int]$ChunkSize = 200000,
  [Nullable[int]]$LimitMissing = $null,
  [switch]$AllowOverwrite,
  [switch]$NoCompress,
  [string]$PrimaryMode = "projected"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[D16-P5A v7] ProjectRoot=$ProjectRoot"
Write-Host "[D16-P5A v7] SoftlabelDir=$SoftlabelDir"
Write-Host "[D16-P5A v7] RunDir=$RunDir"
Write-Host "[D16-P5A v7] ModelDir=$ModelDir"
Write-Host "[D16-P5A v7] Device=$Device BatchSize=$BatchSize ChunkSize=$ChunkSize"
Write-Host "[D16-P5A v7] PrimaryMode=$PrimaryMode"

if (-not (Test-Path $ProjectRoot)) { throw "ProjectRoot not found: $ProjectRoot" }
if (-not (Test-Path $SoftlabelDir)) { throw "SoftlabelDir not found: $SoftlabelDir" }
if (-not (Test-Path $ModelDir)) { throw "ModelDir not found: $ModelDir" }

$ckpt1 = Join-Path $ModelDir "model\best_with_state.pt"
$ckpt2 = Join-Path $ModelDir "best_with_state.pt"
if (-not ((Test-Path $ckpt1) -or (Test-Path $ckpt2))) {
  throw "D15-compatible checkpoint not found. Expected $ckpt1 or $ckpt2"
}

Push-Location $ProjectRoot
try {
  $args = @(
    "scripts\gv1_d16_p5a_fill_missing_fullprofile_v7.py",
    "--softlabel-dir", $SoftlabelDir,
    "--run-dir", $RunDir,
    "--model-dir", $ModelDir,
    "--device", $Device,
    "--batch-size", "$BatchSize",
    "--chunk-size", "$ChunkSize",
    "--primary-mode", $PrimaryMode
  )
  if ($LimitMissing -ne $null) { $args += @("--limit-missing", "$LimitMissing") }
  if ($AllowOverwrite) { $args += "--allow-overwrite" }
  if ($NoCompress) { $args += "--no-compress" }

  Write-Host "[D16-P5A v7] Running Python compact full-profile metrics fill-missing evaluator..."
  & python @args
  $code = $LASTEXITCODE
  if ($code -ne 0) { throw "Python evaluator failed with exit code $code" }
} finally {
  Pop-Location
}

Write-Host "[D16-P5A v7] DONE"
Write-Host "Predictions: $(Join-Path $RunDir 'eval_full_profiles\predictions')"
Write-Host "Scorecard: $(Join-Path $RunDir 'D16_P5A_V7_FINAL_SCORECARD.json')"
