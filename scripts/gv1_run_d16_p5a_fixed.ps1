param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$SoftlabelDir = "",
  [string]$ModelDir = "auto",
  [string]$RunDir = "",
  [string]$Config = "configs\d15_p2_precision_benchmark_config.json",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 65536,
  [int]$EvalStride = 1,
  [ValidateSet("raw", "projected")]
  [string]$PrimaryMode = "projected",
  [switch]$AllowOverwrite,
  [int]$LimitCells = 0,
  [switch]$NoInternalAudit
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($SoftlabelDir)) {
  $SoftlabelDir = Join-Path $CacheRoot "xjtu_softlabels_p2dlite_rg_v1_D15_ALL55_FINAL"
}
if ([string]::IsNullOrWhiteSpace($ModelDir)) {
  $ModelDir = "auto"
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $CacheRoot "xjtu_d16_p5a_FIXED_D15P2_existing_on_ALL55"
}

Write-Host "[D16-P5A fixed v3] ProjectRoot=$ProjectRoot" -ForegroundColor Cyan
Write-Host "[D16-P5A fixed v3] CacheRoot=$CacheRoot" -ForegroundColor Cyan
Write-Host "[D16-P5A fixed v3] SoftlabelDir=$SoftlabelDir" -ForegroundColor Cyan
Write-Host "[D16-P5A fixed v3] ModelDir=$ModelDir" -ForegroundColor Cyan
Write-Host "[D16-P5A fixed v3] RunDir=$RunDir" -ForegroundColor Cyan
Write-Host "[D16-P5A fixed v3] PrimaryMode=$PrimaryMode" -ForegroundColor Cyan

if (-not (Test-Path $SoftlabelDir)) { throw "SoftlabelDir not found: $SoftlabelDir" }
if (-not (Test-Path $CacheRoot)) { throw "CacheRoot not found: $CacheRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }
if (-not (Test-Path "scripts\gv1_d16_p5a_existing_transfer_eval_fixed.py")) { throw "Missing fixed runner script under scripts\. Did you copy the zip into project root?" }

# IMPORTANT: ModelDir is no longer hard-failed here. If it is missing or set to auto,
# the Python runner auto-discovers compatible D15 RG NN checkpoints under CacheRoot and ProjectRoot.
if (($ModelDir -ne "auto") -and (-not (Test-Path $ModelDir))) {
  Write-Host "[D16-P5A fixed v3] Requested ModelDir does not exist; switching to auto-discovery: $ModelDir" -ForegroundColor Yellow
  $ModelDir = "auto"
}

Write-Host "[D16-P5A fixed v3] compile fixed runner" -ForegroundColor Green
python -m py_compile scripts\gv1_d16_p5a_existing_transfer_eval_fixed.py

$argsList = @(
  "scripts\gv1_d16_p5a_existing_transfer_eval_fixed.py",
  "--softlabel-dir", $SoftlabelDir,
  "--cache-root", $CacheRoot,
  "--model-dir", $ModelDir,
  "--run-dir", $RunDir,
  "--config", $Config,
  "--device", $Device,
  "--batch-size", "$BatchSize",
  "--eval-stride", "$EvalStride",
  "--primary-mode", $PrimaryMode
)

if ($AllowOverwrite) { $argsList += "--allow-overwrite" }
if ($LimitCells -gt 0) { $argsList += @("--limit-cells", "$LimitCells") }
if ($NoInternalAudit) { $argsList += "--no-internal-audit" }

Write-Host "[D16-P5A fixed v3] running:" -ForegroundColor Green
Write-Host ("python " + ($argsList -join " "))
python @argsList

Write-Host "[D16-P5A fixed v3] DONE" -ForegroundColor Cyan
Write-Host "Model discovery: $RunDir\D16_P5A_MODEL_DISCOVERY.json" -ForegroundColor Cyan
Write-Host "Scorecard: $RunDir\D16_P5A_FIXED_SCORECARD.json" -ForegroundColor Cyan
Write-Host "Eval summary: $RunDir\eval_full_profiles\D16_P5A_EVAL_SUMMARY.json" -ForegroundColor Cyan
Write-Host "Primary predictions for D15 audit: $RunDir\eval_full_profiles\predictions" -ForegroundColor Cyan
Write-Host "Raw/projected metrics: $RunDir\eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv" -ForegroundColor Cyan
Write-Host "Precision audit summary: $RunDir\precision_audit\D15_P2_PRECISION_AUDIT_SUMMARY.json" -ForegroundColor Cyan
