
param(
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55"
)
$ErrorActionPreference = "Stop"
$PredRoot = Join-Path $RunDir "eval_full_profiles\predictions"
$Scorecard = Join-Path $RunDir "D16_P5A_FINAL_SCORECARD.json"
$Metrics = Join-Path $RunDir "eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv"
$Routing = Join-Path $RunDir "eval_full_profiles\D16_P5A_ROUTING_TABLE.csv"
Write-Host "RunDir: $RunDir"
Write-Host "PredRoot: $PredRoot"
if (Test-Path $PredRoot) {
  $count = (Get-ChildItem $PredRoot -Filter *.npz -File | Measure-Object).Count
  Write-Host "prediction npz count: $count" -ForegroundColor Cyan
} else {
  Write-Host "prediction root missing" -ForegroundColor Red
}
foreach ($p in @($Scorecard,$Metrics,$Routing)) {
  if (Test-Path $p) { Write-Host "FOUND: $p" -ForegroundColor Green }
  else { Write-Host "MISSING: $p" -ForegroundColor Red }
}
if (Test-Path $Scorecard) {
  Get-Content $Scorecard -TotalCount 80
}
