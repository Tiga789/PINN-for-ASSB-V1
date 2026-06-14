param(
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55"
)
$PredDir = Join-Path $RunDir "eval_full_profiles\predictions"
Write-Host "RunDir: $RunDir"
Write-Host "PredDir: $PredDir"
$count = 0
$size = 0.0
if (Test-Path $PredDir) {
  $files = Get-ChildItem $PredDir -Filter "*.npz" -File -ErrorAction SilentlyContinue
  $count = ($files | Measure-Object).Count
  $size = (($files | Measure-Object Length -Sum).Sum / 1GB)
}
Write-Host "prediction npz count: $count"
Write-Host ("prediction size GB: {0:N3}" -f $size)
$filesToCheck = @(
  (Join-Path $RunDir "D16_P5A_V7_FINAL_SCORECARD.json"),
  (Join-Path $RunDir "eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv"),
  (Join-Path $RunDir "eval_full_profiles\D16_P5A_BATCH_METRICS.csv"),
  (Join-Path $RunDir "eval_full_profiles\D16_P5A_PROTOCOL_METRICS.csv"),
  (Join-Path $RunDir "eval_full_profiles\D16_P5A_V7_ROUTING_TABLE.csv"),
  (Join-Path $RunDir "eval_full_profiles\D16_P5A_FAILURES.json")
)
foreach ($f in $filesToCheck) {
  if (Test-Path $f) { Write-Host "FOUND: $f" -ForegroundColor Green }
  else { Write-Host "MISSING: $f" -ForegroundColor Yellow }
}
if (Test-Path (Join-Path $RunDir "D16_P5A_V7_FINAL_SCORECARD.json")) {
  Write-Host "`nScorecard preview:" -ForegroundColor Cyan
  Get-Content (Join-Path $RunDir "D16_P5A_V7_FINAL_SCORECARD.json") -Raw | Select-String "operational_status|prediction_file_count_primary|profile_count_with_metrics|failure_count" -Context 0,1
}
