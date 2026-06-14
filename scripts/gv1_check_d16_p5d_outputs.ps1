param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$RunDir = ""
)
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $CacheRoot "xjtu_d16_p5d_train6_eval49_delta_gauge_FAST"
}
$ModelDir = Join-Path $RunDir "model_train6_delta_gauge_observation_physics"
$EvalDir = Join-Path $RunDir "eval_all55_vs_softlabels"
Write-Host "RunDir: $RunDir"
$paths = @(
  "$RunDir\D16_P5D_TRAIN6_EVAL49_MANIFEST.csv",
  "$ModelDir\D16_P5D_TRAINING_SUMMARY.json",
  "$ModelDir\D16_P5D_TRAIN_INPUT_AUDIT.json",
  "$ModelDir\model\best_with_state.pt",
  "$EvalDir\D16_P5D_FINAL_SCORECARD.json",
  "$EvalDir\D16_P5D_METRICS_BY_PROFILE.csv",
  "$EvalDir\D16_P5D_SPLIT_METRICS.csv",
  "$EvalDir\D16_P5D_BATCH_METRICS.csv",
  "$EvalDir\D16_P5D_PROTOCOL_METRICS.csv"
)
foreach ($p in $paths) {
  if (Test-Path $p) { Write-Host "FOUND: $p" -ForegroundColor Green } else { Write-Host "MISSING: $p" -ForegroundColor Yellow }
}
if (Test-Path "$EvalDir\D16_P5D_FINAL_SCORECARD.json") {
  Write-Host "`nScorecard preview:" -ForegroundColor Cyan
  Get-Content "$EvalDir\D16_P5D_FINAL_SCORECARD.json" -TotalCount 80
}
