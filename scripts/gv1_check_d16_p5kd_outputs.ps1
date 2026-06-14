param(
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kd_train10_generator_aligned_hard_cbar_ocp_FAST",
  [string]$TrainSet = "D_train10_prior_balanced"
)
$StageRunDir = Join-Path $RunDir $TrainSet
$ModelDir = Join-Path $StageRunDir "model_generator_aligned_hard_cbar_ocp_residual"
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"
Write-Host "StageRunDir: $StageRunDir"
Write-Host "ModelDir: $ModelDir"
Write-Host "EvalDir: $EvalDir"
$files = @(
  (Join-Path $StageRunDir "D16_P5KD_${TrainSet}_MANIFEST.csv"),
  (Join-Path $StageRunDir "D16_P5KD_${TrainSet}_MANIFEST_SUMMARY.json"),
  (Join-Path $ModelDir "D16_P5KD_TRAINING_SUMMARY.json"),
  (Join-Path $ModelDir "D16_P5KD_TRAIN_INPUT_AUDIT.json"),
  (Join-Path $ModelDir "model\best_with_state.pt"),
  (Join-Path $EvalDir "D16_P5KD_FINAL_SCORECARD.json"),
  (Join-Path $EvalDir "D16_P5KD_METRICS_BY_PROFILE.csv"),
  (Join-Path $EvalDir "D16_P5KD_SPLIT_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5KD_BATCH_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5KD_PROTOCOL_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5KD_FAILURES.json")
)
foreach ($f in $files) {
  if (Test-Path $f) { $item=Get-Item $f; Write-Host "FOUND: $f | size=$($item.Length)" } else { Write-Host "MISSING: $f" }
}
$score = Join-Path $EvalDir "D16_P5KD_FINAL_SCORECARD.json"
if (Test-Path $score) {
  Write-Host "`nScorecard preview:" -ForegroundColor Cyan
  Get-Content $score -TotalCount 80
}
