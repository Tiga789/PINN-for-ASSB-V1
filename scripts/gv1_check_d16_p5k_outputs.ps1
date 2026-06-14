param(
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST",
  [ValidateSet("A_train6", "B_train8", "C_train10")]
  [string]$TrainSet = "C_train10"
)

$StageRunDir = Join-Path $RunDir $TrainSet
$ModelDir = Join-Path $StageRunDir "model_hard_cbar_ocp_residual"
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"

Write-Host "RunDir: $RunDir"
Write-Host "StageRunDir: $StageRunDir"

$paths = @(
  (Join-Path $StageRunDir "D16_P5K_${TrainSet}_MANIFEST.csv"),
  (Join-Path $StageRunDir "D16_P5K_${TrainSet}_MANIFEST_SUMMARY.json"),
  (Join-Path $ModelDir "D16_P5K_TRAINING_SUMMARY.json"),
  (Join-Path $ModelDir "D16_P5K_TRAIN_INPUT_AUDIT.json"),
  (Join-Path $ModelDir "model\best_with_state.pt"),
  (Join-Path $EvalDir "D16_P5K_FINAL_SCORECARD.json"),
  (Join-Path $EvalDir "D16_P5K_METRICS_BY_PROFILE.csv"),
  (Join-Path $EvalDir "D16_P5K_SPLIT_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5K_BATCH_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5K_PROTOCOL_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5K_FAILURES.json")
)

foreach ($p in $paths) {
  if (Test-Path $p) { Write-Host "FOUND: $p" } else { Write-Host "MISSING: $p" }
}

$manifest = Join-Path $StageRunDir "D16_P5K_${TrainSet}_MANIFEST.csv"
if (Test-Path $manifest) {
  $rows = Import-Csv $manifest
  $train = ($rows | Where-Object {$_.split -eq "train"}).Count
  $eval = ($rows | Where-Object {$_.split -eq "eval"}).Count
  Write-Host "Manifest rows: $($rows.Count); train=$train; eval=$eval"
}

$score = Join-Path $EvalDir "D16_P5K_FINAL_SCORECARD.json"
if (Test-Path $score) {
  Write-Host "`nScorecard preview:`n"
  Get-Content $score -TotalCount 80
}
