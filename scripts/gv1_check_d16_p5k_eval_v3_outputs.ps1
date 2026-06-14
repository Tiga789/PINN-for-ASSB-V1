param(
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5k_train6to10_hard_cbar_ocp_residual_FAST",
  [ValidateSet("A_train6", "B_train8", "C_train10")]
  [string]$TrainSet = "C_train10"
)
$StageRunDir = Join-Path $RunDir $TrainSet
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"
$Score = Join-Path $EvalDir "D16_P5K_FINAL_SCORECARD.json"
Write-Host "StageRunDir: $StageRunDir"
Write-Host "EvalDir: $EvalDir"
$files = @(
  $Score,
  (Join-Path $EvalDir "D16_P5K_METRICS_BY_PROFILE.csv"),
  (Join-Path $EvalDir "D16_P5K_SPLIT_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5K_BATCH_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5K_PROTOCOL_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5K_FAILURES.json")
)
foreach ($f in $files) {
  if (Test-Path $f) { $item = Get-Item $f; Write-Host "FOUND: $f | size=$($item.Length)" } else { Write-Host "MISSING: $f" }
}
if (Test-Path $Score) {
  $j = Get-Content $Score -Raw | ConvertFrom-Json
  $j | Select-Object stage, operational_status, profile_count_requested, profile_count_evaluated, failure_count | Format-List
  if ($j.global_metrics_weighted) { $j.global_metrics_weighted | Format-List }
}
