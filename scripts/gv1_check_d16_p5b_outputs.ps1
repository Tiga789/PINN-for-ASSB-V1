param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$RunDir = ""
)
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $CacheRoot "xjtu_d16_p5b_train6_eval49_observation_physics"
}
$Manifest = Join-Path $RunDir "D16_P5B_TRAIN6_EVAL49_MANIFEST.csv"
$TrainSummary = Join-Path $RunDir "model_train6_observation_physics\D16_P5B_TRAINING_SUMMARY.json"
$Ckpt = Join-Path $RunDir "model_train6_observation_physics\model\best_with_state.pt"
$Score = Join-Path $RunDir "eval_all55_vs_softlabels\D16_P5B_FINAL_SCORECARD.json"
$ProfileCsv = Join-Path $RunDir "eval_all55_vs_softlabels\D16_P5B_METRICS_BY_PROFILE.csv"
$SplitCsv = Join-Path $RunDir "eval_all55_vs_softlabels\D16_P5B_SPLIT_METRICS.csv"
$BatchCsv = Join-Path $RunDir "eval_all55_vs_softlabels\D16_P5B_BATCH_METRICS.csv"
$ProtocolCsv = Join-Path $RunDir "eval_all55_vs_softlabels\D16_P5B_PROTOCOL_METRICS.csv"

Write-Host "RunDir: $RunDir" -ForegroundColor Cyan
foreach ($p in @($Manifest,$TrainSummary,$Ckpt,$Score,$ProfileCsv,$SplitCsv,$BatchCsv,$ProtocolCsv)) {
  if (Test-Path $p) { Write-Host "FOUND: $p" -ForegroundColor Green } else { Write-Host "MISSING: $p" -ForegroundColor Red }
}
if (Test-Path $Manifest) {
  $rows = Import-Csv $Manifest
  Write-Host "Manifest rows: $($rows.Count); train=$(( $rows | Where-Object {$_.split -eq 'train'} | Measure-Object).Count); eval=$(( $rows | Where-Object {$_.split -eq 'eval'} | Measure-Object).Count)" -ForegroundColor Cyan
}
if (Test-Path $Score) {
  Write-Host "`nScorecard preview:" -ForegroundColor Cyan
  Get-Content $Score -TotalCount 80 | ForEach-Object { Write-Host $_ }
}
