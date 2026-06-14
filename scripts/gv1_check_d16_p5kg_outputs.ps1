param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg_rulev2_strict_gate_FAST",
  [ValidateSet("G_train12_rulev2_strict")]
  [string]$TrainSet = "G_train12_rulev2_strict"
)
$StageRunDir = Join-Path $RunDir $TrainSet
$ModelDir = Join-Path $StageRunDir "model_rulev2_strict_gate_hard_cbar_ocp_residual"
$EvalDir = Join-Path $StageRunDir "eval_all55_vs_softlabels"
$BaseDir = Join-Path $StageRunDir "baseline_only_preflight"
Write-Host "StageRunDir: $StageRunDir"
$files = @(
  (Join-Path $StageRunDir "D16_P5KG_${TrainSet}_MANIFEST.csv"),
  (Join-Path $StageRunDir "D16_P5KG_${TrainSet}_MANIFEST_SUMMARY.json"),
  (Join-Path $BaseDir "D16_P5KG_BASELINE_ONLY_SCORECARD.json"),
  (Join-Path $BaseDir "D16_P5KG_BASELINE_ONLY_SPLIT_METRICS.csv"),
  (Join-Path $ModelDir "D16_P5KG_TRAINING_SUMMARY.json"),
  (Join-Path $ModelDir "D16_P5KG_TRAIN_INPUT_AUDIT.json"),
  (Join-Path $ModelDir "model\best_with_state.pt"),
  (Join-Path $EvalDir "D16_P5KG_FINAL_SCORECARD.json"),
  (Join-Path $EvalDir "D16_P5KG_METRICS_BY_PROFILE.csv"),
  (Join-Path $EvalDir "D16_P5KG_SPLIT_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5KG_BATCH_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5KG_PROTOCOL_METRICS.csv"),
  (Join-Path $EvalDir "D16_P5KG_FAILURES.json")
)
foreach ($f in $files) {
  if (Test-Path $f) { Write-Host "FOUND: $f | size=$((Get-Item $f).Length)" }
  else { Write-Host "MISSING: $f" }
}
if (Test-Path (Join-Path $StageRunDir "D16_P5KG_${TrainSet}_MANIFEST.csv")) {
  $m = Import-Csv (Join-Path $StageRunDir "D16_P5KG_${TrainSet}_MANIFEST.csv")
  Write-Host ("Manifest rows: {0}; core_train={1}; hard_probe={2}; train_total={3}; eval={4}" -f $m.Count, ($m|Where-Object {$_.split -eq 'core_train'}).Count, ($m|Where-Object {$_.split -eq 'hard_probe'}).Count, ($m|Where-Object {$_.split -in @('core_train','hard_probe','train')}).Count, ($m|Where-Object {$_.split -eq 'eval'}).Count)
}
if (Test-Path (Join-Path $BaseDir "D16_P5KG_BASELINE_ONLY_SCORECARD.json")) {
  Write-Host "`nBaseline scorecard preview:"
  Get-Content (Join-Path $BaseDir "D16_P5KG_BASELINE_ONLY_SCORECARD.json") -TotalCount 80
}
if (Test-Path (Join-Path $EvalDir "D16_P5KG_FINAL_SCORECARD.json")) {
  Write-Host "`nFinal scorecard preview:"
  Get-Content (Join-Path $EvalDir "D16_P5KG_FINAL_SCORECARD.json") -TotalCount 100
}
