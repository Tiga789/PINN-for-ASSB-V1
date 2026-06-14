param(
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg_baseline_noregression_audit"
)

$Report = Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_AUDIT_REPORT.md"
$Files = @(
  $Report,
  (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_AUDIT_SUMMARY.json"),
  (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_BY_PROFILE.csv"),
  (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_SPLIT_METRICS.csv"),
  (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_MODEL_SUMMARY.csv"),
  (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_BATCH_METRICS.csv"),
  (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_PROTOCOL_METRICS.csv"),
  (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_FAILURES.json")
)

Write-Host "OutDir: $OutDir"
foreach ($f in $Files) {
  if (Test-Path $f) {
    $item = Get-Item $f
    Write-Host "FOUND: $f | size=$($item.Length)"
  } else {
    Write-Host "MISSING: $f"
  }
}

if (Test-Path (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_SPLIT_METRICS.csv")) {
  Write-Host "`nSplit metrics preview:"
  Import-Csv (Join-Path $OutDir "D16_P5KG_BASELINE_NOREGRESSION_SPLIT_METRICS.csv") |
    Select-Object model,group,profile_count,theta_a_mean_mae,theta_a_mean_r2,theta_c_mean_mae,theta_c_mean_r2 |
    Format-Table -AutoSize
}

if (Test-Path $Report) {
  Write-Host "`nReport preview:"
  Get-Content $Report -TotalCount 140
}
