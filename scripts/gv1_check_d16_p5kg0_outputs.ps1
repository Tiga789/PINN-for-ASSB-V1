param(
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg0_baseline_repair_audit"
)

$Report = Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_AUDIT_REPORT.md"
$Files = @(
  $Report,
  (Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_AUDIT_SUMMARY.json"),
  (Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_BY_PROFILE.csv"),
  (Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_SPLIT_METRICS.csv"),
  (Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_MODEL_SUMMARY.csv"),
  (Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_BATCH_METRICS.csv"),
  (Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_PROTOCOL_METRICS.csv"),
  (Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_FAILURES.json")
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

$Split = Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_SPLIT_METRICS.csv"
if (Test-Path $Split) {
  Write-Host "`nSplit metrics preview:"
  Import-Csv $Split |
    Select-Object model,group,profile_count,theta_a_mean_mae,theta_a_mean_r2,theta_c_mean_mae,theta_c_mean_r2,cs_a_mean_r2,cs_c_mean_r2 |
    Format-Table -AutoSize
}

$Fail = Join-Path $OutDir "D16_P5KG0_BASELINE_REPAIR_FAILURES.json"
if (Test-Path $Fail) {
  $fails = Get-Content $Fail -Raw | ConvertFrom-Json
  Write-Host "`nFailure count: $($fails.Count)"
  if ($fails.Count -gt 0) {
    $fails | Select-Object -First 10 model,profile_id,batch,split,error | Format-List
  }
}

if (Test-Path $Report) {
  Write-Host "`nReport preview:"
  Get-Content $Report -TotalCount 180
}
