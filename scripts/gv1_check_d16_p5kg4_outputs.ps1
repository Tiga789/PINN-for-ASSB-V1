param(
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg4_exact_array_audit"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "OutDir: $OutDir"
$files = @(
  "D16_P5KG4_EXACT_ARRAY_AUDIT_REPORT.md",
  "D16_P5KG4_EXACT_ARRAY_SCORECARD.json",
  "D16_P5KG4_EXACT_ARRAY_SPLIT_METRICS.csv",
  "D16_P5KG4_EXACT_ARRAY_BY_PROFILE.csv",
  "D16_P5KG4_EXACT_ARRAY_CANDIDATE_SUMMARY.csv",
  "D16_P5KG4_EXACT_ARRAY_FAILURES.json"
)
foreach ($f in $files) {
  $p = Join-Path $OutDir $f
  if (Test-Path $p) {
    $item = Get-Item $p
    Write-Host "FOUND: $p | size=$($item.Length)"
  } else {
    Write-Host "MISSING: $p"
  }
}
$score = Join-Path $OutDir "D16_P5KG4_EXACT_ARRAY_SCORECARD.json"
if (Test-Path $score) {
  Write-Host "`nScorecard preview:"
  Get-Content $score -Raw | ConvertFrom-Json | Select-Object stage,operational_status,profile_count_requested,profile_count_evaluated,failure_count | Format-List
}
$fail = Join-Path $OutDir "D16_P5KG4_EXACT_ARRAY_FAILURES.json"
if (Test-Path $fail) {
  $j = Get-Content $fail -Raw | ConvertFrom-Json
  Write-Host "Failure count: $($j.Count)"
}
$report = Join-Path $OutDir "D16_P5KG4_EXACT_ARRAY_AUDIT_REPORT.md"
if (Test-Path $report) {
  Write-Host "`nReport head:"
  Get-Content $report -TotalCount 80
}
