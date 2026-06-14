param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$OutDir = ""
)
$ErrorActionPreference = "Continue"
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = Join-Path $CacheRoot "xjtu_d16_p5kg1_observed_theta0_audit"
}
Write-Host "OutDir: $OutDir"
$files = @(
  "D16_P5KG1_OBSERVED_THETA0_AUDIT_REPORT.md",
  "D16_P5KG1_OBSERVED_THETA0_SPLIT_METRICS.csv",
  "D16_P5KG1_OBSERVED_THETA0_BY_PROFILE.csv",
  "D16_P5KG1_OBSERVED_THETA0_MODEL_SUMMARY.csv",
  "D16_P5KG1_OBSERVED_THETA0_ESTIMATOR_SUMMARY.json",
  "D16_P5KG1_OBSERVED_THETA0_FAILURES.json"
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
$fail = Join-Path $OutDir "D16_P5KG1_OBSERVED_THETA0_FAILURES.json"
if (Test-Path $fail) {
  try {
    $j = Get-Content $fail -Raw | ConvertFrom-Json
    Write-Host "Failure count: $($j.Count)"
  } catch { Write-Host "Could not parse failures json: $_" }
}
$report = Join-Path $OutDir "D16_P5KG1_OBSERVED_THETA0_AUDIT_REPORT.md"
if (Test-Path $report) {
  Write-Host "`nReport preview:"
  Get-Content $report -TotalCount 120
}
