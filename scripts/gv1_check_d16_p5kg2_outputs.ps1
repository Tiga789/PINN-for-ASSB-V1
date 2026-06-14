param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = Join-Path $CacheRoot "xjtu_d16_p5kg2_gated_theta0_adapter_audit"
}
Write-Host "OutDir: $OutDir"

$files = @(
  "D16_P5KG2_GATED_THETA0_ADAPTER_AUDIT_REPORT.md",
  "D16_P5KG2_GATED_THETA0_SPLIT_METRICS.csv",
  "D16_P5KG2_GATED_THETA0_BY_PROFILE_SELECTION.csv",
  "D16_P5KG2_GATED_THETA0_CANDIDATE_SUMMARY.csv",
  "D16_P5KG2_GATED_THETA0_FAILURES.json"
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

$fail = Join-Path $OutDir "D16_P5KG2_GATED_THETA0_FAILURES.json"
if (Test-Path $fail) {
  $j = Get-Content $fail -Raw | ConvertFrom-Json
  Write-Host "Failure count: $($j.Count)"
}

$report = Join-Path $OutDir "D16_P5KG2_GATED_THETA0_ADAPTER_AUDIT_REPORT.md"
if (Test-Path $report) {
  Write-Host "`nReport preview:`n"
  Get-Content $report -TotalCount 120
}
