param(
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5kg3_theta0_adapter_v2_audit"
)

$ErrorActionPreference = "Stop"
Write-Host "OutDir: $OutDir"
$files = @(
  "D16_P5KG3_THETA0_ADAPTER_V2_AUDIT_REPORT.md",
  "D16_P5KG3_THETA0_ADAPTER_V2_SPLIT_SUMMARY.csv",
  "D16_P5KG3_THETA0_ADAPTER_V2_BY_PROFILE.csv",
  "D16_P5KG3_THETA0_ADAPTER_V2_CANDIDATE_SUMMARY.csv",
  "D16_P5KG3_THETA0_ADAPTER_V2_SUMMARY.json",
  "D16_P5KG3_THETA0_ADAPTER_V2_FAILURES.json"
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

$fail = Join-Path $OutDir "D16_P5KG3_THETA0_ADAPTER_V2_FAILURES.json"
if (Test-Path $fail) {
  $txt = Get-Content $fail -Raw
  try {
    $j = $txt | ConvertFrom-Json
    if ($null -eq $j) { Write-Host "Failure count: 0" }
    elseif ($j -is [array]) { Write-Host "Failure count: $($j.Count)" }
    else { Write-Host "Failure count: 1" }
  } catch { Write-Host "Could not parse failures json" }
}

$report = Join-Path $OutDir "D16_P5KG3_THETA0_ADAPTER_V2_AUDIT_REPORT.md"
if (Test-Path $report) {
  Write-Host "`nReport preview:`n"
  Get-Content $report -TotalCount 80
}
