param(
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke2_provenance_baseline_only_audit"
)

$Report = Join-Path $OutDir "D16_P5K_E2_PROVENANCE_BASELINE_AUDIT_REPORT.md"
$ByProfile = Join-Path $OutDir "D16_P5K_E2_BASELINE_ONLY_BY_PROFILE.csv"
$ByGroup = Join-Path $OutDir "D16_P5K_E2_BASELINE_ONLY_BY_GROUP.csv"
$Prov = Join-Path $OutDir "D16_P5K_E2_PROVENANCE_SUMMARY.json"
$Fail = Join-Path $OutDir "D16_P5K_E2_FAILURES.json"

Write-Host "OutDir: $OutDir" -ForegroundColor Cyan
foreach ($f in @($Report, $ByProfile, $ByGroup, $Prov, $Fail)) {
  if (Test-Path $f) {
    $item = Get-Item $f
    Write-Host "FOUND: $f | size=$($item.Length)" -ForegroundColor Green
  } else {
    Write-Host "MISSING: $f" -ForegroundColor Red
  }
}

if (Test-Path $ByProfile) {
  try {
    $rows = Import-Csv $ByProfile
    Write-Host "baseline-by-profile rows: $($rows.Count)" -ForegroundColor Cyan
  } catch {}
}
if (Test-Path $Fail) {
  try {
    $fails = Get-Content $Fail -Raw | ConvertFrom-Json
    Write-Host "failure_count: $($fails.Count)" -ForegroundColor Cyan
  } catch {}
}

if (Test-Path $Report) {
  Write-Host "`nReport preview:" -ForegroundColor Cyan
  Get-Content $Report -TotalCount 120
}
