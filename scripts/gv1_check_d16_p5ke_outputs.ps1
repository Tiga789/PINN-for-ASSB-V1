param(
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5ke_diagnostic_first_audit"
)

$Report = Join-Path $OutDir "D16_P5K_E_DIAGNOSTIC_REPORT.md"
Write-Host "OutDir: $OutDir"
if (Test-Path $Report) {
  $item = Get-Item $Report
  Write-Host "FOUND: $Report | size=$($item.Length)"
  Write-Host ""
  Write-Host "Report preview:"
  Get-Content $Report -TotalCount 120
} else {
  Write-Host "MISSING: $Report"
}
