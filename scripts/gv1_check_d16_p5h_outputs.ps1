param(
    [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
    [string]$OutputFile = ""
)

if (-not $OutputFile -or $OutputFile.Trim() -eq "") {
    $OutputFile = Join-Path $CacheRoot "xjtu_d16_p5h_exact_r2_audit\D16_P5H_EXACT_R2_AUDIT_REPORT.md"
}

Write-Host "P5H report: $OutputFile"
if (Test-Path $OutputFile) {
    $item = Get-Item $OutputFile
    Write-Host "FOUND size=$($item.Length) bytes LastWriteTime=$($item.LastWriteTime)"
    Write-Host ""
    Write-Host "First 80 lines:"
    Get-Content $OutputFile -TotalCount 80
} else {
    Write-Host "MISSING: $OutputFile"
}
