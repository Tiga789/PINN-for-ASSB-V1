param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [int]$MaxResults = 50
)
$ErrorActionPreference = "Stop"
$roots = @($CacheRoot, $ProjectRoot) | Where-Object { Test-Path $_ }
$rows = @()
foreach ($root in $roots) {
  Write-Host "Scanning $root ..." -ForegroundColor Cyan
  $files = Get-ChildItem -Path $root -Recurse -File -Include "best_with_state.pt","best.pt" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch "softlabel|softlabels|replay_profiles|__pycache__|precision_audit" } |
    Select-Object -First $MaxResults
  foreach ($f in $files) {
    $dir = Split-Path $f.FullName -Parent
    if ((Split-Path $dir -Leaf) -in @("model","checkpoint","checkpoints")) {
      $modelDir = Split-Path $dir -Parent
    } else {
      $modelDir = $dir
    }
    $rows += [PSCustomObject]@{
      ModelDir = $modelDir
      Checkpoint = $f.FullName
      SizeMB = [Math]::Round($f.Length / 1MB, 3)
      LastWriteTime = $f.LastWriteTime
    }
  }
}
$rows = $rows | Sort-Object LastWriteTime -Descending -Unique
$rows | Format-Table -AutoSize
Write-Host "`nUse one candidate with:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\gv1_run_d16_p5a_fixed.ps1 -AllowOverwrite -LimitCells 2 -ModelDir '<ModelDir from above>'" -ForegroundColor Green
