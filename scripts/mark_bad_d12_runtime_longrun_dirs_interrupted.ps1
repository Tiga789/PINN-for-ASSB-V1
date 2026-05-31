param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = "Stop"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Get-ChildItem $CacheRoot -Directory |
  Where-Object { $_.Name -match '^xjtu_batch134_d12_runtime_metadata_(off|zero|on)_.*_200ks$' } |
  ForEach-Object {
    $NewName = $_.Name + "_INTERRUPTED_BAD_40000epoch_" + $Stamp
    Rename-Item -LiteralPath $_.FullName -NewName $NewName
    Write-Host "Renamed invalid long-run dir:" $_.Name "->" $NewName
  }
