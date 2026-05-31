param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = "Stop"
$CmdRoot = Join-Path $CacheRoot "xjtu_batch134_d12_runtime_metadata_true_smoke_commands"
if (!(Test-Path $CmdRoot)) { throw "Missing TRUE SMOKE command root: $CmdRoot" }
$Scripts = Get-ChildItem $CmdRoot -Filter "run_d12_runtime_TRUE_SMOKE_*.ps1"
if ($Scripts.Count -lt 3) { throw "Expected 3 TRUE SMOKE generated scripts, found $($Scripts.Count)" }
$Bad = Select-String -Path (Join-Path $CmdRoot "*.ps1") -Pattern "--epochs 40000|--time_window_s 200000|--max_time_points 8192|--batch_size 2048|_200ks" -ErrorAction SilentlyContinue
if ($Bad) {
  $Bad | Format-Table Path, LineNumber, Line -AutoSize
  throw "Preflight failed: old long-run parameters or _200ks suffix found. Do not run these scripts."
}
$Params = Select-String -Path (Join-Path $CmdRoot "*.ps1") -Pattern "--epochs|--time_window_s|--max_time_points|--batch_size" 
$Params | Format-Table Path, LineNumber, Line -AutoSize
Write-Host "D12 TRUE SMOKE preflight PASS."
