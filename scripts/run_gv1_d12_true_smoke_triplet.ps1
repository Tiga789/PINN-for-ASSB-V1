param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = "Stop"
$CmdRoot = Join-Path $CacheRoot "xjtu_batch134_d12_runtime_metadata_true_smoke_commands"
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_true_smoke_preflight.ps1" -CacheRoot $CacheRoot
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_runtime_TRUE_SMOKE_metadata_off_1profile.generated.ps1")
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_runtime_TRUE_SMOKE_metadata_zero_1profile.generated.ps1")
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_runtime_TRUE_SMOKE_metadata_on_1profile.generated.ps1")
