param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [int]$ExpectedProfiles = 3
)
$ErrorActionPreference = "Stop"
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_preflight.ps1" -CacheRoot $CacheRoot -ExpectedProfiles $ExpectedProfiles
$CmdRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1_metadata_ablation_commands"
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_s1_metadata_off_$($ExpectedProfiles)profile.generated.ps1")
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_s1_metadata_zero_$($ExpectedProfiles)profile.generated.ps1")
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_s1_metadata_on_$($ExpectedProfiles)profile.generated.ps1")
