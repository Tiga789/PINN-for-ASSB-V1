param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe"
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s2_prepare_balanced_strict_smoke_commands.ps1" -ProjectRoot $ProjectRoot -CacheRoot $CacheRoot -Python $Python
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s2_preflight.ps1" -CacheRoot $CacheRoot
$CmdRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s2_metadata_ablation_commands"
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_s2_metadata_off_6profile.generated.ps1")
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_s2_metadata_zero_6profile.generated.ps1")
powershell -ExecutionPolicy Bypass -File (Join-Path $CmdRoot "run_d12_s2_metadata_on_6profile.generated.ps1")
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s2_collect_scorecard.ps1" -ProjectRoot $ProjectRoot -CacheRoot $CacheRoot -Python $Python
