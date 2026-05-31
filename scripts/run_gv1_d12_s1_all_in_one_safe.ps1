param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [int]$ProfileLimit = 3,
  [int]$Epochs = 100,
  [double]$TimeWindowS = 40000,
  [int]$MaxTimePoints = 1024,
  [int]$BatchSize = 512
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_prepare_strict_smoke_commands.ps1" `
  -ProjectRoot $ProjectRoot `
  -CacheRoot $CacheRoot `
  -Python $Python `
  -ProfileLimit $ProfileLimit `
  -Epochs $Epochs `
  -TimeWindowS $TimeWindowS `
  -MaxTimePoints $MaxTimePoints `
  -BatchSize $BatchSize
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_preflight.ps1" -CacheRoot $CacheRoot -ExpectedProfiles $ProfileLimit -ExpectedEpochs $Epochs -ExpectedTimeWindowS $TimeWindowS -ExpectedMaxTimePoints $MaxTimePoints -ExpectedBatchSize $BatchSize
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_run_triplet.ps1" -CacheRoot $CacheRoot -ExpectedProfiles $ProfileLimit
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d12_s1_collect_scorecard.ps1" -ProjectRoot $ProjectRoot -CacheRoot $CacheRoot -Python $Python
