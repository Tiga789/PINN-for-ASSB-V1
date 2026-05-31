param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [int]$ProfileLimit = 1,
  [int]$Epochs = 50,
  [double]$TimeWindowS = 40000,
  [int]$MaxTimePoints = 1024,
  [int]$BatchSize = 512,
  [int]$Seed = 42,
  [string]$Device = "auto"
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
& $Python "scripts\gv1_d12_prepare_runtime_true_smoke_commands.py" `
  --project_root $ProjectRoot `
  --cache_root $CacheRoot `
  --profile_limit $ProfileLimit `
  --epochs $Epochs `
  --time_window_s $TimeWindowS `
  --max_time_points $MaxTimePoints `
  --batch_size $BatchSize `
  --seed $Seed `
  --device $Device `
  --python $Python
if ($LASTEXITCODE -ne 0) { throw "D12 TRUE SMOKE command preparation failed." }
