param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [int]$Epochs = 150,
  [int]$TimeWindowS = 40000,
  [int]$MaxTimePoints = 1024,
  [int]$BatchSize = 512,
  [int]$Seed = 42
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

& $PythonExe "scripts\gv1_d11_s5a_prepare_lowtarget_diagnosis_commands.py" `
  --project_root $ProjectRoot `
  --cache_root $CacheRoot `
  --python_exe $PythonExe `
  --epochs $Epochs `
  --time_window_s $TimeWindowS `
  --max_time_points $MaxTimePoints `
  --batch_size $BatchSize `
  --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "D11-S5A command preparation failed." }
