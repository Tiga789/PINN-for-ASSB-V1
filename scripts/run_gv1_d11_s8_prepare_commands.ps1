param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache"
)
$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot
& $PythonExe "scripts\gv1_d11_s8_prepare_p2dlike_commands.py" --project_root $ProjectRoot --python_exe $PythonExe --cache_root $CacheRoot
if ($LASTEXITCODE -ne 0) { throw "D11-S8 command preparation failed." }
