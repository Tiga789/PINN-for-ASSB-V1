param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

& $PythonExe .\scripts\gv1_d11_s7_prepare_lowvoltage_escape_commands.py --project_root $ProjectRoot --cache_root $CacheRoot --python_exe $PythonExe
if ($LASTEXITCODE -ne 0) { throw 'D11-S7 command preparation failed.' }
