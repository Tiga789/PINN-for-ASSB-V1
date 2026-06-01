param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

& $PythonExe .\scripts\gv1_d11_s7_apply_lowvoltage_escape_patch.py --project_root $ProjectRoot
if ($LASTEXITCODE -ne 0) { throw 'D11-S7 patch application failed.' }

& $PythonExe -m compileall gv1 scripts
if ($LASTEXITCODE -ne 0) { throw 'compileall failed after D11-S7 patch.' }

Write-Host 'D11-S7 patch applied and compileall passed.'
