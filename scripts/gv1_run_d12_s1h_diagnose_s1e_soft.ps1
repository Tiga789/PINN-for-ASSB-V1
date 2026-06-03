param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$RunsRootS1E = "",
  [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunsRootS1E)) {
  $RunsRootS1E = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_6x40ks"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1h_s1e_soft_high_diagnostic"
}

Write-Host "{"
Write-Host "  \"stage\": \"D12-S1H diagnostic-only S1E-soft high audit\"," 
Write-Host "  \"ProjectRoot\": \"$($ProjectRoot.Replace('\\','\\'))\"," 
Write-Host "  \"CacheRoot\": \"$($CacheRoot.Replace('\\','\\'))\"," 
Write-Host "  \"RunsRootS1E\": \"$($RunsRootS1E.Replace('\\','\\'))\"," 
Write-Host "  \"OutputDir\": \"$($OutputDir.Replace('\\','\\'))\"," 
Write-Host "  \"PythonExe\": \"$($PythonExe.Replace('\\','\\'))\"," 
Write-Host "  \"Training\": false," 
Write-Host "  \"MainlineOverwritten\": false" 
Write-Host "}"

if (!(Test-Path $ProjectRoot)) { throw "ProjectRoot not found: $ProjectRoot" }
if (!(Test-Path $PythonExe)) { throw "PythonExe not found: $PythonExe" }
if (!(Test-Path $RunsRootS1E)) { throw "RunsRootS1E not found: $RunsRootS1E" }

$scriptPath = Join-Path $ProjectRoot "scripts\gv1_d12_s1h_diagnose_s1e_soft.py"
if (!(Test-Path $scriptPath)) { throw "Missing script: $scriptPath" }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

& $PythonExe $scriptPath `
  --runs_root_s1e $RunsRootS1E `
  --output_dir $OutputDir `
  --baseline_mode baseline_d951 `
  --candidate_mode d12s1e_p2d_low_anchor_soft

Write-Host "D12-S1H diagnostic done. Output: $OutputDir"
