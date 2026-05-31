param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11c2_metadata_input_patch_design",
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$PyScript = Join-Path $ProjectRoot "scripts\gv1_d11c2_metadata_input_patch.py"

if (!(Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }
if (!(Test-Path $PyScript)) { throw "Python script not found: $PyScript" }

$argsList = @(
  $PyScript,
  "--cache_root", $CacheRoot,
  "--out_dir", $OutDir,
  "--dry_run"
)
if ($Strict) { $argsList += "--strict" }

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) { throw "D11C2 metadata input patch generation failed." }

Write-Host "D11C2 outputs saved to: $OutDir"
Write-Host "Open recommendation: $(Join-Path $OutDir 'D11C2_RECOMMENDATION.md')"
