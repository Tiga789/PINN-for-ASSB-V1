param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$ProjectRoot = "",
  [string]$OutDir = "",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [switch]$Strict
)

if ($ProjectRoot -eq "") {
  $ProjectRoot = (Get-Location).Path
}

if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

$PythonArgs = @(
  "scripts\gv1_d10_p5_regime_policy_and_d11_plan.py",
  "--cache_root", $CacheRoot,
  "--project_root", $ProjectRoot
)

if ($OutDir -ne "") {
  $PythonArgs += @("--out_dir", $OutDir)
}

if ($Strict) {
  $PythonArgs += @("--strict")
}

& $PythonExe @PythonArgs
if ($LASTEXITCODE -ne 0) {
  throw "D10-P5 regime policy / D11 plan generation failed."
}
