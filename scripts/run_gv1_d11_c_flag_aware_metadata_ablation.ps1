param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$ProjectRoot = "",
  [string]$D10P5Dir = "",
  [string]$D11BDir = "",
  [string]$TrainingReadyDir = "",
  [string]$OutDir = "",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

if ($ProjectRoot -eq "") {
  $ProjectRoot = (Get-Location).Path
}

if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

$PythonArgs = @(
  "scripts\gv1_d11_c_flag_aware_metadata_ablation.py",
  "--cache_root", $CacheRoot,
  "--project_root", $ProjectRoot
)

if ($D10P5Dir -ne "") {
  $PythonArgs += @("--d10p5_dir", $D10P5Dir)
}

if ($D11BDir -ne "") {
  $PythonArgs += @("--d11b_dir", $D11BDir)
}

if ($TrainingReadyDir -ne "") {
  $PythonArgs += @("--training_ready_dir", $TrainingReadyDir)
}

if ($OutDir -ne "") {
  $PythonArgs += @("--out_dir", $OutDir)
}

& $PythonExe @PythonArgs
if ($LASTEXITCODE -ne 0) {
  throw "D11-C flag-aware metadata ablation design generation failed."
}
