param(
  [string]$ProjectRoot = ".",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$D11C2Dir = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11c2_metadata_input_patch_design",
  [string]$TrainingReadyDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_training_ready",
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_ablation_plan",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

& $PythonExe "scripts\gv1_d12_prepare_metadata_on_off_ablation.py" `
  --project_root $ProjectRoot `
  --cache_root $CacheRoot `
  --d11c2_dir $D11C2Dir `
  --training_ready_dir $TrainingReadyDir `
  --out_dir $OutDir

if ($LASTEXITCODE -ne 0) {
  throw "D12 metadata on/off ablation preparation failed."
}
