param(
  [Parameter(Mandatory=$true)][string]$MetadataOffDir,
  [Parameter(Mandatory=$true)][string]$MetadataOnDir,
  [string]$OutDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d12_metadata_on_off_scorecard",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

& $PythonExe "scripts\gv1_d12_collect_on_off_scorecard.py" `
  --metadata_off_dir $MetadataOffDir `
  --metadata_on_dir $MetadataOnDir `
  --out_dir $OutDir

if ($LASTEXITCODE -ne 0) {
  throw "D12 metadata on/off scorecard collection failed."
}
