param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [switch]$RunAblation,
  [switch]$CollectOnly
)

$ErrorActionPreference = 'Stop'

Set-Location $ProjectRoot
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

if ($CollectOnly) {
  Write-Host "==== Collect D12-S3 scorecard only ===="
  & .\scripts\run_gv1_d12_s3_collect_scorecard.ps1
  exit $LASTEXITCODE
}

Write-Host "==== Step 1: source preflight ===="
& .\scripts\run_gv1_d12_s3_preflight_check.ps1 -ProjectRoot $ProjectRoot -CacheRoot $CacheRoot -PythonExe $PythonExe

Write-Host "`n==== Step 2: prepare generated commands ===="
& .\scripts\run_gv1_d12_s3_prepare_commands.ps1

Write-Host "`n==== Step 3: generated command preflight ===="
& .\scripts\run_gv1_d12_s3_preflight_check.ps1 -ProjectRoot $ProjectRoot -CacheRoot $CacheRoot -PythonExe $PythonExe -AfterPrepare

$cmdDir = Join-Path $CacheRoot "xjtu_batch134_d12_s3_metadata_ablation_commands"
if (-not $RunAblation) {
  Write-Host "`nPrepared only. To run all 69 commands later:"
  Write-Host "& `"$cmdDir\run_d12_s3_all_modes_23profile.generated.ps1`""
  Write-Host "Then collect scorecard: .\scripts\run_gv1_d12_s3_collect_scorecard.ps1"
  exit 0
}

Write-Host "`n==== Step 4: run D12-S3 all modes, 69 runs ===="
& "$cmdDir\run_d12_s3_all_modes_23profile.generated.ps1"

Write-Host "`n==== Step 5: collect scorecard ===="
& .\scripts\run_gv1_d12_s3_collect_scorecard.ps1

