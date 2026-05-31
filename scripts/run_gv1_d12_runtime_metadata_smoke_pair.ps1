param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [int]$Epochs = 40000,
  [double]$TimeWindowS = 200000,
  [int]$MaxTimePoints = 8192,
  [int]$BatchSize = 2048,
  [int]$Seed = 42,
  [string]$Device = "auto"
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
& $Python "scripts\gv1_d12_runtime_patch_guardrail.py" --project_root $ProjectRoot --cache_root $CacheRoot
if ($LASTEXITCODE -ne 0) { throw "D12 runtime guardrail failed; smoke pair not started." }
& $Python "scripts\gv1_d12_prepare_runtime_onoff_commands.py" `
  --project_root $ProjectRoot `
  --cache_root $CacheRoot `
  --profile_limit 1 `
  --epochs $Epochs `
  --time_window_s $TimeWindowS `
  --max_time_points $MaxTimePoints `
  --batch_size $BatchSize `
  --seed $Seed `
  --device $Device
if ($LASTEXITCODE -ne 0) { throw "D12 runtime smoke command preparation failed." }
$CmdRoot = Join-Path $CacheRoot "xjtu_batch134_d12_runtime_metadata_ablation_commands"
foreach ($ScriptName in @("run_d12_runtime_metadata_off_1profile.generated.ps1", "run_d12_runtime_metadata_zero_1profile.generated.ps1", "run_d12_runtime_metadata_on_1profile.generated.ps1")) {
  $ScriptPath = Join-Path $CmdRoot $ScriptName
  if (!(Test-Path $ScriptPath)) { throw "Generated script missing: $ScriptPath" }
  powershell -ExecutionPolicy Bypass -File $ScriptPath
  if ($LASTEXITCODE -ne 0) { throw "D12 runtime smoke failed: $ScriptName" }
}
