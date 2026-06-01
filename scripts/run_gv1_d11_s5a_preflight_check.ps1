param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [switch]$AfterPrepare
)

$ErrorActionPreference = 'Stop'
Set-Location $ProjectRoot

Write-Host "==== D11-S5A preflight ===="
Write-Host "ProjectRoot = $ProjectRoot"
Write-Host "CacheRoot   = $CacheRoot"
Write-Host "PythonExe   = $PythonExe"
Write-Host "AfterPrepare= $AfterPrepare"

$required = @(
  "gv1\model.py",
  "gv1\output_transform.py",
  "gv1\losses.py",
  "gv1\trainer.py",
  "scripts\gv1_train_conditioned_pinn.py",
  "scripts\gv1_d11_s5a_prepare_lowtarget_diagnosis_commands.py",
  "scripts\gv1_d11_s5a_scorecard_from_predictions.py"
)
foreach ($p in $required) {
  if (-not (Test-Path $p)) { throw "Missing required file: $p" }
}

Write-Host "==== Compile check ===="
& $PythonExe -m compileall gv1 scripts
if ($LASTEXITCODE -ne 0) { throw "compileall failed with exit code $LASTEXITCODE" }

Write-Host "==== Mainline marker check ===="
$trainText = Get-Content "scripts\gv1_train_conditioned_pinn.py" -Raw
$outText = Get-Content "gv1\output_transform.py" -Raw
$lossText = Get-Content "gv1\losses.py" -Raw
if ($trainText -notmatch "D9\.5\.1") { Write-Warning "Training script does not contain D9.5.1 marker." }
if ($trainText -notmatch "rare") { Write-Warning "Training script does not mention rare-regime terms." }
if ($outText -match "enable_voltage_hard_clamp:\s*bool\s*=\s*True") { throw "Unsafe default: enable_voltage_hard_clamp=True in output_transform.py" }
if ($lossText -notmatch "warmup") { Write-Warning "losses.py does not mention warmup." }

Write-Host "==== Cache path check ===="
if (-not (Test-Path $CacheRoot)) { throw "CacheRoot not found: $CacheRoot" }
$trainingReady = Join-Path $CacheRoot "xjtu_batch134_training_ready"
if (-not (Test-Path $trainingReady)) { throw "training-ready directory not found: $trainingReady" }
$manifestCandidates = @(
  (Join-Path $trainingReady "xjtu_batch134_profile_manifest.csv"),
  (Join-Path $trainingReady "profile_manifest.csv")
)
$manifestOk = $false
foreach ($m in $manifestCandidates) { if (Test-Path $m) { $manifestOk = $true; Write-Host "Found manifest: $m" } }
if (-not $manifestOk) { throw "No profile manifest found under $trainingReady" }

if ($AfterPrepare) {
  Write-Host "==== Generated command safety check ===="
  $cmdDir = Join-Path $CacheRoot "xjtu_batch134_d11_s5a_lowtarget_sign_gate_commands"
  if (-not (Test-Path $cmdDir)) { throw "Generated command dir not found: $cmdDir" }
  $bad = Select-String -Path (Join-Path $cmdDir "*.ps1") -Pattern "epochs 40000","--epochs 40000","time_window_s 200000","--time_window_s 200000","max_time_points 8192","batch_size 2048","enable_voltage_hard_clamp True","--enable_voltage_hard_clamp True" -ErrorAction SilentlyContinue
  if ($bad) {
    $bad | Format-Table -AutoSize
    throw "Unsafe generated command parameter detected. Stop before running D11-S5A."
  }
  $summaryPath = Join-Path $cmdDir "d11_s5a_command_preparation_summary.json"
  if (-not (Test-Path $summaryPath)) { throw "Missing preparation summary: $summaryPath" }
  Write-Host "Generated scripts look safe."
}

Write-Host "D11-S5A preflight PASS."
