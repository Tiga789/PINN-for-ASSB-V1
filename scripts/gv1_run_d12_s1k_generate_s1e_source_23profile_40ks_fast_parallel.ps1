param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [int]$Epochs = 800,
  [int]$BatchSize = 4096,
  [int]$MaxTimePoints = 4096,
  [int]$PredictionTimePoints = 2048,
  [int]$Seed = 42,
  [string]$Device = "auto",
  [int]$MaxParallel = 2,
  [string[]]$Modes = @("baseline_d951", "d12s1e_p2d_low_anchor_soft"),
  [switch]$Clean,
  [switch]$RunSourceScorecard
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$RunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_23x40ks"
$ProfilesRoot = Join-Path $CacheRoot "xjtu_batch134_replay_profiles"
$ScorecardDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_23x40ks_scorecard"

function Canonical-ProfileName([string]$FolderName) {
  if ($FolderName -match '^(\d{4})_') {
    $idx = [int]$matches[1]
    if ($idx -ge 1 -and $idx -le 8) { return "Batch-1_2C_battery-$idx" }
    if ($idx -ge 9 -and $idx -le 16) { return "Batch-3_R2p5_battery-$($idx-8)" }
    if ($idx -ge 17 -and $idx -le 24) { return "Batch-4_R3_battery-$($idx-16)" }
  }
  $safe = $FolderName.Replace('R2.5','R2p5')
  return $safe
}

function Quote-Arg([string]$Value) {
  if ($null -eq $Value) { return '""' }
  $v = [string]$Value
  # Quote every argument to preserve Windows paths with spaces such as
  # E:\XJTU battery dataset\_gv1_cache\...
  $v = $v.Replace('"','\"')
  return '"' + $v + '"'
}

function Get-S1ESoftModeArgs() {
  return @(
    "--rare_loss_warmup_start_frac", "0.12",
    "--rare_loss_warmup_full_frac", "0.65",
    "--rare_loss_start_scale", "0.25",
    "--p2d_transport_scale_V", "0.170",
    "--p2d_max_correction_V", "0.56",
    "--p2d_protocol_gain", "0.14",
    "--p2d_transport_gate_center_V", "3.14",
    "--p2d_transport_gate_width_V", "0.20",
    "--p2d_transport_pred_center_V", "3.64",
    "--p2d_transport_pred_width_V", "0.26",
    "--p2d_low_gate_power", "0.95",
    "--p2d_pred_low_gate_power", "0.95",
    "--p2d_normal_suppression_center_V", "0.0",
    "--p2d_lowtarget_focus_weight", "0.40",
    "--p2d_deep_coverage_weight", "0.120",
    "--p2d_low_residual_anchor_weight", "1.60",
    "--p2d_deep_residual_anchor_weight", "0.90",
    "--p2d_low_anchor_max_V", "0.52",
    "--p2d_low_anchor_huber_delta_V", "0.045",
    "--p2d_normal_preservation_weight", "0.42",
    "--p2d_normal_down_bias_guard_weight", "4.0",
    "--p2d_normal_down_shift_guard_weight", "6.0",
    "--p2d_normal_down_allowed_shift_V", "0.0065",
    "--p2d_normal_regret_guard_weight", "5.0",
    "--p2d_normal_regret_allowed_V", "0.0025",
    "--p2d_nonlow_regret_guard_weight", "2.5",
    "--p2d_nonlow_regret_allowed_V", "0.0035",
    "--p2d_normal_correction_budget_weight", "10.0",
    "--p2d_normal_correction_allowed_V", "0.0045",
    "--p2d_nonlow_correction_budget_weight", "4.0",
    "--p2d_nonlow_correction_allowed_V", "0.0070",
    "--p2d_rest_preservation_weight", "0.16",
    "--p2d_high_preservation_weight", "0.14",
    "--p2d_correction_l2_weight", "0.025",
    "--p2d_preservation_huber_delta_V", "0.045"
  )
}

function Get-TrainArgs([string]$Mode, [string]$Npz, [string]$OutDir) {
  if ($Mode -eq "baseline_d951") {
    return @(
      (Join-Path $ProjectRoot "scripts\gv1_train_conditioned_pinn.py"),
      "--solution_npz", $Npz,
      "--output_dir", $OutDir,
      "--profile_adaptive_mode", "auto",
      "--epochs", "$Epochs",
      "--batch_size", "$BatchSize",
      "--max_time_points", "$MaxTimePoints",
      "--time_window_s", "40000",
      "--prediction_time_points", "$PredictionTimePoints",
      "--prediction_radial_points", "64",
      "--seed", "$Seed",
      "--device", "$Device",
      "--enable_voltage_hard_clamp", "false"
    )
  }
  if ($Mode -eq "d12s1e_p2d_low_anchor_soft") {
    return @(
      (Join-Path $ProjectRoot "scripts\gv1_train_d12_s1_p2d_local.py"),
      "--solution_npz", $Npz,
      "--output_dir", $OutDir,
      "--profile_adaptive_mode", "auto",
      "--epochs", "$Epochs",
      "--batch_size", "$BatchSize",
      "--max_time_points", "$MaxTimePoints",
      "--time_window_s", "40000",
      "--prediction_time_points", "$PredictionTimePoints",
      "--prediction_radial_points", "64",
      "--seed", "$Seed",
      "--device", "$Device",
      "--enable_voltage_hard_clamp", "false"
    ) + (Get-S1ESoftModeArgs)
  }
  throw "Unknown source mode: $Mode"
}

if ($MaxParallel -lt 1) { throw "MaxParallel must be >= 1" }

if ($Clean) {
  if (Test-Path -LiteralPath $RunsRoot) { Remove-Item -Recurse -Force $RunsRoot }
  if (Test-Path -LiteralPath $ScorecardDir) { Remove-Item -Recurse -Force $ScorecardDir }
}
New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null

$Preflight = [ordered]@{
  stage = "D12-S1K v3 source generation: quoted-args S1E-soft + baseline 23-profile 40ks"
  ProjectRoot = $ProjectRoot
  CacheRoot = $CacheRoot
  RunsRoot = $RunsRoot
  ProfilesRoot = $ProfilesRoot
  PythonExe = $PythonExe
  Epochs = $Epochs
  BatchSize = $BatchSize
  MaxTimePoints = $MaxTimePoints
  PredictionTimePoints = $PredictionTimePoints
  TimeWindowSeconds = 40000
  Modes = $Modes
  MaxParallel = $MaxParallel
  ExcludedProfile = "Batch-1_2C_battery-8 only"
  ExpectedPredictionCount = 46
  NoMainlineOverwrite = $true
}
$Preflight | ConvertTo-Json -Depth 6 | Tee-Object -FilePath (Join-Path $RunsRoot "D12_S1K_sourcegen_23x40ks_preflight.json")

& $PythonExe -m py_compile `
  (Join-Path $ProjectRoot "scripts\gv1_train_conditioned_pinn.py") `
  (Join-Path $ProjectRoot "scripts\gv1_train_d12_s1_p2d_local.py")
if ($LASTEXITCODE -ne 0) { throw "py_compile failed for source training scripts" }

$ProfilesDir = Join-Path $ProfilesRoot "profiles"
if (-not (Test-Path -LiteralPath $ProfilesDir -PathType Container)) {
  throw "Cannot find replay profiles folder: $ProfilesDir"
}

$ProfileFiles = Get-ChildItem -LiteralPath $ProfilesDir -Recurse -Filter "solution_replay_profile.npz" -File | Sort-Object FullName
$Profiles = @()
foreach ($pf in $ProfileFiles) {
  $folder = Split-Path -Leaf (Split-Path -Parent $pf.FullName)
  $name = Canonical-ProfileName $folder
  if ($name -eq "Batch-1_2C_battery-8") { continue }
  $Profiles += [PSCustomObject]@{ Name=$name; Npz=$pf.FullName; Folder=$folder }
}
if ($Profiles.Count -ne 23) {
  Write-Warning "Expected 23 profiles after excluding Batch-1_2C_battery-8, found $($Profiles.Count). Continuing, but verify profile discovery."
}

$Tasks = @()
foreach ($p in $Profiles) {
  foreach ($mode in $Modes) {
    $out = Join-Path $RunsRoot ($mode + "__" + $p.Name)
    $Tasks += [PSCustomObject]@{ Mode=$mode; Profile=$p.Name; Npz=$p.Npz; OutDir=$out }
  }
}
Write-Host "D12-S1K source generation queued tasks:" $Tasks.Count "; MaxParallel=$MaxParallel"

$active = New-Object System.Collections.ArrayList
$results = New-Object System.Collections.ArrayList
$taskIndex = 0

function Start-Task($task, [int]$idx, [int]$total) {
  New-Item -ItemType Directory -Force -Path $task.OutDir | Out-Null
  $args = Get-TrainArgs $task.Mode $task.Npz $task.OutDir
  $argString = ($args | ForEach-Object { Quote-Arg $_ }) -join " "
  $argString | Out-File -FilePath (Join-Path $task.OutDir "command.txt") -Encoding utf8
  $stdout = Join-Path $task.OutDir "stdout.log"
  $stderr = Join-Path $task.OutDir "stderr.log"
  Write-Host "START [$idx/$total] $($task.Mode) / $($task.Profile)"
  $proc = Start-Process -FilePath $PythonExe `
    -ArgumentList $argString `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -NoNewWindow
  return [PSCustomObject]@{ Process=$proc; Task=$task; Index=$idx; Stdout=$stdout; Stderr=$stderr }
}

function Complete-Task($item) {
  $proc = $item.Process
  $task = $item.Task
  try { $proc.WaitForExit(); $proc.Refresh() } catch {}
  $pred = Join-Path $task.OutDir "prediction.npz"
  $exit = $proc.ExitCode
  $exitSource = "process_exit_code"

  $console = Join-Path $task.OutDir "console.log"
  "# STDOUT" | Out-File -FilePath $console -Encoding utf8
  if (Test-Path -LiteralPath $item.Stdout) { Get-Content -LiteralPath $item.Stdout | Add-Content -Path $console -Encoding utf8 }
  "`n# STDERR" | Add-Content -Path $console -Encoding utf8
  if (Test-Path -LiteralPath $item.Stderr) { Get-Content -LiteralPath $item.Stderr | Add-Content -Path $console -Encoding utf8 }

  if ($null -eq $exit) {
    if (Test-Path -LiteralPath $pred -PathType Leaf) {
      $exit = 0
      $exitSource = "prediction_exists_exitcode_null"
    } else {
      $exit = 9999
      $exitSource = "missing_prediction_exitcode_null"
    }
  }

  $ok = ((Test-Path -LiteralPath $pred -PathType Leaf) -and ($exit -eq 0))
  if ($ok) {
    Write-Host "DONE  $($task.Mode) / $($task.Profile)"
  } else {
    Write-Host "FAIL  $($task.Mode) / $($task.Profile) exit=$exit source=$exitSource ; see $($task.OutDir)\console.log and stderr.log" -ForegroundColor Red
  }
  return [PSCustomObject]@{
    Mode=$task.Mode
    Profile=$task.Profile
    ExitCode=$exit
    ExitCodeSource=$exitSource
    Ok=$ok
    HasPrediction=(Test-Path -LiteralPath $pred -PathType Leaf)
    OutDir=$task.OutDir
    PredictionNpz=$pred
  }
}

while ($taskIndex -lt $Tasks.Count -or $active.Count -gt 0) {
  while ($taskIndex -lt $Tasks.Count -and $active.Count -lt $MaxParallel) {
    $taskIndex++
    [void]$active.Add((Start-Task $Tasks[$taskIndex-1] $taskIndex $Tasks.Count))
  }
  Start-Sleep -Milliseconds 500
  for ($i = $active.Count - 1; $i -ge 0; $i--) {
    $item = $active[$i]
    if ($item.Process.HasExited) {
      [void]$results.Add((Complete-Task $item))
      $active.RemoveAt($i)
    }
  }
}

$failed = @($results | Where-Object { -not $_.Ok })
$results | Sort-Object Profile,Mode | Format-Table Mode,Profile,ExitCode,Ok,OutDir -AutoSize
$results | ConvertTo-Json -Depth 6 | Out-File -FilePath (Join-Path $RunsRoot "D12_S1K_sourcegen_task_results.json") -Encoding utf8
if ($failed.Count -gt 0) {
  throw "D12-S1K source generation had $($failed.Count) failed task(s). Fix failed run(s) before wrapper confirmation."
}

$predCount = (Get-ChildItem -LiteralPath $RunsRoot -Recurse -Filter "prediction.npz" -File | Measure-Object).Count
Write-Host "D12-S1K source generation done. prediction_count=$predCount ; source root: $RunsRoot"

if ($RunSourceScorecard) {
  & $PythonExe (Join-Path $ProjectRoot "scripts\gv1_scorecard_d12_s1e.py") `
    --runs_root $RunsRoot `
    --output_dir $ScorecardDir `
    --baseline_mode "baseline_d951" `
    --max_global_regress_V 0.005 `
    --max_normal_regress_V 0.005
  if ($LASTEXITCODE -ne 0) { throw "S1E source scorecard failed" }
  Write-Host "S1E source scorecard: $ScorecardDir"
}

Write-Host "Next: run scripts\gv1_run_d12_s1k_apply_23profile_40ks.ps1 with -SourceRunsRoot `"$RunsRoot`""
