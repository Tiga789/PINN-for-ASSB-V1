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
  [string[]]$Modes = @("baseline_d951", "d12s1f_p2d_low_anchor_highsafe_soft", "d12s1f_p2d_low_anchor_highsafe_mid"),
  [int]$MaxParallel = 2,
  [switch]$Clean,
  [switch]$SkipScorecard
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$RunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1f_p2d_highsafe_fast_parallel_6x40ks"
$ProfilesRoot = Join-Path $CacheRoot "xjtu_batch134_replay_profiles"
$ScorecardDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1f_p2d_highsafe_fast_parallel_6x40ks_scorecard"

$Profiles = @(
  @{Name="Batch-1_2C_battery-1"; Patterns=@("*0001*battery-1*2C*battery-1*", "*Batch-1*2C*battery-1*", "*B1*2C*battery-1*")},
  @{Name="Batch-1_2C_battery-2"; Patterns=@("*0002*battery-2*2C*battery-2*", "*Batch-1*2C*battery-2*", "*B1*2C*battery-2*")},
  @{Name="Batch-3_R2p5_battery-1"; Patterns=@("*0009*battery-1*R2.5*battery-1*", "*Batch-3*R2.5*battery-1*", "*Batch-3*R25*battery-1*", "*B3*R25*battery-1*")},
  @{Name="Batch-3_R2p5_battery-2"; Patterns=@("*0010*battery-2*R2.5*battery-2*", "*Batch-3*R2.5*battery-2*", "*Batch-3*R25*battery-2*", "*B3*R25*battery-2*")},
  @{Name="Batch-4_R3_battery-1"; Patterns=@("*0017*battery-1*R3*battery-1*", "*Batch-4*R3*battery-1*", "*B4*R3*battery-1*")},
  @{Name="Batch-4_R3_battery-2"; Patterns=@("*0018*battery-2*R3*battery-2*", "*Batch-4*R3*battery-2*", "*B4*R3*battery-2*")}
)

function Quote-Arg([string]$s) {
  if ($null -eq $s) { return '""' }
  $escaped = $s.Replace('"', '\"')
  if ($escaped -match '[\s`"&|<>]') { return '"' + $escaped + '"' }
  return $escaped
}

function Find-ProfileNpz($profile) {
  $direct = Join-Path (Join-Path $ProfilesRoot $profile.Name) "solution_replay_profile.npz"
  if (Test-Path -LiteralPath $direct -PathType Leaf) { return (Resolve-Path -LiteralPath $direct).Path }

  $AliasMap = @{
    "Batch-1_2C_battery-1"   = "profiles\0001_battery-1_2C_battery-1\solution_replay_profile.npz"
    "Batch-1_2C_battery-2"   = "profiles\0002_battery-2_2C_battery-2\solution_replay_profile.npz"
    "Batch-3_R2p5_battery-1" = "profiles\0009_battery-1_R2.5_battery-1\solution_replay_profile.npz"
    "Batch-3_R2p5_battery-2" = "profiles\0010_battery-2_R2.5_battery-2\solution_replay_profile.npz"
    "Batch-4_R3_battery-1"   = "profiles\0017_battery-1_R3_battery-1\solution_replay_profile.npz"
    "Batch-4_R3_battery-2"   = "profiles\0018_battery-2_R3_battery-2\solution_replay_profile.npz"
  }
  if ($AliasMap.ContainsKey($profile.Name)) {
    $mapped = Join-Path $ProfilesRoot $AliasMap[$profile.Name]
    if (Test-Path -LiteralPath $mapped -PathType Leaf) { return (Resolve-Path -LiteralPath $mapped).Path }
  }

  $realProfiles = Join-Path $ProfilesRoot "profiles"
  if (Test-Path -LiteralPath $realProfiles) {
    foreach ($pat in $profile.Patterns) {
      $hit = Get-ChildItem -LiteralPath $realProfiles -Recurse -Filter "solution_replay_profile.npz" -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like $pat } |
        Select-Object -First 1
      if ($null -ne $hit) { return $hit.FullName }
    }
  }
  throw "Cannot find solution_replay_profile.npz for $($profile.Name) under $ProfilesRoot"
}

function Get-ModeArgs([string]$Mode) {
  switch ($Mode) {
    "baseline_d951" { return @() }
    "d12s1f_p2d_low_anchor_highsafe_soft" {
      return @(
        "--rare_loss_warmup_start_frac", "0.12",
        "--rare_loss_warmup_full_frac", "0.65",
        "--rare_loss_start_scale", "0.25",
        "--p2d_transport_scale_V", "0.160",
        "--p2d_max_correction_V", "0.54",
        "--p2d_protocol_gain", "0.12",
        "--p2d_transport_gate_center_V", "3.14",
        "--p2d_transport_gate_width_V", "0.20",
        "--p2d_transport_pred_center_V", "3.62",
        "--p2d_transport_pred_width_V", "0.25",
        "--p2d_low_gate_power", "0.98",
        "--p2d_pred_low_gate_power", "1.00",
        "--p2d_normal_suppression_center_V", "0.0",
        "--p2d_high_suppression_center_V", "4.03",
        "--p2d_high_suppression_width_V", "0.075",
        "--p2d_high_suppression_power", "1.50",
        "--p2d_allow_upward_correction_V", "0.0",
        "--p2d_lowtarget_focus_weight", "0.38",
        "--p2d_deep_coverage_weight", "0.115",
        "--p2d_low_residual_anchor_weight", "1.45",
        "--p2d_deep_residual_anchor_weight", "0.80",
        "--p2d_low_anchor_max_V", "0.50",
        "--p2d_low_anchor_huber_delta_V", "0.045",
        "--p2d_normal_preservation_weight", "0.46",
        "--p2d_normal_down_bias_guard_weight", "4.5",
        "--p2d_normal_down_shift_guard_weight", "7.0",
        "--p2d_normal_down_allowed_shift_V", "0.0060",
        "--p2d_normal_regret_guard_weight", "5.5",
        "--p2d_normal_regret_allowed_V", "0.0020",
        "--p2d_nonlow_regret_guard_weight", "3.2",
        "--p2d_nonlow_regret_allowed_V", "0.0030",
        "--p2d_normal_correction_budget_weight", "12.0",
        "--p2d_normal_correction_allowed_V", "0.0040",
        "--p2d_nonlow_correction_budget_weight", "6.0",
        "--p2d_nonlow_correction_allowed_V", "0.0060",
        "--p2d_rest_preservation_weight", "0.18",
        "--p2d_high_preservation_weight", "0.35",
        "--p2d_high_regret_guard_weight", "10.0",
        "--p2d_high_regret_allowed_V", "0.0015",
        "--p2d_high_correction_budget_weight", "18.0",
        "--p2d_high_correction_allowed_V", "0.0015",
        "--p2d_high_overshoot_guard_weight", "24.0",
        "--p2d_high_overshoot_threshold_V", "4.35",
        "--p2d_correction_l2_weight", "0.032",
        "--p2d_preservation_huber_delta_V", "0.040"
      )
    }
    "d12s1f_p2d_low_anchor_highsafe_mid" {
      return @(
        "--rare_loss_warmup_start_frac", "0.10",
        "--rare_loss_warmup_full_frac", "0.62",
        "--rare_loss_start_scale", "0.28",
        "--p2d_transport_scale_V", "0.178",
        "--p2d_max_correction_V", "0.56",
        "--p2d_protocol_gain", "0.13",
        "--p2d_transport_gate_center_V", "3.15",
        "--p2d_transport_gate_width_V", "0.205",
        "--p2d_transport_pred_center_V", "3.64",
        "--p2d_transport_pred_width_V", "0.255",
        "--p2d_low_gate_power", "0.95",
        "--p2d_pred_low_gate_power", "0.98",
        "--p2d_normal_suppression_center_V", "0.0",
        "--p2d_high_suppression_center_V", "4.02",
        "--p2d_high_suppression_width_V", "0.075",
        "--p2d_high_suppression_power", "1.65",
        "--p2d_allow_upward_correction_V", "0.0",
        "--p2d_lowtarget_focus_weight", "0.42",
        "--p2d_deep_coverage_weight", "0.125",
        "--p2d_low_residual_anchor_weight", "1.80",
        "--p2d_deep_residual_anchor_weight", "0.95",
        "--p2d_low_anchor_max_V", "0.52",
        "--p2d_low_anchor_huber_delta_V", "0.045",
        "--p2d_normal_preservation_weight", "0.52",
        "--p2d_normal_down_bias_guard_weight", "5.5",
        "--p2d_normal_down_shift_guard_weight", "8.5",
        "--p2d_normal_down_allowed_shift_V", "0.0055",
        "--p2d_normal_regret_guard_weight", "6.5",
        "--p2d_normal_regret_allowed_V", "0.0018",
        "--p2d_nonlow_regret_guard_weight", "4.0",
        "--p2d_nonlow_regret_allowed_V", "0.0028",
        "--p2d_normal_correction_budget_weight", "15.0",
        "--p2d_normal_correction_allowed_V", "0.0038",
        "--p2d_nonlow_correction_budget_weight", "8.0",
        "--p2d_nonlow_correction_allowed_V", "0.0055",
        "--p2d_rest_preservation_weight", "0.20",
        "--p2d_high_preservation_weight", "0.45",
        "--p2d_high_regret_guard_weight", "14.0",
        "--p2d_high_regret_allowed_V", "0.0012",
        "--p2d_high_correction_budget_weight", "24.0",
        "--p2d_high_correction_allowed_V", "0.0012",
        "--p2d_high_overshoot_guard_weight", "32.0",
        "--p2d_high_overshoot_threshold_V", "4.35",
        "--p2d_correction_l2_weight", "0.040",
        "--p2d_preservation_huber_delta_V", "0.038"
      )
    }
    default { throw "Unknown mode: $Mode" }
  }
}

function Start-TrainProcess([object]$Task) {
  $stdout = Join-Path $Task.OutDir "stdout.log"
  $stderr = Join-Path $Task.OutDir "stderr.log"
  $argString = ($Task.Args | ForEach-Object { Quote-Arg $_ }) -join " "
  $argString | Out-File -FilePath (Join-Path $Task.OutDir "command.txt") -Encoding utf8

  $p = Start-Process -FilePath $PythonExe `
    -ArgumentList $argString `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru `
    -NoNewWindow

  return [PSCustomObject]@{
    Process = $p
    Task = $Task
    Stdout = $stdout
    Stderr = $stderr
    StartedAt = Get-Date
  }
}

function Finish-TrainProcess([object]$Running) {
  $p = $Running.Process
  $p.WaitForExit()
  $p.Refresh()
  $outDir = $Running.Task.OutDir
  $console = Join-Path $outDir "console.log"
  "# STDOUT" | Out-File -FilePath $console -Encoding utf8
  if (Test-Path $Running.Stdout) { Get-Content $Running.Stdout | Add-Content -Path $console -Encoding utf8 }
  "`n# STDERR" | Add-Content -Path $console -Encoding utf8
  if (Test-Path $Running.Stderr) { Get-Content $Running.Stderr | Add-Content -Path $console -Encoding utf8 }

  # Some Windows PowerShell + Start-Process combinations return a null ExitCode
  # even after WaitForExit/Refresh. Treat the run as successful only if the
  # required prediction artifact exists. Otherwise keep a nonzero synthetic code.
  $prediction = Join-Path $outDir "prediction.npz"
  $exit = $p.ExitCode
  $exitSource = "process_exit_code"
  if ($null -eq $exit) {
    if (Test-Path -LiteralPath $prediction -PathType Leaf) {
      $exit = 0
      $exitSource = "prediction_exists_exitcode_null"
    } else {
      $exit = 9999
      $exitSource = "missing_prediction_exitcode_null"
    }
  }

  [PSCustomObject]@{
    Mode = $Running.Task.Mode
    Profile = $Running.Task.Profile
    OutDir = $outDir
    ExitCode = $exit
    ExitCodeSource = $exitSource
    HasPrediction = (Test-Path -LiteralPath $prediction -PathType Leaf)
    StartedAt = $Running.StartedAt
    EndedAt = Get-Date
  }
}

if ($MaxParallel -lt 1) { throw "MaxParallel must be >= 1" }

if ($Clean) {
  if (Test-Path $RunsRoot) { Remove-Item -Recurse -Force $RunsRoot }
  if (Test-Path $ScorecardDir) { Remove-Item -Recurse -Force $ScorecardDir }
}
New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null

$Preflight = [ordered]@{
  stage = "D12-S1F fast parallel 6-profile 40ks preflight"
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
  Battery8Excluded = $true
  MetadataOn = $false
  HardClampDisabled = $true
  MainlineOverwritten = $false
  ParallelRunner = $true
  Note = "Runs separate profile/mode processes concurrently to improve GPU occupancy. If CUDA OOM occurs, lower BatchSize or MaxParallel."
}
$Preflight | ConvertTo-Json -Depth 6 | Tee-Object -FilePath (Join-Path $RunsRoot "D12_S1F_6profile_40ks_fast_parallel_preflight.json")

& $PythonExe -m py_compile `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_model.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_transform.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_losses.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_trainer.py") `
  (Join-Path $ProjectRoot "scripts\gv1_train_d12_s1_p2d_local.py") `
  (Join-Path $ProjectRoot "scripts\gv1_scorecard_d12_s1f.py")
if ($LASTEXITCODE -ne 0) { throw "py_compile failed" }

$Tasks = New-Object System.Collections.ArrayList
foreach ($profile in $Profiles) {
  $npz = Find-ProfileNpz $profile
  foreach ($mode in $Modes) {
    $out = Join-Path $RunsRoot ($mode + "__" + $profile.Name)
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    if ($mode -eq "baseline_d951") {
      $args = @(
        (Join-Path $ProjectRoot "scripts\gv1_train_conditioned_pinn.py"),
        "--solution_npz", $npz,
        "--output_dir", $out,
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
    } else {
      $args = @(
        (Join-Path $ProjectRoot "scripts\gv1_train_d12_s1_p2d_local.py"),
        "--solution_npz", $npz,
        "--output_dir", $out,
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
      ) + (Get-ModeArgs $mode)
    }
    [void]$Tasks.Add([PSCustomObject]@{Mode=$mode; Profile=$profile.Name; Npz=$npz; OutDir=$out; Args=$args})
  }
}

$running = New-Object System.Collections.ArrayList
$completed = New-Object System.Collections.ArrayList
$taskIndex = 0
$total = $Tasks.Count
Write-Host "D12-S1F fast parallel queued tasks: $total ; MaxParallel=$MaxParallel"

while (($taskIndex -lt $total) -or ($running.Count -gt 0)) {
  while (($taskIndex -lt $total) -and ($running.Count -lt $MaxParallel)) {
    $task = $Tasks[$taskIndex]
    Write-Host ("START [{0}/{1}] {2} / {3}" -f ($taskIndex + 1), $total, $task.Mode, $task.Profile)
    $rp = Start-TrainProcess $task
    [void]$running.Add($rp)
    $taskIndex += 1
    Start-Sleep -Milliseconds 500
  }

  Start-Sleep -Seconds 3

  for ($i = $running.Count - 1; $i -ge 0; $i--) {
    $rp = $running[$i]
    if ($rp.Process.HasExited) {
      $result = Finish-TrainProcess $rp
      [void]$completed.Add($result)
      $running.RemoveAt($i)
      if ($result.ExitCode -eq 0) {
        Write-Host ("DONE  {0} / {1}" -f $result.Mode, $result.Profile)
      } else {
        Write-Host ("FAIL  {0} / {1} exit={2}; see {3}" -f $result.Mode, $result.Profile, $result.ExitCode, (Join-Path $result.OutDir "console.log")) -ForegroundColor Red
      }
    }
  }
}

$completedPath = Join-Path $RunsRoot "D12_S1F_fast_parallel_completed.csv"
$completed | Export-Csv -NoTypeInformation -Path $completedPath -Encoding UTF8
$failed = @($completed | Where-Object { $_.ExitCode -ne 0 })
if ($failed.Count -gt 0) {
  $failed | Format-Table Mode, Profile, ExitCode, OutDir -AutoSize
  throw "D12-S1F fast parallel had $($failed.Count) failed task(s). Fix failed run(s) before scorecard."
}

if (-not $SkipScorecard) {
  & $PythonExe (Join-Path $ProjectRoot "scripts\gv1_scorecard_d12_s1f.py") `
    --runs_root $RunsRoot `
    --output_dir $ScorecardDir `
    --baseline_mode "baseline_d951" `
    --max_global_regress_V 0.005 `
    --max_normal_regress_V 0.005
  if ($LASTEXITCODE -ne 0) { throw "Scorecard failed" }
  Write-Host "D12-S1F fast parallel done. Scorecard: $ScorecardDir"
} else {
  Write-Host "D12-S1F fast parallel training done. Scorecard skipped by -SkipScorecard."
}
