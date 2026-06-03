param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [int]$Epochs = 1200,
  [int]$BatchSize = 2048,
  [int]$MaxTimePoints = 4096,
  [int]$PredictionTimePoints = 2048,
  [int]$Seed = 42,
  [string]$Device = "auto",
  [string[]]$Modes = @("baseline_d951", "d12s1b_p2d_preserve_light", "d12s1b_p2d_preserve_mid", "d12s1b_p2d_preserve_guarded"),
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$RunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1b_p2d_preservation_6x40ks"
$ProfilesRoot = Join-Path $CacheRoot "xjtu_batch134_replay_profiles"
$ScorecardDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1b_p2d_preservation_6x40ks_scorecard"

$Profiles = @(
  @{Name="Batch-1_2C_battery-1"; Patterns=@("*0001*battery-1*2C*battery-1*", "*Batch-1*2C*battery-1*", "*B1*2C*battery-1*")},
  @{Name="Batch-1_2C_battery-2"; Patterns=@("*0002*battery-2*2C*battery-2*", "*Batch-1*2C*battery-2*", "*B1*2C*battery-2*")},
  @{Name="Batch-3_R2p5_battery-1"; Patterns=@("*0009*battery-1*R2.5*battery-1*", "*Batch-3*R2.5*battery-1*", "*Batch-3*R25*battery-1*", "*B3*R25*battery-1*")},
  @{Name="Batch-3_R2p5_battery-2"; Patterns=@("*0010*battery-2*R2.5*battery-2*", "*Batch-3*R2.5*battery-2*", "*Batch-3*R25*battery-2*", "*B3*R25*battery-2*")},
  @{Name="Batch-4_R3_battery-1"; Patterns=@("*0017*battery-1*R3*battery-1*", "*Batch-4*R3*battery-1*", "*B4*R3*battery-1*")},
  @{Name="Batch-4_R3_battery-2"; Patterns=@("*0018*battery-2*R3*battery-2*", "*Batch-4*R3*battery-2*", "*B4*R3*battery-2*")}
)

function Find-ProfileNpz($profile) {
  # 1) Compatibility alias directory.
  $direct = Join-Path (Join-Path $ProfilesRoot $profile.Name) "solution_replay_profile.npz"
  if (Test-Path -LiteralPath $direct -PathType Leaf) { return (Resolve-Path -LiteralPath $direct).Path }

  # 2) Current D8/D9/D10 replay profile naming.
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

  # 3) Fallback search in the real profiles folder.
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
    "d12s1b_p2d_preserve_light" {
      return @(
        "--p2d_transport_scale_V", "0.075",
        "--p2d_max_correction_V", "0.24",
        "--p2d_protocol_gain", "0.06",
        "--p2d_transport_pred_center_V", "3.45",
        "--p2d_transport_pred_width_V", "0.18",
        "--p2d_low_gate_power", "1.20",
        "--p2d_pred_low_gate_power", "1.45",
        "--p2d_normal_suppression_center_V", "3.38",
        "--p2d_normal_suppression_width_V", "0.16",
        "--p2d_normal_suppression_power", "1.60",
        "--p2d_lowtarget_focus_weight", "0.14",
        "--p2d_deep_coverage_weight", "0.035",
        "--p2d_normal_preservation_weight", "1.20",
        "--p2d_normal_bias_preservation_weight", "6.0",
        "--p2d_normal_shift_guard_weight", "8.0",
        "--p2d_normal_allowed_shift_V", "0.006",
        "--p2d_rest_preservation_weight", "0.24",
        "--p2d_high_preservation_weight", "0.24",
        "--p2d_correction_l2_weight", "0.10",
        "--p2d_preservation_huber_delta_V", "0.035"
      )
    }
    "d12s1b_p2d_preserve_mid" {
      return @(
        "--p2d_transport_scale_V", "0.10",
        "--p2d_max_correction_V", "0.30",
        "--p2d_protocol_gain", "0.10",
        "--p2d_transport_pred_center_V", "3.50",
        "--p2d_transport_pred_width_V", "0.20",
        "--p2d_low_gate_power", "1.10",
        "--p2d_pred_low_gate_power", "1.30",
        "--p2d_normal_suppression_center_V", "3.44",
        "--p2d_normal_suppression_width_V", "0.17",
        "--p2d_normal_suppression_power", "1.35",
        "--p2d_lowtarget_focus_weight", "0.17",
        "--p2d_deep_coverage_weight", "0.050",
        "--p2d_normal_preservation_weight", "1.80",
        "--p2d_normal_bias_preservation_weight", "8.0",
        "--p2d_normal_shift_guard_weight", "12.0",
        "--p2d_normal_allowed_shift_V", "0.006",
        "--p2d_rest_preservation_weight", "0.28",
        "--p2d_high_preservation_weight", "0.28",
        "--p2d_correction_l2_weight", "0.14",
        "--p2d_preservation_huber_delta_V", "0.035"
      )
    }
    "d12s1b_p2d_preserve_guarded" {
      return @(
        "--p2d_transport_scale_V", "0.13",
        "--p2d_max_correction_V", "0.35",
        "--p2d_protocol_gain", "0.16",
        "--p2d_transport_pred_center_V", "3.54",
        "--p2d_transport_pred_width_V", "0.21",
        "--p2d_low_gate_power", "1.05",
        "--p2d_pred_low_gate_power", "1.20",
        "--p2d_normal_suppression_center_V", "3.50",
        "--p2d_normal_suppression_width_V", "0.18",
        "--p2d_normal_suppression_power", "1.20",
        "--p2d_lowtarget_focus_weight", "0.20",
        "--p2d_deep_coverage_weight", "0.065",
        "--p2d_normal_preservation_weight", "2.40",
        "--p2d_normal_bias_preservation_weight", "10.0",
        "--p2d_normal_shift_guard_weight", "16.0",
        "--p2d_normal_allowed_shift_V", "0.006",
        "--p2d_rest_preservation_weight", "0.32",
        "--p2d_high_preservation_weight", "0.30",
        "--p2d_correction_l2_weight", "0.18",
        "--p2d_preservation_huber_delta_V", "0.035"
      )
    }
    default { throw "Unknown mode: $Mode" }
  }
}

if ($Clean) {
  if (Test-Path $RunsRoot) { Remove-Item -Recurse -Force $RunsRoot }
  if (Test-Path $ScorecardDir) { Remove-Item -Recurse -Force $ScorecardDir }
}
New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null

$Preflight = [ordered]@{
  stage = "D12-S1B preservation-tightened 6-profile 40ks preflight"
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
  Battery8Excluded = $true
  MetadataOn = $false
  HardClampDisabled = $true
  MainlineOverwritten = $false
}
$Preflight | ConvertTo-Json -Depth 6 | Tee-Object -FilePath (Join-Path $RunsRoot "D12_S1B_6profile_40ks_preflight.json")

& $PythonExe -m py_compile `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_model.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_transform.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_losses.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_trainer.py") `
  (Join-Path $ProjectRoot "scripts\gv1_train_d12_s1_p2d_local.py") `
  (Join-Path $ProjectRoot "scripts\gv1_scorecard_d12_s1b.py")
if ($LASTEXITCODE -ne 0) { throw "py_compile failed" }

foreach ($profile in $Profiles) {
  $npz = Find-ProfileNpz $profile
  foreach ($mode in $Modes) {
    $out = Join-Path $RunsRoot ($mode + "__" + $profile.Name)
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    if ($mode -eq "baseline_d951") {
      $cmd = @(
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
      $cmd = @(
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
    ($cmd -join " ") | Out-File -FilePath (Join-Path $out "command.txt") -Encoding utf8
    Write-Host "RUN $mode / $($profile.Name)"
    & $PythonExe @cmd 2>&1 | Tee-Object -FilePath (Join-Path $out "console.log")
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $mode / $($profile.Name)" }
  }
}

& $PythonExe (Join-Path $ProjectRoot "scripts\gv1_scorecard_d12_s1b.py") `
  --runs_root $RunsRoot `
  --output_dir $ScorecardDir `
  --baseline_mode "baseline_d951" `
  --max_global_regress_V 0.005 `
  --max_normal_regress_V 0.005
if ($LASTEXITCODE -ne 0) { throw "Scorecard failed" }
Write-Host "D12-S1B 6x40ks done. Scorecard: $ScorecardDir"
