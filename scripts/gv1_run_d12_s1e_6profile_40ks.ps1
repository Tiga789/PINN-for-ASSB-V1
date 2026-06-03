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
  [string[]]$Modes = @("baseline_d951", "d12s1e_p2d_low_anchor_soft", "d12s1e_p2d_low_anchor_mid", "d12s1e_p2d_low_anchor_guarded"),
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$RunsRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_6x40ks"
$ProfilesRoot = Join-Path $CacheRoot "xjtu_batch134_replay_profiles"
$ScorecardDir = Join-Path $CacheRoot "xjtu_batch134_d12_s1e_p2d_anchor_budget_6x40ks_scorecard"

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
    "d12s1e_p2d_low_anchor_soft" {
      # Soft S1E: restore low correction using residual anchoring, with a small
      # normal correction budget. Intended to beat S1D low improvement without
      # repeating S1C normal leakage.
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
    "d12s1e_p2d_low_anchor_mid" {
      # Mid S1E: primary promotion candidate. Low anchor is strong enough to
      # target >=20 mV low improvement, while correction-budget guards target
      # <=5 mV normal/global regression.
      return @(
        "--rare_loss_warmup_start_frac", "0.10",
        "--rare_loss_warmup_full_frac", "0.60",
        "--rare_loss_start_scale", "0.30",
        "--p2d_transport_scale_V", "0.195",
        "--p2d_max_correction_V", "0.60",
        "--p2d_protocol_gain", "0.16",
        "--p2d_transport_gate_center_V", "3.15",
        "--p2d_transport_gate_width_V", "0.21",
        "--p2d_transport_pred_center_V", "3.66",
        "--p2d_transport_pred_width_V", "0.27",
        "--p2d_low_gate_power", "0.90",
        "--p2d_pred_low_gate_power", "0.90",
        "--p2d_normal_suppression_center_V", "0.0",
        "--p2d_lowtarget_focus_weight", "0.46",
        "--p2d_deep_coverage_weight", "0.140",
        "--p2d_low_residual_anchor_weight", "2.40",
        "--p2d_deep_residual_anchor_weight", "1.20",
        "--p2d_low_anchor_max_V", "0.55",
        "--p2d_low_anchor_huber_delta_V", "0.045",
        "--p2d_normal_preservation_weight", "0.50",
        "--p2d_normal_down_bias_guard_weight", "5.5",
        "--p2d_normal_down_shift_guard_weight", "8.0",
        "--p2d_normal_down_allowed_shift_V", "0.0060",
        "--p2d_normal_regret_guard_weight", "6.0",
        "--p2d_normal_regret_allowed_V", "0.0020",
        "--p2d_nonlow_regret_guard_weight", "3.5",
        "--p2d_nonlow_regret_allowed_V", "0.0030",
        "--p2d_normal_correction_budget_weight", "14.0",
        "--p2d_normal_correction_allowed_V", "0.0040",
        "--p2d_nonlow_correction_budget_weight", "6.0",
        "--p2d_nonlow_correction_allowed_V", "0.0060",
        "--p2d_rest_preservation_weight", "0.18",
        "--p2d_high_preservation_weight", "0.16",
        "--p2d_correction_l2_weight", "0.030",
        "--p2d_preservation_huber_delta_V", "0.042"
      )
    }
    "d12s1e_p2d_low_anchor_guarded" {
      # Guarded S1E: stronger low drive with tighter normal correction budget.
      # This tests whether the current branch can satisfy both low and normal
      # constraints at the same time.
      return @(
        "--rare_loss_warmup_start_frac", "0.10",
        "--rare_loss_warmup_full_frac", "0.58",
        "--rare_loss_start_scale", "0.35",
        "--p2d_transport_scale_V", "0.220",
        "--p2d_max_correction_V", "0.65",
        "--p2d_protocol_gain", "0.18",
        "--p2d_transport_gate_center_V", "3.16",
        "--p2d_transport_gate_width_V", "0.22",
        "--p2d_transport_pred_center_V", "3.68",
        "--p2d_transport_pred_width_V", "0.28",
        "--p2d_low_gate_power", "0.85",
        "--p2d_pred_low_gate_power", "0.85",
        "--p2d_normal_suppression_center_V", "0.0",
        "--p2d_lowtarget_focus_weight", "0.52",
        "--p2d_deep_coverage_weight", "0.160",
        "--p2d_low_residual_anchor_weight", "3.00",
        "--p2d_deep_residual_anchor_weight", "1.50",
        "--p2d_low_anchor_max_V", "0.58",
        "--p2d_low_anchor_huber_delta_V", "0.045",
        "--p2d_normal_preservation_weight", "0.60",
        "--p2d_normal_down_bias_guard_weight", "7.0",
        "--p2d_normal_down_shift_guard_weight", "10.0",
        "--p2d_normal_down_allowed_shift_V", "0.0055",
        "--p2d_normal_regret_guard_weight", "8.0",
        "--p2d_normal_regret_allowed_V", "0.0015",
        "--p2d_nonlow_regret_guard_weight", "4.5",
        "--p2d_nonlow_regret_allowed_V", "0.0025",
        "--p2d_normal_correction_budget_weight", "18.0",
        "--p2d_normal_correction_allowed_V", "0.0035",
        "--p2d_nonlow_correction_budget_weight", "8.0",
        "--p2d_nonlow_correction_allowed_V", "0.0055",
        "--p2d_rest_preservation_weight", "0.20",
        "--p2d_high_preservation_weight", "0.18",
        "--p2d_correction_l2_weight", "0.038",
        "--p2d_preservation_huber_delta_V", "0.040"
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
  stage = "D12-S1E low-anchor-budget 6-profile 40ks preflight"
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
  Change = "low residual anchor + normal correction budget relative to baseline; no post-hoc correction; no mainline overwrite"
}
$Preflight | ConvertTo-Json -Depth 6 | Tee-Object -FilePath (Join-Path $RunsRoot "D12_S1E_6profile_40ks_preflight.json")

& $PythonExe -m py_compile `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_model.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_transform.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_losses.py") `
  (Join-Path $ProjectRoot "gv1\d12_s1_p2d_trainer.py") `
  (Join-Path $ProjectRoot "scripts\gv1_train_d12_s1_p2d_local.py") `
  (Join-Path $ProjectRoot "scripts\gv1_scorecard_d12_s1e.py")
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

& $PythonExe (Join-Path $ProjectRoot "scripts\gv1_scorecard_d12_s1e.py") `
  --runs_root $RunsRoot `
  --output_dir $ScorecardDir `
  --baseline_mode "baseline_d951" `
  --max_global_regress_V 0.005 `
  --max_normal_regress_V 0.005
if ($LASTEXITCODE -ne 0) { throw "Scorecard failed" }
Write-Host "D12-S1E 6x40ks done. Scorecard: $ScorecardDir"
