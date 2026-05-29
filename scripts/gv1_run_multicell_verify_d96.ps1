param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProfileRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles",
  [string]$OutRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_d96",
  [string]$ProtocolQuota = "2C:2,R2.5:2,R3:2",
  [int]$MaxProfiles = 6,
  [int]$Epochs = 1000,
  [int]$BatchSize = 4096,
  [int]$MaxTimePoints = 8192,
  [int]$PredictionTimePoints = 4096,
  [double]$TimeWindowS = 40000,
  [double]$Lr = 0.0007,
  [string]$Device = "cuda",
  [string]$ProfileAdaptiveMode = "auto",
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$windowTag = if ($TimeWindowS -ge 1000) { "$([int]($TimeWindowS / 1000))ks" } else { "${TimeWindowS}s" }
$windowTag = $windowTag -replace " ", ""

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$selectionJson = Join-Path $OutRoot "selected_profiles_d96_${windowTag}.json"
$metricsJson = Join-Path $OutRoot "metrics_summary_d96_${windowTag}.json"
$scorecardJson = Join-Path $OutRoot "scorecard_d96_${windowTag}.json"

Write-Host "===== D9.6 profile selection ====="
& $Python .\scripts\gv1_d96_profile_inventory.py `
  --profile_root $ProfileRoot `
  --output_json $selectionJson `
  --quota $ProtocolQuota `
  --max_profiles $MaxProfiles

$payload = Get-Content $selectionJson -Raw | ConvertFrom-Json
if (-not $payload.ok -or $payload.selected_count -lt 1) {
  throw "No profiles selected. Check ProfileRoot: $ProfileRoot"
}

Write-Host ""
Write-Host "Selected profiles:" $payload.selected_count
foreach ($p in @($payload.profiles)) {
  Write-Host " - $($p.run_id) => $($p.solution_npz)"
}

foreach ($p in @($payload.profiles)) {
  $runOut = Join-Path $OutRoot ("$($p.run_id)_${windowTag}")
  Write-Host ""
  Write-Host "===== Training $($p.run_id) | D9.6 $windowTag | D9.5.1 core ====="
  Write-Host $p.solution_npz

  & $Python .\scripts\gv1_train_conditioned_pinn.py `
    --solution_npz $p.solution_npz `
    --output_dir $runOut `
    --profile_adaptive_mode $ProfileAdaptiveMode `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --max_time_points $MaxTimePoints `
    --prediction_time_points $PredictionTimePoints `
    --time_window_s $TimeWindowS `
    --lr $Lr `
    --seed $Seed `
    --device $Device
}

Write-Host ""
Write-Host "===== Collecting D9.6 metrics ====="
& $Python .\scripts\gv1_prediction_metrics.py --root $OutRoot --output_json $metricsJson

Write-Host ""
Write-Host "===== Building D9.6 scorecard ====="
& $Python .\scripts\gv1_multicell_scorecard_d96.py --metrics_json $metricsJson --output_json $scorecardJson

Write-Host ""
Write-Host "Saved selection: $selectionJson"
Write-Host "Saved metrics:   $metricsJson"
Write-Host "Saved scorecard: $scorecardJson"
