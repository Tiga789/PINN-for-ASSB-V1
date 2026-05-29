param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProfileRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles",
  [string]$OutRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_2c_tail_3x200ks_d961",
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

$profiles = Get-ChildItem $ProfileRoot -Recurse -Filter solution_replay_profile.npz |
  Where-Object { $_.FullName -match "_2C_" -and ($_.FullName -match "battery-6" -or $_.FullName -match "battery-7" -or $_.FullName -match "battery-8") } |
  Sort-Object FullName

if ($profiles.Count -lt 1) { throw "No late 2C battery-6/7/8 profiles found under $ProfileRoot" }

foreach ($p in $profiles) {
  $name = Split-Path (Split-Path $p.FullName -Parent) -Leaf
  $runOut = Join-Path $OutRoot ("${name}_200ks_d961")
  Write-Host ""
  Write-Host "===== D9.6.1 training $name 200ks ====="
  Write-Host $p.FullName

  & $Python .\scripts\gv1_train_conditioned_pinn.py `
    --solution_npz $p.FullName `
    --output_dir $runOut `
    --profile_adaptive_mode auto `
    --epochs 1600 `
    --batch_size 4096 `
    --max_time_points 16384 `
    --prediction_time_points 8192 `
    --time_window_s 200000 `
    --lr 0.00055 `
    --seed 42 `
    --device $Device
}

$metricsJson = Join-Path $OutRoot "metrics_summary_d961_2c_tail_200ks.json"
$scorecardJson = Join-Path $OutRoot "scorecard_d961_2c_tail_200ks.json"
& $Python .\scripts\gv1_prediction_metrics.py --root $OutRoot --output_json $metricsJson
& $Python .\scripts\gv1_multicell_scorecard_d961.py --metrics_json $metricsJson --output_json $scorecardJson

Write-Host ""
Write-Host "Saved metrics:   $metricsJson"
Write-Host "Saved scorecard: $scorecardJson"
Get-Content $scorecardJson -Raw
