param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProfileRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles",
  [string]$OutRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_d951",
  [int]$Epochs = 1200,
  [int]$BatchSize = 4096,
  [int]$MaxTimePoints = 8192,
  [int]$PredictionTimePoints = 4096,
  [double]$TimeWindowS = 40000,
  [double]$Lr = 0.0007,
  [string]$Device = "cuda",
  [string]$ProfileAdaptiveMode = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$p2c  = Get-ChildItem $ProfileRoot -Recurse -Filter solution_replay_profile.npz | Where-Object { $_.FullName -match "_2C_" }   | Sort-Object FullName | Select-Object -First 1
$pr25 = Get-ChildItem $ProfileRoot -Recurse -Filter solution_replay_profile.npz | Where-Object { $_.FullName -match "_R2\.5_" } | Sort-Object FullName | Select-Object -First 1
$pr3  = Get-ChildItem $ProfileRoot -Recurse -Filter solution_replay_profile.npz | Where-Object { $_.FullName -match "_R3_" }   | Sort-Object FullName | Select-Object -First 1

if (-not $p2c -or -not $pr25 -or -not $pr3) {
  throw "Could not find one 2C, one R2.5, and one R3 solution_replay_profile.npz under $ProfileRoot"
}

$windowTag = if ($TimeWindowS -ge 1000) { "$( [int]($TimeWindowS / 1000) )ks" } else { "${TimeWindowS}s" }
$windowTag = $windowTag -replace " ", ""

$runs = @(
  @{name="B1_2C_${windowTag}";  path=$p2c.FullName;  out=Join-Path $OutRoot "B1_2C_${windowTag}"},
  @{name="B3_R25_${windowTag}"; path=$pr25.FullName; out=Join-Path $OutRoot "B3_R25_${windowTag}"},
  @{name="B4_R3_${windowTag}";  path=$pr3.FullName;  out=Join-Path $OutRoot "B4_R3_${windowTag}"}
)

foreach ($r in $runs) {
  Write-Host ""
  Write-Host "===== Training $($r.name) | D9.5.1 trend-first warmup=$ProfileAdaptiveMode ====="
  Write-Host $r.path

  & $Python .\scripts\gv1_train_conditioned_pinn.py `
    --solution_npz $r.path `
    --output_dir $r.out `
    --profile_adaptive_mode $ProfileAdaptiveMode `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --max_time_points $MaxTimePoints `
    --prediction_time_points $PredictionTimePoints `
    --time_window_s $TimeWindowS `
    --lr $Lr `
    --device $Device
}

& $Python .\scripts\gv1_prediction_metrics.py --root $OutRoot --output_json (Join-Path $OutRoot "metrics_summary_d951_${windowTag}.json")
