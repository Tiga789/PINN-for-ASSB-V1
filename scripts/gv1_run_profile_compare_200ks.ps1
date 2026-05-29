param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProfileRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles",
  [string]$OutRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_200ks_d91",
  [int]$Epochs = 1200,
  [int]$BatchSize = 4096,
  [int]$MaxTimePoints = 16384,
  [int]$PredictionTimePoints = 8192,
  [double]$TimeWindowS = 200000,
  [double]$Lr = 0.001,
  [string]$Device = "cuda"
)

Set-Location $ProjectRoot

$p2c  = Get-ChildItem $ProfileRoot -Recurse -Filter solution_replay_profile.npz | Where-Object { $_.FullName -match "_2C_" }   | Sort-Object FullName | Select-Object -First 1
$pr25 = Get-ChildItem $ProfileRoot -Recurse -Filter solution_replay_profile.npz | Where-Object { $_.FullName -match "_R2\.5_" } | Sort-Object FullName | Select-Object -First 1
$pr3  = Get-ChildItem $ProfileRoot -Recurse -Filter solution_replay_profile.npz | Where-Object { $_.FullName -match "_R3_" }   | Sort-Object FullName | Select-Object -First 1

$runs = @(
  @{name="B1_2C_midwin_200ks";  path=$p2c.FullName;  out=Join-Path $OutRoot "B1_2C_midwin_200ks"},
  @{name="B3_R25_midwin_200ks"; path=$pr25.FullName; out=Join-Path $OutRoot "B3_R25_midwin_200ks"},
  @{name="B4_R3_midwin_200ks";  path=$pr3.FullName;  out=Join-Path $OutRoot "B4_R3_midwin_200ks"}
)

foreach ($r in $runs) {
  Write-Host ""
  Write-Host "===== Training $($r.name) ====="
  Write-Host $r.path

  & $Python .\scripts\gv1_train_conditioned_pinn.py `
    --solution_npz $r.path `
    --output_dir $r.out `
    --epochs $Epochs `
    --batch_size $BatchSize `
    --max_time_points $MaxTimePoints `
    --prediction_time_points $PredictionTimePoints `
    --time_window_s $TimeWindowS `
    --lr $Lr `
    --device $Device `
    --voltage_range_strategy profile_minmax `
    --voltage_margin_V 0.02 `
    --phis_c_correction_scale_V 0.60
}

& $Python .\scripts\gv1_prediction_metrics.py --root $OutRoot --output_json (Join-Path $OutRoot "metrics_summary.json")
