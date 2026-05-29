param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$SolutionNpz = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles\0008_battery-8_2C_battery-8\solution_replay_profile.npz",
  [string]$OutRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d962",
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

Write-Host "===== D9.6.2 rollback-style targeted repair: B1_2C battery-8 200ks ====="
Write-Host $SolutionNpz

& $Python .\scripts\gv1_train_conditioned_pinn.py `
  --solution_npz $SolutionNpz `
  --output_dir $OutRoot `
  --profile_adaptive_mode late_2c_rollback `
  --epochs 1600 `
  --batch_size 4096 `
  --max_time_points 16384 `
  --prediction_time_points 8192 `
  --time_window_s 200000 `
  --lr 0.00055 `
  --seed 42 `
  --device $Device

$metricsJson = Join-Path $OutRoot "metrics_borderline_200ks_d962.json"
& $Python .\scripts\gv1_prediction_metrics.py --root $OutRoot --output_json $metricsJson

Write-Host ""
Write-Host "Saved metrics: $metricsJson"
Get-Content $metricsJson -Raw
