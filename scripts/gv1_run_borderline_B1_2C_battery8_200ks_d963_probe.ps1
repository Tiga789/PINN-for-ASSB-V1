$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $projectRoot
$python = "D:\Anaconda\envs\torchgpu\python.exe"
$solution = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles\0008_battery-8_2C_battery-8\solution_replay_profile.npz"
$outRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d963_probe"
$baseline = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96\metrics_borderline_200ks.json"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
$runs = @(
  @{name="A_reproduce_d96_seed42_lr7e4"; seed=42; lr=0.0007; epochs=1400},
  @{name="B_lower_lr_seed42_lr5e4";     seed=42; lr=0.0005; epochs=1600},
  @{name="C_lower_lr_seed7_lr5e4";      seed=7;  lr=0.0005; epochs=1600}
)
foreach ($r in $runs) {
  $out = Join-Path $outRoot $r.name
  Write-Host ""; Write-Host "===== D9.6.3 probe $($r.name) ====="
  & $python .\scripts\gv1_train_conditioned_pinn.py `
    --solution_npz $solution `
    --output_dir $out `
    --profile_adaptive_mode auto `
    --epochs $r.epochs `
    --batch_size 4096 `
    --max_time_points 16384 `
    --prediction_time_points 8192 `
    --time_window_s 200000 `
    --lr $r.lr `
    --seed $r.seed `
    --device cuda
  & $python .\scripts\gv1_prediction_metrics.py `
    --root $out `
    --output_json (Join-Path $out "metrics_borderline_200ks_d963.json")
}
$scorecard = Join-Path $outRoot "scorecard_borderline_200ks_d963_probe.json"
if (Test-Path $baseline) {
  & $python .\scripts\gv1_select_borderline_d963.py --root $outRoot --baseline_json $baseline --output_json $scorecard
} else {
  Write-Host "Baseline JSON not found; selecting without baseline comparison: $baseline"
  & $python .\scripts\gv1_select_borderline_d963.py --root $outRoot --output_json $scorecard
}
Write-Host ""; Write-Host "Saved scorecard: $scorecard"
Get-Content $scorecard -Raw
