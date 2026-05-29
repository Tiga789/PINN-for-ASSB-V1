# GV1 D9.7 diagnostic-only run for B1_2C battery-8.
$ErrorActionPreference = "Stop"
$python = "D:\Anaconda\envs\torchgpu\python.exe"
$projectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-Location $projectRoot

$solution = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles\0008_battery-8_2C_battery-8\solution_replay_profile.npz"
$scorecard24x40 = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_24x40ks_d96\scorecard_d96_40ks.json"
$out = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_battery8_outlier_diagnosis"

$roots = @(
  "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d96",
  "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d961",
  "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d961_v2",
  "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d962",
  "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d963_probe\A_reproduce_d96_seed42_lr7e4",
  "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d963_probe\B_lower_lr_seed42_lr5e4",
  "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_borderline_B1_2C_battery8_200ks_d963_probe\C_lower_lr_seed7_lr5e4"
)

$argsList = @(
  ".\scripts\gv1_d97_battery8_outlier_diagnosis.py",
  "--solution_npz", $solution,
  "--scorecard_json", $scorecard24x40,
  "--output_dir", $out,
  "--prediction_roots"
) + $roots

& $python @argsList
Write-Host ""
Write-Host "D9.7 summary path:"
Write-Host (Join-Path $out "d97_battery8_diagnosis_summary.json")
