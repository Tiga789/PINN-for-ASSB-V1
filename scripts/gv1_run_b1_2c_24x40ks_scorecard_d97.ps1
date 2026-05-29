# GV1 D9.7 scorecard-only inspection. Diagnostic only.
$ErrorActionPreference = "Stop"
$python = "D:\Anaconda\envs\torchgpu\python.exe"
$projectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
Set-Location $projectRoot
$solution = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles\0008_battery-8_2C_battery-8\solution_replay_profile.npz"
$scorecard24x40 = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_multicell_24x40ks_d96\scorecard_d96_40ks.json"
$out = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d97_scorecard_only"
& $python .\scripts\gv1_d97_battery8_outlier_diagnosis.py --solution_npz $solution --scorecard_json $scorecard24x40 --output_dir $out
