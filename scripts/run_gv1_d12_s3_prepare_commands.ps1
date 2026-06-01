$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$Python = "D:\Anaconda\envs\torchgpu\python.exe"
& $Python scripts\gv1_d12_s3_prepare_23profile_strict_commands.py `
  --project_root "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  --cache_root "E:\XJTU battery dataset\_gv1_cache" `
  --python $Python `
  --epochs 100 `
  --time_window_s 40000 `
  --max_time_points 1024 `
  --batch_size 512 `
  --prediction_time_points 1024 `
  --prediction_radial_points 32 `
  --seed 42 `
  --device auto
if ($LASTEXITCODE -ne 0) { throw "D12-S3 prepare commands failed" }
