$ErrorActionPreference = 'Stop'
Set-Location "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$Python = "D:\Anaconda\envs\torchgpu\python.exe"
& $Python scripts\gv1_d12_s3_scorecard_from_predictions.py `
  --cache_root "E:\XJTU battery dataset\_gv1_cache" `
  --expected_run_count 69
if ($LASTEXITCODE -ne 0) { throw "D12-S3 collect scorecard failed" }
