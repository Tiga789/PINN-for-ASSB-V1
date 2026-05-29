param(
  [string]$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
  [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProfileRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_replay_profiles\profiles",
  [string]$OutRoot = "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_train_conditioned_pinn_profile_compare_500ks_d92",
  [string]$Device = "cuda"
)

Set-Location $ProjectRoot
& .\scripts\gv1_run_profile_compare_d92.ps1 `
  -ProjectRoot $ProjectRoot `
  -Python $Python `
  -ProfileRoot $ProfileRoot `
  -OutRoot $OutRoot `
  -Epochs 1800 `
  -BatchSize 4096 `
  -MaxTimePoints 32768 `
  -PredictionTimePoints 16384 `
  -TimeWindowS 500000 `
  -Lr 0.0006 `
  -Device $Device
