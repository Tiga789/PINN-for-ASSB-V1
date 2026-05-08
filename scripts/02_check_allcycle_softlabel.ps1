# Check the generated all-cycle solution.npz.
$ErrorActionPreference = "Stop"

cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

D:\Anaconda\envs\torchgpu\python.exe .\inspect_assb_softlabel_solution.py `
  --solution "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_label_allcycle\solution.npz" `
  --expected_cycle_from 5 `
  --expected_cycle_to 522
