# Short smoke generation: cycles 5--20 only.
$ErrorActionPreference = "Stop"

cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

D:\Anaconda\envs\torchgpu\python.exe .\integration_spm\generate_assb_softlabel_allcycle.py `
  --record_csv "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --fixed_alignment_summary ".\Data\assb_soft_labels_cycle5_v4\soft_label_summary.json" `
  --output_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_label_cycle5_20_smoke" `
  --cycle_from 5 `
  --cycle_to 20 `
  --n_r 64 `
  --dtype float32 `
  --overwrite
