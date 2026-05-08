# Run from project root after ModelFin_102 exists.
$env:ASSB_SOFT_LABEL_DIR = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1"
$env:ASSB_OCP_DIR = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_pinn_vs_softlabels.py `
  --model_dir .\ModelFin_102 `
  --soft_label_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --output_dir .\EvalFin_102_cycles5_522_v1_softlabel_only
