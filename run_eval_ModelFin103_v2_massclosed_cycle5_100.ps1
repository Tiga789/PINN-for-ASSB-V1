# Run ModelFin_103 against the v2 mass-closed candidate soft labels, cycle5-100.
# Place this file in the project root or run it from the project root.

cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE="False"
$env:ASSB_EVAL_REFERENCE="soft_labels_only"

D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py `
  --model_dir ModelFin_103 `
  --soft_label_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --cycle_from 5 `
  --cycle_to 100 `
  --output_dir EvalFin_103_cycles5_100_v2_massclosed_candidate_softlabel_only `
  --debug_print_first_batch
