# Run from the project root after extracting this package into:
# C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1

$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$PythonExe   = "D:\Anaconda\envs\torchgpu\python.exe"

cd $ProjectRoot

# Avoid stale cycle5_v4 summary env contamination in long-sequence workflows.
Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue

& $PythonExe .\integration_spm\generate_assb_soft_labels_cycle5_522_v2_massclosed_candidate.py `
  --source_solution_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_lable_cycle5-522_v1" `
  --record_csv "C:\Users\Tiga_QJW\Desktop\ZHB_realDATA\record_extracted.csv" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --output_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate" `
  --cycle_from 5 `
  --cycle_to 522 `
  --overwrite
