$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
$env:ASSB_SOFT_LABEL_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate"
$env:ASSB_OCP_DIR="C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE="False"
$env:ASSB_EVAL_REFERENCE="soft_labels_only"

D:\Anaconda\envs\torchgpu\python.exe .\evaluate_assb_pinn_cycles5_100_v2_massclosed_softlabels.py `
  --model_dir ModelFin_105 `
  --soft_label_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate" `
  --ocp_dir "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs" `
  --cycle_from 5 `
  --cycle_to 100 `
  --output_dir EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only `
  --debug_print_first_batch

D:\Anaconda\envs\torchgpu\python.exe .\diagnose_eval_potential_common_mode.py `
  --eval_dir .\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only

Write-Host ""
Write-Host "Worst per-cycle MAE summary:"
$csv = Import-Csv .\EvalFin_105_cycles5_100_v2_massclosed_candidate_pGauge_softlabel_only\metrics_by_cycle.csv
foreach ($v in "phis_c","phie","theta_a","theta_c","cs_a","cs_c") {
  Write-Host "`n==== $v worst by MAE ===="
  $csv | Where-Object {$_.variable -eq $v} |
    Sort-Object {[double]$_.mae} -Descending |
    Select-Object -First 10 variable,cycle_id,n,mae,rmse,bias_mean,corr,r2,nmae,std_ratio_pred_over_label |
    Format-Table -AutoSize
}
