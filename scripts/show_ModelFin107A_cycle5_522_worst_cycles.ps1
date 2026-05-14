$ErrorActionPreference = "Stop"
$EVAL = "EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only"
$CSV = ".\$EVAL\metrics_by_cycle_corrected.csv"
if (!(Test-Path $CSV)) { throw "Missing $CSV. Run run_all_ModelFin107A_csA_calib5_522_eval5_522.ps1 first." }

$rows = Import-Csv $CSV
foreach ($v in @("phis_c","phie","theta_a","theta_c","cs_a","cs_c")) {
  Write-Host "`n==== $v worst by MAE, ModelFin_107A cycle5-522 ====" -ForegroundColor Yellow
  $rows | Where-Object {$_.variable -eq $v} |
    Sort-Object {[double]$_.mae} -Descending |
    Select-Object -First 12 variable,cycle_id,n,mae,rmse,bias_mean,corr,r2,nmae,std_ratio_pred_over_label |
    Format-Table -AutoSize
}
