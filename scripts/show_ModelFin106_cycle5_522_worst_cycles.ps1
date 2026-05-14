$ErrorActionPreference = "Stop"
$EVAL = "EvalFin_106_cycles5_522_v2_massclosed_candidate_linearCycleGauge_softlabel_only"
$CSV = ".\$EVAL\metrics_by_cycle_corrected.csv"
if (!(Test-Path $CSV)) { throw "Missing $CSV. Run .\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1 first." }
$csv = Import-Csv $CSV
foreach ($v in "phis_c","phie","theta_a","theta_c","cs_a","cs_c") {
  Write-Host "`n==== $v worst by MAE, cycle5-522 ====" -ForegroundColor Cyan
  $csv | Where-Object {$_.variable -eq $v} |
    Sort-Object {[double]$_.mae} -Descending |
    Select-Object -First 12 variable,cycle_id,n,mae,rmse,bias_mean,corr,r2,nmae,std_ratio_pred_over_label |
    Format-Table -AutoSize
}
