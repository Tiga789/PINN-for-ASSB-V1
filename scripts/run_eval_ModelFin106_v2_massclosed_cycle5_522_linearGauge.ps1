$ErrorActionPreference = "Stop"

$PY = "D:\Anaconda\envs\torchgpu\python.exe"
$SOFT = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\assb_soft_labels_cycle5_522_v2_massclosed_candidate"
$OCP = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"

$CYCLE_FROM = 5
$CYCLE_TO = 522
$RAW_EVAL = "EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only"
$CORR_EVAL = "EvalFin_106_cycles5_522_v2_massclosed_candidate_linearCycleGauge_softlabel_only"

# Default concentration sampling rows for full-cycle evaluation.
# Increase to 30000/50000 if you want denser cs/theta per-cycle statistics; set to 0 only if you intentionally want all rows.
if ($env:ASSB_EVAL_MAX_CS_ROWS) {
  $MAX_CS_ROWS = [int]$env:ASSB_EVAL_MAX_CS_ROWS
} else {
  $MAX_CS_ROWS = 20000
}

if (!(Test-Path ".\ModelFin_106\best.pt")) { throw "Missing .\ModelFin_106\best.pt. Build ModelFin_106 first." }
if (!(Test-Path ".\ModelFin_106\gauge_config.json")) { throw "Missing .\ModelFin_106\gauge_config.json. Build ModelFin_106 first." }
if (!(Test-Path $SOFT)) { throw "Missing soft-label directory: $SOFT" }
if (!(Test-Path $OCP)) { throw "Missing OCP directory: $OCP" }

Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
$env:ASSB_SOFT_LABEL_DIR = $SOFT
$env:ASSB_OCP_DIR = $OCP
$env:ASSB_COMPARE_EXPERIMENT_VOLTAGE = "False"
$env:ASSB_EVAL_REFERENCE = "soft_labels_only"

Write-Host "[INFO] Evaluating ModelFin_106 raw outputs on cycle $CYCLE_FROM-$CYCLE_TO ..." -ForegroundColor Cyan
Write-Host "[INFO] max_cs_time_points = $MAX_CS_ROWS"
& $PY .\evaluate_assb_pinn_cycles5_522_v2_massclosed_softlabels.py `
  --model_dir ModelFin_106 `
  --soft_label_dir $SOFT `
  --ocp_dir $OCP `
  --cycle_from $CYCLE_FROM `
  --cycle_to $CYCLE_TO `
  --output_dir $RAW_EVAL `
  --max_time_points 0 `
  --max_cs_time_points $MAX_CS_ROWS `
  --debug_print_first_batch

Write-Host "[INFO] Applying ModelFin_106 linear-cycle common-mode gauge to cycle $CYCLE_FROM-$CYCLE_TO ..." -ForegroundColor Cyan
& $PY .\apply_ModelFin106_linear_cycle_gauge_cycle5_522.py `
  --model_dir ModelFin_106 `
  --eval_dir $RAW_EVAL `
  --output_dir $CORR_EVAL `
  --cycle_from $CYCLE_FROM `
  --cycle_to $CYCLE_TO

Write-Host "`n[OK] Full-cycle ModelFin_106 evaluation finished." -ForegroundColor Green
Write-Host "Raw metrics:       .\$RAW_EVAL\metrics_global.json"
Write-Host "Corrected metrics: .\$CORR_EVAL\metrics_global_corrected.json"
Write-Host "Per-cycle metrics: .\$CORR_EVAL\metrics_by_cycle_corrected.csv"
Write-Host "Gauge diagnostic:  .\$CORR_EVAL\potential_common_mode_diagnostic_before_after.json"

if (Test-Path ".\$CORR_EVAL\metrics_global_corrected.json") {
  $m = Get-Content ".\$CORR_EVAL\metrics_global_corrected.json" -Raw | ConvertFrom-Json
  Write-Host "`n==== Corrected global metrics summary ====" -ForegroundColor Yellow
  foreach ($v in @("phis_c","phie","theta_a","theta_c","cs_a","cs_c")) {
    $row = $m.$v
    if ($null -ne $row) {
      "{0,-8} MAE={1} RMSE={2} R2={3} corr={4} NMAE={5}" -f $v,$row.mae,$row.rmse,$row.r2,$row.corr,$row.nmae | Write-Host
    }
  }
}
