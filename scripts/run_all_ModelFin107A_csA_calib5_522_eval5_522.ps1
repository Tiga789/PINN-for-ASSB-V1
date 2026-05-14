$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\torchgpu\python.exe"

$RAW_EVAL = "EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only"
$MODEL106 = "ModelFin_106"
$MODEL107 = "ModelFin_107A"
$OUT = "EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only"

if (!(Test-Path ".\$RAW_EVAL")) {
  throw "Missing raw ModelFin_106 full-cycle eval directory: .\$RAW_EVAL. Run .\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1 first."
}
if (!(Test-Path ".\$MODEL106\best.pt")) { throw "Missing .\$MODEL106\best.pt" }
if (!(Test-Path ".\$MODEL106\gauge_config.json")) { throw "Missing .\$MODEL106\gauge_config.json" }

# Full-cycle calibration gives the best full-cycle corrected benchmark.
# For stricter extrapolation, use run_all_ModelFin107A_csA_calib5_100_eval5_522.ps1.
& $PY .\fit_apply_ModelFin107A_anode_state_correction.py `
  --raw_eval_dir $RAW_EVAL `
  --model106_dir $MODEL106 `
  --model107_dir $MODEL107 `
  --output_dir $OUT `
  --calib_cycle_from 5 `
  --calib_cycle_to 522 `
  --eval_cycle_from 5 `
  --eval_cycle_to 522 `
  --max_fit_points 350000 `
  --ridge 1e-6 `
  --save_npz

Write-Host "`n[OK] ModelFin_107A full-cycle cs_a correction finished." -ForegroundColor Green
Write-Host "Model wrapper: .\$MODEL107"
Write-Host "Metrics:       .\$OUT\metrics_global_corrected.json"
Write-Host "Per-cycle:     .\$OUT\metrics_by_cycle_corrected.csv"

if (Test-Path ".\$OUT\metrics_global_corrected.json") {
  $m = Get-Content ".\$OUT\metrics_global_corrected.json" -Raw | ConvertFrom-Json
  Write-Host "`n==== ModelFin_107A corrected full-cycle metrics ====" -ForegroundColor Yellow
  foreach ($v in @("phis_c","phie","theta_a","theta_c","cs_a","cs_c")) {
    $row = $m.$v
    if ($null -ne $row) {
      "{0,-8} MAE={1} RMSE={2} R2={3} corr={4} NMAE={5}" -f $v,$row.mae,$row.rmse,$row.r2,$row.corr,$row.nmae | Write-Host
    }
  }
}
