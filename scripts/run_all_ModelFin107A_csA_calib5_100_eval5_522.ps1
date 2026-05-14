$ErrorActionPreference = "Stop"
$PY = "D:\Anaconda\envs\torchgpu\python.exe"

$RAW_EVAL = "EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only"
$MODEL106 = "ModelFin_106"
$MODEL107 = "ModelFin_107A_calib5_100"
$OUT = "EvalFin_107A_calib5_100_eval5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only"

if (!(Test-Path ".\$RAW_EVAL")) {
  throw "Missing raw ModelFin_106 full-cycle eval directory: .\$RAW_EVAL. Run .\scripts\run_eval_ModelFin106_v2_massclosed_cycle5_522_linearGauge.ps1 first."
}

# Stricter setting: fit cs_a correction on cycle5-100 and apply it to the full cycle5-522 range.
& $PY .\fit_apply_ModelFin107A_anode_state_correction.py `
  --raw_eval_dir $RAW_EVAL `
  --model106_dir $MODEL106 `
  --model107_dir $MODEL107 `
  --output_dir $OUT `
  --calib_cycle_from 5 `
  --calib_cycle_to 100 `
  --eval_cycle_from 5 `
  --eval_cycle_to 522 `
  --max_fit_points 250000 `
  --ridge 1e-5 `
  --save_npz

Write-Host "`n[OK] ModelFin_107A calib5-100 -> eval5-522 check finished." -ForegroundColor Green
Write-Host "Model wrapper: .\$MODEL107"
Write-Host "Metrics:       .\$OUT\metrics_global_corrected.json"
Write-Host "Per-cycle:     .\$OUT\metrics_by_cycle_corrected.csv"
