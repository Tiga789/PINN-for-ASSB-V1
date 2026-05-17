# ASSB-111 modified checker for D4 ModelFin_107A assets and optional ModelFin_111 package.
# This script keeps the original ModelFin_107A checks and adds an optional
# strict30 ModelFin_111 reference/state-guard check. It does not modify any
# 107A core files.

param(
  [string]$ModelFin111Dir = ".\ModelFin_111",
  [switch]$CheckModelFin111,
  [switch]$Strict107A,
  [switch]$Strict111
)

$ErrorActionPreference = "Stop"

function Test-RequiredFile {
  param([string]$Path, [string]$Label = "")
  if (Test-Path $Path) {
    if ($Label.Length -gt 0) { Write-Host "OK: $Label -> $Path" -ForegroundColor Green }
    else { Write-Host "OK: $Path" -ForegroundColor Green }
    return $true
  }
  $msg = "Missing: $Path"
  if ($Label.Length -gt 0) { $msg = "Missing ${Label}: $Path" }
  throw $msg
}

function Test-OptionalPath {
  param([string]$Path, [string]$Label = "")
  if (Test-Path $Path) {
    if ($Label.Length -gt 0) { Write-Host "OK: $Label -> $Path" -ForegroundColor Green }
    else { Write-Host "OK: $Path" -ForegroundColor Green }
    return $true
  }
  if ($Label.Length -gt 0) { Write-Host "WARN: missing optional ${Label}: $Path" -ForegroundColor Yellow }
  else { Write-Host "WARN: missing optional path: $Path" -ForegroundColor Yellow }
  return $false
}

Write-Host "[check_ModelFin107A_package] Checking D4/107A support files..." -ForegroundColor Cyan

$required107A = @(
  ".\diagnose_ModelFin106_csA_cbar_radial_fullcycle.py",
  ".\fit_apply_ModelFin107A_anode_state_correction.py",
  ".\scripts\run_diagnose_ModelFin106_csA_fullcycle.ps1",
  ".\scripts\run_all_ModelFin107A_csA_calib5_522_eval5_522.ps1",
  ".\scripts\run_all_ModelFin107A_csA_calib5_100_eval5_522.ps1",
  ".\scripts\show_ModelFin107A_cycle5_522_worst_cycles.ps1"
)

foreach ($f in $required107A) { Test-RequiredFile $f | Out-Null }

$raw106 = ".\EvalFin_106_cycles5_522_v2_massclosed_candidate_linearGauge_raw_softlabel_only"
if (Test-Path $raw106) {
  Write-Host "OK: raw ModelFin_106 full-cycle eval directory exists." -ForegroundColor Green
} elseif ($Strict107A) {
  throw "Missing raw ModelFin_106 full-cycle eval directory: $raw106"
} else {
  Write-Host "WARN: raw ModelFin_106 full-cycle eval directory is missing. Run ModelFin_106 full-cycle eval first." -ForegroundColor Yellow
}

$eval107A = ".\EvalFin_107A_cycles5_522_v2_massclosed_candidate_linearGauge_csACorrected_softlabel_only"
if (Test-Path $eval107A) {
  Write-Host "OK: ModelFin_107A corrected eval directory exists." -ForegroundColor Green
  Test-OptionalPath (Join-Path $eval107A "metrics_global_corrected.json") "107A corrected metrics" | Out-Null
  Test-OptionalPath (Join-Path $eval107A "state_array_alignment_provenance.json") "107A state provenance" | Out-Null
} elseif ($Strict107A) {
  throw "Missing ModelFin_107A corrected eval directory: $eval107A"
} else {
  Write-Host "WARN: ModelFin_107A corrected eval directory is missing." -ForegroundColor Yellow
}

if ($CheckModelFin111 -or (Test-Path $ModelFin111Dir)) {
  Write-Host "[check_ModelFin107A_package] Checking optional ASSB-111 strict30 package..." -ForegroundColor Cyan
  Test-RequiredFile $ModelFin111Dir "ModelFin_111 directory" | Out-Null

  $required111 = @(
    "soh_head.pt",
    "soh_head_config.json",
    "feature_scaler.json",
    "feature_schema.json",
    "split_manifest.json",
    "dataset.csv",
    "state_engine_ref.json",
    "model_manifest.json"
  )
  foreach ($name in $required111) {
    $path = Join-Path $ModelFin111Dir $name
    Test-RequiredFile $path "ModelFin_111/$name" | Out-Null
  }

  $stateRefPath = Join-Path $ModelFin111Dir "state_engine_ref.json"
  try {
    $stateRef = Get-Content $stateRefPath -Raw | ConvertFrom-Json
    if ($stateRef.state_engine_mode -ne "frozen_ModelFin_107A_reference") {
      throw "state_engine_mode is not frozen_ModelFin_107A_reference"
    }
    Write-Host "OK: ModelFin_111 uses frozen ModelFin_107A reference state engine." -ForegroundColor Green
    if ($stateRef.state_model_dir) { Test-OptionalPath $stateRef.state_model_dir "state_model_dir from state_engine_ref" | Out-Null }
    if ($stateRef.state_eval_dir) { Test-OptionalPath $stateRef.state_eval_dir "state_eval_dir from state_engine_ref" | Out-Null }
  } catch {
    throw "Invalid ModelFin_111 state_engine_ref.json: $($_.Exception.Message)"
  }

  $auditPath = Join-Path $ModelFin111Dir "leakage_audit.json"
  if (Test-Path $auditPath) {
    try {
      $audit = Get-Content $auditPath -Raw | ConvertFrom-Json
      if ($audit.ok -eq $false) { throw "leakage_audit.json reports ok=false" }
      Write-Host "OK: ModelFin_111 leakage audit is not failing." -ForegroundColor Green
    } catch {
      throw "Invalid/failing leakage_audit.json: $($_.Exception.Message)"
    }
  } elseif ($Strict111) {
    throw "Missing ModelFin_111 leakage_audit.json"
  } else {
    Write-Host "WARN: ModelFin_111 leakage_audit.json is missing. This is allowed before training/package build, but not for final reporting." -ForegroundColor Yellow
  }
} else {
  Write-Host "INFO: ModelFin_111 package not found and -CheckModelFin111 not set; skipped ASSB-111 checks." -ForegroundColor Cyan
}

Write-Host "[check_ModelFin107A_package] Done." -ForegroundColor Cyan
