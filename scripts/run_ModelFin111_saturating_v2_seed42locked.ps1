param(
  [string]$PythonExe = "D:\Anaconda\envs\torchgpu\python.exe",
  [string]$ProjectRoot = ".",
  [string]$InputFile = "input_assb111_strict30_saturating_v2_seed42locked",
  [string]$BaseRunScript = "scripts\run_ModelFin111_strict30.ps1",
  [string]$SelectionDir = "EvalFin_111_seed42_locked_selection",
  [string]$CandidateRoot = "ASSB111_seed42_locked_candidates",
  [string]$FinalWorkDir = "Data\assb111_seed42_locked",
  [string]$FinalModelDir = "ModelFin_111_seed42_locked",
  [string]$FinalEvalDir = "EvalFin_111_seed42_locked_strict30_test70",
  [int]$Seed = 42,
  [string]$Device = "cuda",
  [switch]$AllowCPU,
  [switch]$SoftFailEval,
  [switch]$RunOverdecayDiagnostics,
  [switch]$ForceClean,
  [switch]$SkipCandidateRuns,
  [switch]$SkipFinalCopy
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $ProjectRoot

# Strictly avoid stale ASSB env vars from earlier cycle5/all-cycle workflows.
Remove-Item Env:ASSB_SOFT_LABEL_SUMMARY -ErrorAction SilentlyContinue
Remove-Item Env:ASSB_SOFT_LABEL_DIR -ErrorAction SilentlyContinue
Remove-Item Env:ASSB_OCP_DIR -ErrorAction SilentlyContinue

function Require-Path([string]$Path) {
  if (-not (Test-Path $Path)) { throw "Required path not found: $Path" }
}

function Script-HasParam([string]$ScriptPath, [string]$ParamName) {
  if (-not (Test-Path $ScriptPath)) { return $false }
  $pat = "`$" + [Regex]::Escape($ParamName) + "\b"
  return (Select-String -Path $ScriptPath -Pattern $pat -Quiet)
}

function Add-ParamIfSupported([object[]]$Args, [string]$ScriptPath, [string]$Name, [object]$Value) {
  if (Script-HasParam $ScriptPath $Name) {
    if ($Value -is [bool]) {
      if ($Value) { return $Args + @("-$Name") }
      return $Args
    }
    return $Args + @("-$Name", $Value)
  }
  return $Args
}

function Invoke-External([string]$Name, [string]$Exe, [object[]]$Args) {
  Write-Host "`n==== $Name ====" -ForegroundColor Cyan
  Write-Host ($Exe + " " + ($Args -join " ")) -ForegroundColor DarkGray
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Require-Path $InputFile
Require-Path $BaseRunScript
Require-Path "scripts\optimize_assb111_seed42_locked_trainval.py"
Require-Path "scripts\compare_assb111_seed42locked_candidates.py"
Require-Path "scripts\diagnose_assb111_seed42locked_selection_audit.py"

New-Item -ItemType Directory -Force $SelectionDir | Out-Null
New-Item -ItemType Directory -Force $CandidateRoot | Out-Null

Write-Host "ASSB-111 seed42-locked saturating_v2 train/val-only optimization" -ForegroundColor Green
Write-Host "ProjectRoot: $(Get-Location)"
Write-Host "InputFile:   $InputFile"
Write-Host "Seed:        $Seed"
Write-Host "Variant:     saturating_v2"
Write-Host "Selection:   train/val only; test metrics forbidden for candidate selection."

# Small, pre-declared grid. This is intentionally narrow; do not add ad hoc candidates after seeing test results.
$candidates = @(
  # c00 is the exact recovery baseline: original saturating_v2 seed42 behavior before the over-complex seed42locked changes.
  @{ tag="c00_repro_lr2e3_e5000";    lr=2e-3; wd=1e-5; epochs=5000; patience=600;  use_ema=$false; topk=$false; dropout=0.05 },
  # Narrow train/val-only small-optimization candidates. Do not add candidates after looking at test metrics.
  @{ tag="c01_lr1e3_e6000";          lr=1e-3; wd=1e-5; epochs=6000; patience=900;  use_ema=$false; topk=$false; dropout=0.05 },
  @{ tag="c02_lr5e4_e7000";          lr=5e-4; wd=1e-5; epochs=7000; patience=1200; use_ema=$false; topk=$false; dropout=0.05 },
  @{ tag="c03_lr5e4_e7000_ema";      lr=5e-4; wd=1e-5; epochs=7000; patience=1200; use_ema=$true;  topk=$false; dropout=0.05 },
  @{ tag="c04_lr1e3_e6000_topk";     lr=1e-3; wd=5e-6; epochs=6000; patience=900;  use_ema=$false; topk=$true;  dropout=0.05 }
)

if ($ForceClean) {
  foreach ($c in $candidates) {
    Remove-Item (Join-Path $CandidateRoot ("Data_" + $c.tag)), `
                (Join-Path $CandidateRoot ("Model_" + $c.tag)), `
                (Join-Path $CandidateRoot ("Eval_" + $c.tag)) `
                -Recurse -Force -ErrorAction SilentlyContinue
  }
  Remove-Item $SelectionDir, $FinalWorkDir, $FinalModelDir, $FinalEvalDir -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $SelectionDir | Out-Null
}

if (-not $SkipCandidateRuns) {
  foreach ($c in $candidates) {
    $workDir = Join-Path $CandidateRoot ("Data_" + $c.tag)
    $modelDir = Join-Path $CandidateRoot ("Model_" + $c.tag)
    $evalDir = Join-Path $CandidateRoot ("Eval_" + $c.tag)

    $args = @(
      $BaseRunScript,
      "-PythonExe", $PythonExe,
      "-ProjectRoot", ".",
      "-InputFile", $InputFile,
      "-WorkDir", $workDir,
      "-ModelDir", $modelDir,
      "-EvalDir", $evalDir,
      "-Epochs", $c.epochs,
      "-Device", $Device,
      "-Seed", $Seed,
      "-SOHModelVariant", "saturating_v2",
      "-SOHFloorPrior", 0.72,
      "-SOHNumericMin", 0.60,
      "-LR", $c.lr,
      "-WeightDecay", $c.wd,
      "-Patience", $c.patience,
      "-MinTrainR2ForBest", 0.990,
      "-MaxTrainMAEForBest", 0.0030,
      "-MaxValMAEForBest", 0.00150
    )
    if ($AllowCPU) { $args += "-AllowCPU" }
    if ($SoftFailEval) { $args += "-SoftFailEval" }
    if ($RunOverdecayDiagnostics) { $args += "-RunOverdecayDiagnostics" }

    # These optional switches are passed only if the modified base script supports them.
    $args = Add-ParamIfSupported $args $BaseRunScript "CandidateTag" $c.tag
    $args = Add-ParamIfSupported $args $BaseRunScript "UseEMA" $c.use_ema
    $args = Add-ParamIfSupported $args $BaseRunScript "TopKCheckpointAvg" $c.topk
    $args = Add-ParamIfSupported $args $BaseRunScript "Dropout" $c.dropout
    $args = Add-ParamIfSupported $args $BaseRunScript "SelectionMode" "visible_train_val_only"
    $args = Add-ParamIfSupported $args $BaseRunScript "ProtocolTag" "seed42_locked_trainval_only"

    Invoke-External ("candidate " + $c.tag) "powershell" @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File") + $args
  }
}

# Candidate selection is visible-only: it reads train_summary/leakage/config files, not final test metrics.
$candidateModels = @($candidates | ForEach-Object { Join-Path $CandidateRoot ("Model_" + $_.tag) })
$compareArgs = @(
  "scripts\compare_assb111_seed42locked_candidates.py",
  "--candidate_model_dirs"
) + $candidateModels + @(
  "--output_dir", $SelectionDir,
  "--input_file", $InputFile,
  "--selection_mode", "visible_train_val_only",
  "--require_no_test_history",
  "--max_val_mae", "0.00150",
  "--min_train_r2", "0.990"
)
Invoke-External "visible-only candidate comparison" $PythonExe $compareArgs

$auditArgs = @(
  "scripts\diagnose_assb111_seed42locked_selection_audit.py",
  "--selection_dir", $SelectionDir,
  "--candidate_root", $CandidateRoot,
  "--output_json", (Join-Path $SelectionDir "selection_audit.json")
)
Invoke-External "selection audit" $PythonExe $auditArgs

$selectedJson = Join-Path $SelectionDir "selected_candidate.json"
Require-Path $selectedJson
$selected = Get-Content $selectedJson | ConvertFrom-Json
Write-Host "`nSelected candidate: $($selected.candidate_tag)" -ForegroundColor Green
Write-Host "Selected model dir: $($selected.model_dir)"
Write-Host "Selection used test metrics: $($selected.selection_used_test_metrics)"

if (-not $SkipFinalCopy) {
  $selModel = [string]$selected.model_dir
  $selTag = [string]$selected.candidate_tag
  $selWork = Join-Path $CandidateRoot ("Data_" + $selTag)
  $selEval = Join-Path $CandidateRoot ("Eval_" + $selTag)

  Remove-Item $FinalWorkDir, $FinalModelDir, $FinalEvalDir -Recurse -Force -ErrorAction SilentlyContinue
  if (Test-Path $selWork)  { Copy-Item $selWork  $FinalWorkDir  -Recurse -Force }
  if (Test-Path $selModel) { Copy-Item $selModel $FinalModelDir -Recurse -Force }
  if (Test-Path $selEval)  { Copy-Item $selEval  $FinalEvalDir  -Recurse -Force }

  Copy-Item $selectedJson (Join-Path $FinalModelDir "selected_candidate.json") -Force -ErrorAction SilentlyContinue
  Copy-Item (Join-Path $SelectionDir "candidate_visible_score.csv") (Join-Path $FinalModelDir "candidate_visible_score.csv") -Force -ErrorAction SilentlyContinue
  Copy-Item (Join-Path $SelectionDir "selection_audit.json") (Join-Path $FinalModelDir "selection_audit.json") -Force -ErrorAction SilentlyContinue
}

Write-Host "`nASSB-111 seed42-locked run completed." -ForegroundColor Green
Write-Host "SelectionDir: $SelectionDir"
Write-Host "FinalModel:   $FinalModelDir"
Write-Host "FinalEval:    $FinalEvalDir"
Write-Host "Note: final test metrics are for reporting only; they were not used for selection."
