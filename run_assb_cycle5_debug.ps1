<#
ASSB cycle5-only physics-only debug runner.
Place this file in the project root, then run one stage at a time.

Examples:
  powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage sanity
  powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage smoke
  powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage baseline
  powershell -ExecutionPolicy Bypass -File .\run_assb_cycle5_debug.ps1 -Stage eval61

This script never enables data loss. The input files all use alpha : 1.0 1.0 0.0 0.0
and w_*_dat : 0.0.
#>

param(
    [ValidateSet("sanity", "smoke", "eval60", "baseline", "eval61", "boundary", "eval62", "lbfgs", "eval63", "all")]
    [string]$Stage = "sanity",

    [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",

    [string]$RepoRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",

    [string]$OcpDir = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

$SoftLabelDir = Join-Path $RepoRoot "Data\assb_soft_labels_cycle5_v3"
$env:ASSB_SOFT_LABEL_DIR = $SoftLabelDir
$env:ASSB_OCP_DIR = $OcpDir
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "RepoRoot            = $RepoRoot"
Write-Host "Python              = $Python"
Write-Host "ASSB_SOFT_LABEL_DIR = $env:ASSB_SOFT_LABEL_DIR"
Write-Host "ASSB_OCP_DIR        = $env:ASSB_OCP_DIR"
Write-Host "Stage               = $Stage"

function Run-Sanity {
    & $Python .\assb_cycle5_softlabel_sanity.py `
        --soft_label_dir "Data\assb_soft_labels_cycle5_v3" `
        --output_dir "Eval_cycle5_softlabel_sanity"
}

function Run-TrainSmoke {
    & $Python .\main.py -i input_assb_cycle5_debug_smoke
}

function Run-TrainBaseline {
    & $Python .\main.py -i input_assb_cycle5_debug_sgd
}

function Run-TrainBoundary {
    if (-not (Test-Path ".\ModelFin_61\best.pt")) {
        throw "ModelFin_61\best.pt not found. Run -Stage baseline first."
    }
    & $Python .\main.py -i input_assb_cycle5_debug_boundary
}

function Run-TrainLbfgs {
    if (-not (Test-Path ".\ModelFin_62\best.pt")) {
        throw "ModelFin_62\best.pt not found. Run -Stage boundary first."
    }
    & $Python .\main.py -i input_assb_cycle5_debug_lbfgs
}

function Run-Eval([string]$ModelDir, [string]$OutDir) {
    if (-not (Test-Path ".\$ModelDir")) {
        throw "$ModelDir not found. Train it before evaluating."
    }
    & $Python .\evaluate_assb_cycle5_pinn_vs_softlabels_debug.py `
        --model_dir $ModelDir `
        --soft_label_dir "Data\assb_soft_labels_cycle5_v3" `
        --ocp_dir $OcpDir `
        --output_dir $OutDir `
        --debug_print_first_batch
}

switch ($Stage) {
    "sanity"   { Run-Sanity }
    "smoke"    { Run-Sanity; Run-TrainSmoke; Run-Eval "ModelFin_60" "EvalFin_60_cycle5_smoke" }
    "eval60"   { Run-Eval "ModelFin_60" "EvalFin_60_cycle5_smoke" }
    "baseline" { Run-Sanity; Run-TrainBaseline; Run-Eval "ModelFin_61" "EvalFin_61_cycle5_baseline" }
    "eval61"   { Run-Eval "ModelFin_61" "EvalFin_61_cycle5_baseline" }
    "boundary" { Run-TrainBoundary; Run-Eval "ModelFin_62" "EvalFin_62_cycle5_boundary" }
    "eval62"   { Run-Eval "ModelFin_62" "EvalFin_62_cycle5_boundary" }
    "lbfgs"    { Run-TrainLbfgs; Run-Eval "ModelFin_63" "EvalFin_63_cycle5_lbfgs" }
    "eval63"   { Run-Eval "ModelFin_63" "EvalFin_63_cycle5_lbfgs" }
    "all"      {
        Run-Sanity
        Run-TrainSmoke
        Run-Eval "ModelFin_60" "EvalFin_60_cycle5_smoke"
        Run-TrainBaseline
        Run-Eval "ModelFin_61" "EvalFin_61_cycle5_baseline"
        Run-TrainBoundary
        Run-Eval "ModelFin_62" "EvalFin_62_cycle5_boundary"
        Run-TrainLbfgs
        Run-Eval "ModelFin_63" "EvalFin_63_cycle5_lbfgs"
    }
}
