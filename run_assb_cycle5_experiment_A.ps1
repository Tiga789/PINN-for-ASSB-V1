param(
    [string]$RepoRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1",
    [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
    [string]$SoftLabelDir = "Data\assb_soft_labels_cycle5_v3",
    [string]$OcpDir = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs",
    [ValidateSet("audit", "train", "eval", "all")]
    [string]$Stage = "all"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$env:ASSB_SOFT_LABEL_DIR = (Resolve-Path $SoftLabelDir).Path
$env:ASSB_OCP_DIR = $OcpDir

Write-Host "RepoRoot            = $RepoRoot"
Write-Host "Python              = $Python"
Write-Host "ASSB_SOFT_LABEL_DIR = $env:ASSB_SOFT_LABEL_DIR"
Write-Host "ASSB_OCP_DIR        = $env:ASSB_OCP_DIR"
Write-Host "Stage               = $Stage"

function Run-Audit {
    & $Python .\assb_cycle5_integration_mass_audit.py `
        --soft_label_dir $SoftLabelDir `
        --ocp_dir $OcpDir `
        --output_dir Eval_cycle5_A_integration_mass_audit
}

function Run-Train {
    & $Python .\main.py -i input_assb_cycle5_A_no_seccons
}

function Run-Eval {
    if (-not (Test-Path .\ModelFin_81)) {
        throw "ModelFin_81 not found. Run Stage train first."
    }
    & $Python .\evaluate_assb_pinn_vs_softlabels.py `
        --model_dir ModelFin_81 `
        --soft_label_dir $SoftLabelDir `
        --ocp_dir $OcpDir `
        --output_dir EvalFin_81_cycle5_A_no_seccons `
        --debug_print_first_batch
}

if ($Stage -eq "audit") { Run-Audit }
elseif ($Stage -eq "train") { Run-Train }
elseif ($Stage -eq "eval") { Run-Eval }
else {
    Run-Audit
    Run-Train
    Run-Eval
}
