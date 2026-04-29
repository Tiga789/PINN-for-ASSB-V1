param(
    [string]$Python = "D:\Anaconda\envs\torchgpu\python.exe",
    [string]$SoftLabelDir = "Data\assb_soft_labels_cycle5_v3",
    [string]$OcpDir = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\ocp_estimation_outputs",
    [string]$ModelDir = "ModelFin_80",
    [string]$OutputDir = "EvalFin_80_cycle5_provenance_evalfix"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$env:ASSB_SOFT_LABEL_DIR = (Resolve-Path $SoftLabelDir).Path
$env:ASSB_OCP_DIR = $OcpDir

& $Python .\evaluate_assb_pinn_vs_softlabels.py `
  --model_dir $ModelDir `
  --soft_label_dir $SoftLabelDir `
  --ocp_dir $OcpDir `
  --output_dir $OutputDir `
  --debug_print_first_batch
