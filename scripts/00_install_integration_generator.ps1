# Copy this patch into the project root without touching training files.
# Run this script from inside the unzipped patch folder.
$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
$PatchRoot = Split-Path -Parent $PSScriptRoot

New-Item -ItemType Directory -Force (Join-Path $ProjectRoot "integration_spm") | Out-Null
Copy-Item (Join-Path $PatchRoot "integration_spm\generate_assb_softlabel_allcycle.py") (Join-Path $ProjectRoot "integration_spm\generate_assb_softlabel_allcycle.py") -Force
Copy-Item (Join-Path $PatchRoot "tools\inspect_assb_softlabel_solution.py") (Join-Path $ProjectRoot "inspect_assb_softlabel_solution.py") -Force

Write-Host "Installed: integration_spm\generate_assb_softlabel_allcycle.py"
Write-Host "Installed: inspect_assb_softlabel_solution.py"
Write-Host "Project root: $ProjectRoot"
