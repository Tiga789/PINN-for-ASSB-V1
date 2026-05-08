$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$patch = Join-Path $repo "patch_files"
if (!(Test-Path $patch)) { throw "patch_files not found. Please extract the zip into the project root first." }

function Backup-And-Copy($relativePath, $backupRelativePath) {
  $src = Join-Path $patch $relativePath
  $dst = Join-Path $repo $relativePath
  $backup = Join-Path $repo $backupRelativePath
  if (!(Test-Path $src)) { throw "Missing patch file: $src" }
  $dstDir = Split-Path -Parent $dst
  if (!(Test-Path $dstDir)) { New-Item -ItemType Directory -Force $dstDir | Out-Null }
  if ((Test-Path $dst) -and !(Test-Path $backup)) {
    Copy-Item $dst $backup -Force
    Write-Host "Backed up $relativePath -> $backupRelativePath"
  }
  Copy-Item $src $dst -Force
  Write-Host "Installed $relativePath"
}

# These three are wrappers: they need the pre102 backup files.
Backup-And-Copy "util\_rescale.py" "util\_rescale__pre102.py"
Backup-And-Copy "util\_losses.py" "util\_losses__pre102.py"
Backup-And-Copy "util\init_pinn.py" "util\init_pinn__pre102.py"

# Direct replacements / additions for ModelFin_102 smoke using existing full soft labels.
Backup-And-Copy "util\spm_assb_train_discharge.py" "util\spm_assb_train_discharge__pre102_smoke.py"
Backup-And-Copy "evaluate_assb_pinn_vs_softlabels.py" "evaluate_assb_pinn_vs_softlabels__pre102_smoke.py"
Backup-And-Copy "input_assb_cycles5to522_v4_continuous_ID102" "input_assb_cycles5to522_v4_continuous_ID102__pre102_smoke"
Backup-And-Copy "input_assb_ModelFin102_102_smoke" "input_assb_ModelFin102_102_smoke__pre102_smoke"

Write-Host "Install finished. Run .\scripts\check_102_smoke_existing_softlabels.ps1 next."
