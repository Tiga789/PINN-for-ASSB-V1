param(
    [string]$PythonExe = "python"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ProjectRoot\VERIFY_D18_S0_S1_FIX_V2.ps1" -PythonExe $PythonExe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
