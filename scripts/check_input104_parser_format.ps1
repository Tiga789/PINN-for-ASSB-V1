$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$InputFile = ".\input_assb_cycles5to522_v2_massclosed_ID104_radial010"
Write-Host "Checking input parser format:" $InputFile
$bad = @()
$i = 0
Get-Content $InputFile | ForEach-Object {
  $i += 1
  if ($_.Trim().Length -eq 0) { return }
  $n = ($_.ToCharArray() | Where-Object {$_ -eq ':'}).Count
  if ($n -ne 1) { $bad += "line $i has $n colon(s): $_" }
}
if ($bad.Count -gt 0) {
  $bad | ForEach-Object { Write-Host $_ }
  throw "Input parser format check failed."
}
Write-Host "OK: every non-empty input line contains exactly one ':' delimiter."
