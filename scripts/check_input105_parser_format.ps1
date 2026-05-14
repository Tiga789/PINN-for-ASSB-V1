$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$InputFile = ".\input_assb_cycles5to522_v2_massclosed_ID105_potentialGauge"
Write-Host "Checking input parser format: $InputFile"
$bad = @()
$i = 0
Get-Content $InputFile | ForEach-Object {
  $i++
  $line = $_.Trim()
  if ($line.Length -gt 0) {
    $count = ([regex]::Matches($line, ':')).Count
    if ($count -ne 1) {
      $bad += "Line $i has $count ':' delimiters: $_"
    }
  }
}
if ($bad.Count -gt 0) {
  $bad | ForEach-Object { Write-Host $_ }
  throw "Input parser check failed. Every non-empty input line must contain exactly one ':' delimiter."
}
Write-Host "OK: every non-empty input line contains exactly one ':' delimiter."
