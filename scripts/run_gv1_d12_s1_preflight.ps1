param(
  [string]$CacheRoot = "E:\XJTU battery dataset\_gv1_cache",
  [int]$ExpectedEpochs = 100,
  [double]$ExpectedTimeWindowS = 40000,
  [int]$ExpectedMaxTimePoints = 1024,
  [int]$ExpectedBatchSize = 512,
  [int]$ExpectedProfiles = 3,
  [string]$TargetProfileId = "Batch-1_2C_battery-8"
)
$ErrorActionPreference = "Stop"
$CmdRoot = Join-Path $CacheRoot "xjtu_batch134_d12_s1_metadata_ablation_commands"
$SummaryPath = Join-Path $CmdRoot "d12_s1_command_preparation_summary.json"
if (!(Test-Path $SummaryPath)) { throw "Missing D12-S1 command summary: $SummaryPath" }
$Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
if ($Summary.target_included -eq $true) { throw "STOP: target battery-8 is included in D12-S1 selected profiles." }
if ($Summary.selected_profile_ids -contains $TargetProfileId) { throw "STOP: selected profile list contains battery-8." }
if ([int]$Summary.profile_limit -ne $ExpectedProfiles) { throw "Expected $ExpectedProfiles profiles, got $($Summary.profile_limit)." }
if ([int]$Summary.epochs -ne $ExpectedEpochs) { throw "Expected epochs=$ExpectedEpochs, got $($Summary.epochs)." }
if ([double]$Summary.time_window_s -ne [double]$ExpectedTimeWindowS) { throw "Expected time_window_s=$ExpectedTimeWindowS, got $($Summary.time_window_s)." }
if ([int]$Summary.max_time_points -ne $ExpectedMaxTimePoints) { throw "Expected max_time_points=$ExpectedMaxTimePoints, got $($Summary.max_time_points)." }
if ([int]$Summary.batch_size -ne $ExpectedBatchSize) { throw "Expected batch_size=$ExpectedBatchSize, got $($Summary.batch_size)." }

$Scripts = Get-ChildItem $CmdRoot -Filter "run_d12_s1_metadata_*_$($ExpectedProfiles)profile.generated.ps1"
if ($Scripts.Count -ne 3) { throw "Expected 3 D12-S1 generated scripts, found $($Scripts.Count)." }

$Hits = Select-String -Path ($Scripts.FullName) -Pattern "--epochs|--time_window_s|--max_time_points|--batch_size|_200ks|40000|200000|8192|2048"
$Hits | ForEach-Object { "{0}:{1}: {2}" -f (Split-Path $_.Path -Leaf), $_.LineNumber, $_.Line.Trim() }

$Bad = $Hits | Where-Object {
  $_.Line -match "--epochs\s+40000|--time_window_s\s+200000|--max_time_points\s+8192|--batch_size\s+2048|_200ks"
}
if ($Bad) {
  $Bad | ForEach-Object { "BAD: {0}:{1}: {2}" -f (Split-Path $_.Path -Leaf), $_.LineNumber, $_.Line.Trim() }
  throw "STOP: D12-S1 generated scripts contain old long-run parameters."
}

$Required = @(
  "--epochs $ExpectedEpochs",
  "--batch_size $ExpectedBatchSize",
  "--max_time_points $ExpectedMaxTimePoints",
  "--time_window_s $([double]$ExpectedTimeWindowS).0"
)
$AllLines = $Hits | ForEach-Object { $_.Line }
foreach ($Needle in $Required) {
  if (-not (($AllLines | Select-String -Pattern ([regex]::Escape($Needle))) )) {
    throw "Expected parameter not found in generated scripts: $Needle"
  }
}

"D12-S1 preflight PASS: generated scripts are strict smoke commands."
