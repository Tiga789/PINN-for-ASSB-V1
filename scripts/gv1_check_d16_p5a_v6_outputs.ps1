param(
    [string]$RunDir = "E:\XJTU battery dataset\_gv1_cache\xjtu_d16_p5a_V4_D15_existing_on_ALL55"
)
$PredDir = Join-Path $RunDir "eval_full_profiles\predictions"
Write-Host "RunDir: $RunDir" -ForegroundColor Cyan
Write-Host "PredDir: $PredDir" -ForegroundColor Cyan
if (Test-Path $PredDir) {
    $files = Get-ChildItem $PredDir -Filter "*.npz" -File -ErrorAction SilentlyContinue
    Write-Host "prediction npz count: $($files.Count)" -ForegroundColor Yellow
    $gb = ($files | Measure-Object Length -Sum).Sum / 1GB
    Write-Host ("prediction size GB: {0:N3}" -f $gb) -ForegroundColor Yellow
} else {
    Write-Host "prediction npz count: 0" -ForegroundColor Red
}
$items = @(
    "D16_P5A_V6_FINAL_SCORECARD.json",
    "eval_full_profiles\D16_P5A_METRICS_BY_PROFILE.csv",
    "eval_full_profiles\D16_P5A_BATCH_METRICS.csv",
    "eval_full_profiles\D16_P5A_PROTOCOL_METRICS.csv",
    "eval_full_profiles\D16_P5A_V6_ROUTING_TABLE.csv",
    "eval_full_profiles\D16_P5A_FAILURES.json"
)
foreach ($rel in $items) {
    $p = Join-Path $RunDir $rel
    if (Test-Path $p) { Write-Host "FOUND: $p" -ForegroundColor Green } else { Write-Host "MISSING: $p" -ForegroundColor Red }
}
