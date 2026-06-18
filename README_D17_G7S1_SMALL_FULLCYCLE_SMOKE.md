# D17-G7-S1 small full-cycle smoke

This package trains a small full-cycle coverage smoke candidate with real-time console progress. It does not run full G6, does not use frozen-test labels, and does not promote directly to S2.

## Run S1

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g7s1_small_fullcycle_smoke.py `
  --config configs/d17_g7s1_small_fullcycle_smoke.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --s0_summary "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s0_fullcycle_sampling_audit/D17_G7S0_FULLCYCLE_SAMPLING_AUDIT_SUMMARY.json" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s1_small_fullcycle_smoke" `
  --train_profile_count 8 `
  --validation_profile_count 2 `
  --internal_heldout_count 2 `
  --max_time_points 4096 `
  --time_window_s 0 `
  --epochs 180 `
  --lr 0.0006 `
  --batch_size 2048 `
  --progress_every 1 `
  --eval_every 10 `
  --device auto
```

## Inspect S1

```powershell
python scripts\d17_g7s1_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g7s1_small_fullcycle_smoke/D17_G7S1_SMALL_FULLCYCLE_SMOKE_SUMMARY.json"
```

## Watch progress from a second PowerShell

```powershell
$h = "E:\XJTU battery dataset\_gv1_cache\xjtu_d17_g\g7s1_small_fullcycle_smoke\D17_G7S1_training_progress_live.csv"
while ($true) {
  Clear-Host
  if (Test-Path $h) { Get-Content $h -Tail 8 } else { Write-Host "No progress CSV yet." }
  Start-Sleep -Seconds 15
}
```

## Required next check before S2

If `selected_cycle_check_ready=true`, run G6F selected-cycle dense checks with this S1 candidate. Only consider G7-S2-mini if selected-cycle dense metrics improve substantially over G21/G4.
