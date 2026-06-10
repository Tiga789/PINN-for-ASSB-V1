# D15-P4D targeted fix: Batch-5_battery-8 weak anode gradient

Purpose: regenerate **only** `Batch-5_battery-8` with a stronger negative-electrode radial-gradient prior, audit it, and overwrite only that cell's old P4D soft-label output after a candidate passes.

This package does **not** include any `gv1/` files and will not overwrite `gv1/__init__.py`.

Default affected final output:

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_softlabels_p2dlite_rg_v1_d15p4d_batch56_remaining14/profiles/Batch-5_battery-8
```

No other cell directory is modified.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File scripts\d15_p4d_target_b5b8_run_all.ps1 -AllowOverwrite
```

Optional backup of old target cell directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4d_target_b5b8_run_all.ps1 -AllowOverwrite -BackupOld
```

## Review output

Upload this file for review:

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_batch5_battery8_targeted_fix_review.zip
```

The main summary is:

```text
E:/XJTU battery dataset/_gv1_cache/xjtu_d15_p4d_batch5_battery8_targeted_fix/D15_P4D_BATCH5_BATTERY8_TARGETED_FIX_SUMMARY.json
```

## What this can claim

If PASS: `Batch-5_battery-8` has been regenerated and now passes targeted radial-gradient audit.

Do not claim experimental internal-state truth or held-out generalization from this targeted fix.
