# D15-P4A-fix: replay profile canonicalization / readiness audit

This package fixes the D15-P4A audit issue where cache directory names such as `xjtu_batch134_replay_profiles` or `xjtu_d14_p3b_batch56_replay_smoke` could be incorrectly canonicalized as `Batch-134_*` or `Batch-56_*`.

## Packaging rule

This package contains **no** `gv1/` files. It will not overwrite `gv1/__init__.py` or any existing GV1 module.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p4a_fix_run_all.ps1
```

If the output directory already exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4a_fix_run_all.ps1 -AllowOverwrite
```

## Upload for review

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_fix_results_for_review.zip
```
