# D15-P4C · Batch-5/6 remaining 14-cell replay-profile completion

This clean package completes replay profiles for the remaining 14 XJTU cells that still lack P2Dlite-RG soft labels:

- Batch-5 battery-1/2/3/4/5/6/8
- Batch-6 battery-1/2/4/5/6/7/8

D15-P4C does **not** generate P2Dlite-RG soft labels and does **not** train any neural network. It prepares the missing replay profiles for the next stage, D15-P4D.

## Important boundaries

- No `gv1/` files are included in this package.
- This package will not overwrite `gv1/__init__.py`.
- This package does not contain `__pycache__` or `.pyc` files.
- GPU is not expected to be used by replay-profile construction.
- The included resource smoke explicitly reports that the current P2Dlite-RG soft-label generator is NumPy/CPU unless rewritten.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p4c_run_all.ps1
```

If output already exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4c_run_all.ps1 -AllowOverwrite
```

Use more workers if raw MAT parsing is slow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4c_run_all.ps1 -AllowOverwrite -Workers 4
```

## Outputs

Default outputs:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch56_remaining14_replay_profiles_d15p4c
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4c_batch56_remaining14_replay_audit
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4c_softlabel_resource_smoke
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4c_batch56_replay_completion_scorecard
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4c_results_for_review.zip
```

Upload this file for review:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4c_results_for_review.zip
```

## What I should check next

The review zip contains:

- `D15_P4C_PREFLIGHT_REPORT.json`
- `D15_P4C_RAW_TARGET_MANIFEST.csv`
- `D15_P4C_REPLAY_BUILD_REPORT.json`
- `xjtu_batch56_remaining14_replay_profile_manifest.csv`
- `D15_P4C_REPLAY_AUDIT_SUMMARY.json`
- `D15_P4C_REPLAY_AUDIT_BY_PROFILE.csv`
- `D15_P4C_RESOURCE_SMOKE_REPORT.json`
- `D15_P4C_FINAL_SCORECARD.json`

If D15-P4C is PASS, the next stage is D15-P4D: generating P2Dlite-RG soft labels for these 14 cells.
