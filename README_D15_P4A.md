# D15-P4A · remaining 32-cell replay-profile audit

This package audits whether the remaining XJTU cells are ready for D15-P4B soft-label completion.
It does not generate soft labels and does not train a neural network.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"
powershell -ExecutionPolicy Bypass -File scripts\d15_p4a_run_all.ps1
```

If rerunning the same output directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\d15_p4a_run_all.ps1 -AllowOverwrite
```

## Upload for review

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_results_for_review.zip
```

## Main outputs

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_remaining32_replay_audit\D15_P4A_FINAL_SCORECARD.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_remaining32_replay_audit\D15_P4A_REMAINING32_CELL_MANIFEST.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d15_p4a_remaining32_replay_audit\D15_P4A_P4B_INPUT_MANIFEST.csv
```

## Status

- PASS: all 32 remaining cells have usable replay profiles.
- REVIEW: counts are right, but one or more replay profiles are missing or need attention.
- FAIL: raw or remaining counts are inconsistent.

Batch-1 battery-8 must remain flagged as the known outlier even if it is listed as P4B-ready.
