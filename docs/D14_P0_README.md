# D14-P0 Freeze / No-Regression Audit Package

This package replaces the earlier incorrect placeholder package. It is the real D14-P0 package for **freezing the accepted XJTU voltage mainline and protecting the existing ASSB baseline**.

D14-P0 does **not** train a new model and does **not** generate SOH labels or P2D state labels. Its job is to verify that the local repository and cache are still consistent with the accepted state before D14-P1/P2 experiments.

## What D14-P0 checks

1. GV1 D9.6/D9.5.1 mainline files exist:
   - `gv1/model.py`
   - `gv1/output_transform.py`
   - `gv1/profile_adaptive.py`
   - `gv1/losses.py`
   - `gv1/trainer.py`
   - `scripts/gv1_train_conditioned_pinn.py`

2. The mainline is not obviously polluted by failed branches:
   - hard voltage clamp enabled by default
   - metadata_on as default training mode
   - high-safe / component-guard failed branch markers as active defaults
   - unflagging or including battery-8 in the mainline

3. The ASSB five-target engineering wrapper is still present:
   - `ModelFin_112_deterministic_wrapper`
   - `EvalFin_112_deterministic_wrapper`

4. XJTU accepted evidence is present:
   - D10-P1 non-outlier 23-profile 200ks output
   - D12-S1K 23x200ks scorecard output

5. Battery-8 remains flagged/excluded from mainline statistics.

6. A fingerprint snapshot is generated for future no-regression checks.

## Installation / overwrite paths

Copy the package contents into your repository root:

```text
C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1
```

Expected new files after copy:

```text
scripts/gv1_d14_p0_freeze_mainline_audit.py
scripts/gv1_d14_p0_verify_outputs.py
scripts/run_gv1_d14_p0_freeze_audit.ps1
configs/d14_p0_expected_mainline.json
docs/D14_P0_README.md
```

These files do not modify the old ASSB mainline and do not overwrite the GV1 model/trainer/loss files.

## Recommended command

From PowerShell:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File .\scripts\run_gv1_d14_p0_freeze_audit.ps1 `
  -ProjectRoot "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  -CacheRoot "E:\XJTU battery dataset\_gv1_cache" `
  -OutputDir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit" `
  -StrictCache `
  -StrictASSB
```

If you do not want missing cache directories to fail the run, remove `-StrictCache`. If you do not want missing ASSB wrapper directories to fail the run, remove `-StrictASSB`.

## Expected output

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit\D14_P0_FREEZE_AUDIT.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit\D14_P0_FREEZE_AUDIT.md
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit\D14_P0_FILE_FINGERPRINTS.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit\D14_P0_BASELINE_FINGERPRINT.json
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit\D14_P0_SCORECARD_INDEX.csv
E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit\D14_P0_FREEZE_AUDIT_console.log
```

## How to interpret status

- `PASS`: D14-P0 freeze audit is clean.
- `WARN`: Usually acceptable but needs manual inspection. Common examples: git info unavailable, candidate names not automatically found in a custom scorecard filename, optional legacy file missing.
- `FAIL`: Do not proceed to D14-P1/P2 until fixed. Typical failures: core GV1 file missing, hard clamp active by default, metadata_on default, battery-8 unflagged, strict cache directory missing.

## Future fingerprint comparison

After a clean audit, keep:

```text
D14_P0_BASELINE_FINGERPRINT.json
```

For future no-regression comparison:

```powershell
python .\scripts\gv1_d14_p0_freeze_mainline_audit.py `
  --project-root "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1" `
  --cache-root "E:\XJTU battery dataset\_gv1_cache" `
  --output-dir "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit_rerun" `
  --baseline-fingerprint "E:\XJTU battery dataset\_gv1_cache\xjtu_d14_p0_freeze_audit\D14_P0_BASELINE_FINGERPRINT.json"
```

