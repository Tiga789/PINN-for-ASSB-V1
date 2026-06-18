# D17-G6.4 P4D/GEO provenance audit

Purpose: stop training/patching until the exact provenance of problematic P4D/GEO soft labels is known.

This package performs a light metadata-only audit:

- no training;
- no checkpoint selection;
- no radial solver;
- no full-array load;
- no 55-cell audit;
- only reads sidecar JSON, NPZ headers, small scalar/string metadata, and local generator/config files.

It is intended after G6.3 formula forensics returned `patch_ready=false`, meaning the candidate current-integral formulas did not reproduce `Batch-6_GEO_battery-2/5` inventory trajectories.

## Run

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

python scripts\d17_g64_p4d_provenance_audit.py `
  --project_root "." `
  --config configs/d17_g64_p4d_provenance_audit.json `
  --split_manifest "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_pinn_rebuild/split/d17_split_manifest.json" `
  --g0_profile_semantics_csv "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g0_generator_equivalence_audit/D17_G0_PROFILE_SEMANTICS.csv" `
  --out_dir "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g64_p4d_provenance_audit" `
  --profile_contains "Batch-6_GEO_battery-2" `
  --profile_contains "Batch-6_GEO_battery-5"
```

Inspect:

```powershell
python scripts\d17_g64_inspect_summary.py `
  "E:/XJTU battery dataset/_gv1_cache/xjtu_d17_g/g64_p4d_provenance_audit/D17_G64_P4D_PROVENANCE_AUDIT_SUMMARY.json"
```

## Interpretation

- `provenance_ready=true`: selected soft labels contain enough script/config/hash/source metadata to attempt a version-matched formula equivalence test.
- `provenance_ready=false`: do not train, do not patch, do not run full G6. Read the profile details and identify the missing or mismatched provenance.

Important output files:

- `D17_G64_P4D_PROVENANCE_AUDIT_SUMMARY.json`
- `D17_G64_PROFILE_PROVENANCE_DETAILS.json`
- `D17_G64_LOCAL_GENERATOR_CODE_SCAN.json`
- `D17_G64_LOCAL_P4D_CONFIG_SCAN.json`
- `D17_G64_PROFILE_PROVENANCE_INDEX.csv`
