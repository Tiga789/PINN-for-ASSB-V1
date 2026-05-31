# GV1 D11C2 Flag-aware Metadata Input Patch Package

This package generates a design-only metadata input patch for the D11C2 stage.

## Files

```text
scripts/gv1_d11c2_metadata_input_patch.py
scripts/run_gv1_d11c2_metadata_input_patch.ps1
manifests/d11c2_expected_outputs.json
docs/D11C2_SCOPE.md
README_GV1_D11C2.md
```

## Run

From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d11c2_metadata_input_patch.ps1"
```

Or directly with Python:

```powershell
D:\Anaconda\envs\torchgpu\python.exe "scripts\gv1_d11c2_metadata_input_patch.py" --dry_run
```

## Default output

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11c2_metadata_input_patch_design
```

## Guardrails

- Keep D9.6/D9.5.1 frozen.
- Keep B1_2C battery-8 flagged/excluded.
- Do not launch 24-profile 200ks training.
- Do not adopt hard guard/component clamp/D10-P3 correction.
- Use the generated manifest only after manual review.
