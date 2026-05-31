# GV1 D11-C flag-aware metadata ablation design package

This package implements **D11-C: flag-aware metadata ablation design-only**.

It reads the D10-P5 policy, D11-B feature-distance audit, and GV1 training-ready manifest. It then generates a flag-aware profile metadata manifest, candidate route matrix, feature-group summary, and guardrail checklist.

It **does not train a model** and **does not modify** the D9.6/D9.5.1 mainline.

## Files

```text
scripts/gv1_d11_c_flag_aware_metadata_ablation.py
scripts/gv1_d11_c_flag_aware_metadata_ablation_design.py
scripts/run_gv1_d11_c_flag_aware_metadata_ablation.ps1
scripts/run_gv1_d11_c_flag_aware_metadata_ablation_design.ps1
manifests/d11_c_expected_outputs.json
docs/D11_C_SCOPE.md
README_GV1_D11_C.md
package_info.json
```

The `*_design.py` and `*_design.ps1` files are compatibility aliases. The preferred run script is:

```text
scripts/run_gv1_d11_c_flag_aware_metadata_ablation.ps1
```

## Run

From the project root:

```powershell
cd "C:\Users\Tiga_QJW\Desktop\ASSB_Scheme_V1\PINN-for-ASSB-V1"

powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d11_c_flag_aware_metadata_ablation.ps1"
```

Compatibility command:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts\run_gv1_d11_c_flag_aware_metadata_ablation_design.ps1"
```

Default output directory:

```text
E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_c_flag_aware_metadata_ablation_design
```

## Main outputs

```text
D11_C_RECOMMENDATION.md
d11_c_flag_aware_metadata_ablation_summary.json
d11_c_profile_metadata_manifest.csv
d11_c_candidate_routes.csv
d11_c_guardrail_checklist.csv
d11_c_feature_group_summary.csv
d11_c_metadata_patch_design.md
```

## Check outputs

```powershell
Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_c_flag_aware_metadata_ablation_design\D11_C_RECOMMENDATION.md" -Raw

Get-Content "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_c_flag_aware_metadata_ablation_design\d11_c_flag_aware_metadata_ablation_summary.json" -Raw

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_c_flag_aware_metadata_ablation_design\d11_c_candidate_routes.csv" |
  Format-Table route_id,route_name,status,allowed,risk -AutoSize

Import-Csv "E:\XJTU battery dataset\_gv1_cache\xjtu_batch134_d11_c_flag_aware_metadata_ablation_design\d11_c_profile_metadata_manifest.csv" |
  Where-Object { $_.is_B1_2C_battery8_target -eq "1" } |
  Format-Table profile_id,batch_id,battery_id,protocol,metadata_group,d11c_role -AutoSize
```

## Expected verdict

```text
d11_c_design_only_flag_aware_metadata_ablation_plan_ready
```

If context files or manifests are missing, expected fallback verdict:

```text
d11_c_incomplete_context_keep_design_only_do_not_train
```

## Guardrails

- Keep `GV1 D9.6 / D9.5.1 trend-first warmup rare-regime` frozen.
- Keep `B1_2C battery-8` flagged/excluded while unresolved.
- Do not run direct 24-profile 200ks mainline claim from this package.
- Do not use hard voltage clamp or component clamp repair.
- Do not use same-window target voltage features as predictive metadata for voltage prediction.
- Do not treat this package as a training implementation; it is a D11-C design/audit artifact generator.
